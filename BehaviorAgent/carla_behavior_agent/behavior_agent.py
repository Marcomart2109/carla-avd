# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import math
import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal
from overtake import Overtake 
from misc import *
import logging
import time

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='cautious', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._approach_speed = 10
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5
        self._blocked = False

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

        self._overtake = Overtake(self._vehicle, opt_dict)
        
        # Aggiungi una variabile per tenere traccia dello stato dell'offset laterale
        self._lateral_offset_active = False
        self._original_max_throttle = None  # To store the original max_throttle
        self._overtake_throttle_modified = False  # Flag to indicate if throttle was changed during overtake

        # Configurazione standard del logger
        self.logger = logging.getLogger('behavior_agent')
        self.logger.setLevel(logging.DEBUG)

        # Assicurati che la directory dei log esista
        import os
        log_dir = './carla_behavior_agent/logs/'
        os.makedirs(log_dir, exist_ok=True)

        # Create a file handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            f'{log_dir}behavior_agent.log',
            maxBytes=5*1024*1024,
            backupCount=5,
            mode="w"
        )
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set it for the handler
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(self.formatter)
        
        # Add the handler to the logger 
        self.logger.addHandler(file_handler)
        
        # Imposta propagate a False per evitare log duplicati
        self.logger.propagate = False

        self.logger.info(f"Behavior agent initialized with behavior: {behavior}")
    
        # Riferimento allo streamer (da impostare esternamente)
        self._streamer = None
        
        # Aggiungi un log di test (verrà inviato quando viene impostato lo streamer)
        self._test_log_sent = False

        # Tracciamento oggetti per evitare log duplicati
        self._logged_objects = {
            "pedestrian": {},
            "bicycle": {},
            "vehicle": {},
            "obstacle": {},
            "traffic_light": False,
            "stop_sign": {}
        }
        self._log_timeout = 5.0  # Secondi prima di poter loggare nuovamente lo stesso oggetto

    ################################
    ########### RUN STEP ###########
    ################################

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        # --- Restoration logic for max_throttle ---
        # Check if an overtake was active and throttle modified, but overtake is now finished.
        if not self._overtake.in_overtake and self._overtake_throttle_modified:
            if self._original_max_throttle is not None:
                self._local_planner._vehicle_controller.max_throt = self._original_max_throttle
                self.logger.info(f"Overtake finished: Max throttle restored to {self._local_planner._vehicle_controller.max_throt:.2f}.")
                print(f"--- Overtake finished: Max throttle restored to {self._local_planner._vehicle_controller.max_throt:.2f}.")
            else:
                # Fallback: if original was somehow not stored, log a warning.
                # Consider setting to a default if necessary, e.g., self._local_planner._vehicle_controller.max_throt = 0.75 (controller's default)
                self.logger.warning("Overtake finished: _original_max_throttle was None. Throttle may not be correctly restored.")
            self._overtake_throttle_modified = False
            self._original_max_throttle = None # Clear stored value

        # Update the information regarding the ego vehicle
        self._update_information()

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        # Get the current waypoint of the ego vehicle
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        
        
        #######################################
        ######### RED LIGHTS SCENARIO #########
        #######################################
        

        # Check if the vehicle is near a traffic light
        # and if it is red
        if self.traffic_light_manager():
            # TODO: Add a behaviour for more than one traffic light
            return self.emergency_stop()


        ##############################################
        ####### Pedestrian Avoidance Behaviors #######
        ##############################################
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the car and the walker,
            # we use bounding boxes to calculate the actual distance
            distance = max(0,w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x))
            msg = f"[DEBUG - Pedestrian Avoidance] Pedestrian {walker} detected, distance 1: {distance}"
            print(msg)
            self.logger.critical(msg)
            
            # Emergency brake if the car is very close.
            if  distance < 6 * self._behavior.braking_distance:
                target_speed = self._vehicle.get_velocity().length() / 2
                self.set_target_speed(target_speed)
                msg = f"[INFO - Walker Obstacle Avoidance] Target speed set to:{target_speed}\n[INFO] Distance to static obstacle: {distance}"
                print(msg)
                # Complete stop only if extremely close
                if distance < 2 * self._behavior.braking_distance:
                    msg = f"[WARNING - Walker Obstacle Avoidance] Emergency stop due to static obstacle proximity"
                    print(msg)
                    self.logger.critical(msg)
                    return self.emergency_stop()



        ##################################################
        ########## Obstacle Avoidance Behaviors ##########
        ##################################################
        so_state, static_obstacle, so_distance = self.static_obstacle_avoid_manager(ego_vehicle_wp)
        if so_state is True:
            
            extent_obstacle = max(static_obstacle.bounding_box.extent.y, static_obstacle.bounding_box.extent.x)
            extent_vehicle = max(self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            msg = f"[Distance Information]so_distance: {so_distance}, extent_obstacle: {extent_obstacle}, extent_vehicle: {extent_vehicle}"
            print(msg)
            self.logger.info(msg)
            distance = so_distance - extent_obstacle - extent_vehicle
            msg = f"[Static Obstacle Avoidance] Static obstacle:{static_obstacle} detected, distance 1: {distance}"
            print(msg)
            self.logger.warning(msg)
        
            distance = compute_distance_from_center(actor1 =self._vehicle, actor2 = static_obstacle, distance=so_distance)
            msg = f"[Static Obstacle Avoidance] Static obstacle:{static_obstacle} detected, distance 2: {distance}"
            print(msg)
            self.logger.critical(msg)


            msg = f"Static obstacle detected, distance: {distance}"
            print(f"[DEBUG - Static Obstacle Avoidance] {msg}")

            overtake_path = self._overtake.run_step(
                object_to_overtake=static_obstacle, ego_vehicle_wp=ego_vehicle_wp, distance_same_lane=1, distance_other_lane=20, distance_from_object=so_distance, speed_limit = self._speed_limit
            )

            if overtake_path:
                print("[INFO - Static Obstacle Avoidance] Overtake path found")
                self.__update_global_plan(overtake_path=overtake_path)
            
            msg = f"[DEBUG - Static Obstacle Avoidance] in overtake: {self._overtake.in_overtake}, so_distance: {so_distance}"
            print(msg)
            self.logger.debug(msg)

            if not self._overtake.in_overtake and distance < 2 * self._behavior.braking_distance:
                self.set_target_speed(self._approach_speed)
                msg = f"[INFO - Static Obstacle Avoidance] Target speed set to:{self._approach_speed}\n[INFO] Distance to static obstacle: {distance}"
                print(msg)
                # Complete stop only if extremely close
                if distance < self._behavior.braking_distance:
                    msg = "[WARNING - Static Obstacle Avoidance] Emergency stop due to static obstacle proximity"
                    print(msg)
                    self.logger.critical(msg)
                    return self.emergency_stop()
                
                return self._local_planner.run_step()


        #################################################
        ######### Bicycle Avoidance Behaviors ##########
        ################################################
        bicycle_state, bicycle, b_distance = self.bycicle_avoid_manager(ego_vehicle_wp)
        
        if bicycle_state:
            # Ottieni l'ID della corsia del veicolo ego e della bicicletta
            ego_lane_id = ego_vehicle_wp.lane_id
            bicycle_wp = self._map.get_waypoint(bicycle.get_location())
            bicycle_lane_id = bicycle_wp.lane_id

            # Get the yaw of the ego vehicle and the vehicle in front.
            ego_current_yaw = self._vehicle.get_transform().rotation.yaw # Mantenere il segno per calcoli di angolo più precisi se necessario
            bicycle_current_yaw = bicycle.get_transform().rotation.yaw

            # Calcola la differenza di yaw, normalizzandola tra -180 e 180
            diff_yaw = (bicycle_current_yaw - ego_current_yaw + 180) % 360 - 180
            
            # Definisci una soglia per considerare una bicicletta "allineata" (es. +/- 20 gradi)
            # e una soglia per considerarla "attraversante" (es. al di fuori di +/- 45 gradi dalla direzione opposta)
            is_aligned_threshold = 20  # Gradi
            is_crossing_threshold = 45 # Gradi (per rilevare attraversamenti più diretti)

            # Controlla se i veicoli sono approssimativamente allineati (stessa direzione di marcia o quasi)
            if abs(diff_yaw) < is_aligned_threshold:                   
                # Controlla se la bicicletta è nella STESSA corsia del veicolo ego
                if ego_lane_id == bicycle_lane_id:
                    self.logger.info(f"Bicycle {bicycle.id} detected in the SAME lane ({ego_lane_id}), aligned. Distance: {b_distance:.2f}m.")
                    print(f"--- Bicycle {bicycle.id} detected in the same lane, aligned.")
                    # Controlla se la bicicletta è vicina al centro della corsia. In questo caso, l'agente sorpasserà la bicicletta.
                    if is_bicycle_near_center(vehicle_location=bicycle.get_location(), ego_vehicle_wp=ego_vehicle_wp) and get_speed(self._vehicle) < 0.1:
                        self.logger.info(f"Bicycle {bicycle.id} is near the center of our lane. Attempting to overtake.")
                        print("--- Bicycle is near the center of the lane! We can try to overtake it.")

                        overtake_path = self._overtake.run_step(
                            object_to_overtake=bicycle, ego_vehicle_wp=ego_vehicle_wp, distance_same_lane=1, distance_from_object=b_distance, speed_limit = self._speed_limit
                        )                                               
                        if overtake_path:
                            self.logger.info(f"Overtake path found for bicycle {bicycle.id}.")
                            self.__update_global_plan(overtake_path=overtake_path)
                        if not self._overtake.in_overtake: 
                             self.logger.warning(f"Overtake of bicycle {bicycle.id} not active. Emergency stop.")
                             print(f"--- Overtake of bicycle {bicycle.id} not active. Emergency stop.")
                             return self.emergency_stop()
                    else: 
                        self.logger.info(f"Bicycle {bicycle.id} is in our lane, but off-center. Applying lateral offset.")
                        print("--- Bicycle is in our lane, not centered. Applying lateral offset to avoid it.")
                        new_offset = -(2.5 * bicycle.bounding_box.extent.y + self._vehicle.bounding_box.extent.y)
                        self._local_planner.set_lateral_offset(new_offset)
                        self._lateral_offset_active = True
                        target_speed = min([
                            self._behavior.max_speed,
                            self._speed_limit - self._behavior.speed_lim_dist])
                        self._local_planner.set_speed(target_speed)
                        control = self._local_planner.run_step(debug=debug)
                else: # La bicicletta è allineata (stessa direzione) ma in una corsia DIVERSA
                    self.logger.info(f"Bicycle {bicycle.id} detected in a different lane (Ego: {ego_lane_id}, Bicycle: {bicycle_lane_id}), aligned. No offset applied.")
                    print(f"--- Bicycle detected in a different lane (Ego: {ego_lane_id}, Bicycle: {bicycle_lane_id}), aligned. No offset applied.")
                    if self._lateral_offset_active:
                        self.logger.info("Resetting previously active lateral offset as current bicycle is in a different lane.")
                        print("--- Resetting previamente active lateral offset.")
                        self._local_planner.set_lateral_offset(0.0)
                        self._lateral_offset_active = False
            
            # CASO: Bicicletta NON allineata (potenzialmente attraversante o in direzione opposta)
            else:
                self.logger.info(f"Bicycle {bicycle.id} is NOT ALIGNED (yaw_diff: {diff_yaw:.1f} deg). Distance: {b_distance:.2f}m. Speed: {get_speed(bicycle):.1f} km/h.")
                print(f"--- Bicycle {bicycle.id} is NOT ALIGNED (yaw_diff: {diff_yaw:.1f} deg). Distance: {b_distance:.2f}m.")

                # Verifica se la bicicletta sta effettivamente attraversando la nostra traiettoria
                # Questo è un controllo semplificato. Una predizione di traiettoria sarebbe più robusta.
                # Consideriamo "attraversante" se non è né allineata né in direzione opposta quasi parallela.
                is_potentially_crossing = abs(diff_yaw) > is_aligned_threshold and abs(abs(diff_yaw) - 180) > is_aligned_threshold 

                if is_potentially_crossing and b_distance < self._behavior.braking_distance * 3: # Distanza di sicurezza per reagire
                    self.logger.warning(f"Bicycle {bicycle.id} is potentially CROSSING at {b_distance:.2f}m. Initiating emergency stop or significant slowdown.")
                    print(f"--- Bicycle {bicycle.id} is potentially CROSSING at {b_distance:.2f}m. EMERGENCY STOP / SLOWDOWN.")
                    
                    # Se molto vicina, arresto di emergenza completo
                    if b_distance < self._behavior.braking_distance * 1.5:
                         self.logger.critical(f"Bicycle {bicycle.id} CRITICAL proximity while crossing. EMERGENCY STOP.")
                         return self.emergency_stop()
                    else:
                        # Altrimenti, rallenta drasticamente per valutare meglio
                        target_speed = min(self._approach_speed / 2, get_speed(self._vehicle) / 2) # Rallenta significativamente
                        target_speed = max(target_speed, 0) # Non andare in retromarcia
                        self.set_target_speed(target_speed)
                        self.logger.info(f"Slowing down significantly to {target_speed:.1f} km/h for crossing bicycle.")
                        control = self._local_planner.run_step(debug=debug)

                elif get_speed(bicycle) < 1 and b_distance < self._behavior.braking_distance * 2: # Non allineata, lenta/ferma e vicina
                    self.logger.info(f"Bicycle {bicycle.id} is not aligned, slow/stopped, and close. Decelerating to approach speed.")
                    print("--- Bicycle is not aligned, slow/stopped, and close. Decelerating.")
                    self.set_target_speed(self._approach_speed)
                    control = self._local_planner.run_step(debug=debug)
                
                else: # Bicicletta non allineata, ma più lontana o in movimento in modo non immediatamente critico
                      # In questo caso, potrebbe essere una bicicletta sul lato opposto della strada che si allontana
                      # o una che ha già attraversato. Mantenere una velocità prudente.
                    self.logger.info(f"Bicycle {bicycle.id} not aligned, but not an immediate crossing threat. Maintaining cautious speed.")
                    print("--- Bicycle not aligned, but not an immediate crossing threat. Maintaining cautious speed.")
                    # Non fare nulla di specifico qui lascerebbe il controllo al comportamento di default successivo
                    # Oppure, si potrebbe impostare una velocità leggermente ridotta per precauzione
                    # target_speed = min(self._speed_limit - self._behavior.speed_lim_dist, self._behavior.max_speed)
                    # self._local_planner.set_speed(target_speed)
                    # control = self._local_planner.run_step(debug=debug)
                    # Per ora, lasciamo che la logica di fallback gestisca questo,
                    # ma se 'control' non è ancora impostato, verrà gestito dal car_following o dal lane_following.

        elif self._lateral_offset_active: # Nessuna bicicletta rilevata, ma l'offset era attivo
            self.logger.info("No bicycle detected. Resetting previously active lateral offset.")
            print("--- No bicycle detected, resetting lateral offset to 0") # Messaggio originale
            self._local_planner.set_lateral_offset(0.0)
            self._lateral_offset_active = False
                

        #############################################
        ######## Vehicle Avoidance Behaviors ########
        #############################################
        # 1: Check for stop signs
        stop_sign_state, stop_sign, stop_sign_distance = self.stop_sign_manager(self._vehicle)
        if stop_sign_state:
            # Distance is computed from the center of the car and the stop sign,
            # we use bounding boxes to calculate the actual distance
            distance = stop_sign_distance
            msg = f"[DEBUG - Stop Sign Avoidance] Stop sign {stop_sign} detected, distance: {distance}"
            print(msg)
            self.logger.debug(msg)

            # Slow down if the stop sign is getting closer
            if distance < 2 * self._behavior.braking_distance:
                self.set_target_speed(self._approach_speed)
                msg = f"[INFO - Stop Sign Avoidance] Target speed set to:{self._approach_speed}\n[INFO] Distance to stop sign: {distance}"
                print(msg)
                
                # # Complete stop only if extremely close
                # if distance < self._behavior.braking_distance:
                #     msg = "[WARNING - Stop Sign Avoidance] Emergency stop due to stop sign proximity"
                #     print(msg)
                #     self.logger.critical(msg)
                #     return self.emergency_stop()
                
                # return self._local_planner.run_step()



        # 2: Check for vehicles
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)


        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            
            # Check if the car is legitimately stopped
            vehicle_wp, is_abandoned = self.is_vehicle_legitimately_stopped(vehicle)
            msg = f"[DEBUG - Vehicle Avoidance] Vehicle {vehicle} detected, distance: {distance}, is it abandoned? {is_abandoned}"
            print(msg)
            self.logger.debug(msg)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance and not self._blocked:
                msg= f"[INFO - Vehicle Avoidance] Emergency stop due to vehicle proximity"
                print(msg)
                self.logger.critical(msg)
                self._blocked = True
                return self.emergency_stop()

            if is_abandoned:
                # If the vehicle is considered parked/abandoned, we can try to overtake it.
                msg = f"[DEBUG - Vehicle Avoidance] Vehicle {vehicle} is parked/abandoned, attempting to overtake."
                print(msg)
                self.logger.debug(msg)
                
                overtake_path_generated_this_step = False
                if not self._overtake.in_overtake: # Only attempt to start a new overtake if not already in one
                    # Calculate a dynamic distance_same_lane, e.g., 1 second of travel at current speed, min 5m.
                    # Convert current speed from m/s to km/h for self._speed, then back to m/s for distance.
                    # get_speed returns m/s.
                    safe_approach_distance = max(5.0, get_speed(self._vehicle) * 1.0) 

                    overtake_path = self._overtake.run_step(
                        object_to_overtake=vehicle, 
                        ego_vehicle_wp=ego_vehicle_wp, 
                        distance_same_lane=safe_approach_distance,
                        distance_from_object=distance, 
                        speed_limit=self._vehicle.get_speed_limit() # Use current actual speed limit from vehicle
                    )
                    if overtake_path:
                        msg = f"[INFO - Vehicle Avoidance] Overtake path found. Initiating overtake."
                        print(msg)
                        self.logger.info(msg)
                        self.__update_global_plan(overtake_path=overtake_path)
                        # self._overtake.in_overtake and self._overtake.overtake_cnt are set by self._overtake.run_step()
                        overtake_path_generated_this_step = True
                    else:
                        # Failed to find an overtake path, stop.
                        msg = "[WARNING - Vehicle Avoidance] Failed to find an overtake path. Stopping."
                        print(msg)
                        self.logger.warning(msg)
                        return self.emergency_stop()
                
                # If now in an overtake maneuver (either newly initiated or ongoing)
                if self._overtake.in_overtake:
                    # Set a higher speed for overtaking.
                    # The _update_information method (called at the start of run_step)
                    # is responsible for checking oncoming traffic and aborting if necessary by changing the global plan.
                    
                    current_actual_speed_limit = self._vehicle.get_speed_limit()
                    
                    effective_max_speed = self._behavior.max_speed
                    overtake_speed_margin_kmh = getattr(self._behavior, 'overtake_speed_margin_kmh', 10.0) 

                    # If current behavior is Cautious, consider using Normal's speed parameters for a more effective overtake.
                    if isinstance(self._behavior, Cautious):
                        # Create a temporary Normal behavior instance to access its parameters
                        temp_normal_behavior = Normal()
                        overtake_speed_margin_kmh = getattr(temp_normal_behavior, 'overtake_speed_margin_kmh', 10.0) # Use Normal's margin
                        effective_max_speed = temp_normal_behavior.max_speed # Use Normal's max_speed
                        self.logger.info("Overtake: Current behavior is Cautious. Using Normal's speed parameters for overtake.")
                        print("--- Overtake: Current behavior is Cautious. Using Normal's speed parameters for overtake.")

                        # ---- Modify max_throttle if Cautious during overtake ----
                        if not self._overtake_throttle_modified:
                            self._original_max_throttle = self._local_planner._vehicle_controller.max_throt
                            self._local_planner._vehicle_controller.max_throt = 1.0
                            self._overtake_throttle_modified = True
                            self.logger.info("Overtake (Cautious): Max throttle temporarily set to 1.0 for faster acceleration.")
                            print("--- Overtake (Cautious): Max throttle temporarily set to 1.0 for faster acceleration.")
                                
                    desired_overtake_speed = current_actual_speed_limit + overtake_speed_margin_kmh
                    
                    # Ensure target speed is at least the current speed limit (if not already exceeding it)
                    # and not more than effective_max_speed.
                    target_speed = min(effective_max_speed, desired_overtake_speed)
                    target_speed = max(target_speed, current_actual_speed_limit) # Ensure at least speed limit

                    current_speed_kmh = get_speed(self._vehicle)
                    # If just starting the overtake and current speed is well below the desired overtake speed,
                    # ensure the target_speed encourages acceleration.
                    # If already moving fast, maintain speed or adjust to target_speed.
                    if overtake_path_generated_this_step and current_speed_kmh < target_speed:
                        pass # target_speed is already set to encourage acceleration
                    elif current_speed_kmh > target_speed and self._overtake.in_overtake : # if already faster than calculated target during overtake
                         target_speed = min(effective_max_speed, current_speed_kmh) # maintain current speed if safe, capped by max_speed

                    self._local_planner.set_speed(target_speed)
                    msg = f"[INFO - Overtake] Active overtake. Target speed: {target_speed:.2f} km/h. Current speed: {current_speed_kmh:.2f} km/h. Limit: {current_actual_speed_limit:.2f} km/h."
                    print(msg)
                    self.logger.info(msg)
                # else:
                    # If not self._overtake.in_overtake here, it implies the overtake either:
                    # 1. Failed to start (emergency_stop was called).
                    # 2. Finished in a previous step (normal speed logic will apply).
                    # No specific speed setting here if not actively overtaking.

                control = self._local_planner.run_step(debug=debug) # This will use the speed set above if overtaking
            
            else: # Vehicle is not abandoned
                # If the vehicle is not parked, we can follow it
                msg = f"[DEBUG - Vehicle Avoidance] Vehicle {vehicle} is not parked, following."
                print(msg)
                self.logger.debug(msg)
                control = self.car_following_manager(vehicle, distance, debug=debug)

        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
            msg = f"[DEBUG - Intersection Behavior] Target speed set to:{target_speed}"
            print(msg)
            self.logger.debug(msg)
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
            self._blocked = False
        return control

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

        # Decrease the overtake counter if the agent is in the middle of an overtake maneuver.
        if self._overtake.overtake_cnt > 0:
            self._overtake.overtake_cnt -= 1
            self._local_planner.draw_waypoints(color=carla.Color(0, 255, 0))
        else: 
            self._overtake.in_overtake = False

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(self._vehicle, lights_list)

        if affected:
            if self._should_log_object("traffic_light"):
                self.send_log("TRAFFIC_LIGHT", "Red light detected", "WARNING")
        else:
            # Reset quando non siamo più influenzati dal semaforo
            self._logged_objects["traffic_light"] = False
            
        return affected
    
    def stop_sign_manager(self, vehicle : carla.Vehicle, sign_distance : int = 20) -> bool:
        """
        This method is in charge of behaviors for stop signs.
        
            :param vehicle (carla.Vehicle): vehicle object to be checked.
            :param sign_distance (int): distance to the stop sign.
            
            :return affected (bool): True if the vehicle is affected by the stop sign, False otherwise.
        """
        affected, signal = self._affected_by_sign(vehicle=vehicle, sign_type="206", max_distance=sign_distance)
        distance = -1 if not affected else dist(a=vehicle, b=signal)
        return affected, signal, distance

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def bycicle_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any bicycle.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a bicycle nearby, False if not
            :return vehicle: nearby bicycle
            :return distance: distance to nearby bicycle
        """
        BICYCLES_ID = ['vehicle.bh.crossbike','vehicle.diamondback.century', 'vehicle.gazelle.omafiets']
        
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        bicycle_list = [v for v in vehicle_list if v.type_id in BICYCLES_ID]
        bicycle_list = [b for b in bicycle_list if is_within_distance(b.get_transform(), self._vehicle.get_transform(), 7, angle_interval=[0, 90])]
            
        if not bicycle_list or len(bicycle_list) == 0:
            return False, None, -1
    
        bicycle_list = sorted(bicycle_list, key=lambda x: dist(x, waypoint))
        bike_state = True
        distance = dist(bicycle_list[0], waypoint)
        
        if self._should_log_object("bicycle", bicycle_list[0].id):
            self.send_log("BICYCLE", "Bicycle detected", "WARNING")
        
        return bike_state, bicycle_list[0], distance

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if dist(v,waypoint) < 15 and v.id != self._vehicle.id]
        

        if not vehicle_list:
            return False, None, -1

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        if vehicle_state:
            if self._should_log_object("vehicle", vehicle.id):
                self.send_log("VEHICLE", "Vehicle detected ahead", "INFO")
    
        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")        
        walker_list = [w for w in walker_list if is_within_distance(w.get_transform(), self._vehicle.get_transform(), 15, angle_interval=[0, 90])]

        if not walker_list:
            return False, None, -1

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        if walker_state:
            if self._should_log_object("pedestrian", walker.id):
                self.send_log("PEDESTRIAN", "Pedestrian detected", "WARNING")
    
        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        # Get the vehicle control
        control = carla.VehicleControl()
        # Set the throttle and brake values to 0.0 and self._max_brake, respectively.
        control.throttle = 0.0
        control.brake = self._max_brake
        # Set the hand brake to False.
        control.hand_brake = False
        return control

    def static_obstacle_avoid_manager(self, waypoint, obj_filter="*static.prop*"):
        """
        Handles avoidance behavior for static obstacles.
        """
        # Filter relevant static props
        props = self._world.get_actors().filter(obj_filter)
        props = [p for p in props if is_within_distance(p.get_transform(), self._vehicle.get_transform(), 40, angle_interval=[0, 90])]

        if not props:
            return False, None, -1

        props = sorted(props,key = lambda x: dist(x,waypoint))

            
        o_state, o, o_distance = self._vehicle_obstacle_detected(
            props, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=60
        )

        if o_state:
            if self._should_log_object("obstacle", o.id):
                self.send_log("OBSTACLE", "Static obstacle detected", "INFO")
    
        return o_state, o, o_distance

    
    def __update_global_plan(self, overtake_path : list) -> None:
        """
        This method updates the global plan of the agent in order to overtake the vehicle in front of the agent.
        
            :param overtake_path (list): path to overtake the vehicle.
        """
        # Set the overtake plan and the target speed of the agent.
        new_plan = self._local_planner.set_overtake_plan(overtake_plan=overtake_path, overtake_distance=self._overtake.overtake_ego_distance)
        # Update the speed of the agent in order to overtake the vehicle.
        self.set_target_speed(2 * self._speed_limit)
        # Set the global plan of the agent.
        self.set_global_plan(new_plan)

    def set_streamer(self, streamer):
        """Imposta lo streamer per l'invio dei log al server Flask"""
        self._streamer = streamer
    
        # Reset dei log quando si imposta lo streamer
        if self._streamer:
            self._streamer.reset_logs()
            
        # Invia un log di test per verificare il funzionamento
        if not self._test_log_sent:
            self.send_log("SYSTEM", "BehaviorAgent initialized and connected to streamer", "INFO")
            self._test_log_sent = True

    def send_log(self, category, message, level="INFO"):
        """
        Invia un log sia al file di log che allo streamer (se disponibile)
        """
        # Log al file normale
        if level == "DEBUG":
            self.logger.debug(message)
        elif level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "CRITICAL":
            self.logger.critical(message)
            
        # Log allo streamer se disponibile
        if self._streamer:
            self._streamer.add_log(category, message, level)

    def _should_log_object(self, category, object_id=None):
        """
        Determina se un oggetto dovrebbe essere loggato, evitando duplicati.
        
        Args:
            category: Categoria dell'oggetto (pedestrian, bicycle, vehicle, obstacle, ecc.)
            object_id: ID dell'oggetto, None per oggetti senza ID
        
        Returns:
            bool: True se l'oggetto deve essere loggato, False altrimenti
        """
        current_time = time.time()
        
        # Per oggetti booleani (es. traffic light)
        if object_id is None:
            if not self._logged_objects[category]:
                self._logged_objects[category] = True
                return True
            return False
        
        # Per oggetti con ID
        if object_id not in self._logged_objects[category] or \
           current_time - self._logged_objects[category][object_id] > self._log_timeout:
            self._logged_objects[category][object_id] = current_time
            return True
        return False