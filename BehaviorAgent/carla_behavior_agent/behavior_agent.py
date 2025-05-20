""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal
from overtake import Overtake 
from misc import *
import logging
import time
from log_manager import ACTION, DedupFilter
import os 

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
        self._blocked = False

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

        self._overtake = Overtake(self._vehicle, opt_dict)
        
        # Added a variable to track the state of the lateral offset
        self._lateral_offset_active = False

        # Logger Configuration
        self.logger = logging.getLogger('behavior_agent')
        self.logger.setLevel(logging.DEBUG)
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
        file_handler.addFilter(DedupFilter(interval=10.0))
        
        # Add the handler to the logger 
        self.logger.addHandler(file_handler)
        
        # set propage to false to avoid double logging
        self.logger.propagate = False
    
        # Streamer Refer
        self._streamer = None
        
        # Test Log
        self._test_log_sent = False

        # Track the objects that have been logged to avoid duplicates
        self._logged_objects = {
            "pedestrian": {},
            "bicycle": {},
            "vehicle": {},
            "obstacle": {},
            "traffic_light": False,
            "stop_sign": {},
            "overtake": {}
        }
        
        # Stop sign handling
        self._at_stop_sign = False
        self._stop_sign_wait_time = 0
        self._stop_sign_waiting = False
        self._min_stop_time = 3.0  # Tempo minimo di attesa in secondi allo stop
        self._stop_sign_respected = {}  # Dizionario per tenere traccia degli stop già rispettati

    ################################
    ########### RUN STEP ###########
    ################################

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

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
            self.send_log_to_web("TRAFFIC_LIGHT", "Red light detected", "ACTION")
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
            
            # Debug message
            msg = f"[DEBUG - Pedestrian Avoidance] Pedestrian {walker} detected, distance 1: {distance}"
            self.send_log_to_console(msg)
            
            # Brake moment
            if  distance < 8 * self._behavior.braking_distance:
                target_speed = self._vehicle.get_velocity().length() / 2
                self.set_target_speed(target_speed)
                print(f"[INFO - Walker Obstacle Avoidance] Target speed set to:{target_speed}\n[INFO] Distance to static obstacle: {distance}")
                
                # Complete stop only if extremely close
                if distance < 2 * self._behavior.braking_distance:
                    # Debug message
                    msg = f"[WARNING - Walker Obstacle Avoidance] Emergency stop due to static obstacle proximity"
                    self.send_log_to_console(msg, "WARNING")
                    self.send_log_to_web("Pedestrian", f"Emergency Stop for the Pedestrian: {walker.id}", "ACTION")

                    return self.emergency_stop()



        ##################################################
        ########## Obstacle Avoidance Behaviors ##########
        ##################################################
        so_state, static_obstacle, so_distance = self.static_obstacle_avoid_manager(ego_vehicle_wp)
        if so_state is True:        
            distance = compute_distance_from_center(actor1 =self._vehicle, actor2 = static_obstacle, distance=so_distance)

            # Debug message
            msg = f"[Static Obstacle Avoidance] Static obstacle:{static_obstacle} detected, distance: {distance}"
            print(msg)
            self.logger.critical(msg)

            if not ego_vehicle_wp.is_junction:
                overtake_path = self._overtake.run_step(
                    object_to_overtake=static_obstacle, ego_vehicle_wp=ego_vehicle_wp, distance_same_lane=1, distance_other_lane=20, distance_from_object=so_distance, speed_limit = self._speed_limit
                )

            # Check if the overtake path was generated and start the overtake
            if overtake_path: 
                self.send_log_to_web("overtake", "Overtake Path Found", "INFO")
                self.__update_global_plan(overtake_path=overtake_path)
                self.send_log_to_web("overtake", "Starting the overtake", "ACTION")
            

            if not self._overtake.in_overtake and distance < 3 * self._behavior.braking_distance:
                self.set_target_speed(self._approach_speed*2)

                if distance < 2 * self._behavior.braking_distance:
                    self.set_target_speed(self._approach_speed)

                    if distance < self._behavior.braking_distance:
                        msg = "[WARNING - Static Obstacle Avoidance] Emergency stop due to static obstacle proximity"
                        self.send_log_to_console(msg, "WARNING")
                        self.send_log_to_web("obstacle", "Emergency Stop for the Obstacle", "ACTION")
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
                    msg = "Bicycle {bicycle.id} detected in the SAME lane ({ego_lane_id}), aligned. Distance: {b_distance:.2f}m."
                    self.send_log_to_console(msg)
                    # Controlla se la bicicletta è vicina al centro della corsia. In questo caso, l'agente sorpasserà la bicicletta.
                    if is_bicycle_near_center(vehicle_location=bicycle.get_location(), ego_vehicle_wp=ego_vehicle_wp) and get_speed(self._vehicle) < 0.1:
                        self.logger.info(f"Bicycle {bicycle.id} is near the center of our lane. Attempting to overtake.")
                        print("--- Bicycle is near the center of the lane! We can try to overtake it.")

                        if not ego_vehicle_wp.is_junction:
                            overtake_path = self._overtake.run_step(
                                object_to_overtake=bicycle, ego_vehicle_wp=ego_vehicle_wp, distance_same_lane=1, distance_from_object=b_distance, speed_limit = self._speed_limit
                            )                                               
                        if overtake_path:
                            msg = f"Overtake path found for bicycle {bicycle.id}. Initiating overtake."
                            self.send_log_to_console(msg)
                            self.__update_global_plan(overtake_path=overtake_path)
                            self.send_log_to_web("overtake", f"Starting the overtake of bicycle {bicycle.id}.", "ACTION")

                        if not self._overtake.in_overtake: 
                            msg = f"Overtake path not found for bicycle {bicycle.id}. Emergency stop."
                            self.send_log_to_web("overtake", msg, "ACTION")
                            self.send_log_to_console(msg, "WARNING")
                            return self.emergency_stop()
                    else: 
                        msg = f"Bicycle {bicycle.id} is in our lane, but not centered. Applying lateral offset."
                        self.send_log_to_console(msg)
                        new_offset = -(2.5 * bicycle.bounding_box.extent.y + self._vehicle.bounding_box.extent.y)
                        self.send_log_to_web("bicycle", f"Applying lateral offset of {new_offset:.2f}m to avoid bicycle {bicycle.id}", "ACTION")
                        self._local_planner.set_lateral_offset(new_offset)
                        self._lateral_offset_active = True
                        target_speed = min([
                            self._behavior.max_speed,
                            self._speed_limit - self._behavior.speed_lim_dist])
                        self._local_planner.set_speed(target_speed)
                        control = self._local_planner.run_step(debug=debug)
                else: # La bicicletta è allineata (stessa direzione) ma in una corsia DIVERSA
                    msg = f"Bicycle {bicycle.id} detected in a different lane (Ego: {ego_lane_id}, Bicycle: {bicycle_lane_id}), aligned. No offset applied."
                    self.send_log_to_console(msg)
                    if self._lateral_offset_active:
                        msg = "Resetting previously active lateral offset as current bicycle is in a different lane."
                        self.send_log_to_console(msg)
                        self.send_log_to_web("bicycle", msg, "ACTION")
                        self._local_planner.set_lateral_offset(0.0)
                        self._lateral_offset_active = False
            
            # CASO: Bicicletta NON allineata (potenzialmente attraversante o in direzione opposta)
            else:
                msg = f"Bicycle {bicycle.id} detected in a different lane (Ego: {ego_lane_id}, Bicycle: {bicycle_lane_id}), NOT aligned. Distance: {b_distance:.2f}m."
                self.send_log_to_console(msg)

                # Verifica se la bicicletta sta effettivamente attraversando la nostra traiettoria
                # Questo è un controllo semplificato. Una predizione di traiettoria sarebbe più robusta.
                # Consideriamo "attraversante" se non è né allineata né in direzione opposta quasi parallela.
                is_potentially_crossing = abs(diff_yaw) > is_aligned_threshold and abs(abs(diff_yaw) - 180) > is_aligned_threshold 

                if is_potentially_crossing and b_distance < self._behavior.braking_distance * 3: # Distanza di sicurezza per reagire
                    msg = f"Bicycle {bicycle.id} is potentially CROSSING at {b_distance:.2f}m. Initiating emergency stop or significant slowdown."
                    self.send_log_to_console(msg, "WARNING")
                    # Se molto vicina, arresto di emergenza completo
                    if b_distance < self._behavior.braking_distance * 1.5:
                        msg = f"Bicycle {bicycle.id} CRITICAL proximity while crossing. EMERGENCY STOP."
                        self.send_log_to_console(msg, "CRITICAL")
                        return self.emergency_stop()
                    else:
                        # Altrimenti, rallenta drasticamente per valutare meglio
                        target_speed = min(self._approach_speed / 2, get_speed(self._vehicle) / 2) # Rallenta significativamente
                        target_speed = max(target_speed, 0) # Non andare in retromarcia
                        self.set_target_speed(target_speed)
                        msg = f"Bicycle {bicycle.id} is crossing. Slowing down to {target_speed:.1f} km/h."
                        self.send_log_to_console(msg)
                        control = self._local_planner.run_step(debug=debug)

                elif get_speed(bicycle) < 1 and b_distance < self._behavior.braking_distance * 2: # Non allineata, lenta/ferma e vicina
                    msg = f"Bicycle {bicycle.id} is NOT aligned, slow/stopped, and close. Decelerating to approach speed."
                    self.send_log_to_console(msg)
                    self.set_target_speed(self._approach_speed)
                    control = self._local_planner.run_step(debug=debug)
                
                else: # Bicicletta non allineata, ma più lontana o in movimento in modo non immediatamente critico
                      # In questo caso, potrebbe essere una bicicletta sul lato opposto della strada che si allontana
                      # o una che ha già attraversato. Mantenere una velocità prudente.
                    msg = f"Bicycle {bicycle.id} is NOT aligned, but not an immediate crossing threat. Maintaining cautious speed."
                    self._send_log_to_console(msg)
                    # Non fare nulla di specifico qui lascerebbe il controllo al comportamento di default successivo
                    # Oppure, si potrebbe impostare una velocità leggermente ridotta per precauzione
                    # target_speed = min(self._speed_limit - self._behavior.speed_lim_dist, self._behavior.max_speed)
                    # self._local_planner.set_speed(target_speed)
                    # control = self._local_planner.run_step(debug=debug)
                    # Per ora, lasciamo che la logica di fallback gestisca questo,
                    # ma se 'control' non è ancora impostato, verrà gestito dal car_following o dal lane_following.

        elif self._lateral_offset_active: # Nessuna bicicletta rilevata, ma l'offset era attivo
            msg = f"Bicycle offset was active, but no bicycle detected. Resetting lateral offset."
            self.send_log_to_console(msg)
            self._local_planner.set_lateral_offset(0.0)
            self._lateral_offset_active = False
                

        
        ##############################################
        ################ STOP SIGN ###################
        ##############################################
        stop_sign_state, stop_sign, stop_sign_distance = self.stop_sign_manager(self._vehicle)
        is_junction = ego_vehicle_wp.is_junction

        # Gestione incrocio e segnale di stop
        if is_junction:
            self.send_log_to_console("Vehicle is in a junction", "INFO")
            
            # Disabilitiamo il sorpasso negli incroci
            if self._overtake.in_overtake:
                self.send_log_to_console("Aborting overtake in junction", "WARNING")
                self._overtake.in_overtake = False
                self._overtake.overtake_cnt = 0
                # Reimpostiamo la velocità a un valore sicuro per l'incrocio
                self.set_target_speed(min(self._approach_speed, self._speed_limit - 10))
                self.send_log_to_web("JUNCTION", "Sorpasso annullato nell'incrocio", "ACTION")
            
            # Imposta la velocità di approccio all'incrocio (ridotta)
            if not self._stop_sign_waiting:
                target_speed = min(self._approach_speed, self._speed_limit - 10)
                self.set_target_speed(target_speed)
                self.send_log_to_console(f"Setting junction approach speed to {target_speed} km/h", "INFO")

        if stop_sign_state:
            # Distance is computed from the center of the car and the stop sign
            distance = stop_sign_distance
            self.send_log_to_console(f"Stop sign: {stop_sign} detected, distance: {distance}", "INFO")
            
            # Se siamo molto vicini allo stop e non ci siamo ancora fermati
            if distance < self._behavior.braking_distance * 1.5 and not self._stop_sign_waiting:
                self.send_log_to_console("Approaching stop sign, slowing down", "INFO")
                self.set_target_speed(self._approach_speed / 2)
                
                # MODIFICA CRUCIALE: controlla se lo stop è già stato rispettato
                if distance < self._behavior.braking_distance and not (stop_sign and stop_sign.id in self._stop_sign_respected):
                    self.send_log_to_console("Stopping at stop sign", "ACTION")
                    self.send_log_to_web("STOP_SIGN", f"Arresto completo allo stop", "ACTION")
                    self._stop_sign_waiting = True
                    self._stop_sign_wait_time = time.time()
                    return self.emergency_stop()
                elif stop_sign and stop_sign.id in self._stop_sign_respected:
                    # Log che stiamo ignorando uno stop già rispettato
                    self.send_log_to_console(f"Stop sign {stop_sign.id} già rispettato, proseguiamo", "DEBUG")
                    self.send_log_to_web("STOP_SIGN", f"Stop sign già rispettato, proseguiamo", "INFO")
    
            # Se ci siamo già fermati allo stop, controlliamo se possiamo ripartire
            elif self._stop_sign_waiting:
                # Controlliamo se abbiamo aspettato il tempo minimo
                wait_time = time.time() - self._stop_sign_wait_time
                
                if wait_time < self._min_stop_time:
                    self.send_log_to_console(f"Waiting at stop sign for {wait_time:.1f}s", "INFO")
                    return self.emergency_stop()
                else:
                    # Controlla se ci sono veicoli che hanno la precedenza
                    vehicle_list = self._world.get_actors().filter("*vehicle*")
                    
                    # Filtra solo veicoli nelle vicinanze e non noi stessi
                    # Usiamo la location del veicolo invece del waypoint per il calcolo della distanza
                    ego_location = self._vehicle.get_location()
                    nearby_vehicles = []
                    
                    for v in vehicle_list:
                        if v.id != self._vehicle.id:
                            # Calcola la distanza effettiva tra i veicoli
                            distance = ego_location.distance(v.get_location())
                            if distance < 15:  # Ridotto da 20m a 15m
                                # Controlla se il veicolo è davvero all'incrocio (non dietro o su altre strade)
                                v_wp = self._map.get_waypoint(v.get_location())
                                if v_wp.is_junction and v_wp.is_junction == ego_vehicle_wp.is_junction:
                                    nearby_vehicles.append(v)
                    
                    # Se ci sono veicoli con precedenza, continua ad aspettare
                    if nearby_vehicles:
                        self.send_log_to_console(f"Vehicles with right of way detected: {len(nearby_vehicles)} vehicles, waiting at stop", "INFO")
                        return self.emergency_stop()
                    else:
                        # Se non ci sono veicoli, possiamo ripartire
                        self.send_log_to_console("No vehicles with right of way, proceeding after stop", "ACTION")
                        self.send_log_to_web("STOP_SIGN", "Ripartenza dopo lo stop", "ACTION")
                        
                        # AGGIUNTA CRITICA: Salva l'ID dello stop nei segnali rispettati
                        if stop_sign and hasattr(stop_sign, 'id'):
                            self._stop_sign_respected[stop_sign.id] = time.time()
                            self.send_log_to_console(f"Stop sign {stop_sign.id} marked as respected", "DEBUG")
                            self.send_log_to_web("STOP_SIGN", f"Stop sign {stop_sign.id} rispettato", "DEBUG")
                        
                        self._stop_sign_waiting = False
                        target_speed = min(self._approach_speed, self._speed_limit - 10)
                        self.set_target_speed(target_speed)
                        return self._local_planner.run_step()
    
        #############################################
        ######## Vehicle Avoidance Behaviors ########
        #############################################
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
            self.send_log_to_console(msg,"DEBUG")

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance and not self._blocked:
                msg= f"[INFO - Vehicle Avoidance] Emergency stop due to vehicle proximity"
                self.send_log_to_console(msg, "CRITICAL")
                self._blocked = True
                return self.emergency_stop()

            if is_abandoned:
                # If the vehicle is considered parked/abandoned, we can try to overtake it.
                msg = f"[DEBUG - Vehicle Avoidance] Vehicle {vehicle} is parked/abandoned, attempting to overtake."
                self.send_log_to_console(msg, "DEBUG")
                
                overtake_path_generated_this_step = False
                if not self._overtake.in_overtake: # Only attempt to start a new overtake if not already in one
                    # Calculate a dynamic distance_same_lane, e.g., 1 second of travel at current speed, min 5m.
                    # Convert current speed from m/s to km/h for self._speed, then back to m/s for distance.
                    # get_speed returns m/s.
                    safe_approach_distance = max(3.0, get_speed(self._vehicle) * 1.0) 

                    if not ego_vehicle_wp.is_junction:
                        overtake_path = self._overtake.run_step(
                            object_to_overtake=vehicle, 
                            ego_vehicle_wp=ego_vehicle_wp, 
                            distance_same_lane=safe_approach_distance,
                            distance_from_object=distance, 
                            speed_limit=self._vehicle.get_speed_limit() # Use current actual speed limit from vehicle
                        )
                    if overtake_path:
                        msg = f" Overtake path found. Initiating overtake."
                        self.send_log_to_console(msg)
                        self.send_log_to_web("overtake", msg, "ACTION")
                        self.__update_global_plan(overtake_path=overtake_path)
                        # self._overtake.in_overtake and self._overtake.overtake_cnt are set by self._overtake.run_step()
                        overtake_path_generated_this_step = True
                    else:
                        # Failed to find an overtake path, stop.
                        msg = "Failed to find an overtake path. Waiting."
                        self.send_log_to_console(msg, "WARNING")
                        self.send_log_to_web("OVERTAKE", msg, "INFO")
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
                        msg = f"Current behavior is Cautious. Using Normal's speed parameters for overtake."
                        self.send_log_to_console(msg)
                                
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
                    self.send_log_to_console(msg)
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
            # Se siamo in attesa a uno stop, non procediamo
            if self._stop_sign_waiting:
                return self.emergency_stop()
                
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
            msg = f"[DEBUG - Intersection Behavior] Target speed set to:{target_speed}"
            self.send_log_to_console(msg, "DEBUG")
            self._local_planner.set_speed(target_speed)            # Aggiungi questo nella sezione relativa agli incroci in run_step
            if stop_sign_state and distance < self._behavior.braking_distance:
                self.send_log_to_console("Stopping at stop sign", "ACTION")
                self.send_log_to_web("STOP_SIGN", "Arresto completo allo stop", "ACTION")
                self._stop_sign_waiting = True
                self._stop_sign_wait_time = time.time()
                return self.emergency_stop()
            elif is_junction:
                # Controlla se possiamo attraversare l'incrocio
                can_cross = self.junction_manager(ego_vehicle_wp)
                if not can_cross:
                    return self.emergency_stop()
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            if not self._overtake.in_overtake:               
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                self._blocked = False
            else:
                control = self._local_planner.run_step(debug=debug)
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

        # Reset stop sign waiting state if we've moved away from any stop sign or if we're waiting too long
        if self._stop_sign_waiting:
            stop_sign_present, _, distance = self.stop_sign_manager(self._vehicle, sign_distance=30)
            current_time = time.time()
            wait_duration = current_time - self._stop_sign_wait_time
            
            # Resettiamo se ci siamo allontanati o se abbiamo aspettato troppo (30 secondi max)
            if not stop_sign_present or distance > 15 or wait_duration > 30.0:
                reason = "moved away from stop" if not stop_sign_present or distance > 15 else "timeout after 30s"
                self.send_log_to_console(f"Stop sign no longer relevant ({reason}), resetting state", "INFO")
                self.send_log_to_web("STOP_SIGN", f"Resettato stato di stop ({reason})", "ACTION")
                self._stop_sign_waiting = False
                # Impostiamo una velocità bassa ma non zero per assicurare che il veicolo si muova
                target_speed = min(self._approach_speed / 2, self._speed_limit / 2)
                self.set_target_speed(target_speed)

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(self._vehicle, lights_list)

        if affected:
            if self._should_log_object("traffic_light"):
                self.send_log_to_web("TRAFFIC_LIGHT", "Red light detected", "WARNING")
        else:
            # Reset quando non siamo più influenzati dal semaforo
            self._logged_objects["traffic_light"] = False
            
        return affected
    
    def stop_sign_manager(self, vehicle : carla.Vehicle, sign_distance : int = 20):
        """
        This method is in charge of behaviors for stop signs.
        """
        affected, signal = self._affected_by_sign(vehicle=vehicle, sign_type="206", max_distance=sign_distance)
        distance = -1 if not affected else dist(a=vehicle, b=signal)
        
        # Se il segnale è stato rispettato di recente, lo ignoriamo
        if affected and signal and signal.id in self._stop_sign_respected:
            last_respect_time = self._stop_sign_respected[signal.id]
            if time.time() - last_respect_time < 30.0:  # 30 secondi di "immunità" dopo aver rispettato lo stop
                return False, None, -1
    
        if affected and self._should_log_object("stop_sign", signal.id):
            self.send_log_to_web("STOP_SIGN", f"Stop sign detected at {distance:.2f}m", "INFO")
    
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
        
        if bike_state:
            properties = {
                'is_moving': get_speed(bicycle_list[0]) > 0.1,
                'relative_position': self._get_relative_position(bicycle_list[0])
            }
            
            if self._should_log_object("bicycle", bicycle_list[0].id, properties):
                self.send_log_to_web("BICYCLE", f"Bicycle detected (ID: {bicycle_list[0].id})", "Info")
    
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
            properties = {
                'is_moving': get_speed(vehicle) > 0.1,
                'relative_position': self._get_relative_position(vehicle),
                'lane_id': self._map.get_waypoint(vehicle.get_location()).lane_id
            }
            
            if self._should_log_object("vehicle", vehicle.id, properties):
                self.send_log_to_web("VEHICLE", f"Vehicle detected (ID: {vehicle.id})", "INFO")
    
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
            # Crea un dizionario con le proprietà rilevanti che determinano se è una nuova detection
            properties = {
                'is_moving': get_speed(walker) > 0.1,
                'relative_position': 'front' if is_within_distance(walker.get_transform(), self._vehicle.get_transform(), 10, angle_interval=[0, 30]) else 'side'
            }
            
            if self._should_log_object("pedestrian", walker.id, properties):
                self.send_log_to_web("PEDESTRIAN", f"Pedestrian detected (ID: {walker.id})", "WARNING")
    
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

    def static_obstacle_avoid_manager(self, waypoint, obj_filter="*static.prop*", obj_ignore = ["static.prop.dirtdebris01","static.prop.dirtdebris02","static.prop.dirtdebris03"]):
        """
        Handles avoidance behavior for static obstacles.
        """
        # Filter relevant static props
        props = self._world.get_actors().filter(obj_filter)
        props = [p for p in props if is_within_distance(p.get_transform(), self._vehicle.get_transform(), 40, angle_interval=[0, 90])]

        # Filtra gli oggetti da ignorare
        props = [p for p in props if not any(ignore_type in p.type_id for ignore_type in obj_ignore)]

        if not props:
            return False, None, -1

        props = sorted(props,key = lambda x: dist(x,waypoint))
        
        o_state, o, o_distance = self._vehicle_obstacle_detected(
            props, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=60
        )

        if o_state:
            if self._should_log_object("obstacle", o.id):
                self.send_log_to_web("OBSTACLE", f"Static obstacle detected; {o}", "INFO")

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
            self.send_log_to_web("SYSTEM", "BehaviorAgent initialized and connected to streamer", "DEBUG")
            self._test_log_sent = True
            
    def send_log_to_web(self, category, message, level="INFO", object_id=None, properties=None):
        """Invia un log allo streamer."""
        # Logga con il logger standard
        if level == "INFO":
            self.logger.info(f"[{category}] {message}")
        elif level == "WARNING":
            self.logger.warning(f"[{category}] {message}")
        elif level == "ERROR":
            self.logger.error(f"[{category}] {message}")
        elif level == "ACTION": 
            self.logger.action(f"[{category}] {message}")
        elif level == "DEBUG":
            self.logger.debug(f"[{category}] {message}")
    
        # Invia il log allo streamer se disponibile
        if self._streamer:
            self._streamer.add_log(category, message, level, object_id, properties)

    def _should_log_object(self, category, object_id=None, properties=None):
        """
        Determina se un oggetto dovrebbe essere loggato, evitando duplicati per ID.
    
        Args:
            category: Categoria dell'oggetto
            object_id: ID dell'oggetto
            properties: Proprietà dell'oggetto (ignorate per la deduplicazione)
    
        Returns:
            bool: True se l'oggetto deve essere loggato, False altrimenti
        """
        if object_id is None:
            # Per oggetti senza ID (come semafori)
            if not self._logged_objects[category]:
                self._logged_objects[category] = True
                return True
            return False
    
        # Per oggetti con ID, controlliamo semplicemente se è già stato loggato
        # Non consideriamo le proprietà: una volta loggato, non viene più loggato
        if object_id not in self._logged_objects[category]:
            # Prima volta che vediamo questo oggetto
            self._logged_objects[category][object_id] = {
                'timestamp': time.time(),
                'properties': properties or {}
            }
            return True
    
        # L'oggetto è già stato loggato, non generiamo un nuovo log
        return False

    def _get_relative_position(self, actor):
        """
        Determina la posizione relativa di un attore rispetto al veicolo ego.
        Restituisce: 'front', 'behind', 'left', 'right'
        """
        ego_transform = self._vehicle.get_transform()
        actor_transform = actor.get_transform()
        
        # Calcola il vettore dall'ego all'attore
        forward_vector = ego_transform.get_forward_vector()
        right_vector = ego_transform.get_right_vector()
        
        # Calcola il vettore dall'ego all'attore nel sistema di coordinate dell'ego
        to_actor = actor_transform.location - ego_transform.location
        
        # Proietta questo vettore sui vettori dell'ego
        forward_proj = to_actor.x * forward_vector.x + to_actor.y * forward_vector.y
        right_proj = to_actor.x * right_vector.x + to_actor.y * right_vector.y
        
        # Determina la posizione relativa
        if abs(right_proj) < abs(forward_proj):
            return 'front' if forward_proj > 0 else 'behind'
        else:
            return 'right' if right_proj > 0 else 'left'
        
    def send_log_to_console(self,message,level="INFO"):
        """
        Logga un messaggio sulla console.
        """
        print(f"{message}")
        if level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "ACTION": 
            self.logger.action(message)
        elif level == "DEBUG":
            self.logger.debug(message)

    def junction_manager(self, ego_vehicle_wp):
        """
        Gestisce la logica di attraversamento degli incroci.
    
        :param ego_vehicle_wp: waypoint del veicolo
        :return: (bool) True se è sicuro attraversare, False altrimenti
        """
        if not ego_vehicle_wp.is_junction:
            return True
    
        # Ottieni il tipo di incrocio e la direzione pianificata
        junction = ego_vehicle_wp.get_junction()
        _, direction = self._local_planner.get_incoming_waypoint_and_direction(steps=self._look_ahead_steps)
    
        # Se siamo fermi a uno stop, controlliamo se è sicuro ripartire
        if self._stop_sign_waiting:
            wait_time = time.time() - self._stop_sign_wait_time
        
            # Dobbiamo aspettare almeno il tempo minimo
            if wait_time < self._min_stop_time:
                self.send_log_to_console(f"Attendendo allo stop per {wait_time:.1f}s", "INFO")
                return False
        
            # Prima di tutto, ricerchiamo lo stop sign che stiamo rispettando per salvarlo
            # come rispettato e non doverci fermare di nuovo
            stop_sign_state, stop_sign, _ = self.stop_sign_manager(self._vehicle, sign_distance=30)
            if stop_sign and hasattr(stop_sign, 'id'):
                # Marchiamo subito come rispettato, prima di controllare i veicoli
                self._stop_sign_respected[stop_sign.id] = time.time()
                self.send_log_to_console(f"Stop sign {stop_sign.id} marcato come rispettato", "DEBUG")
                self.send_log_to_web("STOP_SIGN", f"Stop sign {stop_sign.id} marcato come rispettato", "DEBUG")
        
            # Controlliamo i veicoli nelle vicinanze dell'incrocio
            junction_radius = math.sqrt(junction.bounding_box.extent.y**2 + junction.bounding_box.extent.x**2)
        
            # Ottieni i veicoli distinti per direzione
            vehicles = self._categorize_vehicles_at_junction(junction, ego_vehicle_wp)
        
            # Applica le regole di precedenza in base alla direzione pianificata
            can_proceed = self._check_junction_rules(vehicles, direction)
        
            if can_proceed:
                self.send_log_to_console("Nessun veicolo con precedenza, attraversamento sicuro", "ACTION")
                self.send_log_to_web("STOP_SIGN", "Ripartenza dopo lo stop", "ACTION")
                
                self._stop_sign_waiting = False
                return True
            else:
                self.send_log_to_console("Veicoli con precedenza rilevati, attendere", "INFO")
                return False
    
        return True

    def _categorize_vehicles_at_junction(self, junction, ego_vehicle_wp):
        """
        Categorizza i veicoli nell'incrocio per posizione (destra, sinistra, frontale).
        """
        # Ottieni il punto centrale dell'incrocio
        pivot = carla.Transform(junction.bounding_box.location, ego_vehicle_wp.transform.rotation)
    
        # Raggio di rilevamento
        junction_radius = math.sqrt(junction.bounding_box.extent.y**2 + junction.bounding_box.extent.x**2)
    
        # Ottieni tutti i veicoli vicini
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if v.id != self._vehicle.id]
    
        # Calcola l'orientamento rispetto all'incrocio
        yaw = math.radians(pivot.rotation.yaw)
        front_vector = np.array([np.cos(yaw), np.sin(yaw), 0])
    
        # Dizionario per veicoli in ogni direzione
        vehicles = {"left": [], "right": [], "front": []}
    
        for vehicle in vehicle_list:
            if vehicle.id == self._vehicle.id:
                continue
            
            # Controlla se il veicolo è effettivamente all'incrocio
            vehicle_wp = self._map.get_waypoint(vehicle.get_location())
            if not vehicle_wp.is_junction or vehicle_wp.get_junction().id != junction.id:
                continue
            
            # Calcola la posizione relativa del veicolo
            vehicle_pos = np.array([vehicle.get_location().x, vehicle.get_location().y, vehicle.get_location().z])
            pivot_pos = np.array([pivot.location.x, pivot.location.y, pivot.location.z])
            vec = vehicle_pos - pivot_pos
        
            # Calcola prodotto vettoriale e scalare
            cross = np.cross(front_vector, vec)[2]
            dot = np.dot(front_vector, vec[:2])
        
            # Categorizza in base alla posizione
            if abs(cross) < abs(dot):
                if dot > 0:
                    vehicles["front"].append(vehicle)
            else:
                if cross < 0:
                    vehicles["left"].append(vehicle)
                else:
                    vehicles["right"].append(vehicle)
    
        # Aggiungi questi log di debug alla fine del metodo
        self.send_log_to_console(f"Veicoli categorizzati all'incrocio: sinistra:{len(vehicles['left'])}, destra:{len(vehicles['right'])}, frontali:{len(vehicles['front'])}", "DEBUG")
        self.send_log_to_web("JUNCTION", f"Veicoli all'incrocio - sinistra:{len(vehicles['left'])}, destra:{len(vehicles['right'])}, fronte:{len(vehicles['front'])}", "DEBUG")
        
        return vehicles

    def _check_junction_rules(self, vehicles, planned_direction):
        """
        Applica le regole di precedenza negli incroci.
    
        :param vehicles: dizionario con veicoli classificati per direzione
        :param planned_direction: direzione pianificata (RoadOption)
        :return: True se è sicuro procedere, False altrimenti
        """
        # Ottieni il primo veicolo in ogni direzione (più vicino)
        vehicle_left = vehicles["left"][0] if vehicles["left"] else None
        vehicle_right = vehicles["right"][0] if vehicles["right"] else None
        vehicle_front = vehicles["front"][0] if vehicles["front"] else None
    
        # Se non ci sono veicoli, è sicuro procedere
        if not vehicle_left and not vehicle_right and not vehicle_front:
            return True
    
        # Logica per direzione DIRITTA
        if planned_direction == RoadOption.STRAIGHT:
            # Controlla i veicoli da sinistra (potrebbero attraversare)
            left_is_safe = vehicle_left is None or not self._is_vehicle_going_straight(vehicle_left)
            # Controlla i veicoli da destra (potrebbero svoltare a sinistra)
            right_is_safe = vehicle_right is None or not self._is_vehicle_turning_left(vehicle_right)
            # Controlla i veicoli frontali (potrebbero svoltare a sinistra)
            front_is_safe = vehicle_front is None or not self._is_vehicle_turning_left(vehicle_front)
        
            result = left_is_safe and right_is_safe and front_is_safe
            self.send_log_to_web("JUNCTION", f"Direzione DRITTO - Sicurezza: sinistra:{left_is_safe}, destra:{right_is_safe}, fronte:{front_is_safe}", "DEBUG")
            return result
    
        # Logica per SVOLTA A SINISTRA
        elif planned_direction == RoadOption.LEFT:
            # Controlla i veicoli frontali (che vanno dritto o a destra)
            front_is_safe = vehicle_front is None or not (self._is_vehicle_going_straight(vehicle_front) or 
                                                        self._is_vehicle_turning_right(vehicle_front))
            # Controlla i veicoli da destra (che vanno dritto o a sinistra)
            right_is_safe = vehicle_right is None or not (self._is_vehicle_going_straight(vehicle_right) or 
                                                         self._is_vehicle_turning_left(vehicle_right))
        
            result = front_is_safe and right_is_safe
            self.send_log_to_web("JUNCTION", f"Direzione SINISTRA - Sicurezza: fronte:{front_is_safe}, destra:{right_is_safe}", "DEBUG")
            return result
    
        # Logica per SVOLTA A DESTRA
        elif planned_direction == RoadOption.RIGHT:
            # Controlla i veicoli frontali (che svoltano a sinistra)
            front_is_safe = vehicle_front is None or not self._is_vehicle_turning_left(vehicle_front)
            # Veicoli da sinistra che vanno dritto
            left_is_safe = vehicle_left is None or not self._is_vehicle_going_straight(vehicle_left)
        
            result = front_is_safe and left_is_safe
            self.send_log_to_web("JUNCTION", f"Direzione DESTRA - Sicurezza: fronte:{front_is_safe}, sinistra:{left_is_safe}", "DEBUG")
            return result
    
        # Per altre situazioni, sii cauto
        return False

    def _is_vehicle_turning_left(self, vehicle):
        """Verifica se un veicolo sta attivando l'indicatore di svolta a sinistra."""
        if not vehicle:
            return False
    
        light_state = vehicle.get_light_state()
        return bool(light_state & carla.libcarla.VehicleLightState.LeftBlinker)

    def _is_vehicle_turning_right(self, vehicle):
        """Verifica se un veicolo sta attivando l'indicatore di svolta a destra."""
        if not vehicle:
            return False
    
        light_state = vehicle.get_light_state()
        return bool(light_state & carla.libcarla.VehicleLightState.RightBlinker)

    def _is_vehicle_going_straight(self, vehicle):
        """Verifica se un veicolo sta andando dritto (nessun indicatore attivo)."""
        if not vehicle:
            return False
    
        light_state = vehicle.get_light_state()
        return not (bool(light_state & carla.libcarla.VehicleLightState.LeftBlinker) or 
                   bool(light_state & carla.libcarla.VehicleLightState.RightBlinker))