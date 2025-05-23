""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import numpy as np
import carla
import math
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal
from misc import *
import time
from BehaviorAgent.carla_behavior_agent.object_detector import ObjectDetector
from BehaviorAgent.carla_behavior_agent.overtake_maneuver import OvertakingManeuver
from BehaviorAgent.carla_behavior_agent.logger import Logger

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

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
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
        self._approach_speed = 15
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self.initialized = False

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()
            
        # Inizializza il gestore di sorpasso
        self._overtake_manager = OvertakingManeuver(
            self._vehicle, 
            self._world, 
            self._map, 
            self._local_planner,
            self._sampling_resolution
        )

        # Added a variable to track the state of the lateral offset
        self._lateral_offset_active = False

        # Inizializza il rilevatore di oggetti
        
        self._object_detector = ObjectDetector(self._world, self._vehicle, self._behavior)
        
        # Dopo aver inizializzato il local planner, passa il riferimento all'object detector
        self._object_detector.set_local_planner(self._local_planner)
    
        # Logger Configuration
        self.logger = Logger(name="behavior_agent", log_dir="./carla_behavior_agent/logs/")


        # Stop sign handling
        self._at_stop_sign = False
        self._stop_sign_wait_time = 0
        self._stop_sign_waiting = False
        self._min_stop_time = 3.0  # Tempo minimo di attesa in secondi allo stop
        self._stop_sign_respected = {}  # Dizionario per tenere traccia degli stop già rispettati
    
        # Timer per i veicoli temporaneamente fermi
        self._stopped_vehicles_timer = {}
        self._timer_cleanup_interval = 30.0  # Pulisci i timer ogni 30 secondi
        self._last_timer_cleanup = time.time()

        # Nuove variabili di stato per la gestione degli incroci
        self._in_tate = False
        self._in_junction_grace_period = False
        self._junction_grace_duration = 4.0  # Durata del periodo di grazia in secondi
        
        #Offset 
        self._lateral_offset_active = False
        self._lateral_offset_type = None  # 'cone', 'bicycle', None

    
    ################################
    ########### RUN STEP ###########
    ################################

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        if not self.initialized:
            self.logger.web_debug("BehaviorAgent", f"Initializing Behavior Agent set to: {self._behavior}")
            self.initialized = True

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
            self.logger.web_info("TRAFFIC_LIGHT", "Red light detected", "ACTION")
            return self.emergency_stop()


        ##############################################
        ####### Pedestrian Avoidance Behaviors #######
        ##############################################
        walker_state, walker, w_distance = self._object_detector.detect_pedestrians(self._direction)

        if walker_state:
            # Distance is computed from the center of the car and the walker,
            # we use bounding boxes to calculate the actual distance
            
            distance = compute_distance_from_center(actor1=self._vehicle, actor2=walker, distance=w_distance)
            
            self.logger.debug(f"Pedestrian {walker} detected, distance: {distance}")
            
            # Brake moment
            if distance <= 3:
                self.logger.web_debug("PEDESTRIAN", f"MI FERMO PER IL PEDONE")
                return self.emergency_stop()

            else:
                self.logger.web_debug("PEDESTRIAN", f"HO VISTO UN PEDONE, RALLENTO!")
                self.set_target_speed(self._approach_speed)
                return self._local_planner.run_step()



        ##################################################
        ########## Obstacle Avoidance Behaviors ##########
        ##################################################
        # DA CONTROLLARE
        so_state, static_obstacle, so_distance = self._object_detector.detect_static_obstacles(ego_vehicle_wp)
        cone_state, static_obstacle_cone, so_distance_cone = self._object_detector.detect_static_obstacles(ego_vehicle_wp,max_distance= 20, obj_filter="*static.prop.constructioncone*")

        if so_state:        
            distance = compute_distance_from_center(actor1 =self._vehicle, actor2 = static_obstacle, distance=so_distance)

            self.logger.critical(f"Static obstacle:{static_obstacle} detected, distance: {distance}")

            overtake_path = None
            if not ego_vehicle_wp.is_junction:
                overtake_path = self._overtake_manager.run_overtake_step(
                    object_to_overtake=static_obstacle, 
                    ego_vehicle_wp=ego_vehicle_wp, 
                    distance_same_lane=1, 
                    distance_other_lane=20, 
                    distance_from_object=so_distance, 
                    speed_limit=self._speed_limit
                )
    
            # Check if the overtake path was generated and start the overtake
            if overtake_path: 
                self.logger.web_info("OVERTAKE", "Overtake Path Found - OBSTACLE", "INFO")
                self._overtake_manager.update_overtake_plan(overtake_path=overtake_path)
                self.logger.web_action("OVERTAKE", "Starting the overtake", "ACTION")
            

            if not self._overtake_manager.in_overtake and distance < 3 * self._behavior.braking_distance:
                self.set_target_speed(self._approach_speed*2)

                if distance < 2 * self._behavior.braking_distance:
                    self.set_target_speed(self._approach_speed)

                    if distance < self._behavior.braking_distance:
                        self.logger.both_action("OBSTACLE", "[WARNING - Static Obstacle Avoidance] Emergency stop due to static obstacle proximity")
                        return self.emergency_stop()
                
                return self._local_planner.run_step()
            
        if cone_state and not self._overtake_manager.in_overtake:
            o_loc = static_obstacle_cone.get_location()
            o_wp = self._map.get_waypoint(o_loc)
            if o_wp.lane_id == ego_vehicle_wp.lane_id:
                self._local_planner.set_lateral_offset(-2 * static_obstacle_cone.bounding_box.extent.y - self._vehicle.bounding_box.extent.y)
                self.logger.web_debug("STATIC_OBSTACLE", f"Static obstacle {static_obstacle_cone} detected in the same lane, applying lateral offset")
                self._lateral_offset_active = True
                self._lateral_offset_type = 'cone'
            else:
                self._local_planner.set_lateral_offset(.1 * static_obstacle_cone.bounding_box.extent.y + self._vehicle.bounding_box.extent.y)
                self.logger.web_debug("STATIC_OBSTACLE", f"Static obstacle {static_obstacle_cone} detected in a different lane")
                self._lateral_offset_active = True
                self._lateral_offset_type = 'cone'

        elif self._lateral_offset_active and not cone_state and self._lateral_offset_type == 'cone':
            self.logger.web_debug("STATIC_OBSTACLE", "No static obstacle detected, resetting cone lateral offset")
            self._local_planner.set_lateral_offset(0.0)
            self._lateral_offset_active = False
            self._lateral_offset_type = None
        


        #################################################
        ######### Bicycle Avoidance Behaviors ##########
        ################################################
        # DA CONTROLLARE
        bicycle_state, bicycle, b_distance = self._object_detector.detect_bicycles(ego_vehicle_wp)
        
        if bicycle_state:
            b_distance = compute_distance_from_center(actor1=self._vehicle, actor2=bicycle, distance=b_distance)
            # Ottieni l'ID della corsia del veicolo ego e della bicicletta
            ego_lane_id = ego_vehicle_wp.lane_id
            bicycle_wp = self._map.get_waypoint(bicycle.get_location())
            bicycle_lane_id = bicycle_wp.lane_id

            # Get the yaw of the ego vehicle and the vehicle in front.
            ego_current_yaw = self._vehicle.get_transform().rotation.yaw
            bicycle_current_yaw = bicycle.get_transform().rotation.yaw

            # Calcola la differenza di yaw, normalizzandola tra -180 e 180
            diff_yaw = (bicycle_current_yaw - ego_current_yaw + 180) % 360 - 180
            
            # Definisci una soglia per considerare una bicicletta "allineata"
            is_aligned_threshold = 20  # Gradi

            # Controlla se i veicoli sono approssimativamente allineati
            if abs(diff_yaw) < is_aligned_threshold:                   
                # Controlla se la bicicletta è nella STESSA corsia del veicolo ego
                if ego_lane_id == bicycle_lane_id:
                    self.logger.info("Bicycle {bicycle.id} detected in the SAME lane ({ego_lane_id}), aligned. Distance: {b_distance:.2f}m.")
                    # Controlla se la bicicletta è vicina al centro della corsia
                    if is_bicycle_near_center(vehicle_location=bicycle.get_location(), ego_vehicle_wp=ego_vehicle_wp) and get_speed(self._vehicle) < 0.1:
                        self.logger.info(f"Bicycle {bicycle.id} is near the center of our lane. Attempting to overtake.")

                        overtake_path = None
                        if not ego_vehicle_wp.is_junction:
                            overtake_path = self._overtake_manager.run_overtake_step(
                                object_to_overtake=bicycle, ego_vehicle_wp=ego_vehicle_wp, distance_same_lane=1, distance_from_object=b_distance, speed_limit = self._speed_limit
                            )                                               
    
                        if overtake_path:
                            self.logger.both_action("BICYCLE", f"Overtake path found for bicycle {bicycle.id}. Initiating overtake.")
                            self._overtake_manager.update_overtake_plan(overtake_path=overtake_path)

                        if not self._overtake_manager.in_overtake: 
                            self.logger.both_info("OVERTAKE", f"Overtake path not found for bicycle {bicycle.id}. Emergency stop.")
                            return self.emergency_stop()
                    else: 
                        self.logger.info(f"Bicycle {bicycle.id} is in our lane, but not centered. Applying lateral offset.")
                        new_offset = -(2.5 * bicycle.bounding_box.extent.y + self._vehicle.bounding_box.extent.y)
                        self.logger.web_action("BICYCLE", f"Applying lateral offset of {new_offset:.2f}m to avoid bicycle {bicycle.id}")
                        
                        # Solo se non c'è un offset per coni attivo
                        if self._lateral_offset_type != 'cone':
                            self._local_planner.set_lateral_offset(new_offset)
                            self._lateral_offset_active = True
                            self._lateral_offset_type = 'bicycle'
                        
                        target_speed = min([
                            self._behavior.max_speed,
                            self._speed_limit - self._behavior.speed_lim_dist])
                        self._local_planner.set_speed(target_speed)
                        control = self._local_planner.run_step(debug=debug)
                else: # La bicicletta è allineata (stessa direzione) ma in una corsia DIVERSA
                    self.logger.info(f"Bicycle {bicycle.id} detected in a different lane (Ego: {ego_lane_id}, Bicycle: {bicycle_lane_id}), aligned. No offset applied.")
                    # Reset solo se non ci sono coni e l'offset era attivo per le biciclette
                    if self._lateral_offset_active and not cone_state:
                        self.logger.both_action("BICYCLE", "Resetting previously active lateral offset as current bicycle is in a different lane.")
                        self._local_planner.set_lateral_offset(0.0)
                        self._lateral_offset_active = False
            
            # CASO: Bicicletta NON allineata
            else:
                self.logger.info(f"Bicycle {bicycle.id} detected in a different lane (Ego: {ego_lane_id}, Bicycle: {bicycle_lane_id}), NOT aligned. Distance: {b_distance:.2f}m.")
                is_potentially_crossing = abs(diff_yaw) > is_aligned_threshold and abs(abs(diff_yaw) - 180) > is_aligned_threshold 

                if is_potentially_crossing and b_distance < self._behavior.braking_distance * 2.5:
                    self.logger.warning(f"Bicycle {bicycle.id} is potentially CROSSING at {b_distance:.2f}m. Initiating emergency stop or significant slowdown.")
                    if b_distance < self._behavior.braking_distance:
                        self.logger.critical(f"Bicycle {bicycle.id} CRITICAL proximity while crossing. EMERGENCY STOP.")
                        return self.emergency_stop()
                    else:
                        target_speed = min(self._approach_speed / 2, get_speed(self._vehicle) / 2)
                        target_speed = max(target_speed, 0)
                        self.set_target_speed(target_speed)
                        self.logger.info(f"Bicycle {bicycle.id} is crossing. Slowing down to {target_speed:.1f} km/h.")
                        control = self._local_planner.run_step(debug=debug)

                elif get_speed(bicycle) < 1 and b_distance < self._behavior.braking_distance * 2:
                    self.logger.info(f"Bicycle {bicycle.id} is NOT aligned, slow/stopped, and close. Decelerating to approach speed.")
                    self.set_target_speed(self._approach_speed)
                    control = self._local_planner.run_step(debug=debug)
                
                else:
                    self.logger.info(f"Bicycle {bicycle.id} is NOT aligned, but not an immediate crossing threat. Maintaining cautious speed.")

        # Reset del lateral offset solo se non ci sono né coni né biciclette rilevanti
        elif self._lateral_offset_active and not cone_state:
            self.logger.info(f"No bicycle or cone detected, resetting lateral offset.")
            self._local_planner.set_lateral_offset(0.0)
            self._lateral_offset_active = False
                

        
    

    
        # Se stiamo per entrare in un incrocio o siamo già in uno
        # Gestione della velocità negli incroci

        # Inizializza la variabile di stato se non esiste
        if not hasattr(self, '_in_junction_state'):
            self._in_junction_state = False
            self._junction_exit_time = 0

        # Periodo di grazia dopo l'uscita dall'incrocio (2 secondi)
        current_time = time.time()

        # Rileva l'ingresso e l'uscita dall'incrocio
        
        # Gestione della velocità in base allo stato dell'incrocio

        # if self._in_junction_state or (current_time - self._junction_exit_time < junction_grace_period):
        #     # Siamo nell'incrociante o nel periodo di grazia: usa approach_speed

        #     self.logger.web_debug("JUNCTION", f"Velocità mantenuta a {self._approach_speed} km/h nell'incrocio")
        #     self.set_target_speed(self._approach_speed)
        #     return self._local_planner.run_step()
            
        #############################################
        ######## Vehicle Avoidance Behaviors ########
        #############################################
        vehicle_state, vehicle, distance = self._object_detector.detect_vehicles(
            ego_vehicle_wp, 
            max_distance=30, 
            lane_offset=0 if self._direction == RoadOption.LANEFOLLOW else 
                         (-1 if self._direction == RoadOption.CHANGELANELEFT else 1),
            angle_th=30 if self._direction == RoadOption.LANEFOLLOW else 180
        )

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = compute_distance_from_center(actor1=self._vehicle, actor2=vehicle, distance=distance)
            
            # Check if the car is legitimately stopped
            _, is_abandoned = self.is_vehicle_legitimately_stopped(vehicle)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance and not is_abandoned:
                self.logger.web_debug("VEHICLE", f"VEICOLO DAVANTI, MI FERMO!")
                self.logger.critical(f"Emergency stop due to vehicle proximity")
                return self.emergency_stop()
            elif distance < self._behavior.min_proximity_threshold*2 and not is_abandoned:
                self.logger.web_debug("VEHICLE", f"VEICOLO DAVANTI, RALLENTO!")
                target_speed = self._approach_speed
                self._local_planner.set_speed(target_speed)

            if is_abandoned:
                # If the vehicle is considered parked/abandoned, we can try to overtake it.
                self.logger.debug(f"Vehicle {vehicle} is parked/abandoned, attempting to overtake.")
                
                # Inizializza il dizionario di veicoli temporaneamente fermi se non esiste
                if not hasattr(self, '_stopped_vehicles_timer'):
                    self._stopped_vehicles_timer = {}
                
                current_time = time.time()
                vehicle_id = vehicle.id
                wait_time_before_overtake = 15.0  # Tempo di attesa in secondi prima di avviare il sorpasso
                
                # Registra il tempo di fermo del veicolo se non è già registrato
                if vehicle_id not in self._stopped_vehicles_timer:
                    self._stopped_vehicles_timer[vehicle_id] = current_time
                    return self.emergency_stop()  # Fermati e aspetta mentre verifichi
                
                # Calcola quanto tempo è passato da quando il veicolo è stato rilevato come fermo
                elapsed_time = current_time - self._stopped_vehicles_timer[vehicle_id]
                
                # Se non è passato abbastanza tempo, continua ad aspettare
                if elapsed_time < wait_time_before_overtake:
                    return self.emergency_stop()  # Continua a fermarti e aspetta
                
                overtake_path_generated_this_step = False
                if not self._overtake_manager.in_overtake: # Only attempt to start a new overtake if not already in one
                    # Calculate a dynamic distance_same_lane, e.g., 1 second of travel at current speed, min 5m.
                    # Convert current speed from m/s to km/h for self._speed, then back to m/s for distance.
                    # get_speed returns m/s.
                    safe_approach_distance = max(1.0, get_speed(self._vehicle) * 1.0) 
                    
                    overtake_path = None
                    if not ego_vehicle_wp.is_junction:
                        overtake_path = self._overtake_manager.run_overtake_step(
                            object_to_overtake=vehicle, 
                            ego_vehicle_wp=ego_vehicle_wp, 
                            distance_same_lane=safe_approach_distance,
                            distance_from_object=distance, 
                            speed_limit=self._vehicle.get_speed_limit() # Use current actual speed limit from vehicle
                        )
                        
                    if overtake_path: 
                        self.logger.both_action("OVERTAKE", f" Sorpasso possibile, avvio sorpasso")
                        self._overtake_manager.update_overtake_plan(overtake_path=overtake_path)
                        overtake_path_generated_this_step = True
                    else:
                        # Failed to find an overtake path, stop.
                        self.logger.both_warning("OVERTAKE", "IL VEICOLO E' DAVVERO FERMO, CERCO UN PERCORSO DI SORPASSO")
                        return self.emergency_stop()

                if self._overtake_manager.in_overtake:  
                    self.logger.web_debug("OVERTAKE", "SORPASSO IN CORSO")                 
                    current_actual_speed_limit = self._vehicle.get_speed_limit()
                    
                    effective_max_speed = self._behavior.max_speed
                    overtake_speed_margin_kmh = getattr(self._behavior, 'overtake_speed_margin_kmh', 10.0) 

                    # If current behavior is Cautious, consider using Normal's speed parameters for a more effective overtake.
                    if isinstance(self._behavior, Cautious):
                        # Create a temporary Normal behavior instance to access its parameters
                        temp_normal_behavior = Normal()
                        overtake_speed_margin_kmh = getattr(temp_normal_behavior, 'overtake_speed_margin_kmh', 10.0) # Use Normal's margin
                        effective_max_speed = temp_normal_behavior.max_speed # Use Normal's max_speed
                        self.logger.info(f"Current behavior is Cautious. Using Normal's speed parameters for overtake.")
                                
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
                    elif current_speed_kmh > target_speed and self._overtake_manager.in_overtake : # if already faster than calculated target during overtake
                         target_speed = min(effective_max_speed, current_speed_kmh) # maintain current speed if safe, capped by max_speed
                         self.logger.web_debug("OVERTAKE - ADJUSTING", f"Adjusting speed during overtake. Current speed: {current_speed_kmh:.2f} km/h, target speed: {target_speed:.2f} km/h.")

                    self._local_planner.set_speed(target_speed)
                    self.logger.info(f"[INFO - Overtake] Active overtake. Target speed: {target_speed:.2f} km/h. Current speed: {current_speed_kmh:.2f} km/h. Limit: {current_actual_speed_limit:.2f} km/h.")
                    

                control = self._local_planner.run_step(debug=debug) # This will use the speed set above if overtaking
            
            else: # Vehicle is not abandoned
                # If the vehicle is not parked, we can follow it
                self.logger.debug(f"[DEBUG - Vehicle Avoidance] Vehicle {vehicle} is not parked, following.")
                self.logger.web_debug("VEHICLE FOLLOWING", f"Vehicle following")
                control = self.car_following_manager(vehicle, distance, debug=debug)
        
            return control
        
            ##############################################
        ################ STOP SIGN ###################
        ##############################################
        stop_sign_state, stop_sign, stop_sign_distance = self.stop_sign_manager(self._vehicle)

        if stop_sign_state:
            # Calcola la distanza effettiva tenendo conto delle dimensioni dei veicoli
            distance = compute_distance_from_center(actor1=self._vehicle, distance=stop_sign_distance)
            
            # Se siamo già in attesa ad uno stop
            if self._stop_sign_waiting:
                # Calcola quanto tempo abbiamo aspettato
                wait_time = time.time() - self._stop_sign_wait_time
                self.logger.web_debug("STOP_SIGN", f"Attesa allo stop: {wait_time:.1f}s / {self._min_stop_time}s")
                
                # Se abbiamo aspettato abbastanza
                if wait_time >= self._min_stop_time:
                    # Segna lo stop come rispettato
                    if stop_sign and hasattr(stop_sign, 'id'):
                        self._stop_sign_respected[stop_sign.id] = time.time()
                        self.logger.both_action("STOP_SIGN", f"Stop sign {stop_sign.id} rispettato, ripartenza")
                    
                    # Ripristina lo stato e imposta una velocità moderata per ripartire
                    self._stop_sign_waiting = False
                    self.set_target_speed(self._approach_speed)
                    
                    # Continua con l'esecuzione normale
                    control = self._local_planner.run_step(debug=debug)
                    return control
                else:
                    # Non è ancora passato abbastanza tempo, continua a fermarti
                    return self.emergency_stop()
            
            # Se non siamo in attesa ma siamo vicini a uno stop non ancora rispettato
            elif distance < 3 and stop_sign.id not in self._stop_sign_respected:
                self.logger.both_action("STOP_SIGN", "Fermata allo stop")
                self._stop_sign_waiting = True
                self._stop_sign_wait_time = time.time()
                return self.emergency_stop()
            
            # NUOVO: Fase di rallentamento anticipato
            # Se rilevo il segnale in lontananza, inizio a rallentare gradualmente
            elif distance < self._behavior.braking_distance * 2.5 and stop_sign.id not in self._stop_sign_respected:

                self.logger.web_debug("STOP_SIGN", f"Rallentamento anticipato per il segnale di stop {stop_sign.id}, distanza: {distance:.1f}m, velocità target: {target_speed:.1f} km/h")
                self.logger.info(f"Approaching stop sign at {distance:.1f}m, slowing down to {target_speed:.1f} km/h")
                self.set_target_speed(self._approach_speed)

            return self._local_planner.run_step(debug=debug)

        if (self._incoming_waypoint.is_junction or ego_vehicle_wp.is_junction):
            # Prima di entrare nell'incrocio, verifica se è sicuro attraversarlo
            can_cross = self.junction_manager(ego_vehicle_wp)
            if not can_cross:
                self.logger.web_action("JUNCTION", "Non è sicuro attraversare l'incrocio, mi fermo")
                return self.emergency_stop()
            
            # Se è sicuro, procedi con il comportamento normale
            if (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]) or (self._direction in [RoadOption.LEFT, RoadOption.RIGHT]):
                self._in_junction_state = True
                self.set_target_speed(self._approach_speed)  
                self.logger.web_action("JUNCTION", f"Ingresso nell'incrocio, velocità ridotta a {self._approach_speed} km/h")
                return self._local_planner.run_step()
            else:
                # Se non stiamo girando, manteniamo la velocità normale
                target_speed = self._approach_speed * 2
                self._local_planner.set_speed(target_speed)
                self.logger.web_debug("JUNCTION", f"Non sto girando, mantengo la velocità a {target_speed:.2f} km/h")
                return self._local_planner.run_step()

        # Nuova parte: controllo continuo durante l'attraversamento dell'incrocio
        # elif ego_vehicle_wp.is_junction and self._in_junction_state:
        #     # Controlla continuamente se è ancora sicuro attraversare
        #     can_cross = self.junction_manager(ego_vehicle_wp)
        #     if not can_cross:
        #         self.logger.web_action("JUNCTION", "Condizioni cambiate, non è più sicuro proseguire nell'incrocio")
        #         return self.emergency_stop()
            
        #     # Se è ancora sicuro, continua con velocità ridotta
        #     self.set_target_speed(self._approach_speed)
        #     self.logger.web_debug("JUNCTION", f"Attraversamento dell'incrocio in corso, velocità mantenuta a {self._approach_speed} km/h")
        #     return self._local_planner.run_step()
            

        elif not ego_vehicle_wp.is_junction and self._in_junction_state:
            # Uscita dall'incrocio
            self._in_junction_state = False
            self._in_junction_grace_period = True  # Inizia il periodo di grazia
            self._junction_exit_time = current_time
            self.set_target_speed(self._approach_speed)
            self.logger.web_debug("JUNCTION", f"Uscita dall'incrocio, mantengo velocità ridotta a {self._approach_speed} km/h per periodo di grazia di {self._junction_grace_duration} secondi")
            return self._local_planner.run_step()

        # Aggiungi un nuovo caso per gestire il periodo di grazia
        elif self._in_junction_grace_period:
            # Controlla se il periodo di grazia è terminato
            if current_time - self._junction_exit_time >= self._junction_grace_duration:
                # Periodo di grazia terminato, torna alla velocità normale
                self._in_junction_grace_period = False
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                self.logger.web_action("JUNCTION", f"Periodo di grazia terminato, velocità ripristinata a {target_speed:.2f} km/h")
            else:
                # Ancora nel periodo di grazia, mantieni la velocità ridotta
                self.set_target_speed(self._approach_speed)
                remaining_time = self._junction_grace_duration - (current_time - self._junction_exit_time)
                self.logger.web_debug("JUNCTION", f"Periodo di grazia: rimangono {remaining_time:.1f} secondi")
    
            return self._local_planner.run_step()

        # 4: Normal behavior
        else:
            if not self._overtake_manager.in_overtake:         
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                self.logger.web_debug("Behavior", f"Nessun Evento Rilevato! Ho settato la velocità a: {target_speed:.2f} km/h, il limite della strada è {self._speed_limit:.2f} km/h")
            else:
                control = self._local_planner.run_step(debug=debug)
        return control

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        # Cleanup dei timer dei veicoli fermi
        self._cleanup_stopped_vehicles_timer()
        
        # Cleanup degli stop sign rispettati dopo un certo tempo
        current_time = time.time()
        expired_stops = []
        for stop_id, timestamp in self._stop_sign_respected.items():
            # Consideriamo "scaduto" un segnale di stop dopo 30 secondi
            if current_time - timestamp > 30.0:
                expired_stops.append(stop_id)
    
        for stop_id in expired_stops:
            del self._stop_sign_respected[stop_id]
            self.logger.debug(f"Stop sign {stop_id} rimosso dalla lista dei rispettati (scaduto)")
    
        # Resto del codice esistente...
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
        if self._overtake_manager.overtake_cnt > 0:
            self._overtake_manager.overtake_cnt -= 1
            self._local_planner.draw_waypoints(color=carla.Color(0, 255, 0))
        else: 
            self._overtake_manager.in_overtake = False

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(self._vehicle, lights_list)

        if affected:
            self.logger.web_warning("TRAFFIC_LIGHT", "Red light detected")
            
        return affected
    
    def stop_sign_manager(self, vehicle: carla.Vehicle, sign_distance: int = 40):
        """
        Gestisce il rilevamento dei segnali di stop.
    
        :param vehicle: veicolo dell'agente
        :param sign_distance: distanza massima di rilevamento del segnale
        :return: (affected, signal, distance) tuple con stato, segnale e distanza
        """
        affected, signal = self._affected_by_sign(vehicle=vehicle, sign_type="206", max_distance=sign_distance)
        distance = -1 if not affected else dist(a=vehicle, b=signal)
        
        # Se il segnale è stato rispettato di recente, lo notifichiamo ma non ci fermiamo
        if affected and signal and signal.id in self._stop_sign_respected:
            last_respect_time = self._stop_sign_respected[signal.id]
            if time.time() - last_respect_time < 30.0:  # 30 secondi di "immunità" dopo aver rispettato lo stop
                self.logger.debug(f"Stop sign {signal.id} già rispettato recentemente")
    
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
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

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
        # Get the vehicle control
        control = carla.VehicleControl()
        # Set the throttle and brake values to 0.0 and self._max_brake, respectively.
        control.throttle = 0.0
        control.brake = self._max_brake
        # Set the hand brake to False.
        control.hand_brake = False
        return control

    def junction_manager(self, ego_vehicle_wp):
        """
        Gestisce l'attraversamento degli incroci.
    
        :param ego_vehicle_wp: waypoint attuale del veicolo
        :return: True se è sicuro attraversare, False altrimenti
        """
        # Verifica se siamo in un incrocio
        if not ego_vehicle_wp.is_junction:
            return True
    
        # Ottieni l'oggetto junction
        junction = ego_vehicle_wp.get_junction()
        if not junction:
            return True
    
        # Ottieni la direzione pianificata dal local planner
        planned_direction = None
        if self._local_planner.target_road_option in [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]:
            planned_direction = self._local_planner.target_road_option
        else:
            # Se non abbiamo una direzione chiara, usiamo STRAIGHT come default
            planned_direction = RoadOption.STRAIGHT
    
        # Categorizza i veicoli nell'incrocio
        vehicles = self._categorize_vehicles_at_junction(junction, ego_vehicle_wp)
        
        # Verifica se possiamo attraversare in sicurezza
        can_cross = self._check_junction_rules(vehicles, planned_direction)
        
        # Log della decisione
        if can_cross:
            self.logger.web_action("JUNCTION", f"Attraversamento sicuro dell'incrocio (direzione: {planned_direction})")
        else:
            self.logger.web_debug("JUNCTION", f"Attesa all'incrocio (direzione: {planned_direction})")
    
        return can_cross

    def _categorize_vehicles_at_junction(self, junction, ego_vehicle_wp):
        """
        Categorizza i veicoli nell'incrocio in base alla loro posizione relativa.
    
        :param junction: oggetto junction
        :param ego_vehicle_wp: waypoint del veicolo ego
        :return: dizionario con veicoli categorizzati per direzione
        """
        vehicles = {"left": [], "right": [], "front": []}
    
        # Ottieni la posizione e la direzione del veicolo
        ego_vehicle = self._vehicle
        ego_location = ego_vehicle.get_location()
        ego_transform = ego_vehicle.get_transform()
        ego_forward = ego_transform.get_forward_vector()
    
        # Raggruppa tutti i veicoli in prossimità dell'incrocio (entro 25 metri)
        vehicles_list = []
        for vehicle in self._world.get_actors().filter('vehicle.*'):
            if vehicle.id == ego_vehicle.id:
                continue  # Salta il veicolo ego
        
            vehicle_location = vehicle.get_location()
            vehicle_wp = self._map.get_waypoint(vehicle_location)
        
            # Calcola la distanza dal veicolo ego
            distance = vehicle_location.distance(ego_location)
        
            # Considera solo veicoli entro 25 metri e in prossimità dell'incrocio
            if distance <= 25.0 and (vehicle_wp.is_junction or 
                                    distance_to_junction(vehicle_wp) < 10.0):
                vehicles_list.append((vehicle, distance))
    
        # Ordina i veicoli per distanza
        vehicles_list.sort(key=lambda x: x[1])
    
        # Categorizza i veicoli in base alla loro posizione relativa
        for vehicle, _ in vehicles_list:
            vehicle_location = vehicle.get_location()
            vehicle_transform = vehicle.get_transform()
            vehicle_forward = vehicle_transform.get_forward_vector()
        
            # Calcola il vettore dal veicolo ego al veicolo target
            to_vehicle = vehicle_location - ego_location
            to_vehicle = carla.Vector3D(to_vehicle.x, to_vehicle.y, 0)
            to_vehicle_norm = math.sqrt(to_vehicle.x**2 + to_vehicle.y**2)
        
            if to_vehicle_norm < 0.001:
                continue  # Evita la divisione per zero
        
            to_vehicle = carla.Vector3D(to_vehicle.x/to_vehicle_norm, to_vehicle.y/to_vehicle_norm, 0)
        
            # Calcola i prodotti scalari per determinare la posizione relativa
            forward_dot = ego_forward.x * to_vehicle.x + ego_forward.y * to_vehicle.y
            right_vector = carla.Vector3D(ego_forward.y, -ego_forward.x, 0)
            right_dot = right_vector.x * to_vehicle.x + right_vector.y * to_vehicle.y
        
            # Determina la categoria in base agli angoli
            if forward_dot >= 0.7:  # Angolo piccolo, veicolo davanti
                vehicles["front"].append(vehicle)
            elif right_dot >= 0.7:  # Veicolo a destra
                vehicles["right"].append(vehicle)
            elif right_dot <= -0.7:  # Veicolo a sinistra
                vehicles["left"].append(vehicle)
            elif forward_dot <= -0.7:  # Veicolo dietro (non ci interessa)
                pass
            else:
                # Casi intermedi, classifichiamo in base all'angolo maggiore
                if abs(right_dot) > abs(forward_dot):
                    if right_dot > 0:
                        vehicles["right"].append(vehicle)
                    else:
                        vehicles["left"].append(vehicle)
                else:
                    if forward_dot > 0:
                        vehicles["front"].append(vehicle)
    
        # Aggiungi questi log di debug alla fine del metodo
        self.logger.debug(f"Veicoli categorizzati all'incrocio: sinistra:{len(vehicles['left'])}, destra:{len(vehicles['right'])}, frontali:{len(vehicles['front'])}")
        self.logger.web_debug("JUNCTION", f"Veicoli all'incrocio - sinistra:{len(vehicles['left'])}, destra:{len(vehicles['right'])}, fronte:{len(vehicles['front'])}")
        
        return vehicles

    def _check_junction_rules(self, vehicles, planned_direction):
        """
        Applica le regole di precedenza negli incroci, considerando solo i veicoli
        che possono interferire con la nostra traiettoria.
        
        :param vehicles: dizionario con veicoli classificati per direzione
        :param planned_direction: direzione pianificata (RoadOption)
        :return: True se è sicuro procedere, False altrimenti
        """
        # Filtra i veicoli prima di fare i controlli di sicurezza
        filtered_vehicles = self._filter_relevant_vehicles(vehicles, planned_direction)
        
        # Log dei veicoli rilevanti dopo il filtraggio
        self.logger.web_debug("JUNCTION", f"Veicoli rilevanti dopo filtraggio - sinistra:{len(filtered_vehicles['left'])}, "
                                        f"destra:{len(filtered_vehicles['right'])}, fronte:{len(filtered_vehicles['front'])}")
        
        # Ottieni il primo veicolo in ogni direzione (più vicino)
        vehicle_left = filtered_vehicles["left"][0] if filtered_vehicles["left"] else None
        vehicle_right = filtered_vehicles["right"][0] if filtered_vehicles["right"] else None
        vehicle_front = filtered_vehicles["front"][0] if filtered_vehicles["front"] else None

        # Se non ci sono veicoli rilevanti, è sicuro procedere
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
            self.logger.web_debug("JUNCTION", f"Direzione DRITTO - Sicurezza: sinistra:{left_is_safe}, destra:{right_is_safe}, fronte:{front_is_safe}")
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
            self.logger.web_debug("JUNCTION", f"Direzione SINISTRA - Sicurezza: fronte:{front_is_safe}, destra:{right_is_safe}")
            return result

        # Logica per SVOLTA A DESTRA
        elif planned_direction == RoadOption.RIGHT:
            # Controlla i veicoli frontali (che svoltano a sinistra)
            front_is_safe = vehicle_front is None or not self._is_vehicle_turning_left(vehicle_front)
            # Veicoli da sinistra che vanno dritto
            left_is_safe = vehicle_left is None or not self._is_vehicle_going_straight(vehicle_left)
        
            result = front_is_safe and left_is_safe
            self.logger.web_debug("JUNCTION", f"Direzione DESTRA - Sicurezza: fronte:{front_is_safe}, sinistra:{left_is_safe}")
            return result

        # Per altre situazioni, sii cauto
        return False
    
    def _filter_relevant_vehicles(self, vehicles, planned_direction):
        """
        Filtra i veicoli per considerare solo quelli che potrebbero interferire con la nostra traiettoria.
        
        :param vehicles: dizionario con veicoli classificati per direzione
        :param planned_direction: direzione pianificata (RoadOption)
        :return: dizionario con veicoli filtrati
        """
        filtered = {"left": [], "right": [], "front": []}
        ego_vehicle = self._vehicle
        ego_location = ego_vehicle.get_location()
        ego_waypoint = self._map.get_waypoint(ego_location)
        
        # Ottieni la traiettoria pianificata attraverso l'incrocio
        planned_path = self._get_junction_path(ego_waypoint, planned_direction)
        
        # Funzione per verificare se un veicolo può interferire con la nostra traiettoria
        def could_interfere(vehicle):
            # Ottieni waypoint del veicolo
            vehicle_location = vehicle.get_location()
            vehicle_waypoint = self._map.get_waypoint(vehicle_location)
            
            # Se il veicolo è fermo e non è sulla nostra corsia, probabilmente non interferisce
            vehicle_speed = get_speed(vehicle)
            if vehicle_speed < 0.5:  # quasi fermo (0.5 m/s ≈ 1.8 km/h)
                # Verifica se è sulla nostra stessa strada/corsia
                if not self._is_on_same_lane(vehicle_waypoint, planned_path):
                    self.logger.web_debug("JUNCTION", f"Veicolo {vehicle.id} ignorato: fermo e non sulla nostra traiettoria")
                    return False
            
            # Per veicoli in movimento, verifica potenziale intersezione con la nostra traiettoria
            return self._could_paths_intersect(vehicle, planned_path)
        
        # Filtra i veicoli in ciascuna direzione
        for direction in ["left", "right", "front"]:
            filtered[direction] = [v for v in vehicles[direction] if could_interfere(v)]
        
        return filtered
    

    def _is_on_same_lane(self, vehicle_waypoint, planned_path):
        """
        Verifica se un waypoint è sulla stessa corsia del percorso pianificato.
        
        :param vehicle_waypoint: waypoint del veicolo
        :param planned_path: lista di waypoint del percorso pianificato
        :return: True se è sulla stessa corsia, False altrimenti
        """
        # Se il percorso è vuoto, consideriamo che non sia sulla stessa corsia
        if not planned_path:
            return False
        
        # Controlla se il waypoint del veicolo è sulla stessa corsia di uno dei waypoint pianificati
        for wp in planned_path:
            # Verifica se hanno lo stesso road_id e lane_id
            if (vehicle_waypoint.road_id == wp.road_id and 
                vehicle_waypoint.lane_id == wp.lane_id):
                return True
        
        return False
    

    def _could_paths_intersect(self, vehicle, planned_path):
        """
        Verifica se la traiettoria del veicolo potrebbe intersecare il nostro percorso pianificato.
        
        :param vehicle: veicolo da controllare
        :param planned_path: lista di waypoint del percorso pianificato
        :return: True se le traiettorie potrebbero intersecarsi, False altrimenti
        """
        # Se non abbiamo un percorso pianificato, assumiamo che potrebbero intersecarsi
        if not planned_path:
            return True
        
        vehicle_location = vehicle.get_location()
        vehicle_wp = self._map.get_waypoint(vehicle_location)
        
        # Ottieni la traiettoria prevista del veicolo
        vehicle_future_waypoints = []
        next_wp = vehicle_wp
        
        # Proietta la traiettoria del veicolo per 10 metri
        for _ in range(5):
            next_wps = next_wp.next(2.0)
            if not next_wps:
                break
            next_wp = next_wps[0]
            vehicle_future_waypoints.append(next_wp)
        
        # Verifica se le traiettorie si intersecano
        for v_wp in vehicle_future_waypoints:
            for p_wp in planned_path:
                # Calcola distanza tra i waypoint
                dist = v_wp.transform.location.distance(p_wp.transform.location)
                if dist < 3.0:  # Soglia di 3 metri per considerare un'intersezione
                    return True
        
        return False
    

    def _get_junction_path(self, start_waypoint, direction):
        """
        Ottiene il percorso pianificato attraverso l'incrocio.
        
        :param start_waypoint: waypoint di partenza
        :param direction: direzione pianificata (RoadOption)
        :return: lista di waypoint che attraversano l'incrocio
        """
        # Ottieni il percorso dal local planner
        if hasattr(self._local_planner, 'get_waypoints_in_junction'):
            return self._local_planner.get_waypoints_in_junction()
        
        # Se non è disponibile, genera un percorso semplice
        waypoints = []
        next_wp = start_waypoint
        
        # Cerca il primo waypoint dell'incrocio
        while not next_wp.is_junction:
            next_wps = next_wp.next(2.0)
            if not next_wps:
                return waypoints
            next_wp = next_wps[0]
        
        # Aggiungi il primo waypoint dell'incrocio
        junction_wp = next_wp
        waypoints.append(junction_wp)
        
        # Attraversa l'incrocio nella direzione specificata
        while junction_wp.is_junction:
            # Ottieni tutti i waypoint successivi
            next_options = junction_wp.next(2.0)
            if not next_options:
                break
            
            # Scegli il waypoint appropriato in base alla direzione
            if direction == RoadOption.LEFT:
                # Prendi il waypoint più a sinistra
                next_options.sort(key=lambda wp: wp.transform.rotation.yaw)
                junction_wp = next_options[0]  # Il più a sinistra
            elif direction == RoadOption.RIGHT:
                # Prendi il waypoint più a destra
                next_options.sort(key=lambda wp: wp.transform.rotation.yaw, reverse=True)
                junction_wp = next_options[0]  # Il più a destra
            else:  # STRAIGHT
                # Prendi il waypoint con la direzione più simile
                current_yaw = junction_wp.transform.rotation.yaw
                junction_wp = min(next_options, key=lambda wp: 
                                abs((wp.transform.rotation.yaw - current_yaw + 180) % 360 - 180))
            
            waypoints.append(junction_wp)
        
        # Aggiungi alcuni waypoint dopo l'incrocio
        for _ in range(3):
            next_wps = junction_wp.next(2.0)
            if not next_wps:
                break
            junction_wp = next_wps[0]
            waypoints.append(junction_wp)
        
        return waypoints

    def _is_vehicle_turning_left(self, vehicle):
        """
        Verifica se un veicolo sta svoltando a sinistra.
    
        :param vehicle: il veicolo da controllare
        :return: True se sta svoltando a sinistra, False altrimenti
        """
        # Ottieni la posizione attuale e la direzione del veicolo
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_wp = self._map.get_waypoint(vehicle_location)
    
        # Ottieni il prossimo waypoint nella direzione di marcia
        vehicle_forward = vehicle_transform.get_forward_vector()
        next_wp = vehicle_wp.next(2.0)[0]  # Guarda avanti di 2 metri
    
        if not next_wp:
            return False
    
        # Calcola la differenza di heading tra il waypoint attuale e il prossimo
        current_yaw = vehicle_wp.transform.rotation.yaw
        next_yaw = next_wp.transform.rotation.yaw
    
        # Normalizza gli angoli
        current_yaw = (current_yaw + 360) % 360
        next_yaw = (next_yaw + 360) % 360
    
        # Calcola la differenza tra gli angoli (tenendo conto del wrap-around)
        yaw_diff = (next_yaw - current_yaw + 180) % 360 - 180
    
        # Se la differenza è negativa, il veicolo sta girando a sinistra
        return -45 > yaw_diff > -135

    def _is_vehicle_turning_right(self, vehicle):
        """
        Verifica se un veicolo sta svoltando a destra.
    
        :param vehicle: il veicolo da controllare
        :return: True se sta svoltando a destra, False altrimenti
        """
        # Ottieni la posizione attuale e la direzione del veicolo
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_wp = self._map.get_waypoint(vehicle_location)
    
        # Ottieni il prossimo waypoint nella direzione di marcia
        vehicle_forward = vehicle_transform.get_forward_vector()
        next_wp = vehicle_wp.next(2.0)[0]  # Guarda avanti di 2 metri
    
        if not next_wp:
            return False
    
        # Calcola la differenza di heading tra il waypoint attuale e il prossimo
        current_yaw = vehicle_wp.transform.rotation.yaw
        next_yaw = next_wp.transform.rotation.yaw
    
        # Normalizza gli angoli
        current_yaw = (current_yaw + 360) % 360
        next_yaw = (next_yaw + 360) % 360
    
        # Calcola la differenza tra gli angoli (tenendo conto del wrap-around)
        yaw_diff = (next_yaw - current_yaw + 180) % 360 - 180
    
        # Se la differenza è positiva, il veicolo sta girando a destra
        return 45 < yaw_diff < 135

    def _is_vehicle_going_straight(self, vehicle):
        """
        Verifica se un veicolo sta procedendo dritto.
    
        :param vehicle: il veicolo da controllare
        :return: True se sta procedendo dritto, False altrimenti
        """
        # Ottieni la posizione attuale e la direzione del veicolo
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_wp = self._map.get_waypoint(vehicle_location)
    
        # Ottieni il prossimo waypoint nella direzione di marcia
        vehicle_forward = vehicle_transform.get_forward_vector()
        next_wp = vehicle_wp.next(2.0)[0]  # Guarda avanti di 2 metri
    
        if not next_wp:
            return True  # Assumiamo che stia andando dritto se non c'è un next waypoint
    
        # Calcola la differenza di heading tra il waypoint attuale e il prossimo
        current_yaw = vehicle_wp.transform.rotation.yaw
        next_yaw = next_wp.transform.rotation.yaw
    
        # Normalizza gli angoli
        current_yaw = (current_yaw + 360) % 360
        next_yaw = (next_yaw + 360) % 360
    
        # Calcola la differenza tra gli angoli (tenendo conto del wrap-around)
        yaw_diff = (next_yaw - current_yaw + 180) % 360 - 180
    
        # Se la differenza è piccola, il veicolo sta andando dritto
        return abs(yaw_diff) < 20
    
    def set_streamer(self, streamer):
        """Imposta lo streamer per l'invio dei log al server Flask"""
        self.logger.set_streamer(streamer)

    def _cleanup_stopped_vehicles_timer(self):
        """Rimuove i timer dei veicoli che non sono più rilevanti"""
        current_time = time.time()
        
        # Esegui la pulizia solo ogni tot secondi per evitare calcoli inutili
        if current_time - self._last_timer_cleanup < self._timer_cleanup_interval:
            return
            
        self._last_timer_cleanup = current_time
        
        # Rimuovi i timer più vecchi di X secondi (ad esempio 60 secondi)
        expired_ids = []
        for vehicle_id, timestamp in self._stopped_vehicles_timer.items():
            if current_time - timestamp > 60.0:
                expired_ids.append(vehicle_id)
                
        for vehicle_id in expired_ids:
            del self._stopped_vehicles_timer[vehicle_id]
            
        self.logger.debug(f"Cleaned up {len(expired_ids)} expired vehicle timers")