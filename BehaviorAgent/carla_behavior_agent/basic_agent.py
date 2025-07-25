# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
from shapely.geometry import Polygon

from local_planner import LocalPlanner, RoadOption
from global_route_planner import GlobalRoutePlanner
from misc import (get_speed, is_within_distance,
                               get_trafficlight_trigger_location,
                               compute_distance,dist)
# from perception.perfectTracker.gt_tracker import PerfectTracker

class BasicAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._ignore_obstacles = False
        self._use_bbs_detection = False
        self._target_speed = 5.0
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._speed_ratio = 1
        self._max_brake = 0.5
        self._offset = 0

        # Change parameters according to the dictionary
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'use_bbs_detection' in opt_dict:
            self._use_bbs_detection = opt_dict['use_bbs_detection']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'detection_speed_ratio' in opt_dict:
            self._speed_ratio = opt_dict['detection_speed_ratio']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'offset' in opt_dict:
            self._offset = opt_dict['offset']
        
        # Initialize the planners
        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        else:
            self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def get_speed(self):
        """
        Returns the current speed of the vehicle
            :return: speed in Km/h
        """
        return get_speed(self._vehicle) / 3.6

    def get_target_speed(self):
        """
        Returns the target speed of the vehicle
            :return: target speed in Km/h
        """
        return self._target_speed

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        """Execute one step of navigation."""
        hazard_detected = False

        #####
        #  Retrieve all relevant actors
        #####
        # Basic Agent :
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        ### 

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control
    
    def reset(self):
        pass

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        speed = self._vehicle.get_velocity().length()
        path = self._local_planner._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")

        self.set_global_plan(path)

    def _affected_by_traffic_light(self, vehicle, lights_list=None, max_distance=None):
        """
        Method to check if a specific vehicle is affected by a red traffic light.

        :param vehicle: The vehicle to check (carla.Vehicle)
        :param lights_list: list containing TrafficLight objects.
            If None, all traffic lights in the scene are used
        :param max_distance: max distance for traffic lights to be considered relevant.
            If None, the base threshold value is used
        :return: Tuple of (bool, carla.TrafficLight) - If affected, returns (True, traffic_light),
                 otherwise returns (False, None)
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        vehicle_location = vehicle.get_location()
        vehicle_waypoint = self._map.get_waypoint(vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(vehicle_location) > max_distance:
                continue

            if trigger_wp.road_id != vehicle_waypoint.road_id:
                continue

            ve_dir = vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(trigger_wp.transform, vehicle.get_transform(), max_distance, [0, 90]):
                return (True, traffic_light)

        return (False, None)

    def _vehicle_obstacle_detected_old(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

        return (False, None, -1)

    def _get_route_polygon(self, max_distance=None):
        """
        Crea un poligono che rappresenta il percorso del veicolo basato sui waypoint seguiti
        dal piano di navigazione.

        :param max_distance: distanza massima per cui generare il percorso.
        :return: un poligono che rappresenta il percorso del veicolo.
        """
        if not max_distance:
            max_distance = self._base_vehicle_threshold  # Usa una soglia predefinita se max_distance è None

        route_bb = []
        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        extent_y = self._vehicle.bounding_box.extent.y  # Estensione lungo l'asse y della bounding box
        r_ext = extent_y + self._offset  # Distanza a destra
        l_ext = -extent_y + self._offset  # Distanza a sinistra
        r_vec = ego_transform.get_right_vector()  # Vettore di direzione verso destra

        # Aggiungi i punti iniziali per il poligono del percorso
        p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
        p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
        route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

        # Aggiungi i waypoint del piano di navigazione al percorso, limitato dalla distanza massima
        for wp, _ in self._local_planner.get_plan():
            if ego_location.distance(wp.transform.location) > max_distance:
                break  # Interrompi il percorso se è oltre la distanza massima

            r_vec = wp.transform.get_right_vector()  # Vettore di direzione a destra del waypoint
            p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

        # Se ci sono meno di 3 punti, non possiamo creare un poligono valido
        if len(route_bb) < 3:
            return None

        return Polygon(route_bb)  # Crea un poligono usando i punti raccolti

    
    def _get_ordered_vehicles(self, reference : carla.Actor, max_distance: float) -> list:
        # Get all vehicles in the scene
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        # Sort the vehicles by distance to the vehicle in question
        # Remove the reference vehicle from the list
        if isinstance(reference, carla.Actor):
            vehicle_list = [
                v for v in vehicle_list
                if v.id != reference.id and 0.1 < dist(v, reference) < max_distance
            ]
        else:
            vehicle_list = [
                v for v in vehicle_list
                if 0.1 < dist(v, reference) < max_distance
            ]
        
        # Sort the vehicles by distance to the ego vehicle
        vehicle_list.sort(key=lambda v: dist(v, reference))
        return vehicle_list
    
    def is_vehicle_legitimately_stopped(self, vehicle: carla.Vehicle) -> tuple:
        '''
        Determina se un veicolo è fermo per una ragione legittima o se è parcheggiato/abbandonato.
        
        :param vehicle: Il veicolo da valutare (quello davanti all'agente)
        :return: (vehicle_wp, is_abandoned) - waypoint del veicolo e se è considerato abbandonato
        '''
        vehicle_loc = vehicle.get_location()
        vehicle_wp = self._map.get_waypoint(vehicle_loc)
        
        # Controlli esistenti per semafori e segnali di stop
        lights_list_for_vehicle = self._world.get_actors().filter("*traffic_light*")
        lights_list_for_vehicle = [
            l for l in lights_list_for_vehicle 
            if is_within_distance(l.get_transform(), vehicle.get_transform(), 50, angle_interval=[0, 90])
        ]
        
        affected_by_traffic_light, _ = self._affected_by_traffic_light(vehicle, lights_list_for_vehicle) 
        affected_by_stop_sign, _ = self._affected_by_sign(vehicle)

        # Se il veicolo è direttamente influenzato da un semaforo o stop, è legittimamente fermo
        if affected_by_traffic_light or affected_by_stop_sign:
            self.logger.debug(f"Veicolo {vehicle.id} fermo a un semaforo o stop")
            return vehicle_wp, False
        
        # NUOVO: Controlla se c'è un semaforo rosso più avanti nella stessa corsia
        if get_speed(vehicle) < 0.5:  # Solo se il veicolo è praticamente fermo
            next_wp = vehicle_wp
            for i in range(10):  # Guarda avanti per circa 20-30m
                next_wps = next_wp.next(3.0)
                if not next_wps:
                    break
                next_wp = next_wps[0]
                
                # Controlla se c'è un semaforo rosso che influenza questo waypoint
                for traffic_light in lights_list_for_vehicle:
                    if traffic_light.state != carla.TrafficLightState.Red:
                        continue
                    
                    # Ottieni i waypoint influenzati dal semaforo
                    affecting_waypoints = traffic_light.get_affected_lane_waypoints()
                    for wp in affecting_waypoints:
                        # Se il waypoint influenzato è vicino al nostro waypoint proiettato
                        if wp.road_id == next_wp.road_id and wp.lane_id == next_wp.lane_id:
                            if next_wp.transform.location.distance(wp.transform.location) < 10.0:
                                self.logger.debug(f"Veicolo {vehicle.id} in coda verso un semaforo rosso")
                                return vehicle_wp, False  # È in coda a un semaforo, non è abbandonato

        
        # Logica originale
        if get_speed(vehicle) < 0.1 and not vehicle_wp.is_junction:
            return vehicle_wp, True  # È abbandonato/parcheggiato
        
        return vehicle_wp, False  # È fermo per una ragione legittima o in movimento
    

    def _affected_by_sign(self, vehicle , sign_type= "206", max_distance = None):
        '''
        This method checks if the vehicle is affected by a stop sign of a specific type.
        Default type is "206", which is the type of the stop sign.
        
            :param vehicle (carla.Vehicle): vehicle to check if it is affected by a stop sign
            :param sign_type (str): type of the stop sign
            :param max_distance (float): max distance for stop signs to be considered relevant.
            
            :return (bool, carla.Actor): a tuple containing a boolean indicating if the vehicle is affected by a stop sign
        '''
        
        if self._ignore_stop_signs:
            return False, None

        if max_distance is None:
            max_distance = 20

        # defines the last traffic light to which pay attention - case there is no sign registered
        target_vehicle_wp = self._map.get_waypoint(vehicle.get_location())
        signs_list = target_vehicle_wp.get_landmarks_of_type(max_distance, type=sign_type, stop_at_junction=False)

        # stop sign
        if sign_type == '206':
            signs_list = [sign for sign in signs_list if
                          self._map.get_waypoint(vehicle.get_location()).lane_id == sign.waypoint.lane_id]

            if signs_list:
                return True, signs_list[0]  # return type Landmark

        return False, None


