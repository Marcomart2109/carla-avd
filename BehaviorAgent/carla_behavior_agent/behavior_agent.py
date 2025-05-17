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

        self.logger = logging.getLogger('behavior_agent')
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler
        file_handler = logging.FileHandler('./carla_behavior_agent/logs/behavior_agent.log', mode = "w")
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set it for the handler
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(self.formatter)
        # Add the handler to the logger 
        self.logger.addHandler(file_handler)

        self.logger.info(f"Behavior agent initialized with behavior: {behavior}")

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


            msg = f"[DEBUG - Static Obstacle Avoidance] Static obstacle detected, distance: {distance}"
            print(msg)

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
            # Distance is computed from the center of the car and the bicycle,
            # we use bounding boxes to calculate the actual distance
            # Get the yaw of the ego vehicle and the vehicle in front.
            ego_yaw = abs(self._vehicle.get_transform().rotation.yaw)
            vehicle_yaw = abs(bicycle.get_transform().rotation.yaw)
            
            # Check if the vehicles are approximately aligned
            if abs(ego_yaw - vehicle_yaw) < 10:                   
                # Check if the bicycle is near the center of the lane. In this case, the agent will overtake the bicycle.
                if is_bicycle_near_center(vehicle_location=bicycle.get_location(), ego_vehicle_wp=ego_vehicle_wp) and get_speed(self._vehicle) < 0.1:
                    print("--- Bicycle is near the center of the lane! We can try to overtake it.")

                    overtake_path = self._overtake.run_step(
                        object_to_overtake=bicycle, ego_vehicle_wp=ego_vehicle_wp, distance_same_lane=1, distance_from_object=b_distance, speed_limit = self._speed_limit
                    )                                               
                    if overtake_path:
                        self.__update_global_plan(overtake_path=overtake_path)
                    if not self._overtake.in_overtake:
                        return self.emergency_stop()
                # If the bicycle is not near the center of the lane, the agent will follow the lane. In particular, the agent will offset 
                # the vehicle if the road is straight.
                else:
                    print("--- Bicycle is not near the center of the lane! We can move of an offset to avoid it.")
                    new_offset = -(2.5 * bicycle.bounding_box.extent.y + self._vehicle.bounding_box.extent.y)
                    self._local_planner.set_lateral_offset(new_offset)
                    self._lateral_offset_active = True
                    target_speed = min([
                        self._behavior.max_speed,
                        self._speed_limit - self._behavior.speed_lim_dist])
                    self._local_planner.set_speed(target_speed)
                    control = self._local_planner.run_step(debug=debug)
            elif get_speed(bicycle) < 1:
                # Set the target speed of the agent.
                print("--- Bicycle is not moving! We decelerate.")
                self.set_target_speed(self._approach_speed)
                return self._local_planner.run_step()
            else:
                print("--- Road is not straight! We follow it until the road is straight.")
                control = self.car_following_manager(bicycle, b_distance, debug=debug)
        elif self._lateral_offset_active:
            # Resetta l'offset solo se era attivo e non ci sono più biciclette
            print("--- No bicycle detected, resetting lateral offset to 0")
            self._local_planner.set_lateral_offset(0.0)
            self._lateral_offset_active = False
                

        # 2.2: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
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

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

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
        bicycle_list = [b for b in bicycle_list if is_within_distance(b.get_transform(), self._vehicle.get_transform(), 30, angle_interval=[0, 90])]
            
        if not bicycle_list or len(bicycle_list) == 0:
            return False, None, -1
        
        bicycle_list = sorted(bicycle_list, key=lambda x: dist(x, waypoint))
        return True, bicycle_list[0], dist(bicycle_list[0], waypoint)

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
        Handles avoidance behavior for static obstacles such as signs,
        barriers, or construction props. Attempts lane changes if possible;
        otherwise, performs an emergency stop only if the obstacle is blocking the lane.
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

        return o_state,o, o_distance

    
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