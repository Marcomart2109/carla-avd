from BehaviorAgent.carla_behavior_agent.local_planner import RoadOption
import carla
import math
from shapely.geometry import Polygon
from misc import dist, is_within_distance, compute_distance_from_center, get_speed, compute_distance

class ObjectDetector:
    """Handles detection of various objects in the environment."""
    
    def __init__(self, world, vehicle, behavior):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self._behavior = behavior
        # Riferimento al local planner, da impostare dopo la creazione dell'oggetto
        self._local_planner = None
        # Altre proprietà necessarie per la compatibilità 
        self._ignore_vehicles = False
        self._base_vehicle_threshold = 5.0
    
    def set_local_planner(self, local_planner):
        """Set the local planner reference."""
        self._local_planner = local_planner
        
    def detect_vehicles(self, waypoint, max_distance=30, lane_offset=0, angle_th=30):
        """Detects vehicles around the ego vehicle."""
        vehicle_list = self.world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if dist(v, waypoint) < max_distance and v.id != self.vehicle.id]
        
        if not vehicle_list:
            return False, None, -1
            
        # Detect vehicles in front of us
        vehicle_state, vehicle, distance = self.vehicle_obstacle_detected(
            vehicle_list, 
            max(self._behavior.min_proximity_threshold, self._behavior.max_speed / 3),
            up_angle_th=angle_th,
            lane_offset=lane_offset
        )

        direction = self._local_planner.target_road_option
        speed = get_speed(self.vehicle)
        # Check for tailgating
        if not vehicle_state and direction == RoadOption.LANEFOLLOW \
                and not waypoint.is_junction and speed > 10 \
                and self._behavior.tailgate_counter == 0:
            self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance
    
    def detect_pedestrians(self, direction, max_distance=12):
        """Detects pedestrians around the ego vehicle."""
        walker_list = self.world.get_actors().filter("*walker.pedestrian*")
        max_distance = 15 if self.map.get_waypoint(self.vehicle.get_location()).is_junction else 30

        
        walker_list = [w for w in walker_list if 
                      is_within_distance(w.get_transform(), self.vehicle.get_transform(), 
                                       max_distance, angle_interval=[0, 90])]
        
        if not walker_list:
            return False, None, -1
        
        speed_limit = self.vehicle.get_speed_limit()

        if self.map.get_waypoint(self.vehicle.get_location()).is_junction:
            return True, walker_list[0], dist(walker_list[0], self.map.get_waypoint(self.vehicle.get_location()))
        

        if direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self.pedestrian_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self.pedestrian_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self.pedestrian_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180)
        
        return walker_state, walker, distance
    
    def detect_bicycles(self, waypoint, max_distance=10):
        """Detects bicycles around the ego vehicle."""
        BICYCLES_ID = ['vehicle.bh.crossbike','vehicle.diamondback.century', 'vehicle.gazelle.omafiets']
        
        vehicle_list = self.world.get_actors().filter("*vehicle*")
        bicycle_list = [v for v in vehicle_list if v.type_id in BICYCLES_ID]
        bicycle_list = [b for b in bicycle_list if 
                        is_within_distance(b.get_transform(), self.vehicle.get_transform(), 
                                         max_distance, angle_interval=[0, 90])]
        
        if not bicycle_list or len(bicycle_list) == 0:
            return False, None, -1
        
        bicycle_list = sorted(bicycle_list, key=lambda x: dist(x, waypoint))
        
        return True, bicycle_list[0], dist(bicycle_list[0], waypoint)

    def detect_static_obstacles(self, waypoint, max_distance=40, obj_filter="*static.prop*", 
                              obj_ignore=["static.prop.dirtdebris01","static.prop.dirtdebris02","static.prop.dirtdebris03","static.prop.mesh"]):
        """Detects static obstacles around the ego vehicle."""
        props = self.world.get_actors().filter(obj_filter)
        props = [p for p in props if is_within_distance(p.get_transform(), self.vehicle.get_transform(), 
                                                      max_distance, angle_interval=[0, 90])]
        
        props = [p for p in props if not any(ignore_type in p.type_id for ignore_type in obj_ignore)]
        
        if not props:
            return False, None, -1
            
        props = sorted(props, key=lambda x: dist(x, waypoint))
        
        if obj_filter == "*static.prop.constructioncone*" and len(props) > 0:
            return True, props[0], dist(props[0], waypoint) 
        # Use the vehicle detection logic for static obstacles
        o_state, o, o_distance = self.vehicle_obstacle_detected(
            props, max(self._behavior.min_proximity_threshold, self._behavior.max_speed / 3), up_angle_th=60
        )
        
        return o_state, o, o_distance
    
    def pedestrian_obstacle_detected(self, pedestrian_list, max_distance=None, up_angle_th=90, low_angle_th=-90, lane_offset=1):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        if self._ignore_vehicles:
            return (False, None, -1)

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self.vehicle.get_transform()
        ego_wpt = self.map.get_waypoint(self.vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self.vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        for pedestrian in pedestrian_list:
            target_transform = pedestrian.get_transform()
            target_wpt = self.map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # Simplified version for outside junctions
            if not ego_wpt.is_junction or not target_wpt.is_junction:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = pedestrian.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, pedestrian, compute_distance(target_transform.location, ego_transform.location))

            # Waypoints aren't reliable, check the proximity of the vehicle to the route
            else:
                print("ENTROOOOO : 2")
                route_bb = []
                ego_location = ego_transform.location
                extent_y = self.vehicle.bounding_box.extent.y
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

                for wp, _ in self._local_planner.get_plan():
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break

                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])

                if len(route_bb) < 3:
                    # 2 points don't create a polygon, nothing to check
                    return (False, None, -1)
                ego_polygon = Polygon(route_bb)

                # Compare the two polygons
                for pedestrian in pedestrian_list:
                    target_extent = pedestrian.bounding_box.extent.x
                    if pedestrian.id == self.vehicle.id:
                        continue
                    if ego_location.distance(pedestrian.get_location()) > max_distance:
                        continue

                    target_bb = pedestrian.bounding_box
                    target_vertices = target_bb.get_world_vertices(pedestrian.get_transform())
                    target_list = [[v.x, v.y, v.z] for v in target_vertices]
                    target_polygon = Polygon(target_list)

                    if ego_polygon.intersects(target_polygon):
                        return (True, pedestrian, compute_distance(pedestrian.get_location(), ego_location))

                return (False, None, -1)

        return (False, None, -1)
    
    def vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self.world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self.vehicle.get_transform()
        ego_wpt = self.map.get_waypoint(self.vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self.vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        for target_vehicle in vehicle_list:
            target_transform = target_vehicle.get_transform()
            target_wpt = self.map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            # Waypoints aren't reliable, check the proximity of the vehicle to the route
            route_bb = []
            ego_location = ego_transform.location
            extent_y = self.vehicle.bounding_box.extent.y
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
            p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
            route_bb.append([p1.x, p1.y, p1.z])
            route_bb.append([p2.x, p2.y, p2.z])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

            if len(route_bb) < 3:
                # 2 points don't create a polygon, nothing to check
                return (False, None, -1)
            ego_polygon = Polygon(route_bb)

            # Compare the two polygons
            for target_vehicle in vehicle_list:
                target_extent = target_vehicle.bounding_box.extent.x
                if target_vehicle.id == self.vehicle.id:
                    continue
                if ego_location.distance(target_vehicle.get_location()) > max_distance:
                    continue

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if ego_polygon.intersects(target_polygon):
                    return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

            return (False, None, -1)

        return (False, None, -1)
    
    def _is_target_behind(self, target_location, ego_transform):
        """
        Check if a target object is behind our vehicle.
        """
        ego_location = ego_transform.location
        ego_orientation = ego_transform.rotation.yaw
        
        # Vector pointing from ego to target
        target_vector = target_location - ego_location
        norm_target = math.sqrt(target_vector.x**2 + target_vector.y**2)
        
        # Convert to normalized vector
        if norm_target > 0:
            target_vector = carla.Vector3D(target_vector.x / norm_target, target_vector.y / norm_target, 0)
        else:
            target_vector = carla.Vector3D(1, 0, 0)  # Default forward vector
        
        # Forward vector of the vehicle
        forward_vector = ego_transform.get_forward_vector()
        forward_vector = carla.Vector3D(forward_vector.x, forward_vector.y, 0)
        
        # Calculate dot product
        d_angle = math.degrees(math.acos(forward_vector.x * target_vector.x + forward_vector.y * target_vector.y))
        
        return d_angle > 90, d_angle
    
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

            speed = get_speed(self.vehicle)
            speed_limit = self.vehicle.get_speed_limit()

            behind_vehicle_state, behind_vehicle, _ = self.vehicle_obstacle_detected(vehicle_list, max(
                self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180, low_angle_th=160)
            if behind_vehicle_state and speed < get_speed(behind_vehicle):
                if (right_turn == carla.LaneChange.Right or right_turn ==
                        carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                    new_vehicle_state, _, _ = self.vehicle_obstacle_detected(vehicle_list, max(
                        self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180, lane_offset=1)
                    if not new_vehicle_state:
                        print("Tailgating, moving to the right!")
                        end_waypoint = self._local_planner.target_waypoint
                        self._behavior.tailgate_counter = 200
                        self.set_destination(end_waypoint.transform.location,
                                            right_wpt.transform.location)
                elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                    new_vehicle_state, _, _ = self.self.vehicle_obstacle_detected(vehicle_list, max(
                        self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180, lane_offset=-1)
                    if not new_vehicle_state:
                        print("Tailgating, moving to the left!")
                        end_waypoint = self._local_planner.target_waypoint
                        self._behavior.tailgate_counter = 200
                        self.set_destination(end_waypoint.transform.location,
                                            left_wpt.transform.location)