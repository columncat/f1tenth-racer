#!/usr/bin/env python3
import numpy as np
import math
import os
from typing import Union
import scipy.spatial
from PIL import Image
import yaml
import cv2
from scipy.linalg import solve_continuous_are
import scipy.linalg as la

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from queue import Queue


WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)



def safe_changeIdx(length, inp, plus):
    return (inp + plus + length) % (length)

def read_map(map_name, map_img_ext):
    map_img_path = map_name + map_img_ext
    map_img_path = os.path.join("src", "pure_pursuit", "maps", map_img_path)
    print(map_img_path)

    map_cfg_path = map_name + '.yaml'
    map_cfg_path = os.path.join("src", "pure_pursuit", "maps", map_cfg_path)
    print(map_cfg_path)

    map_img = Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM)
    map_img = np.asarray(map_img).astype(np.float64)
    map_img[map_img <= 128.] = 0.
    map_img[map_img > 128.] = 255.

    map_height = map_img.shape[0]
    map_width = map_img.shape[1]

    # load map yaml
    map_metadata = yaml.safe_load(open(map_cfg_path, 'r'))
    map_resolution = map_metadata['resolution']
    origin = map_metadata['origin']
    origin_x = origin[0]
    origin_y = origin[1]

    image_xs, image_ys = np.meshgrid(np.arange(map_width), np.arange(map_height))
    map_xs = image_xs * map_resolution + origin_x
    map_ys = image_ys * map_resolution + origin_y

    map_vs = np.where(map_img > 0, 0.0, 1.0)  # 1: occupied  0: free

    return np.dstack((map_vs, map_xs, map_ys)), (map_height, map_width, map_resolution, origin_x, origin_y)

def transform_coords_to_img(path, height, s, tx, ty):
    new_path_x = (path[:, 0] - tx) / s
    new_path_y = height - (path[:, 1] - ty) / s
    return np.vstack((new_path_x, new_path_y)).T.astype(np.int32)

def show_result(imgs, title):
    if not imgs:
        return False
    height, width = imgs[0].shape[:2]
    w_show = 800
    scale_percent = float(w_show / width)
    h_show = int(scale_percent * height)
    dim = (w_show, h_show)
    img_resizes = []
    for img in imgs:
        img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_resizes.append(img_resize)
    img_show = cv2.hconcat(img_resizes)
    cv2.imshow(title, img_show)

    print("Press Q to abort / other keys to proceed")
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        return False
    else:
        cv2.destroyAllWindows()
        return True
    
def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

def angle_mod(x, zero_2_2pi=False, degree=False):
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle

def get_curvature(curr_lane_pos, following_idxs):
    p1, p2, p3 = [curr_lane_pos[i] for i in following_idxs]
    a = np.hypot(p1[1] - p2[1], p1[0] - p2[0])
    b = np.hypot(p2[1] - p3[1], p2[0] - p3[0])
    c = np.hypot(p1[1] - p3[1], p1[0] - p3[0])
    s = (a + b + c) / 2
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    return (4 * area) / (a * b * c)


class HybridNode(Node):
    def __init__(self):
        super().__init__("hybrid_node")

        # Controller Params
        w_cte, w_dcte  = 1000.0, 0.0
        w_yaw, w_dyaw  = 10.0, 0.1
        w_steer = 0.1
        self.max_calc_time = 10.0   # ms
        self.lqr_gain = 1.0

        self.stanley_gain = 0.8
        self.stanley_k = 1.8
        self.stanley_ks = 1.3

        self.speed_gain = 0.75
        self.corner_decel = 1.0     # given in ratio
        self.obstacle_decel = 0.75  # given in ratio
        self.curvature_decel = 0.75
        self.curvature_threshold = 0.50
        
        self.cte_threshold = 0.15
        self.yaw_threshold = np.pi / 8

        self.obs2lane_dist = 0.3    # m
        self.obs_ignore_dist = 7.5    # cm

        self.stanley_const_speed = 3.0
        self.stanley_decel = 0.75
        self.pure_pursuit_const_speed = 2.0
        self.pure_pursuit_decel = 0.75

        self.dt = 0.1
        self.prev_ditem = 0.0
        self.prev_steer_error = 0.0
        self.corner_wpIdx = []

        self.L = 0.3302  # Wheelbase length
        self.Q = np.eye(4)
        self.Q[0, 0] = w_cte
        self.Q[1, 1] = w_dcte
        self.Q[2, 2] = w_yaw
        self.Q[3, 3] = w_dyaw
        self.R = np.eye(1)
        self.R[0, 0] = w_steer
        
        # ROS Params
        self.declare_parameter("visualize")

        self.declare_parameter("obs_activate_dist")

        self.declare_parameter("real_test")
        self.declare_parameter("map_name")
        self.declare_parameter("num_lanes")
        self.declare_parameter("lane_files")
        self.declare_parameter("traj_file")

        self.declare_parameter("lookahead_distance")
        self.declare_parameter("lookahead_attenuation")
        self.declare_parameter("lookahead_idx")
        self.declare_parameter("lookbehind_idx")

        self.declare_parameter("kp_steer")
        self.declare_parameter("ki_steer")
        self.declare_parameter("kd_steer")
        self.declare_parameter("max_steer")
        self.declare_parameter("alpha_steer")

        self.declare_parameter("kp_pos")
        self.declare_parameter("ki_pos")
        self.declare_parameter("kd_pos")

        self.declare_parameter("follow_speed")
        self.declare_parameter("lane_dist_thresh")

        self.declare_parameter("grid_xmin")
        self.declare_parameter("grid_xmax")
        self.declare_parameter("grid_ymin")
        self.declare_parameter("grid_ymax")
        self.declare_parameter("grid_resolution")
        self.declare_parameter("plot_resolution")
        self.declare_parameter("grid_safe_dist")
        self.declare_parameter("goal_safe_dist")


        # interp
        self.declare_parameter('minL')
        self.declare_parameter('maxL')
        self.declare_parameter('minP')
        self.declare_parameter('maxP')
        self.declare_parameter('interpScale')
        self.declare_parameter('Pscale')
        self.declare_parameter('Lscale')
        self.declare_parameter('D')
        self.declare_parameter('vel_scale')

        self.declare_parameter('minL_corner')
        self.declare_parameter('maxL_corner')
        self.declare_parameter('minP_corner')
        self.declare_parameter('maxP_corner')
        self.declare_parameter('Pscale_corner')
        self.declare_parameter('Lscale_corner')

        self.declare_parameter('avoid_v_diff')
        self.declare_parameter('avoid_L_scale')
        self.declare_parameter('pred_v_buffer')
        self.declare_parameter('avoid_buffer')
        self.declare_parameter('avoid_span')

        self.maxL_corner = self.get_parameter('maxL_corner').get_parameter_value().double_value
        self.minL_corner = self.get_parameter('minL_corner').get_parameter_value().double_value
        self.Lscale_corner = self.get_parameter('Lscale_corner').get_parameter_value().double_value
        self.maxL = self.get_parameter('maxL').get_parameter_value().double_value
        self.minL = self.get_parameter('minL').get_parameter_value().double_value
        self.Lscale = self.get_parameter('Lscale').get_parameter_value().double_value
        self.vel_scale = self.get_parameter('vel_scale').get_parameter_value().double_value
        self.interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value
        self.visualize_target = self.get_parameter("visualize").get_parameter_value().bool_value
        self.visualize_obstacle = True

        # Global Map Params
        self.real_test = self.get_parameter("real_test").get_parameter_value().bool_value
        self.map_name = self.get_parameter("map_name").get_parameter_value().string_value
        self.map_img_ext = '.pgm'
        self.map, self.map_metadata = read_map(self.map_name, self.map_img_ext)

        self.track = np.load(f'src/pure_pursuit/csv/{self.map_name}/track.npy')
        self.inner_bound = np.load(f'src/pure_pursuit/csv/{self.map_name}/inner_bound.npy')
        self.outer_bound = np.load(f'src/pure_pursuit/csv/{self.map_name}/outer_bound.npy')
        print(self.map_name)

        # Obstacle detection
        self.grid = None
        self.obstacles = []
        self.xmin = self.get_parameter("grid_xmin").get_parameter_value().double_value
        self.xmax = self.get_parameter("grid_xmax").get_parameter_value().double_value
        self.ymin = self.get_parameter("grid_ymin").get_parameter_value().double_value
        self.ymax = self.get_parameter("grid_ymax").get_parameter_value().double_value
        self.resolution = self.get_parameter("grid_resolution").get_parameter_value().double_value
        self.grid_safe_dist = self.get_parameter("grid_safe_dist").get_parameter_value().double_value

        # Ego Car State Variables
        self.curr_pos = None
        self.curr_yaw = 0.0

        # Lanes Waypoints
        self.lane_files = self.get_parameter("lane_files").get_parameter_value().string_array_value
        self.num_lanes = len(self.lane_files)
        self.lane_free = [True] * self.num_lanes
        self.queue_lane_freed = []

        self.num_lane_pts = []
        self.lane_x = []
        self.lane_y = []
        self.lane_v = []
        self.lane_pos = []

        for i in range(self.num_lanes):
            lane_csv_loc = os.path.join("src", "pure_pursuit", "csv", self.map_name, self.lane_files[i] + ".csv")
            lane_data = np.loadtxt(lane_csv_loc, delimiter=",")
            self.num_lane_pts.append(len(lane_data))
            self.lane_x.append(lane_data[:, 0])
            self.lane_y.append(lane_data[:, 1])
            if self.lane_files[i] == 'lane_optimal':
                self.lane_v = lane_data[:, 2]
            self.lane_pos.append(np.vstack((self.lane_x[-1], self.lane_y[-1]), ).T)

        # Car Status Variables
        self.curr_lane = -1
        self.curr_vel = 0.0
        self.pose_vels = []
        self.odom_vel = 0.0
        self.closest_wps = [(0.0, 0.0)] * self.num_lanes
        self.target_wps  = [(0.0, 0.0)] * self.num_lanes
        self.ctes        = [0.0] * self.num_lanes
        self.yaw_errors  = [0.0] * self.num_lanes
        self.pctes       = [0.0] * self.num_lanes
        self.pyaw_errors = [0.0] * self.num_lanes
        self.is_on_lanes = [False] * self.num_lanes
        self.prev_steer = 0.0
        self.prev_speed = 0.0
        self.message_queue = Queue()

        # Result Params
        self.lap_start = self.get_clock().now().nanoseconds
        self.lap_cte = 0.0
        self.lap_maxsteer = 0.0

        # Topics & Subs, Pubs
        pose_topic = "/pf/viz/inferred_pose" if self.real_test else "/ego_racecar/odom"
        odom_topic = "/odom" if self.real_test else "/ego_racecar/odom"
        drive_topic = "/drive"
        scan_topic = "/scan"
        waypoint_topic = "/waypoint"
        obstacle_topic = '/obstacle'

        if self.real_test:
            self.pose_sub_ = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 10)
        else:
            self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 10)
        self.odom_sub_     = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.scan_sub_     = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.drive_pub_    = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)
        self.obstacle_pub_ = self.create_publisher(MarkerArray, obstacle_topic, 10)
        print('node_init_files')

        # timestamped
        self.scan_timestamped_nanosec = None
        self.scan_timestamped_sec = None
        self.callback_count = 0
    
    def pose_callback(self, pose_msg: Union[PoseStamped, Odometry]):
        #### Read pose data
        start = self.get_clock().now().nanoseconds
        prev_pos = self.curr_pos
        if self.real_test:
            curr_x = pose_msg.pose.position.x
            curr_y = pose_msg.pose.position.y
            curr_pos = np.array([curr_x, curr_y])
            curr_quat = pose_msg.pose.orientation
        else:
            curr_x = pose_msg.pose.pose.position.x
            curr_y = pose_msg.pose.pose.position.y
            curr_pos = np.array([curr_x, curr_y])
            curr_quat = pose_msg.pose.pose.orientation

        curr_yaw = math.atan2(2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
                            1 - 2 * (curr_quat.y ** 2 + curr_quat.z ** 2))
        self.curr_pos = curr_pos
        self.curr_yaw = curr_yaw

        cur_speed = 0.0
        if curr_pos is not None and prev_pos is not None:
            cur_speed = np.linalg.norm(curr_pos - prev_pos) / self.dt
            self.pose_vels.append(cur_speed)
            if len(self.pose_vels) > 5:
                cur_speed = np.average(self.pose_vels[-5:])
        else:
            self.curr_vel = 0.0
        # self.curr_vel = np.average([cur_speed, self.odom_vel])
        self.curr_vel = self.odom_vel

        #### Get Closest Waypoint and Target Waypoint
        lanes = [i for i in range(self.num_lanes)]
        closest_idxs = [np.argmin(np.linalg.norm(self.lane_pos[i][:, :2] - curr_pos, axis=1)) for i in lanes]
        
        traj_distances = [np.linalg.norm(self.lane_pos[i][:, :2] - self.lane_pos[i][closest_idxs[i], :2], axis=1) for i in lanes]
        segment_ends = [np.argmin(traj_distances[i]) for i in lanes]
        L = self.get_L_w_speed(cur_speed)
        # if not self.is_on_lanes[self.curr_lane]:
        #     L = L * 1.5
        interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value
        for i in lanes:
            while traj_distances[i][segment_ends[i]] <= L:
                segment_ends[i] = safe_changeIdx(len(traj_distances[i]), segment_ends[i], 1)
        segment_begins = [safe_changeIdx(len(traj_distances[i]), segment_ends[i], -1) for i in lanes]
        x_arrays = [np.linspace(self.lane_x[i][segment_begins[i]], self.lane_x[i][segment_ends[i]], interpScale) for i in lanes]
        y_arrays = [np.linspace(self.lane_y[i][segment_begins[i]], self.lane_y[i][segment_ends[i]], interpScale) for i in lanes]
        v_array = np.linspace(self.lane_v[segment_begins[-1]], self.lane_v[segment_ends[-1]], interpScale)
        xy_interps = [np.vstack([x_arrays[i], y_arrays[i]]).T for i in lanes]
        dist_interps = [np.linalg.norm(xy_interps[i] - curr_pos, axis=1) - L for i in lanes]
        i_interps = [np.argmin(dist_interps[i]) for i in lanes]
        target_v = v_array[i_interps[-1]]

        self.closest_wps = [self.lane_pos[lane][wpidx] for lane, wpidx in enumerate(closest_idxs)]
        self.target_wps  = [np.array([x_arrays[i][i_interps[i]], y_arrays[i][i_interps[i]]]) for i in lanes]
        path_yaws = [np.arctan2(self.target_wps[i][1] - self.closest_wps[i][1], self.target_wps[i][0] - self.closest_wps[i][0]) for i in lanes]
        
        #### Calculate cte and yaw error
        cte_sizes = [np.linalg.norm(self.closest_wps[i] - curr_pos) for i in lanes]
        cte_signs = [np.sign(np.cross(np.array([np.cos(path_yaws[i]), np.sin(path_yaws[i])]), curr_pos - self.closest_wps[i])) for i in lanes]
        self.pctes = self.ctes
        self.pyaw_errors = self.yaw_errors
        self.ctes = [cte_sizes[i] * cte_signs[i] for i in lanes]
        self.yaw_errors = [normalize_angle(path_yaw - curr_yaw) for path_yaw in path_yaws]

        #### Check if vehicle is on the lane ####
        # if more than half of the queue_lane_freed is False, set lane_free to False and vice versa
        if len(self.queue_lane_freed) > 10:
            for lane in lanes:
                freed_count = 0
                for i in range(1, 6):
                    if self.queue_lane_freed[-i][lane]:
                        freed_count += 1
                self.lane_free[lane] = freed_count > 3

        dist_to_lanes = [np.linalg.norm(curr_pos - self.closest_wps[i]) for i in lanes]
        self.is_on_lanes = [self.is_on_lane(dist_to_lanes[i], self.yaw_errors[i]) for i in lanes]
        on_lanes = [i for i, is_on_lane in enumerate(self.is_on_lanes) if is_on_lane and self.lane_free[i]]
        free_lanes = [i for i, is_free in enumerate(self.lane_free) if is_free]

        #### Select lane to follow ####
        steer_type = 'lqr'
        speed = target_v * self.vel_scale * self.speed_gain
        last_lane = self.curr_lane
        if self.lane_free[-1] and self.is_on_lanes[-1]:
            steer_type = 'lqr'
            self.curr_lane = -1
        elif self.lane_free[-1]:
            steer_type = 'sty'
            self.curr_lane = -1
        elif self.lane_free[last_lane] and self.is_on_lanes[last_lane]:
            steer_type = 'lqr'
            self.curr_lane = last_lane
        else:
            # get lane with min distance among on_lanes
            dist = 1e9
            if len(on_lanes) > 0:
                steer_type = 'sty'
                for lane in on_lanes:
                    if dist_to_lanes[lane] < dist:
                        dist = dist_to_lanes[lane]
                        self.curr_lane = lane
            elif len(free_lanes) > 0:
                steer_type = 'pps'
                for lane in free_lanes:
                    if dist_to_lanes[lane] < dist:
                        dist = dist_to_lanes[lane]
                        self.curr_lane = lane
            else:
                # there is no free lane
                # go backwards
                if self.message_queue.empty():
                    for _ in range(5):
                        self.message_queue.put((0.0, -1.0))

        #### get steer values ####
        cte = self.ctes[self.curr_lane]
        yaw_error = self.yaw_errors[self.curr_lane]
        curr_lane_pos = self.lane_pos[self.curr_lane]
        following_idxs = [safe_changeIdx(len(self.lane_pos[self.curr_lane]), closest_idxs[self.curr_lane], i) for i in range(3)]
        curvature = get_curvature(curr_lane_pos, following_idxs)
        if curvature > 0.5:
            steer_type = 'pps`'
        
        # steer_type = 'sty'
        if steer_type == 'lqr':
            # steer = self.lqr_steering_control(curvature)
            steer = self.get_stanley_steer(cte, yaw_error, cur_speed)
        elif steer_type == 'sty':
            steer = self.get_stanley_steer(cte, yaw_error, cur_speed)
        else:
            steer = self.get_pure_pursuit_steer(curr_pos, curr_yaw, cur_speed, self.target_wps[self.curr_lane], curvature)

        #### Publish the command ####
        if closest_idxs[-1] in self.corner_wpIdx:
            speed = speed * self.corner_decel
        if len(free_lanes) != self.num_lanes:
            speed = speed * self.obstacle_decel
        if curvature > self.curvature_threshold:
            speed = speed * self.curvature_decel
        # if steer_type == 'sty':
        #     # speed = speed * self.stanley_decel
        #     speed = self.stanley_const_speed
        # if steer_type == 'pps':
        #     # speed = speed * self.pure_pursuit_decel
        #     speed = self.pure_pursuit_const_speed

        #### check collision between vehicle and obstacle
        u_vehicle = np.array([np.cos(curr_yaw), np.sin(curr_yaw)])
        if self.obstacles is not None:
            for obs in self.obstacles:
                dist2obs = np.linalg.norm(obs - curr_pos)
                u_v2obs = np.array(obs - curr_pos) / dist2obs
                inner_prod = np.inner(u_vehicle, u_v2obs)
                if inner_prod > 0.9 and (dist2obs < 0.5 or (dist2obs < 1.0 and self.prev_speed < 0.0)):
                    if self.message_queue.empty():
                        for _ in range(5):
                            self.message_queue.put((0.0, -1.0))
                    break
        
        # if not self.message_queue.empty():
        #     steer, speed = self.message_queue.get()
        #     print('getting queued command')
        self.pub_cmd(steer, speed)
        self.prev_steer = steer
        self.prev_speed = speed
        str_lane_free = ''.join(['O' if lane_free else 'X' for lane_free in self.lane_free])
        str_steer = ('l' if steer > 0 else 'r') + f'{abs(steer):.2f}'
        calc_time = (self.get_clock().now().nanoseconds - start) / 10**6
        print(f'{steer_type},  steer:{str_steer},\tspeed:{speed:.2f},\tlane:{self.curr_lane}, wp:{closest_idxs[-1]}, curve:{curvature:.2f}, lane_free:{str_lane_free}, time:{calc_time:.1f}')
        # print(f'odom:{self.odom_vel:.2f}, pose:{self.curr_vel:.2f}')
        self.print_results(closest_idxs[-1], self.target_wps[self.curr_lane], cte_sizes[self.curr_lane] , steer)

    def solve_DARE(self, A, B, Q, R):
        """
        solve a discrete time_Algebraic Riccati equation (DARE)
        """
        X = Q
        Xn = Q
        # max_iter = 20
        # eps = 0.1
        # for i in range(max_iter):
        #     Xn = A.T @ X @ A - A.T @ X @ B @ \
        #         la.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        #     if (abs(Xn - X)).max() < eps:
        #         break
        #     X = Xn

        start = self.get_clock().now().nanoseconds
        while (self.get_clock().now().nanoseconds - start) / 10**6 < self.max_calc_time:
            Xn = A.T @ X @ A - A.T @ X @ B @ \
                la.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
            X = Xn

        return Xn

    def dlqr(self, A, B, Q, R):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        # ref Bertsekas, p.151
        """

        # first, try to solve the ricatti equation
        X = self.solve_DARE(A, B, Q, R)

        # compute the LQR gain
        K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

        eigVals, eigVecs = la.eig(A - B @ K)

        return K, X, eigVals

    def lqr_steering_control(self, curvature):
        cte = self.ctes[self.curr_lane]
        yaw_error = -self.yaw_errors[self.curr_lane]
        if self.callback_count != 0:
            prev_cte = self.pctes[self.curr_lane]
            prev_yaw_error = -self.pyaw_errors[self.curr_lane]
        else:
            prev_cte = 0.0
            prev_yaw_error = 0.0

        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[0, 1] = self.dt
        A[1, 2] = self.curr_vel
        A[2, 2] = 1.0
        A[2, 3] = self.dt

        B = np.zeros((4, 1))
        B[3, 0] = self.curr_vel / self.L

        K, _, _ = self.dlqr(A, B, self.Q, self.R)

        x = np.zeros((4, 1))

        x[0, 0] = cte
        x[1, 0] = (cte - prev_cte) / self.dt
        x[2, 0] = yaw_error
        x[3, 0] = (yaw_error - prev_yaw_error) / self.dt

        feedforward = math.atan2(self.L * curvature, 1)
        feedback = angle_mod((-K @ x)[0, 0])

        delta_yaw = feedforward + feedback
        steer = self.lqr_gain * (delta_yaw * self.L) / max(self.curr_vel, 1.0) / 0.36
        return steer

    def get_stanley_steer(self, cte, yaw_err, speed):
        steer = self.stanley_gain * (yaw_err - np.arctan2(self.stanley_k * cte, speed + self.stanley_ks))
        return steer

    def get_pure_pursuit_steer(self, curr_pos, curr_yaw, cur_speed, target_point, curvature):
        R = np.array([[np.cos(curr_yaw), np.sin(curr_yaw)],
                      [-np.sin(curr_yaw), np.cos(curr_yaw)]])
        _, target_y = R @ np.array([target_point[0] - curr_pos[0],
                                           target_point[1] - curr_pos[1]])
        L = np.linalg.norm(curr_pos - target_point)
        gamma = 2 / L ** 2
        error = gamma * target_y
        steer = self.get_steer_w_speed(cur_speed, error, curvature > 0.5)
        return steer

    def odom_callback(self, odom_msg: Odometry):
        # self.curr_vel = odom_msg.twist.twist.linear.x
        self.odom_vel = odom_msg.twist.twist.linear.x
        pass

    def scan_callback(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        ranges = np.clip(ranges, scan_msg.range_min, scan_msg.range_max)


        nx = int((self.xmax - self.xmin) / self.resolution) + 1
        ny = int((self.ymax - self.ymin) / self.resolution) + 1

        x = np.linspace(self.xmin, self.xmax, nx)
        y = np.linspace(self.ymin, self.ymax, ny)
        y, x = np.meshgrid(y, x)
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)

        ray_idx = ((phi - scan_msg.angle_min) / scan_msg.angle_increment).astype(int)
        obs_rho = ranges[ray_idx]

        self.grid = np.where(np.abs(rho - obs_rho) < self.grid_safe_dist, 1.0, 0.0)  # 1: occupied  0: free

        # cut angle
        # self.grid = np.where(-0.8 < phi, self.grid, 0.0)
        # self.grid = np.where(phi < 0.8, self.grid, 0.0)

        self.grid = np.dstack((self.grid, x, y))  # (h, w, 3)

        # current time
        # cur_time = scan_msg.header.stamp.nanosec/1e9 + scan_msg.header.stamp.sec
        self.scan_timestamped_nanosec = scan_msg.header.stamp.nanosec
        self.scan_timestamped_sec = scan_msg.header.stamp.sec

        if self.curr_pos is not None:
            self.get_obstacle()

    def get_obstacle(self):
        if self.grid is None:
            return
        # self.visualize_occupancy_grid()
        # Calculate wall points
        map_v = self.map[:, :, 0].flatten()
        map_x = self.map[:, :, 1].flatten()
        map_y = self.map[:, :, 2].flatten()

        map_x = map_x[map_v > 0]
        map_y = map_y[map_v > 0]
        map_point = np.vstack((map_x, map_y)).T

        # Calculate grid points in map frame
        R = np.array([[np.cos(self.curr_yaw), -np.sin(self.curr_yaw)],
                      [np.sin(self.curr_yaw), np.cos(self.curr_yaw)]])

        grid_v = self.grid[:, :, 0].flatten()
        grid_x = self.grid[:, :, 1].flatten()
        grid_y = self.grid[:, :, 2].flatten()

        grid_x = grid_x[grid_v > 0]
        grid_y = grid_y[grid_v > 0]
        grid_x, grid_y = R @ np.vstack((grid_x, grid_y)) + self.curr_pos.reshape(-1, 1)
        grid_point = np.vstack((grid_x, grid_y)).T

        grid_point_on_img = transform_coords_to_img(grid_point,
                                                         self.map_metadata[0],
                                                         self.map_metadata[2],
                                                         self.map_metadata[3],
                                                         self.map_metadata[4])

        # Find obstacle that is on the track
        obstacle_idx = []
        for idx in range(len(grid_point_on_img)):
            x = int(grid_point_on_img[idx, 0])
            y = int(grid_point_on_img[idx, 1])
            inside_outer_bound = cv2.pointPolygonTest(self.outer_bound, (x, y), True)
            outside_inner_bound = cv2.pointPolygonTest(self.inner_bound, (x, y), True)
            if inside_outer_bound > self.obs_ignore_dist and outside_inner_bound < -self.obs_ignore_dist:
                obstacle_idx.append(idx)

        self.obstacles = [grid_point[idx] for idx in obstacle_idx if np.linalg.norm(self.curr_pos - grid_point[idx]) < 1.5]
        if len(self.obstacles) == 0:
            self.obstacles = None
            self.queue_lane_freed.append([True] * self.num_lanes)
            return
        # print(f'obstacle_grid_num: {len(obstacle_idx)}')

        lane_freed = []
        for i in range(self.num_lanes):
            d = scipy.spatial.distance.cdist(self.lane_pos[i], self.obstacles)
            lane_freed.append(np.min(d) > self.obs2lane_dist)
        self.queue_lane_freed.append(lane_freed)

        if self.visualize_obstacle:
            if self.obstacles is None:
                return

            marker_arr = MarkerArray()
            for i, pt in enumerate(self.obstacles):
                marker = Marker()
                marker.header.frame_id = "/map"
                marker.id = i
                marker.ns = "obstacle_%u" % i
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                marker.pose.position.x = pt[0]
                marker.pose.position.y = pt[1]

                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
                marker.color.a = 1.0

                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1

                marker.lifetime.nanosec = int(1e8)

                marker_arr.markers.append(marker)
            self.obstacle_pub_.publish(marker_arr)

    def get_L_w_speed(self, speed, corner=False):
        if corner:
            return (self.maxL_corner-self.minL_corner) / self.Lscale_corner * speed + self.minL_corner
        else:
            return (self.maxL-self.minL) / self.Lscale * speed + self.minL

    def get_steer_w_speed(self, speed, error, is_corner):
        if is_corner:
            maxP = self.get_parameter('maxP_corner').get_parameter_value().double_value
            minP = self.get_parameter('minP_corner').get_parameter_value().double_value
            Pscale = self.get_parameter('Pscale_corner').get_parameter_value().double_value
        else:
            maxP = self.get_parameter('maxP').get_parameter_value().double_value
            minP = self.get_parameter('minP').get_parameter_value().double_value
            Pscale = self.get_parameter('Pscale').get_parameter_value().double_value

        interp_P_scale = (maxP-minP) / Pscale
        cur_P = maxP - speed * interp_P_scale
        max_control = self.get_parameter("max_steer").get_parameter_value().double_value
        kd = self.get_parameter('D').get_parameter_value().double_value

        d_error = error - self.prev_steer_error
        # print(f'd_error: {d_error}')
        if not self.real_test:
            if d_error == 0:
                d_error = self.prev_ditem
            else:
                self.prev_ditem = d_error
                self.prev_steer_error = error
        else:
            self.prev_ditem = d_error
            self.prev_steer_error = error
        if is_corner:
            steer = cur_P * error
        else:
            steer = cur_P * error + kd * d_error
        # print(f'cur_p_item:{cur_P * error},  cur_d_item:{kd * d_error}')
        new_steer = np.clip(steer, -max_control, max_control)
        return new_steer

    def print_results(self, curr_idx, target_wp, cte, steer):
        self.callback_count += 1
        if curr_idx < 5:
            now = self.get_clock().now().nanoseconds
            lap_time = (now - self.lap_start) / 10**9
            if lap_time > 5.0:
                print(f'laptime: {lap_time:.2f}s, cte: {self.lap_cte:.2f}m, maxsteer: {self.lap_maxsteer:.2f}')
                self.lap_start = now
                self.lap_cte = 0.0
                self.lap_maxsteer = 0.0
        else:
            self.lap_cte += abs(cte)
            self.lap_maxsteer = max(self.lap_maxsteer, abs(steer))
        if self.visualize_target:
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.id = 0
            marker.ns = "target_waypoint"
            marker.type = 1
            marker.action = 0
            marker.pose.position.x = target_wp[0]
            marker.pose.position.y = target_wp[1]

            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            this_scale = 0.2
            marker.scale.x = this_scale
            marker.scale.y = this_scale
            marker.scale.z = this_scale

            marker.pose.orientation.w = 1.0

            marker.lifetime.nanosec = int(1e8)

            self.waypoint_pub_.publish(marker)

    def pub_cmd(self, steer, speed):
        #### Publish drive message ####
        steer_norm = np.clip(steer, -1.0, 1.0)
        message = AckermannDriveStamped()
        message.drive.speed = speed * self.vel_scale
        message.drive.steering_angle = steer_norm
        self.drive_pub_.publish(message)
        # print(f'{speed:.2f}, {steer:.2f}')

    def is_on_lane(self, dist, yaw_err):
        return dist < self.cte_threshold and abs(yaw_err) < self.yaw_threshold


def main(args=None):
    rclpy.init(args=args)
    print("Pure Pursuit Initialized")
    pure_pursuit_node = HybridNode()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
