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
# 10.30 sec
# stable


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
    area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
    return (4 * area) / (a * b * c)


class TimetrialNode(Node):
    def __init__(self):
        super().__init__("timetrial_node")

        # Controller Params
        self.stanley_overall_gain = 1.2
        self.stanley_yaw_gain = 3.0
        self.stanley_cte_gain = 1.0
        self.stanley_k = 2.5
        self.stanley_ks = 1.6

        self.speed_gain = 1.0
        self.curvature_decel = 0.6
        self.curvature_threshold = 0.35
        
        self.simul_speed_compensate = 1.0
        
        self.cte_threshold = 0.15
        self.yaw_threshold = np.pi / 8

        self.obs2lane_dist = 0.3    # m
        self.obs_ignore_dist = 7.5  # cm
        self.obs_recog_dist = 2.5   # m

        self.L = 0.3302  # Wheelbase length
        self.dt = 0.1
        self.prev_ditem = 0.0
        self.prev_steer_error = 0.0
        self.corner_wpIdx = []
        self.corner_decel = 1.0     # given in ratio
        
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
        self.lane_files = ['lane_optimal']
        self.num_lanes = len(self.lane_files)
        self.lane_free = [True] * self.num_lanes
        self.queue_lane_freed = []

        self.num_lane_pts = []
        self.lane_x = []
        self.lane_y = []
        self.lane_pos = []

        for i in range(self.num_lanes):
            lane_csv_loc = os.path.join("src", "pure_pursuit", "csv", self.map_name, self.lane_files[i] + ".csv")
            lane_data = np.loadtxt(lane_csv_loc, delimiter=",")
            self.num_lane_pts.append(len(lane_data))
            self.lane_x.append(lane_data[:, 0])
            self.lane_y.append(lane_data[:, 1])
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
            self.pose_sub_ = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 1)
        else:
            self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 1)
        self.odom_sub_     = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.drive_pub_    = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)
        self.obstacle_pub_ = self.create_publisher(MarkerArray, obstacle_topic, 10)
        print('node_init_files')

        # timestamped
        self.scan_timestamped_nanosec = None
        self.scan_timestamped_sec = None
        self.callback_count = 0
        self.lap_count = 0
    
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
        self.curr_vel = self.odom_vel

        #### Get Closest Waypoint and Target Waypoint
        lanes = [i for i in range(self.num_lanes)]
        closest_idxs = [np.argmin(np.linalg.norm(self.lane_pos[i][:, :2] - curr_pos, axis=1)) for i in lanes]
        
        traj_distances = [np.linalg.norm(self.lane_pos[i][:, :2] - self.lane_pos[i][closest_idxs[i], :2], axis=1) for i in lanes]
        segment_ends = [np.argmin(traj_distances[i]) for i in lanes]
        L = self.get_L_w_speed(self.curr_vel)
        interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value
        for i in lanes:
            while traj_distances[i][segment_ends[i]] <= L:
                segment_ends[i] = safe_changeIdx(len(traj_distances[i]), segment_ends[i], 1)
        segment_begins = [safe_changeIdx(len(traj_distances[i]), segment_ends[i], -1) for i in lanes]
        x_arrays = [np.linspace(self.lane_x[i][segment_begins[i]], self.lane_x[i][segment_ends[i]], interpScale) for i in lanes]
        y_arrays = [np.linspace(self.lane_y[i][segment_begins[i]], self.lane_y[i][segment_ends[i]], interpScale) for i in lanes]
        xy_interps = [np.vstack([x_arrays[i], y_arrays[i]]).T for i in lanes]
        dist_interps = [np.linalg.norm(xy_interps[i] - curr_pos, axis=1) - L for i in lanes]
        i_interps = [np.argmin(dist_interps[i]) for i in lanes]

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

        #### get steer values ####
        cte = self.ctes[self.curr_lane]
        yaw_error = self.yaw_errors[self.curr_lane]
        curr_lane_pos = self.lane_pos[self.curr_lane]
        following_idxs = [safe_changeIdx(len(self.lane_pos[self.curr_lane]), closest_idxs[self.curr_lane], i) for i in range(3)]
        curvature = get_curvature(curr_lane_pos, following_idxs)
        
        steer = self.get_stanley_steer(cte, yaw_error, self.curr_vel, curvature)

        #### Publish the command ####
        speed = max(9.0 - curvature * 9, 4.0)
        # speed = 3.0
        # if self.lap_count >= 0:
        #     # speed = 4.5
        if not self.real_test:
            speed = speed * self.simul_speed_compensate
        # if closest_idxs[-1] in self.corner_wpIdx:
        #     speed = speed * self.corner_decel
        # if curvature > self.curvature_threshold:
        #     speed = speed * self.curvature_decel

        self.pub_cmd(steer, speed)
        self.prev_steer = steer
        self.prev_speed = speed
        str_lane_free = ''.join(['O' if lane_free else 'X' for lane_free in self.lane_free])
        str_steer = ('l' if steer > 0 else 'r') + f'{abs(steer):.2f}'
        calc_time = (self.get_clock().now().nanoseconds - start) / 10**6
        # print(f'steer:{str_steer}, speed:{speed:.2f}, lane:{self.curr_lane}, wp:{closest_idxs[-1]}, curve:{curvature:.2f}, time:{calc_time:.1f}, lap:{self.lap_count}')
        # print(f'odom:{self.odom_vel:.2f}, pose:{self.curr_vel:.2f}')
        self.print_results(closest_idxs[-1], self.target_wps[self.curr_lane], cte_sizes[self.curr_lane] , steer)

    def get_stanley_steer(self, cte, yaw_err, speed, curvature):
        delta = (self.stanley_yaw_gain - curvature * 1.0) * yaw_err - self.stanley_cte_gain * np.arctan2(self.stanley_k * cte, speed + self.stanley_ks)
        steer = self.stanley_overall_gain * (delta * self.L) / max(self.curr_vel, 1.0) / 0.36
        return steer

    def odom_callback(self, odom_msg: Odometry):
        # self.curr_vel = odom_msg.twist.twist.linear.x
        self.odom_vel = odom_msg.twist.twist.linear.x
        pass

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
                self.lap_count = self.lap_count + 1
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
        message = AckermannDriveStamped()
        message.drive.speed = speed
        message.drive.steering_angle = np.clip(steer, -1.0, 1.0)
        self.drive_pub_.publish(message)


def main(args=None):
    rclpy.init(args=args)
    print("Pure Pursuit Initialized")
    pure_pursuit_node = TimetrialNode()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
