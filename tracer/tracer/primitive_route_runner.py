#!/usr/bin/env python3

import os
import math
import time
import yaml

import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def yaw_from_qz_qw(qz, qw):
    return math.atan2(
        2.0 * qw * qz,
        1.0 - 2.0 * qz * qz
    )


def quat_from_yaw(yaw):
    z = math.sin(yaw / 2.0)
    w = math.cos(yaw / 2.0)
    return z, w


class PrimitiveRouteRunner(Node):
    """
    myAGV primitive based route runner.

    목적:
      Nav2 / DWB가 만든 애매한 /cmd_vel_nav 대신,
      route/goal을 AMCL feedback으로 따라가면서 명확한 primitive cmd_vel_nav를 발행한다.

    Motion primitive:
      1) ROTATE_TO_HEADING : 목표점 방향으로 제자리 회전
      2) DRIVE_STRAIGHT    : angular.z = 0으로 직진
      3) ROTATE_TO_GOAL_YAW: 목표 yaw로 제자리 회전
      4) STOP_HOLD         : 다음 동작 전 완전 정지 유지

    Subscribe:
      /primitive_route_cmd std_msgs/String
        - "<route_name>"        : routes에 있는 route 실행
        - "route <route_name>"  : route 실행
        - "goal <goal_name>"    : 단일 goal 실행
        - "stop" / "cancel"    : 정지
        - "reload"             : goals_yaml 다시 로드

      /amcl_pose geometry_msgs/PoseWithCovarianceStamped

    Publish:
      /cmd_vel_nav geometry_msgs/Twist
      /debug/nav_status std_msgs/String
      /debug/current_goal_name std_msgs/String
      /debug/current_goal_pose geometry_msgs/PoseStamped
      /motion_mode std_msgs/String
    """

    def __init__(self):
        super().__init__('primitive_route_runner')

        # ============================================================
        # Parameters: files / topics
        # ============================================================
        self.declare_parameter(
            'goals_yaml',
            '/home/er/myagv_ros2/src/tracer/config/goals.yaml'
        )
        self.declare_parameter('route_name', 'full_mission_b')
        self.declare_parameter('autostart', False)

        self.declare_parameter('cmd_topic', '/primitive_route_cmd')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_nav')
        self.declare_parameter('status_topic', '/debug/nav_status')
        self.declare_parameter('amcl_topic', '/amcl_pose')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('motion_mode_topic', '/motion_mode')
        self.declare_parameter('current_goal_name_topic', '/debug/current_goal_name')
        self.declare_parameter('current_goal_pose_topic', '/debug/current_goal_pose')

        # ArUco final align hook.
        # If enabled, when a specified goal such as to_obj is reached,
        # the runner publishes /aruco_align_cmd "start" and waits for
        # /aruco_align_done true before continuing to the next waypoint.
        self.declare_parameter('enable_aruco_after_goal', True)
        self.declare_parameter('aruco_goal_names', 'to_obj,to_qr_a,to_qr_b,to_qr_c')  # comma-separated
        self.declare_parameter('aruco_cmd_topic', '/aruco_align_cmd')
        self.declare_parameter('aruco_done_topic', '/aruco_align_done')
        self.declare_parameter('aruco_start_delay_sec', 0.80)
        self.declare_parameter('aruco_timeout_sec', 60.0)

        # ============================================================
        # Parameters: runtime
        # ============================================================
        self.declare_parameter('timer_hz', 20.0)
        self.declare_parameter('amcl_timeout_sec', 5.0)

        # AMCL은 update_min_d/update_min_a 조건 때문에 드문드문 나올 수 있다.
        # 그래서 최초 AMCL pose를 map 기준 anchor로 잡고,
        # 그 사이의 짧은 pose 변화는 /odometry/filtered delta로 보간한다.
        self.declare_parameter('use_odom_between_amcl', True)
        self.declare_parameter('odom_timeout_sec', 0.80)

        self.declare_parameter('goal_timeout_sec', 120.0)
        self.declare_parameter('continue_on_failure', False)

        # ============================================================
        # Parameters: primitive behavior
        # ============================================================
        self.declare_parameter('drive_vx', 0.18)
        self.declare_parameter('min_drive_vx', 0.07)
        self.declare_parameter('slow_down_dist', 0.0)
        self.declare_parameter('max_vx', 0.20)

        # myAGV 회전은 실험상 |wz| ~= 0.40 근처부터 의미 있게 먹었으므로 기본 0.40
        self.declare_parameter('turn_wz', 0.45)
        self.declare_parameter('max_wz', 0.45)

        # 목표점 방향으로 먼저 돌 때 허용 yaw error
        self.declare_parameter('heading_tolerance', 0.10)        # rad, about 9.2 deg

        # 직진 중 heading error가 이보다 커지면 멈추고 다시 rotate_to_heading
        self.declare_parameter('drive_heading_tolerance', 0.20)  # rad, about 12.6 deg

        # 목표 xy 도착 판정
        self.declare_parameter('xy_tolerance', 0.13)

        # 목표 yaw 도착 판정
        self.declare_parameter('final_yaw_tolerance', 0.18)      # rad, about 16 deg
        self.declare_parameter('do_final_yaw', True)

        # 같은 goal 안에서 dist가 매우 짧으면 DRIVE를 생략하고 final yaw만 수행
        self.declare_parameter('drive_skip_dist', 0.08)

        # primitive 전환 시 완전 정지 유지 시간
        self.declare_parameter('stop_hold_sec', 0.60)
        self.declare_parameter('state_debug_period_sec', 0.50)

        # ============================================================
        # Load parameters
        # ============================================================
        self.goals_yaml = os.path.expanduser(self.get_parameter('goals_yaml').value)
        self.default_route_name = self.get_parameter('route_name').value
        self.autostart = bool(self.get_parameter('autostart').value)

        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.amcl_topic = self.get_parameter('amcl_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.motion_mode_topic = self.get_parameter('motion_mode_topic').value
        self.current_goal_name_topic = self.get_parameter('current_goal_name_topic').value
        self.current_goal_pose_topic = self.get_parameter('current_goal_pose_topic').value
        self.aruco_cmd_topic = self.get_parameter('aruco_cmd_topic').value
        self.aruco_done_topic = self.get_parameter('aruco_done_topic').value

        self.timer_hz = float(self.get_parameter('timer_hz').value)

        # ============================================================
        # YAML data
        # ============================================================
        self.frame_id = 'map'
        self.targets = {}
        self.routes = {}
        self.load_goals_yaml()

        # ============================================================
        # Robot pose
        # ============================================================
        # current map-frame pose used by the primitive controller.
        # Raw AMCL updates this directly, and odom delta can keep it fresh
        # between sparse AMCL updates.
        self.amcl_x = None
        self.amcl_y = None
        self.amcl_yaw = None
        self.last_amcl_time = 0.0
        self.last_pose_time = 0.0
        self.pose_source = 'NONE'

        # odom-frame pose from /odometry/filtered
        self.odom_x = None
        self.odom_y = None
        self.odom_yaw = None
        self.last_odom_time = 0.0

        # anchor pair: map pose from latest AMCL and odom pose at that moment
        self.anchor_map_x = None
        self.anchor_map_y = None
        self.anchor_map_yaw = None
        self.anchor_odom_x = None
        self.anchor_odom_y = None
        self.anchor_odom_yaw = None

        # ============================================================
        # Route state
        # ============================================================
        self.active = False
        self.state = 'IDLE'
        self.prev_state = 'IDLE'

        self.current_route_name = ''
        self.current_route = []
        self.current_index = 0
        self.goal_start_time = 0.0

        self.target_name = ''
        self.target_pose = None
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_yaw = 0.0

        self.stop_hold_until = 0.0
        self.last_state_debug_time = 0.0

        # ArUco wait state
        self.aruco_waiting_goal_name = ''
        self.aruco_started = False
        self.aruco_done = False
        self.aruco_wait_start_time = 0.0
        self.aruco_start_time = 0.0

        # ============================================================
        # ROS interfaces
        # ============================================================
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, cmd_qos)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.motion_mode_pub = self.create_publisher(String, self.motion_mode_topic, 10)
        self.goal_name_pub = self.create_publisher(String, self.current_goal_name_topic, 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, self.current_goal_pose_topic, 10)
        self.aruco_cmd_pub = self.create_publisher(String, self.aruco_cmd_topic, 10)

        self.cmd_sub = self.create_subscription(String, self.cmd_topic, self.cmd_cb, 10)
        self.aruco_done_sub = self.create_subscription(Bool, self.aruco_done_topic, self.aruco_done_cb, 10)
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_topic,
            self.amcl_cb,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_cb,
            10
        )

        period = 1.0 / max(self.timer_hz, 1.0)
        self.timer = self.create_timer(period, self.timer_cb)

        self.publish_status(
            f'primitive_ready:goals={self.goals_yaml}:cmd_topic={self.cmd_topic}:'
            f'cmd_vel_topic={self.cmd_vel_topic}:targets={len(self.targets)}:'
            f'routes={list(self.routes.keys())}'
        )
        self.get_logger().info('primitive_route_runner started')
        self.get_logger().info(f'goals_yaml={self.goals_yaml}')
        self.get_logger().info(f'cmd_topic={self.cmd_topic}')
        self.get_logger().info(f'cmd_vel_topic={self.cmd_vel_topic}')
        self.get_logger().info(f'status_topic={self.status_topic}')
        self.get_logger().info(f'odom_topic={self.odom_topic}')
        self.get_logger().info(f'aruco_cmd_topic={self.aruco_cmd_topic}')
        self.get_logger().info(f'aruco_done_topic={self.aruco_done_topic}')

        if self.autostart:
            # AMCL이 아직 안 들어왔을 수 있으므로 timer에서 기다리게 그냥 active 시작
            self.start_route(self.default_route_name)

    # ============================================================
    # YAML
    # ============================================================
    def load_goals_yaml(self):
        if not os.path.exists(self.goals_yaml):
            self.get_logger().error(f'goals_yaml not found: {self.goals_yaml}')
            self.targets = {}
            self.routes = {}
            return

        with open(self.goals_yaml, 'r') as f:
            data = yaml.safe_load(f) or {}

        self.frame_id = data.get('frame_id', 'map')
        self.targets = {}

        for section in ['goals', 'waypoints']:
            items = data.get(section, {}) or {}
            for name, pose in items.items():
                self.targets[str(name)] = pose

        self.routes = data.get('routes', {}) or {}

        self.get_logger().info(
            f'loaded goals_yaml | frame_id={self.frame_id} | '
            f'targets={len(self.targets)} | routes={list(self.routes.keys())}'
        )

    def get_target_pose(self, name):
        if name not in self.targets:
            return None
        return self.targets[name]

    def pose_to_xy_yaw(self, pose):
        x = float(pose['x'])
        y = float(pose['y'])

        if 'yaw' in pose:
            yaw = float(pose['yaw'])
        elif 'qz' in pose and 'qw' in pose:
            yaw = yaw_from_qz_qw(float(pose['qz']), float(pose['qw']))
        else:
            yaw = 0.0

        return x, y, normalize_angle(yaw)

    # ============================================================
    # ROS callbacks
    # ============================================================
    def cmd_cb(self, msg):
        raw = msg.data.strip()
        if not raw:
            raw = self.default_route_name

        lower = raw.lower()

        if lower in ['stop', 'cancel']:
            self.cancel_route('manual_cancel')
            return

        if lower == 'reload':
            self.load_goals_yaml()
            self.publish_status('primitive_reload_done')
            return

        # 명시형: "route full_mission_b", "goal to_obj"
        parts = raw.split()
        if len(parts) >= 2 and parts[0].lower() == 'route':
            self.start_route(parts[1])
            return

        if len(parts) >= 2 and parts[0].lower() == 'goal':
            self.start_single_goal(parts[1])
            return

        # 축약형: route 이름이면 route, target 이름이면 single goal
        if raw in self.routes:
            self.start_route(raw)
            return

        if raw in self.targets:
            self.start_single_goal(raw)
            return

        self.publish_status(f'primitive_error:unknown_command_or_target:{raw}')
        self.get_logger().error(f'Unknown command/route/goal: {raw}')

    def amcl_cb(self, msg):
        """Raw AMCL pose in map frame.

        AMCL can be sparse. Every AMCL update becomes a new map/odom anchor.
        Between anchors, odom_cb integrates odom-frame delta onto this map pose.
        """
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        now = time.monotonic()

        self.amcl_x = float(p.x)
        self.amcl_y = float(p.y)
        self.amcl_yaw = normalize_angle(yaw_from_quat(q))
        self.last_amcl_time = now
        self.last_pose_time = now
        self.pose_source = 'AMCL'

        # If odom is already available, bind this AMCL pose to the current odom pose.
        if self.odom_x is not None and self.odom_y is not None and self.odom_yaw is not None:
            self.anchor_map_x = self.amcl_x
            self.anchor_map_y = self.amcl_y
            self.anchor_map_yaw = self.amcl_yaw
            self.anchor_odom_x = self.odom_x
            self.anchor_odom_y = self.odom_y
            self.anchor_odom_yaw = self.odom_yaw

    def odom_cb(self, msg):
        """Use odom delta to keep the map-frame pose fresh between AMCL updates."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        now = time.monotonic()

        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        self.odom_yaw = normalize_angle(yaw_from_quat(q))
        self.last_odom_time = now

        if not bool(self.get_parameter('use_odom_between_amcl').value):
            return

        # Need at least one AMCL/odom anchor pair first.
        if (
            self.anchor_map_x is None or self.anchor_map_y is None or
            self.anchor_map_yaw is None or self.anchor_odom_x is None or
            self.anchor_odom_y is None or self.anchor_odom_yaw is None
        ):
            return

        dx_o = self.odom_x - self.anchor_odom_x
        dy_o = self.odom_y - self.anchor_odom_y
        dyaw_o = normalize_angle(self.odom_yaw - self.anchor_odom_yaw)

        # Rotate odom-frame delta into map frame using the yaw offset at anchor.
        theta = normalize_angle(self.anchor_map_yaw - self.anchor_odom_yaw)
        c = math.cos(theta)
        s = math.sin(theta)

        self.amcl_x = self.anchor_map_x + c * dx_o - s * dy_o
        self.amcl_y = self.anchor_map_y + s * dx_o + c * dy_o
        self.amcl_yaw = normalize_angle(self.anchor_map_yaw + dyaw_o)
        self.last_pose_time = now
        self.pose_source = 'ODOM_DELTA'

    def aruco_done_cb(self, msg):
        if bool(msg.data):
            self.aruco_done = True
            self.publish_status(f'aruco_done_received:{self.current_route_name}:{self.target_name}')

    def aruco_goal_set(self):
        raw = str(self.get_parameter('aruco_goal_names').value)
        return {x.strip() for x in raw.split(',') if x.strip()}

    def should_run_aruco_for_current_goal(self):
        if not bool(self.get_parameter('enable_aruco_after_goal').value):
            return False
        return self.target_name in self.aruco_goal_set()

    def publish_aruco_cmd(self, cmd):
        msg = String()
        msg.data = str(cmd)
        self.aruco_cmd_pub.publish(msg)
        self.publish_status(f'aruco_cmd:{cmd}:{self.current_route_name}:{self.target_name}')

    # ============================================================
    # Route control
    # ============================================================
    def start_single_goal(self, goal_name):
        if goal_name not in self.targets:
            self.publish_status(f'primitive_error:unknown_goal:{goal_name}')
            return

        self.current_route_name = f'single_{goal_name}'
        self.current_route = [goal_name]
        self.current_index = 0
        self.active = True
        self.set_state('LOAD_NEXT_GOAL')
        self.publish_status(f'route_started:{self.current_route_name}:count=1')

    def start_route(self, route_name):
        if route_name not in self.routes:
            self.publish_status(f'primitive_error:unknown_route:{route_name}')
            self.get_logger().error(f'Unknown route: {route_name}')
            return

        route = self.routes.get(route_name, [])
        if not route:
            self.publish_status(f'primitive_error:empty_route:{route_name}')
            return

        # 현재 goals.yaml은 list[str] 구조.
        # mission_save_node가 만든 list[dict] 구조도 최소 호환한다.
        normalized = []
        for i, item in enumerate(route):
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                name = item.get('name', f'{route_name}_wp{i + 1:02d}')
                self.targets[name] = item
                normalized.append(name)
            else:
                self.publish_status(f'primitive_error:bad_route_item:{route_name}:{i}')
                return

        self.current_route_name = route_name
        self.current_route = normalized
        self.current_index = 0
        self.active = True
        self.set_state('LOAD_NEXT_GOAL')
        self.publish_status(f'route_started:{route_name}:count={len(self.current_route)}')

    def cancel_route(self, reason):
        self.publish_aruco_cmd('stop')
        self.publish_stop()
        self.active = False
        self.set_state('IDLE')
        self.publish_motion_mode('STOP')
        self.publish_status(
            f'route_cancelled:{self.current_route_name}:{self.target_name}:{reason}'
        )

    def finish_route(self):
        self.publish_stop()
        route_name = self.current_route_name
        self.active = False
        self.set_state('IDLE')
        self.publish_motion_mode('STOP')
        self.publish_status(f'route_finished:{route_name}')

    def fail_or_skip_goal(self, reason):
        self.publish_stop()
        self.publish_status(
            f'route_goal_failed:{self.current_route_name}:{self.target_name}:{reason}'
        )

        if bool(self.get_parameter('continue_on_failure').value):
            self.current_index += 1
            self.set_state('LOAD_NEXT_GOAL')
        else:
            self.cancel_route(reason)

    # ============================================================
    # Timer state machine
    # ============================================================
    def timer_cb(self):
        now = time.monotonic()

        # stop hold 중에는 active 여부와 관계없이 0을 계속 발행
        if now < self.stop_hold_until:
            self.publish_stop()
            return

        if not self.active:
            return

        if not self.has_recent_amcl(now):
            self.publish_stop()
            self.publish_motion_mode('WAIT_AMCL')
            self.publish_debug_throttled('primitive_wait_amcl')
            return

        if self.state == 'LOAD_NEXT_GOAL':
            self.load_next_goal()
            return

        if self.target_pose is None:
            self.fail_or_skip_goal('target_pose_none')
            return

        goal_timeout = float(self.get_parameter('goal_timeout_sec').value)
        if self.goal_start_time > 0.0 and (now - self.goal_start_time) > goal_timeout:
            self.fail_or_skip_goal(f'goal_timeout_{goal_timeout:.1f}s')
            return

        dist, heading_error, final_yaw_error = self.compute_errors()

        if self.state == 'BEGIN_GOAL':
            self.handle_begin_goal(dist, heading_error, final_yaw_error)
            return

        if self.state == 'ROTATE_TO_HEADING':
            self.handle_rotate_to_heading(dist, heading_error)
            return

        if self.state == 'DRIVE_STRAIGHT':
            self.handle_drive_straight(dist, heading_error)
            return

        if self.state == 'ROTATE_TO_GOAL_YAW':
            self.handle_rotate_to_goal_yaw(dist, final_yaw_error)
            return

        if self.state == 'WAIT_ARUCO_ALIGN':
            self.handle_wait_aruco_align()
            return

        if self.state == 'GOAL_STOP_HOLD':
            # stop_hold_until은 transition에서 설정되므로 보통 여기로 오래 머물지 않음.
            self.finish_current_goal()
            return

        self.publish_stop()
        self.publish_debug_throttled(f'primitive_unknown_state:{self.state}')

    def load_next_goal(self):
        if self.current_index >= len(self.current_route):
            self.finish_route()
            return

        name = self.current_route[self.current_index]
        pose = self.get_target_pose(name)
        if pose is None:
            self.target_name = name
            self.fail_or_skip_goal(f'unknown_target_{name}')
            return

        self.target_name = name
        self.target_pose = pose
        self.target_x, self.target_y, self.target_yaw = self.pose_to_xy_yaw(pose)
        self.goal_start_time = time.monotonic()

        self.publish_current_goal()
        self.publish_status(
            f'route_goal_sent:{self.current_route_name}:'
            f'{self.current_index + 1}/{len(self.current_route)}:{self.target_name}'
        )
        self.set_state('BEGIN_GOAL')

    def handle_begin_goal(self, dist, heading_error, final_yaw_error):
        xy_tol = float(self.get_parameter('xy_tolerance').value)
        drive_skip_dist = float(self.get_parameter('drive_skip_dist').value)
        heading_tol = float(self.get_parameter('heading_tolerance').value)
        do_final_yaw = bool(self.get_parameter('do_final_yaw').value)
        final_yaw_tol = float(self.get_parameter('final_yaw_tolerance').value)

        if dist <= xy_tol or dist <= drive_skip_dist:
            if do_final_yaw and abs(final_yaw_error) > final_yaw_tol:
                self.transition_with_stop('ROTATE_TO_GOAL_YAW')
            else:
                self.goal_reached(dist, final_yaw_error)
            return

        if abs(heading_error) > heading_tol:
            self.transition_with_stop('ROTATE_TO_HEADING')
        else:
            self.transition_with_stop('DRIVE_STRAIGHT')

    def handle_rotate_to_heading(self, dist, heading_error):
        xy_tol = float(self.get_parameter('xy_tolerance').value)
        heading_tol = float(self.get_parameter('heading_tolerance').value)

        if dist <= xy_tol:
            self.transition_with_stop('ROTATE_TO_GOAL_YAW')
            return

        if abs(heading_error) <= heading_tol:
            self.transition_with_stop('DRIVE_STRAIGHT')
            return

        wz = self.turn_command(heading_error)
        self.publish_cmd(0.0, 0.0, wz)
        self.publish_motion_mode('ROTATE_TO_HEADING')
        self.publish_debug_throttled(
            f'primitive_state:{self.current_route_name}:{self.target_name}:'
            f'ROTATE_TO_HEADING:dist={dist:.3f}:heading_err={heading_error:.3f}:wz={wz:.3f}'
        )

    def handle_drive_straight(self, dist, heading_error):
        xy_tol = float(self.get_parameter('xy_tolerance').value)
        drive_heading_tol = float(self.get_parameter('drive_heading_tolerance').value)

        if dist <= xy_tol:
            self.transition_with_stop('ROTATE_TO_GOAL_YAW')
            return

        # 직진 primitive에서는 angular.z를 섞지 않는다.
        # heading이 틀어지면 멈추고 다시 회전 상태로 전이한다.
        if abs(heading_error) > drive_heading_tol:
            self.transition_with_stop('ROTATE_TO_HEADING')
            return

        vx = self.drive_command(dist)
        self.publish_cmd(vx, 0.0, 0.0)
        self.publish_motion_mode('DRIVE_STRAIGHT')
        self.publish_debug_throttled(
            f'primitive_state:{self.current_route_name}:{self.target_name}:'
            f'DRIVE_STRAIGHT:dist={dist:.3f}:heading_err={heading_error:.3f}:vx={vx:.3f}'
        )

    def handle_rotate_to_goal_yaw(self, dist, final_yaw_error):
        do_final_yaw = bool(self.get_parameter('do_final_yaw').value)
        final_yaw_tol = float(self.get_parameter('final_yaw_tolerance').value)

        if (not do_final_yaw) or abs(final_yaw_error) <= final_yaw_tol:
            self.goal_reached(dist, final_yaw_error)
            return

        wz = self.turn_command(final_yaw_error)
        self.publish_cmd(0.0, 0.0, wz)
        self.publish_motion_mode('ROTATE_TO_GOAL_YAW')
        self.publish_debug_throttled(
            f'primitive_state:{self.current_route_name}:{self.target_name}:'
            f'ROTATE_TO_GOAL_YAW:dist={dist:.3f}:yaw_err={final_yaw_error:.3f}:wz={wz:.3f}'
        )

    def handle_wait_aruco_align(self):
        now = time.monotonic()

        # Before giving control to aruco_align_node, hold stop briefly.
        # After start is sent, do NOT keep publishing /cmd_vel_nav here,
        # otherwise this runner would fight with aruco_align_node.
        if not self.aruco_started:
            if now < self.stop_hold_until:
                self.publish_stop()
                self.publish_motion_mode('PRE_ARUCO_STOP_HOLD')
                return

            self.aruco_done = False
            self.aruco_started = True
            self.aruco_start_time = now
            self.publish_aruco_cmd('start')
            self.publish_motion_mode('ARUCO_ALIGN')
            self.publish_status(
                f'aruco_align_started:{self.current_route_name}:{self.target_name}'
            )
            return

        self.publish_motion_mode('ARUCO_ALIGN')

        if self.aruco_done:
            self.publish_aruco_cmd('stop')
            self.publish_stop()
            self.publish_status(
                f'aruco_align_succeeded:{self.current_route_name}:{self.target_name}'
            )
            self.current_index += 1
            self.set_state('LOAD_NEXT_GOAL')
            stop_hold = float(self.get_parameter('stop_hold_sec').value)
            self.stop_hold_until = time.monotonic() + max(0.0, stop_hold)
            return

        timeout = float(self.get_parameter('aruco_timeout_sec').value)
        if timeout > 0.0 and (now - self.aruco_start_time) > timeout:
            self.publish_aruco_cmd('stop')
            self.fail_or_skip_goal(f'aruco_timeout_{timeout:.1f}s')
            return

        self.publish_debug_throttled(
            f'aruco_align_waiting:{self.current_route_name}:{self.target_name}:'
            f'elapsed={now - self.aruco_start_time:.2f}'
        )

    # ============================================================
    # Error / command helpers
    # ============================================================
    def compute_errors(self):
        dx = self.target_x - self.amcl_x
        dy = self.target_y - self.amcl_y
        dist = math.hypot(dx, dy)

        if dist > 1e-6:
            target_heading = math.atan2(dy, dx)
        else:
            target_heading = self.amcl_yaw

        heading_error = normalize_angle(target_heading - self.amcl_yaw)
        final_yaw_error = normalize_angle(self.target_yaw - self.amcl_yaw)

        return dist, heading_error, final_yaw_error

    def turn_command(self, yaw_error):
        turn_wz = abs(float(self.get_parameter('turn_wz').value))
        max_wz = abs(float(self.get_parameter('max_wz').value))
        wz_mag = clamp(turn_wz, 0.0, max_wz)

        if yaw_error > 0.0:
            return wz_mag
        if yaw_error < 0.0:
            return -wz_mag
        return 0.0

    def drive_command(self, dist):
        drive_vx = float(self.get_parameter('drive_vx').value)
        min_drive_vx = float(self.get_parameter('min_drive_vx').value)
        slow_down_dist = float(self.get_parameter('slow_down_dist').value)
        max_vx = float(self.get_parameter('max_vx').value)

        drive_vx = clamp(drive_vx, 0.0, max_vx)
        min_drive_vx = clamp(min_drive_vx, 0.0, drive_vx)

        if slow_down_dist > 1e-6 and dist < slow_down_dist:
            vx = drive_vx * (dist / slow_down_dist)
            vx = max(min_drive_vx, vx)
        else:
            vx = drive_vx

        return clamp(vx, 0.0, max_vx)

    def has_recent_amcl(self, now):
        # 이름은 기존 코드 호환 때문에 유지하지만,
        # 실제 의미는 "primitive 제어에 사용할 pose가 최근에 있는가"이다.
        if self.amcl_x is None or self.amcl_y is None or self.amcl_yaw is None:
            return False

        if bool(self.get_parameter('use_odom_between_amcl').value):
            odom_timeout = float(self.get_parameter('odom_timeout_sec').value)
            if (now - self.last_pose_time) <= odom_timeout:
                return True

        # odom 보간을 끄거나 odom이 끊긴 경우에는 AMCL freshness만 본다.
        amcl_timeout = float(self.get_parameter('amcl_timeout_sec').value)
        if amcl_timeout <= 0.0:
            return True
        return (now - self.last_amcl_time) <= amcl_timeout

    # ============================================================
    # Transition / publish helpers
    # ============================================================
    def transition_with_stop(self, next_state):
        self.publish_stop()
        stop_hold = float(self.get_parameter('stop_hold_sec').value)
        self.stop_hold_until = time.monotonic() + max(0.0, stop_hold)
        self.set_state(next_state)
        self.publish_status(
            f'primitive_transition:{self.current_route_name}:{self.target_name}:{next_state}'
        )

    def goal_reached(self, dist, yaw_error):
        self.publish_stop()
        self.publish_status(
            f'route_goal_succeeded:{self.current_route_name}:{self.target_name}:'
            f'dist={dist:.3f}:yaw_error={yaw_error:.3f}'
        )

        if self.should_run_aruco_for_current_goal():
            self.aruco_waiting_goal_name = self.target_name
            self.aruco_started = False
            self.aruco_done = False
            self.aruco_wait_start_time = time.monotonic()
            delay = float(self.get_parameter('aruco_start_delay_sec').value)
            self.stop_hold_until = time.monotonic() + max(0.0, delay)
            self.set_state('WAIT_ARUCO_ALIGN')
            self.publish_status(
                f'route_goal_wait_aruco:{self.current_route_name}:{self.target_name}:delay={delay:.2f}'
            )
            return

        self.current_index += 1
        self.set_state('LOAD_NEXT_GOAL')
        stop_hold = float(self.get_parameter('stop_hold_sec').value)
        self.stop_hold_until = time.monotonic() + max(0.0, stop_hold)

    def finish_current_goal(self):
        self.current_index += 1
        self.set_state('LOAD_NEXT_GOAL')

    def set_state(self, new_state):
        if new_state != self.state:
            self.prev_state = self.state
            self.state = new_state
            self.get_logger().info(f'state change: {self.prev_state} -> {self.state}')

    def publish_cmd(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz)
        self.cmd_vel_pub.publish(msg)

    def publish_stop(self):
        self.publish_cmd(0.0, 0.0, 0.0)

    def publish_status(self, text):
        msg = String()
        msg.data = str(text)
        self.status_pub.publish(msg)
        self.get_logger().info(str(text))

    def publish_debug_throttled(self, text):
        now = time.monotonic()
        period = float(self.get_parameter('state_debug_period_sec').value)
        if (now - self.last_state_debug_time) < period:
            return
        self.last_state_debug_time = now
        self.publish_status(text)

    def publish_motion_mode(self, mode):
        msg = String()
        msg.data = str(mode)
        self.motion_mode_pub.publish(msg)

    def publish_current_goal(self):
        name_msg = String()
        name_msg.data = self.target_name
        self.goal_name_pub.publish(name_msg)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = self.target_x
        pose_msg.pose.position.y = self.target_y
        pose_msg.pose.position.z = 0.0

        qz, qw = quat_from_yaw(self.target_yaw)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.goal_pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PrimitiveRouteRunner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
