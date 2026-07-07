#!/usr/bin/env python3

import os
import csv
import math
import yaml
from datetime import datetime

import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


def quat_to_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def yaw_from_qz_qw(qz, qw):
    return math.atan2(
        2.0 * qw * qz,
        1.0 - 2.0 * qz * qz
    )


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class NavDebugLogger(Node):
    def __init__(self):
        super().__init__('nav_debug_logger')

        # =========================
        # Parameters
        # =========================
        self.declare_parameter('debug_dir', '/home/er/nav_debug')
        self.declare_parameter('log_rate_hz', 10.0)

        self.declare_parameter(
            'goals_yaml',
            '/home/er/myagv_ros2/src/tracer/config/goals.yaml'
        )

        self.declare_parameter('cmd_vel_nav_topic', '/cmd_vel_nav')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('amcl_topic', '/amcl_pose')
        self.declare_parameter('odom_topic', '/odometry/filtered')

        self.declare_parameter('current_goal_pose_topic', '/debug/current_goal_pose')
        self.declare_parameter('current_goal_name_topic', '/debug/current_goal_name')
        self.declare_parameter('nav_status_topic', '/debug/nav_status')

        self.debug_dir = self.get_parameter('debug_dir').value
        self.log_rate_hz = float(self.get_parameter('log_rate_hz').value)

        self.goals_yaml = os.path.expanduser(
            self.get_parameter('goals_yaml').value
        )

        self.cmd_vel_nav_topic = self.get_parameter('cmd_vel_nav_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.amcl_topic = self.get_parameter('amcl_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value

        self.current_goal_pose_topic = self.get_parameter(
            'current_goal_pose_topic'
        ).value
        self.current_goal_name_topic = self.get_parameter(
            'current_goal_name_topic'
        ).value
        self.nav_status_topic = self.get_parameter('nav_status_topic').value

        os.makedirs(self.debug_dir, exist_ok=True)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(
            self.debug_dir,
            f'nav_debug_{stamp}.csv'
        )

        # =========================
        # Loaded goals
        # =========================
        self.targets = {}
        self.load_goals_yaml()

        # =========================
        # Latest data holders
        # =========================
        self.cmd_nav = Twist()
        self.cmd_out = Twist()

        self.amcl_pose = None
        self.odom = None

        # target info
        self.target_pose_msg = None
        self.target_name = ''
        self.target_source = ''

        self.route_name = ''
        self.goal_index = ''
        self.goal_count = ''

        self.nav_status = ''

        # =========================
        # QoS
        # =========================
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # =========================
        # Subscriptions
        # =========================
        self.create_subscription(
            Twist,
            self.cmd_vel_nav_topic,
            self.cmd_nav_cb,
            best_effort_qos
        )

        self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_out_cb,
            best_effort_qos
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_topic,
            self.amcl_cb,
            reliable_qos
        )

        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_cb,
            reliable_qos
        )

        # optional topics.
        # debug_route_runner가 나중에 publish하도록 바뀌어도 호환됨.
        self.create_subscription(
            PoseStamped,
            self.current_goal_pose_topic,
            self.target_pose_cb,
            reliable_qos
        )

        self.create_subscription(
            String,
            self.current_goal_name_topic,
            self.target_name_cb,
            reliable_qos
        )

        self.create_subscription(
            String,
            self.nav_status_topic,
            self.nav_status_cb,
            reliable_qos
        )

        # =========================
        # CSV
        # =========================
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            't_sec',

            'route_name',
            'goal_index',
            'goal_count',

            'target_name',
            'target_source',
            'nav_status',

            'target_x',
            'target_y',
            'target_yaw',

            'amcl_x',
            'amcl_y',
            'amcl_yaw',

            'dist_error',
            'yaw_error_rad',
            'yaw_error_deg',

            'cmd_vel_nav_linear_x',
            'cmd_vel_nav_linear_y',
            'cmd_vel_nav_angular_z',

            'cmd_vel_linear_x',
            'cmd_vel_linear_y',
            'cmd_vel_angular_z',

            'odom_linear_x',
            'odom_linear_y',
            'odom_angular_z',

            'delta_linear_x_nav_minus_out',
            'delta_linear_y_nav_minus_out',
            'delta_angular_z_nav_minus_out',

            'amcl_cov_xx',
            'amcl_cov_yy',
            'amcl_cov_yaw'
        ])

        period = 1.0 / max(self.log_rate_hz, 0.1)
        self.timer = self.create_timer(period, self.log_row)

        self.get_logger().info('nav_debug_logger started')
        self.get_logger().info(f'CSV: {self.csv_path}')
        self.get_logger().info(f'goals_yaml={self.goals_yaml}')
        self.get_logger().info(f'loaded_targets={len(self.targets)}')
        self.get_logger().info(f'cmd_vel_nav_topic={self.cmd_vel_nav_topic}')
        self.get_logger().info(f'cmd_vel_topic={self.cmd_vel_topic}')
        self.get_logger().info(f'amcl_topic={self.amcl_topic}')
        self.get_logger().info(f'odom_topic={self.odom_topic}')
        self.get_logger().info(f'nav_status_topic={self.nav_status_topic}')

    # =========================
    # YAML loader
    # =========================
    def load_goals_yaml(self):
        self.targets = {}

        if not self.goals_yaml:
            self.get_logger().warn('goals_yaml parameter is empty')
            return

        if not os.path.exists(self.goals_yaml):
            self.get_logger().warn(f'goals_yaml not found: {self.goals_yaml}')
            return

        try:
            with open(self.goals_yaml, 'r') as f:
                data = yaml.safe_load(f) or {}

            for section in ['goals', 'waypoints']:
                items = data.get(section, {}) or {}
                for name, pose in items.items():
                    self.targets[str(name)] = pose

        except Exception as e:
            self.get_logger().error(f'failed to load goals_yaml: {e}')

    # =========================
    # Callbacks
    # =========================
    def cmd_nav_cb(self, msg):
        self.cmd_nav = msg

    def cmd_out_cb(self, msg):
        self.cmd_out = msg

    def amcl_cb(self, msg):
        self.amcl_pose = msg

    def odom_cb(self, msg):
        self.odom = msg

    def target_pose_cb(self, msg):
        # debug_route_runner가 current_goal_pose를 publish하는 경우를 위한 호환.
        self.target_pose_msg = msg
        if self.target_source == '':
            self.target_source = 'topic'

    def target_name_cb(self, msg):
        # debug_route_runner가 current_goal_name을 publish하는 경우를 위한 호환.
        name = msg.data.strip()
        if name:
            self.set_target_name(name, source='topic')

    def nav_status_cb(self, msg):
        self.nav_status = msg.data
        self.parse_nav_status(msg.data)

    # =========================
    # Status parser
    # =========================
    def parse_nav_status(self, text):
        if not text:
            return

        parts = text.split(':')
        event = parts[0] if len(parts) > 0 else ''

        # route_started:debug_full_route:count=19
        if event == 'route_started':
            if len(parts) >= 2:
                self.route_name = parts[1]

            if len(parts) >= 3 and parts[2].startswith('count='):
                self.goal_count = parts[2].replace('count=', '')
                self.goal_index = ''

            self.target_name = ''
            self.target_source = ''
            return

        # route_finished:debug_full_route
        if event == 'route_finished':
            if len(parts) >= 2:
                self.route_name = parts[1]

            self.goal_index = ''
            self.goal_count = ''
            self.target_name = ''
            self.target_source = ''
            return

        # route_goal_sent:debug_full_route:2/19:way1
        if event == 'route_goal_sent':
            if len(parts) >= 2:
                self.route_name = parts[1]

            if len(parts) >= 3 and '/' in parts[2]:
                idx_count = parts[2].split('/')
                if len(idx_count) == 2:
                    self.goal_index = idx_count[0]
                    self.goal_count = idx_count[1]

            if len(parts) >= 4:
                target = parts[3].strip()
                if target:
                    self.set_target_name(target, source='status')

            return

        # route_goal_accepted:debug_full_route:way1
        # route_goal_succeeded:debug_full_route:way1
        # route_goal_near_enter:debug_full_route:way1:dist=0.150
        # route_goal_fine_timeout:debug_full_route:way1:fine_elapsed=20.20:dist=0.150
        # route_goal_cancel_requested:debug_full_route:way1:fine_timeout
        # route_goal_cancelled_and_skip:debug_full_route:way1:fine_timeout
        # route_goal_hard_timeout:debug_full_route:way1:elapsed=...:dist=...
        # route_goal_failed:::status=5  <-- target 비어 있으므로 무시
        if event.startswith('route_goal_'):
            if len(parts) >= 2 and parts[1].strip():
                self.route_name = parts[1].strip()

            if len(parts) >= 3 and parts[2].strip():
                self.set_target_name(parts[2].strip(), source='status')

            return

        # route_cancel_requested:debug_full_route:way1:manual_cancel
        # route_cancelled:debug_full_route:way1:manual_cancel
        if event.startswith('route_cancel'):
            if len(parts) >= 2 and parts[1].strip():
                self.route_name = parts[1].strip()

            if len(parts) >= 3 and parts[2].strip():
                self.set_target_name(parts[2].strip(), source='status')

            return

    def set_target_name(self, name, source='status'):
        if not name:
            return

        self.target_name = name

        if name in self.targets:
            self.target_source = 'yaml'
        else:
            self.target_source = source

    # =========================
    # Target helpers
    # =========================
    def get_target_xy_yaw(self):
        """
        return: target_x, target_y, target_yaw, source
        없으면 '', '', '', ''
        """

        # 1순위: goals.yaml에서 target_name 기반으로 계산
        if self.target_name and self.target_name in self.targets:
            pose = self.targets[self.target_name]

            try:
                tx = float(pose.get('x', 0.0))
                ty = float(pose.get('y', 0.0))

                # 대부분 qz/qw만 저장되어 있음
                qz = float(pose.get('qz', 0.0))
                qw = float(pose.get('qw', 1.0))

                tyaw = yaw_from_qz_qw(qz, qw)

                return tx, ty, tyaw, 'yaml'

            except Exception:
                return '', '', '', ''

        # 2순위: /debug/current_goal_pose topic
        if self.target_pose_msg is not None:
            try:
                tp = self.target_pose_msg.pose
                tx = tp.position.x
                ty = tp.position.y
                tyaw = quat_to_yaw(tp.orientation)

                return tx, ty, tyaw, 'topic'

            except Exception:
                return '', '', '', ''

        return '', '', '', ''

    # =========================
    # Logging
    # =========================
    def log_row(self):
        now = self.get_clock().now().nanoseconds / 1e9

        target_x = ''
        target_y = ''
        target_yaw = ''
        target_source = self.target_source

        amcl_x = ''
        amcl_y = ''
        amcl_yaw = ''

        dist_error = ''
        yaw_error = ''
        yaw_error_deg = ''

        odom_linear_x = ''
        odom_linear_y = ''
        odom_angular_z = ''

        cov_xx = ''
        cov_yy = ''
        cov_yaw = ''

        # target from yaml or topic
        target_x, target_y, target_yaw, source_from_target = self.get_target_xy_yaw()
        if source_from_target:
            target_source = source_from_target

        # AMCL pose
        if self.amcl_pose is not None:
            ap = self.amcl_pose.pose.pose

            amcl_x = ap.position.x
            amcl_y = ap.position.y
            amcl_yaw = quat_to_yaw(ap.orientation)

            cov = self.amcl_pose.pose.covariance
            cov_xx = cov[0]
            cov_yy = cov[7]
            cov_yaw = cov[35]

        # odometry velocity
        if self.odom is not None:
            odom_linear_x = self.odom.twist.twist.linear.x
            odom_linear_y = self.odom.twist.twist.linear.y
            odom_angular_z = self.odom.twist.twist.angular.z

        # errors
        if target_x != '' and amcl_x != '':
            dx = float(target_x) - float(amcl_x)
            dy = float(target_y) - float(amcl_y)

            dist_error = math.sqrt(dx * dx + dy * dy)

            yaw_error = normalize_angle(float(target_yaw) - float(amcl_yaw))
            yaw_error_deg = math.degrees(yaw_error)

        self.writer.writerow([
            f'{now:.3f}',

            self.route_name,
            self.goal_index,
            self.goal_count,

            self.target_name,
            target_source,
            self.nav_status,

            target_x,
            target_y,
            target_yaw,

            amcl_x,
            amcl_y,
            amcl_yaw,

            dist_error,
            yaw_error,
            yaw_error_deg,

            self.cmd_nav.linear.x,
            self.cmd_nav.linear.y,
            self.cmd_nav.angular.z,

            self.cmd_out.linear.x,
            self.cmd_out.linear.y,
            self.cmd_out.angular.z,

            odom_linear_x,
            odom_linear_y,
            odom_angular_z,

            self.cmd_nav.linear.x - self.cmd_out.linear.x,
            self.cmd_nav.linear.y - self.cmd_out.linear.y,
            self.cmd_nav.angular.z - self.cmd_out.angular.z,

            cov_xx,
            cov_yy,
            cov_yaw
        ])

        self.csv_file.flush()

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NavDebugLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
