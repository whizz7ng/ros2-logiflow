#!/usr/bin/env python3

import os
import math
import time
import yaml

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class DebugRouteRunner(Node):
    def __init__(self):
        super().__init__('debug_route_runner')

        # =========================
        # Parameters
        # =========================
        self.declare_parameter('goals_file', '')
        self.declare_parameter('goals_yaml', '')
        self.declare_parameter('route_name', 'debug_to_obj_only')
        self.declare_parameter('autostart', False)

        # legacy compatibility:
        # 기존 launch에서 goal_timeout_sec만 넘기고 있어도 hard timeout으로 사용
        self.declare_parameter('goal_timeout_sec', 90.0)
        self.declare_parameter('hard_timeout_sec', 0.0)

        # 새 timeout 구조
        self.declare_parameter('near_goal_dist', 0.18)
        self.declare_parameter('fine_timeout_sec', 15.0)
        self.declare_parameter('near_goal_reset_margin', 0.05)

        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('cmd_topic', '/debug_route_cmd')
        self.declare_parameter('status_topic', '/debug/nav_status')
        self.declare_parameter('amcl_topic', '/amcl_pose')
        self.declare_parameter('timer_hz', 5.0)

        self.route_name = self.get_parameter('route_name').value
        self.autostart = bool(self.get_parameter('autostart').value)

        legacy_timeout = float(self.get_parameter('goal_timeout_sec').value)
        hard_timeout = float(self.get_parameter('hard_timeout_sec').value)
        self.hard_timeout_sec = hard_timeout if hard_timeout > 0.0 else legacy_timeout

        self.near_goal_dist = float(self.get_parameter('near_goal_dist').value)
        self.fine_timeout_sec = float(self.get_parameter('fine_timeout_sec').value)
        self.near_goal_reset_margin = float(self.get_parameter('near_goal_reset_margin').value)

        self.frame_id = self.get_parameter('frame_id').value
        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.amcl_topic = self.get_parameter('amcl_topic').value
        self.timer_hz = float(self.get_parameter('timer_hz').value)

        goals_file = self.get_parameter('goals_file').value
        goals_yaml = self.get_parameter('goals_yaml').value

        if goals_file:
            self.goals_path = os.path.expanduser(goals_file)
        elif goals_yaml:
            self.goals_path = os.path.expanduser(goals_yaml)
        else:
            self.goals_path = os.path.expanduser('~/nav_debug/goals.yaml')

        # =========================
        # State
        # =========================
        self.goals = {}
        self.routes = {}

        self.active = False
        self.current_route = []
        self.current_route_name = ''
        self.current_index = 0

        self.target_name = ''
        self.target_pose_dict = None

        self.goal_handle = None
        self.goal_token = 0
        self.cancel_requested = False

        self.goal_sent_time = None
        self.near_since = None
        self.in_near_goal = False

        self.latest_amcl_x = None
        self.latest_amcl_y = None
        self.latest_amcl_time = None

        # =========================
        # ROS interfaces
        # =========================
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.cmd_sub = self.create_subscription(
            String,
            self.cmd_topic,
            self.cmd_cb,
            10
        )

        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_topic,
            self.amcl_cb,
            10
        )

        period = 1.0 / max(self.timer_hz, 1.0)
        self.timer = self.create_timer(period, self.timer_cb)

        self.load_goals_yaml()

        self.get_logger().info(
            f'debug_route_runner ready | goals={self.goals_path} | '
            f'cmd_topic={self.cmd_topic} | status_topic={self.status_topic} | '
            f'hard_timeout={self.hard_timeout_sec:.1f}s | '
            f'near_goal_dist={self.near_goal_dist:.3f}m | '
            f'fine_timeout={self.fine_timeout_sec:.1f}s'
        )

        if self.autostart:
            self.start_route(self.route_name)

    # =========================
    # YAML
    # =========================
    def load_goals_yaml(self):
        if not os.path.exists(self.goals_path):
            self.get_logger().error(f'goals yaml not found: {self.goals_path}')
            return

        with open(self.goals_path, 'r') as f:
            data = yaml.safe_load(f) or {}

        self.goals = {}

        # goals + waypoints를 하나의 target table로 합침
        for section in ['goals', 'waypoints']:
            items = data.get(section, {}) or {}
            for name, pose in items.items():
                self.goals[name] = pose

        self.routes = data.get('routes', {}) or {}

        self.get_logger().info(
            f'loaded targets={len(self.goals)}, routes={list(self.routes.keys())}'
        )

    # =========================
    # Callbacks
    # =========================
    def cmd_cb(self, msg: String):
        cmd = msg.data.strip()

        if cmd in ['cancel', 'stop']:
            self.cancel_current_goal_and_stop_route('manual_cancel')
            return

        if cmd == '':
            cmd = self.route_name

        self.start_route(cmd)

    def amcl_cb(self, msg: PoseWithCovarianceStamped):
        self.latest_amcl_x = msg.pose.pose.position.x
        self.latest_amcl_y = msg.pose.pose.position.y
        self.latest_amcl_time = time.monotonic()

    def timer_cb(self):
        if not self.active:
            return

        if self.goal_sent_time is None:
            return

        if self.cancel_requested:
            return

        now = time.monotonic()
        elapsed = now - self.goal_sent_time

        # 1) hard timeout: goal 전체가 너무 오래 걸릴 때만 skip
        if elapsed > self.hard_timeout_sec:
            dist = self.current_dist_to_target()
            dist_txt = 'nan' if dist is None else f'{dist:.3f}'
            self.publish_status(
                f'route_goal_hard_timeout:{self.current_route_name}:{self.target_name}:'
                f'elapsed={elapsed:.2f}:dist={dist_txt}'
            )
            self.cancel_current_goal_and_skip('hard_timeout')
            return

        # 2) near-goal fine timeout: 목표 근처에 들어온 뒤 미세 보정만 오래 걸릴 때 skip
        dist = self.current_dist_to_target()
        if dist is None:
            return

        near_enter = self.near_goal_dist
        near_exit = self.near_goal_dist + self.near_goal_reset_margin

        if dist <= near_enter:
            if not self.in_near_goal:
                self.in_near_goal = True
                self.near_since = now
                self.publish_status(
                    f'route_goal_near_enter:{self.current_route_name}:{self.target_name}:'
                    f'dist={dist:.3f}'
                )

            fine_elapsed = now - self.near_since
            if fine_elapsed > self.fine_timeout_sec:
                self.publish_status(
                    f'route_goal_fine_timeout:{self.current_route_name}:{self.target_name}:'
                    f'fine_elapsed={fine_elapsed:.2f}:dist={dist:.3f}'
                )
                self.cancel_current_goal_and_skip('fine_timeout')
                return

        elif dist > near_exit:
            if self.in_near_goal:
                self.publish_status(
                    f'route_goal_near_reset:{self.current_route_name}:{self.target_name}:'
                    f'dist={dist:.3f}'
                )
            self.in_near_goal = False
            self.near_since = None

    # =========================
    # Route control
    # =========================
    def start_route(self, route_name: str):
        if route_name not in self.routes:
            self.publish_status(f'route_not_found:{route_name}')
            self.get_logger().error(f'route not found: {route_name}')
            return

        if self.active:
            self.cancel_current_goal_and_stop_route('new_route_requested')

        self.current_route_name = route_name
        self.current_route = list(self.routes[route_name])
        self.current_index = 0
        self.active = True

        self.publish_status(
            f'route_started:{self.current_route_name}:count={len(self.current_route)}'
        )

        self.send_next_goal()

    def send_next_goal(self):
        if not self.active:
            return

        if self.current_index >= len(self.current_route):
            self.publish_status(f'route_finished:{self.current_route_name}')
            self.reset_goal_state()
            self.active = False
            return

        self.target_name = str(self.current_route[self.current_index])

        if self.target_name not in self.goals:
            self.publish_status(
                f'route_goal_missing:{self.current_route_name}:'
                f'{self.current_index + 1}/{len(self.current_route)}:{self.target_name}'
            )
            self.current_index += 1
            self.send_next_goal()
            return

        self.target_pose_dict = self.goals[self.target_name]

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.publish_status('navigate_to_pose_server_unavailable')
            self.get_logger().error('navigate_to_pose action server unavailable')
            self.active = False
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(self.target_pose_dict.get('x', 0.0))
        goal_msg.pose.pose.position.y = float(self.target_pose_dict.get('y', 0.0))
        goal_msg.pose.pose.position.z = float(self.target_pose_dict.get('z', 0.0))

        goal_msg.pose.pose.orientation.x = float(self.target_pose_dict.get('qx', 0.0))
        goal_msg.pose.pose.orientation.y = float(self.target_pose_dict.get('qy', 0.0))
        goal_msg.pose.pose.orientation.z = float(self.target_pose_dict.get('qz', 0.0))
        goal_msg.pose.pose.orientation.w = float(self.target_pose_dict.get('qw', 1.0))

        self.goal_token += 1
        token = self.goal_token

        self.goal_handle = None
        self.cancel_requested = False
        self.goal_sent_time = time.monotonic()
        self.near_since = None
        self.in_near_goal = False

        self.publish_status(
            f'route_goal_sent:{self.current_route_name}:'
            f'{self.current_index + 1}/{len(self.current_route)}:{self.target_name}'
        )

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            lambda future: self.goal_response_cb(future, token)
        )

    def goal_response_cb(self, future, token: int):
        if token != self.goal_token:
            return

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.publish_status(
                f'route_goal_rejected:{self.current_route_name}:{self.target_name}'
            )
            self.current_index += 1
            self.send_next_goal()
            return

        self.goal_handle = goal_handle

        self.publish_status(
            f'route_goal_accepted:{self.current_route_name}:{self.target_name}'
        )

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future: self.goal_result_cb(future, token)
        )

    def goal_result_cb(self, future, token: int):
        if token != self.goal_token:
            return

        if self.cancel_requested:
            return

        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.publish_status(
                f'route_goal_succeeded:{self.current_route_name}:{self.target_name}'
            )
        else:
            self.publish_status(
                f'route_goal_failed:{self.current_route_name}:{self.target_name}:status={status}'
            )

        self.current_index += 1
        self.reset_goal_state(keep_active=True)
        self.send_next_goal()

    # =========================
    # Cancel logic
    # =========================
    def cancel_current_goal_and_skip(self, reason: str):
        if self.cancel_requested:
            return

        self.cancel_requested = True

        self.publish_status(
            f'route_goal_cancel_requested:{self.current_route_name}:{self.target_name}:{reason}'
        )

        token = self.goal_token

        if self.goal_handle is not None:
            cancel_future = self.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(
                lambda future: self.cancel_skip_done_cb(future, token, reason)
            )
        else:
            self.cancel_skip_done_cb(None, token, reason)

    def cancel_skip_done_cb(self, future, token: int, reason: str):
        if token != self.goal_token:
            return

        self.publish_status(
            f'route_goal_cancelled_and_skip:{self.current_route_name}:{self.target_name}:{reason}'
        )

        self.current_index += 1
        self.reset_goal_state(keep_active=True)
        self.send_next_goal()

    def cancel_current_goal_and_stop_route(self, reason: str):
        if not self.active:
            self.publish_status(f'route_cancel_ignored:not_active:{reason}')
            return

        self.cancel_requested = True
        token = self.goal_token

        self.publish_status(
            f'route_cancel_requested:{self.current_route_name}:{self.target_name}:{reason}'
        )

        if self.goal_handle is not None:
            cancel_future = self.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(
                lambda future: self.cancel_stop_done_cb(future, token, reason)
            )
        else:
            self.cancel_stop_done_cb(None, token, reason)

    def cancel_stop_done_cb(self, future, token: int, reason: str):
        if token != self.goal_token:
            return

        self.publish_status(
            f'route_cancelled:{self.current_route_name}:{self.target_name}:{reason}'
        )

        self.reset_goal_state()
        self.active = False

    # =========================
    # Helpers
    # =========================
    def current_dist_to_target(self):
        if self.latest_amcl_x is None or self.latest_amcl_y is None:
            return None

        if self.target_pose_dict is None:
            return None

        tx = float(self.target_pose_dict.get('x', 0.0))
        ty = float(self.target_pose_dict.get('y', 0.0))

        dx = self.latest_amcl_x - tx
        dy = self.latest_amcl_y - ty

        return math.sqrt(dx * dx + dy * dy)

    def reset_goal_state(self, keep_active=False):
        self.goal_handle = None
        self.cancel_requested = False

        self.goal_sent_time = None
        self.near_since = None
        self.in_near_goal = False

        self.target_name = ''
        self.target_pose_dict = None

        if not keep_active:
            self.current_route = []
            self.current_route_name = ''
            self.current_index = 0

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = DebugRouteRunner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
