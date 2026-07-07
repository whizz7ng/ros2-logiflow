#!/usr/bin/env python3

import os
import math
import yaml
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class GoalManager(Node):
    def __init__(self):
        super().__init__('goal_manager')

        # =========================
        # Parameters
        # =========================
        self.declare_parameter(
            'goals_yaml',
            '/home/er/myagv_ros2/src/tracer/config/goals.yaml'
        )
        self.declare_parameter('default_frame_id', 'map')
        self.declare_parameter('autostart_goal', '')
        self.declare_parameter('autostart_route', '')
        self.declare_parameter('wait_action_timeout_sec', 10.0)

        self.goals_yaml = self.get_parameter('goals_yaml').value
        self.default_frame_id = self.get_parameter('default_frame_id').value
        self.autostart_goal = self.get_parameter('autostart_goal').value
        self.autostart_route = self.get_parameter('autostart_route').value
        self.wait_action_timeout_sec = float(
            self.get_parameter('wait_action_timeout_sec').value
        )

        # =========================
        # Internal state
        # =========================
        self.data = {}
        self.frame_id = self.default_frame_id

        self.active = False
        self.current_route_name = ''
        self.current_route_list: List[str] = []
        self.current_route_index = 0
        self.current_goal_name = ''

        # =========================
        # Nav2 Action Client
        # =========================
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # =========================
        # Topics
        # =========================
        self.goal_sub = self.create_subscription(
            String,
            '/goal_cmd',
            self.goal_cmd_cb,
            10
        )

        self.route_sub = self.create_subscription(
            String,
            '/route_cmd',
            self.route_cmd_cb,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/nav_status',
            10
        )

        # =========================
        # Load YAML
        # =========================
        self.load_goals_yaml()

        # autostart는 Nav2 action server 기다린 뒤 실행
        self.create_timer(1.0, self.startup_once)
        self.startup_done = False

        self.get_logger().info('goal_manager started')
        self.get_logger().info(f'goals_yaml = {self.goals_yaml}')
        self.get_logger().info('Subscribe: /goal_cmd, /route_cmd')
        self.get_logger().info('Publish: /nav_status')
        self.get_logger().info('Action: /navigate_to_pose')

    # =========================================================
    # YAML
    # =========================================================
    def load_goals_yaml(self):
        if not os.path.exists(self.goals_yaml):
            self.get_logger().error(f'goals.yaml not found: {self.goals_yaml}')
            raise FileNotFoundError(self.goals_yaml)

        with open(self.goals_yaml, 'r') as f:
            self.data = yaml.safe_load(f)

        if self.data is None:
            self.data = {}

        self.frame_id = self.data.get('frame_id', self.default_frame_id)

        goals = self.data.get('goals', {})
        waypoints = self.data.get('waypoints', {})
        routes = self.data.get('routes', {})

        self.get_logger().info(f'Loaded frame_id: {self.frame_id}')
        self.get_logger().info(f'Loaded goals: {list(goals.keys())}')
        self.get_logger().info(f'Loaded waypoints: {list(waypoints.keys())}')
        self.get_logger().info(f'Loaded routes: {list(routes.keys())}')

    def get_pose_dict(self, name: str) -> Optional[Dict]:
        goals = self.data.get('goals', {})
        waypoints = self.data.get('waypoints', {})

        if name in goals:
            return goals[name]

        if name in waypoints:
            return waypoints[name]

        return None

    def get_route_list(self, route_name: str) -> Optional[List[str]]:
        routes = self.data.get('routes', {})

        if route_name not in routes:
            return None

        return routes[route_name]

    # =========================================================
    # Pose creation
    # =========================================================
    def make_pose_stamped(self, name: str) -> Optional[PoseStamped]:
        pose_info = self.get_pose_dict(name)

        if pose_info is None:
            self.get_logger().error(f'Unknown goal/waypoint name: {name}')
            return None

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = pose_info.get('frame_id', self.frame_id)

        pose.pose.position.x = float(pose_info['x'])
        pose.pose.position.y = float(pose_info['y'])
        pose.pose.position.z = float(pose_info.get('z', 0.0))

        # qz/qw가 있으면 그대로 사용
        if 'qz' in pose_info and 'qw' in pose_info:
            pose.pose.orientation.x = float(pose_info.get('qx', 0.0))
            pose.pose.orientation.y = float(pose_info.get('qy', 0.0))
            pose.pose.orientation.z = float(pose_info['qz'])
            pose.pose.orientation.w = float(pose_info['qw'])

        # yaw가 있으면 yaw -> quaternion 변환
        elif 'yaw' in pose_info:
            yaw = float(pose_info['yaw'])
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)

        else:
            self.get_logger().warn(
                f'{name} has no qz/qw or yaw. Use identity orientation.'
            )
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0

        return pose

    # =========================================================
    # Topic callbacks
    # =========================================================
    def goal_cmd_cb(self, msg: String):
        goal_name = msg.data.strip()

        if not goal_name:
            self.get_logger().warn('Empty /goal_cmd received')
            return

        self.get_logger().info(f'/goal_cmd received: {goal_name}')
        self.send_single_goal(goal_name)

    def route_cmd_cb(self, msg: String):
        route_name = msg.data.strip()

        if not route_name:
            self.get_logger().warn('Empty /route_cmd received')
            return

        self.get_logger().info(f'/route_cmd received: {route_name}')
        self.start_route(route_name)

    # =========================================================
    # Startup autostart
    # =========================================================
    def startup_once(self):
        if self.startup_done:
            return

        self.startup_done = True

        if self.autostart_route:
            self.get_logger().info(f'autostart_route: {self.autostart_route}')
            self.start_route(self.autostart_route)
            return

        if self.autostart_goal:
            self.get_logger().info(f'autostart_goal: {self.autostart_goal}')
            self.send_single_goal(self.autostart_goal)
            return

    # =========================================================
    # Goal sending
    # =========================================================
    def wait_nav_server(self) -> bool:
        self.get_logger().info('Waiting for Nav2 NavigateToPose action server...')

        ok = self.nav_client.wait_for_server(
            timeout_sec=self.wait_action_timeout_sec
        )

        if not ok:
            self.get_logger().error(
                'NavigateToPose action server not available. '
                'Check Nav2 is running.'
            )
            return False

        return True

    def send_single_goal(self, goal_name: str):
        if self.active:
            self.get_logger().warn(
                f'Already active. Ignore goal request: {goal_name}'
            )
            return

        pose = self.make_pose_stamped(goal_name)
        if pose is None:
            self.publish_status(f'goal_failed_unknown:{goal_name}')
            return

        if not self.wait_nav_server():
            self.publish_status(f'goal_failed_no_action_server:{goal_name}')
            return

        self.active = True
        self.current_route_name = ''
        self.current_route_list = []
        self.current_route_index = 0
        self.current_goal_name = goal_name

        self.get_logger().info(
            f'Sending single goal: {goal_name} '
            f'x={pose.pose.position.x:.3f}, y={pose.pose.position.y:.3f}'
        )
        self.publish_status(f'goal_sent:{goal_name}')

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.single_goal_response_cb)

    def single_goal_response_cb(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(f'Goal rejected: {self.current_goal_name}')
            self.publish_status(f'goal_rejected:{self.current_goal_name}')
            self.active = False
            return

        self.get_logger().info(f'Goal accepted: {self.current_goal_name}')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.single_goal_result_cb)

    def single_goal_result_cb(self, future):
        result = future.result()
        status = result.status

        goal_name = self.current_goal_name

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Goal succeeded: {goal_name}')
            self.publish_status(f'goal_succeeded:{goal_name}')
        else:
            self.get_logger().warn(
                f'Goal failed: {goal_name}, status={status}'
            )
            self.publish_status(f'goal_failed:{goal_name}:status={status}')

        self.active = False
        self.current_goal_name = ''

    # =========================================================
    # Route sending
    # =========================================================
    def start_route(self, route_name: str):
        if self.active:
            self.get_logger().warn(
                f'Already active. Ignore route request: {route_name}'
            )
            return

        route_list = self.get_route_list(route_name)

        if route_list is None:
            self.get_logger().error(f'Unknown route name: {route_name}')
            self.publish_status(f'route_failed_unknown:{route_name}')
            return

        if len(route_list) == 0:
            self.get_logger().error(f'Empty route: {route_name}')
            self.publish_status(f'route_failed_empty:{route_name}')
            return

        if not self.wait_nav_server():
            self.publish_status(f'route_failed_no_action_server:{route_name}')
            return

        self.active = True
        self.current_route_name = route_name
        self.current_route_list = route_list
        self.current_route_index = 0

        self.get_logger().info(
            f'Start route: {route_name} -> {self.current_route_list}'
        )
        self.publish_status(f'route_started:{route_name}')

        self.send_next_route_goal()

    def send_next_route_goal(self):
        if self.current_route_index >= len(self.current_route_list):
            self.get_logger().info(f'Route completed: {self.current_route_name}')
            self.publish_status(f'route_succeeded:{self.current_route_name}')

            self.active = False
            self.current_route_name = ''
            self.current_route_list = []
            self.current_route_index = 0
            self.current_goal_name = ''
            return

        goal_name = self.current_route_list[self.current_route_index]
        pose = self.make_pose_stamped(goal_name)

        if pose is None:
            self.get_logger().error(
                f'Route failed. Unknown goal in route: {goal_name}'
            )
            self.publish_status(
                f'route_failed_unknown_goal:{self.current_route_name}:{goal_name}'
            )
            self.active = False
            return

        self.current_goal_name = goal_name

        self.get_logger().info(
            f'Route {self.current_route_name} '
            f'[{self.current_route_index + 1}/{len(self.current_route_list)}] '
            f'→ {goal_name} '
            f'x={pose.pose.position.x:.3f}, y={pose.pose.position.y:.3f}'
        )
        self.publish_status(
            f'route_goal_sent:{self.current_route_name}:{goal_name}'
        )

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.route_goal_response_cb)

    def route_goal_response_cb(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                f'Route goal rejected: {self.current_goal_name}'
            )
            self.publish_status(
                f'route_goal_rejected:{self.current_route_name}:{self.current_goal_name}'
            )
            self.active = False
            return

        self.get_logger().info(f'Route goal accepted: {self.current_goal_name}')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.route_goal_result_cb)

    def route_goal_result_cb(self, future):
        result = future.result()
        status = result.status

        goal_name = self.current_goal_name

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Route goal succeeded: {goal_name}')
            self.publish_status(
                f'route_goal_succeeded:{self.current_route_name}:{goal_name}'
            )

            self.current_route_index += 1
            self.send_next_route_goal()

        else:
            self.get_logger().warn(
                f'Route goal failed: {goal_name}, status={status}'
            )
            self.publish_status(
                f'route_goal_failed:{self.current_route_name}:{goal_name}:status={status}'
            )
            self.active = False

    # =========================================================
    # Status
    # =========================================================
    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = GoalManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
