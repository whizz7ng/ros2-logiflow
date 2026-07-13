#!/usr/bin/env python3

import math
import time
from typing import Dict, List, Optional

import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool, String


def split_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(',') if x.strip()]


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(float(yaw) * 0.5)
    q.w = math.cos(float(yaw) * 0.5)
    return q


class Nav2RouteRunnerNode(Node):
    """
    Nav2 기반 route runner.

    mission_brain_node와의 호환을 위해 기존 command/status protocol은 유지한다.

    Subscribe:
      /nav2_route_cmd std_msgs/String
        - "goal <goal_name>"
        - "route <route_name>"
        - "stop" / "cancel"

    Publish:
      /debug/nav_status std_msgs/String
        - route_started:<route_name>:...
        - route_goal_started:<route_name>:<idx>/<total>:<goal_name>
        - route_goal_succeeded:<route_name>:<goal_name>
        - route_goal_wait_aruco:<route_name>:<goal_name>:timeout=<sec>
        - aruco_align_succeeded:<route_name>:<goal_name>
        - aruco_timeout:<route_name>:<goal_name>:timeout=<sec>
        - route_finished:<route_name>
        - route_goal_failed:<route_name>:<goal_name>:...
        - route_cancelled:<route_name>:...

    Action:
      /navigate_to_pose nav2_msgs/action/NavigateToPose

    ArUco:
      terminal goal(to_obj, to_qr_a/b/c 등)에서 Nav2 도착 후 /aruco_align_cmd "start"를 보내고
      /aruco_align_done true를 기다린다.
    """

    def __init__(self):
        super().__init__('nav2_route_runner_node')

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter('goals_yaml', '/home/er/myagv_ros2/src/tracer_nav2/config/goals.yaml')
        self.declare_parameter('command_topic', '/nav2_route_cmd')
        self.declare_parameter('status_topic', '/debug/nav_status')
        self.declare_parameter('action_name', '/navigate_to_pose')

        self.declare_parameter('frame_id', '')  # empty면 goals.yaml의 frame_id 사용
        self.declare_parameter('goal_timeout_sec', 180.0)
        self.declare_parameter('server_wait_sec', 10.0)

        self.declare_parameter('enable_aruco_after_goal', True)
        self.declare_parameter('aruco_goal_names', 'to_obj,to_qr_a,to_qr_b,to_qr_c')
        self.declare_parameter('aruco_cmd_topic', '/aruco_align_cmd')
        self.declare_parameter('aruco_done_topic', '/aruco_align_done')
        self.declare_parameter('aruco_timeout_sec', 20.0)
        self.declare_parameter('continue_on_aruco_timeout', True)

        self.declare_parameter('print_debug', True)

        self.goals_yaml = str(self.p('goals_yaml'))
        self.command_topic = str(self.p('command_topic'))
        self.status_topic = str(self.p('status_topic'))
        self.action_name = str(self.p('action_name'))

        self.yaml_frame_id = 'map'
        self.goals: Dict[str, dict] = {}
        self.routes: Dict[str, List[str]] = {}
        self.load_goals_yaml()

        # ------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.aruco_cmd_pub = self.create_publisher(String, str(self.p('aruco_cmd_topic')), 10)

        self.cmd_sub = self.create_subscription(
            String,
            self.command_topic,
            self.command_cb,
            10,
        )

        self.aruco_done_sub = self.create_subscription(
            Bool,
            str(self.p('aruco_done_topic')),
            self.aruco_done_cb,
            10,
        )

        self.nav_client = ActionClient(self, NavigateToPose, self.action_name)

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------
        self.active = False
        self.state = 'IDLE'  # IDLE, SENDING, NAVIGATING, WAIT_ARUCO, CANCELLING
        self.current_route_name = ''
        self.current_goals: List[str] = []
        self.current_index = -1
        self.current_goal_name = ''

        self.goal_handle = None
        self.goal_sent_time = 0.0
        self.aruco_wait_start = 0.0

        # Invalidates late callbacks from a cancelled/preempted Nav2 goal.
        self.command_generation = 0

        self.timer = self.create_timer(0.10, self.timer_cb)

        self.publish_status(
            f'nav2_route_runner_ready | cmd={self.command_topic} status={self.status_topic} '
            f'action={self.action_name} goals={len(self.goals)} routes={len(self.routes)}'
        )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def p(self, name):
        return self.get_parameter(name).value

    def publish_status(self, text: str):
        msg = String()
        msg.data = str(text)
        self.status_pub.publish(msg)
        self.get_logger().info(str(text))

    def load_goals_yaml(self):
        with open(self.goals_yaml, 'r') as f:
            data = yaml.safe_load(f) or {}

        self.yaml_frame_id = str(data.get('frame_id', 'map'))
        self.goals = data.get('goals', {}) or {}
        self.routes = data.get('routes', {}) or {}

        if not isinstance(self.goals, dict):
            raise RuntimeError('goals_yaml: "goals" must be a dictionary')
        if not isinstance(self.routes, dict):
            raise RuntimeError('goals_yaml: "routes" must be a dictionary')

    def effective_frame_id(self):
        frame = str(self.p('frame_id')).strip()
        return frame if frame else self.yaml_frame_id

    def make_pose(self, goal_name: str) -> PoseStamped:
        if goal_name not in self.goals:
            raise KeyError(f'unknown goal: {goal_name}')

        g = self.goals[goal_name]
        pose = PoseStamped()
        pose.header.frame_id = self.effective_frame_id()
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(g.get('x', 0.0))
        pose.pose.position.y = float(g.get('y', 0.0))
        pose.pose.position.z = float(g.get('z', 0.0))

        if 'qz' in g and 'qw' in g:
            pose.pose.orientation.x = float(g.get('qx', 0.0))
            pose.pose.orientation.y = float(g.get('qy', 0.0))
            pose.pose.orientation.z = float(g.get('qz', 0.0))
            pose.pose.orientation.w = float(g.get('qw', 1.0))
        else:
            pose.pose.orientation = yaw_to_quaternion(float(g.get('yaw', 0.0)))

        return pose

    def aruco_goal_set(self):
        return set(split_csv(str(self.p('aruco_goal_names'))))

    def should_wait_aruco(self, goal_name: str) -> bool:
        return bool(self.p('enable_aruco_after_goal')) and goal_name in self.aruco_goal_set()

    def send_aruco_cmd(self, command: str):
        msg = String()
        msg.data = str(command)
        self.aruco_cmd_pub.publish(msg)
        self.publish_status(f'aruco_cmd:{command}:route={self.current_route_name}:goal={self.current_goal_name}')

    # ------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------
    def command_cb(self, msg: String):
        text = str(msg.data).strip()
        if not text:
            return

        parts = text.split()
        cmd = parts[0].lower()

        if cmd in ['stop', 'cancel']:
            self.cancel_current(reason=f'manual_{cmd}')
            return

        if cmd == 'goal' and len(parts) >= 2:
            goal_name = parts[1].strip()
            self.start_goal_command(goal_name)
            return

        if cmd == 'route' and len(parts) >= 2:
            route_name = parts[1].strip()
            self.start_route_command(route_name)
            return

        self.publish_status(f'route_cmd_rejected:unknown_command:{text}')

    def start_goal_command(self, goal_name: str):
        if goal_name not in self.goals:
            self.publish_status(f'route_goal_failed:single_{goal_name}:{goal_name}:unknown_goal')
            return

        self.start_route(route_name=f'single_{goal_name}', goals=[goal_name])

    def start_route_command(self, route_name: str):
        if route_name not in self.routes:
            self.publish_status(f'route_goal_failed:{route_name}:unknown:unknown_route')
            return

        goals = [str(x).strip() for x in self.routes[route_name] if str(x).strip()]
        missing = [g for g in goals if g not in self.goals]
        if missing:
            self.publish_status(f'route_goal_failed:{route_name}:{missing[0]}:unknown_goal')
            return

        self.start_route(route_name=route_name, goals=goals)

    def start_route(self, route_name: str, goals: List[str]):
        if self.active:
            self.cancel_current(reason='new_command_preempt')
            # cancel은 비동기지만, myAGV 프로젝트에서는 새 명령 우선으로 바로 덮어쓴다.

        if not goals:
            self.publish_status(f'route_goal_failed:{route_name}:empty:empty_route')
            return

        self.command_generation += 1
        self.active = True
        self.state = 'IDLE'
        self.current_route_name = route_name
        self.current_goals = list(goals)
        self.current_index = -1
        self.current_goal_name = ''
        self.goal_handle = None
        self.goal_sent_time = 0.0
        self.aruco_wait_start = 0.0

        self.publish_status(f'route_started:{route_name}:goals={",".join(goals)}')
        self.advance_to_next_goal()

    def advance_to_next_goal(self):
        self.current_index += 1

        if self.current_index >= len(self.current_goals):
            route = self.current_route_name
            self.reset_state()
            self.publish_status(f'route_finished:{route}')
            return

        self.current_goal_name = self.current_goals[self.current_index]
        self.send_nav2_goal(self.current_goal_name)

    # ------------------------------------------------------------
    # Nav2 action
    # ------------------------------------------------------------
    def send_nav2_goal(self, goal_name: str):
        route = self.current_route_name
        total = len(self.current_goals)
        idx = self.current_index + 1

        try:
            pose = self.make_pose(goal_name)
        except Exception as e:
            self.publish_status(f'route_goal_failed:{route}:{goal_name}:pose_error:{e}')
            self.reset_state()
            return

        if not self.nav_client.wait_for_server(timeout_sec=float(self.p('server_wait_sec'))):
            self.publish_status(f'route_goal_failed:{route}:{goal_name}:nav2_action_server_unavailable')
            self.reset_state()
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.state = 'SENDING'
        self.goal_sent_time = time.monotonic()

        self.publish_status(
            f'route_goal_started:{route}:{idx}/{total}:{goal_name}:'
            f'x={pose.pose.position.x:.3f}:y={pose.pose.position.y:.3f}'
        )

        generation = self.command_generation
        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            lambda future, gen=generation: self.goal_response_cb(future, gen)
        )

    def goal_response_cb(self, future, generation):
        try:
            goal_handle = future.result()
        except Exception as e:
            if generation == self.command_generation and self.active:
                self.publish_status(
                    f'route_goal_failed:{self.current_route_name}:{self.current_goal_name}:'
                    f'send_exception:{e}'
                )
                self.reset_state()
            return

        # A stop/preempt may arrive while the action request is still SENDING.
        # If Nav2 accepts that stale goal later, cancel it immediately.
        if generation != self.command_generation or not self.active:
            if goal_handle is not None and goal_handle.accepted:
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
            return

        if not goal_handle.accepted:
            self.publish_status(f'route_goal_failed:{self.current_route_name}:{self.current_goal_name}:rejected_by_nav2')
            self.reset_state()
            return

        self.goal_handle = goal_handle
        self.state = 'NAVIGATING'
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future, gen=generation: self.goal_result_cb(future, gen)
        )

    def goal_result_cb(self, future, generation):
        if generation != self.command_generation or not self.active:
            return

        try:
            result = future.result()
            status = int(result.status)
        except Exception as e:
            self.publish_status(f'route_goal_failed:{self.current_route_name}:{self.current_goal_name}:result_exception:{e}')
            self.reset_state()
            return

        route = self.current_route_name
        goal = self.current_goal_name

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.publish_status(f'route_goal_succeeded:{route}:{goal}')

            if self.should_wait_aruco(goal):
                self.start_aruco_wait()
            else:
                self.advance_to_next_goal()
            return

        # cancel 중이면 cancel status를 따로 낸다.
        if self.state == 'CANCELLING':
            self.publish_status(f'route_cancelled:{route}:goal={goal}:status={status}')
            self.reset_state()
            return

        self.publish_status(f'route_goal_failed:{route}:{goal}:nav2_status={status}')
        self.reset_state()

    # ------------------------------------------------------------
    # ArUco wait
    # ------------------------------------------------------------
    def start_aruco_wait(self):
        self.state = 'WAIT_ARUCO'
        self.aruco_wait_start = time.monotonic()
        timeout = float(self.p('aruco_timeout_sec'))

        self.publish_status(
            f'route_goal_wait_aruco:{self.current_route_name}:{self.current_goal_name}:'
            f'timeout={timeout:.1f}'
        )
        self.send_aruco_cmd('start')

    def aruco_done_cb(self, msg: Bool):
        if not self.active or self.state != 'WAIT_ARUCO':
            return

        if not bool(msg.data):
            return

        route = self.current_route_name
        goal = self.current_goal_name
        self.publish_status(f'aruco_align_succeeded:{route}:{goal}')
        self.send_aruco_cmd('stop')
        self.advance_to_next_goal()

    def timer_cb(self):
        if not self.active:
            return

        now = time.monotonic()

        if self.state == 'NAVIGATING':
            timeout = float(self.p('goal_timeout_sec'))
            if timeout > 0.0 and self.goal_sent_time > 0.0:
                elapsed = now - self.goal_sent_time
                if elapsed >= timeout:
                    self.publish_status(
                        f'route_goal_failed:{self.current_route_name}:{self.current_goal_name}:'
                        f'goal_timeout:{elapsed:.1f}s'
                    )
                    self.cancel_current(reason='goal_timeout')
            return

        if self.state == 'WAIT_ARUCO':
            timeout = float(self.p('aruco_timeout_sec'))
            if timeout > 0.0 and self.aruco_wait_start > 0.0:
                elapsed = now - self.aruco_wait_start
                if elapsed >= timeout:
                    route = self.current_route_name
                    goal = self.current_goal_name
                    self.publish_status(f'aruco_timeout:{route}:{goal}:timeout={timeout:.1f}:elapsed={elapsed:.1f}')
                    self.send_aruco_cmd('stop')

                    if bool(self.p('continue_on_aruco_timeout')):
                        self.advance_to_next_goal()
                    else:
                        self.publish_status(f'route_goal_failed:{route}:{goal}:aruco_timeout')
                        self.reset_state()

    # ------------------------------------------------------------
    # Cancel / reset
    # ------------------------------------------------------------
    def cancel_current(self, reason='manual'):
        # Invalidate callbacks belonging to the old goal before cancelling it.
        self.command_generation += 1

        if not self.active:
            self.publish_status(f'route_cancelled:none:{reason}')
            return

        route = self.current_route_name
        goal = self.current_goal_name

        if self.state == 'WAIT_ARUCO':
            self.send_aruco_cmd('stop')

        if self.goal_handle is not None:
            try:
                self.state = 'CANCELLING'
                self.goal_handle.cancel_goal_async()
            except Exception:
                pass

        self.publish_status(f'route_cancelled:{route}:goal={goal}:reason={reason}')
        self.reset_state()

    def reset_state(self):
        self.active = False
        self.state = 'IDLE'
        self.current_route_name = ''
        self.current_goals = []
        self.current_index = -1
        self.current_goal_name = ''
        self.goal_handle = None
        self.goal_sent_time = 0.0
        self.aruco_wait_start = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = Nav2RouteRunnerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cancel_current(reason='shutdown')
            time.sleep(0.05)
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
