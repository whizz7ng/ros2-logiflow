#!/usr/bin/env python3

import time
import json
import re
from typing import List, Optional, Dict

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Empty, Bool
from geometry_msgs.msg import PoseStamped


def split_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(',') if x.strip()]


def parse_key_value_map(text: str) -> Dict[str, str]:
    """
    "A:A,B:B,C:C" -> {"A": "A", "B": "B", "C": "C"}
    "A:88,B:87,C:73" 같은 형태도 string value로 저장.
    """
    out = {}
    for item in str(text).split(','):
        item = item.strip()
        if not item or ':' not in item:
            continue
        k, v = item.split(':', 1)
        k = k.strip().upper()
        v = v.strip().upper()
        if k and v:
            out[k] = v
    return out


def extract_target_token(raw: str, valid_targets) -> Optional[str]:
    """
    Orin brain_node / WMS 쪽에서 어떤 형식으로 target을 보내도 A/B/C만 뽑는다.

    지원 예:
      "B"
      "target:B"
      "place_target=B"
      '{"target":"B"}'
      '{"place_target":"B"}'
      '{"destination":"QR_B"}'
      "go B"
      "QR_B"
    """
    text = str(raw).strip()
    valid = {str(x).strip().upper() for x in valid_targets}

    if not text:
        return None

    up = text.upper().strip()
    if up in valid:
        return up

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ['target', 'place_target', 'destination', 'qr', 'zone', 'data']:
                if key in obj:
                    cand = extract_target_token(str(obj[key]), valid)
                    if cand:
                        return cand
        elif isinstance(obj, str):
            cand = extract_target_token(obj, valid)
            if cand:
                return cand
    except Exception:
        pass

    for target in sorted(valid):
        patterns = [
            rf'(^|[^A-Z0-9]){re.escape(target)}([^A-Z0-9]|$)',
            rf'QR[_\- ]?{re.escape(target)}',
            rf'TARGET[_\- :=]+{re.escape(target)}',
            rf'PLACE[_\- ]?TARGET[_\- :=]+{re.escape(target)}',
            rf'DESTINATION[_\- :=]+{re.escape(target)}',
        ]
        for pat in patterns:
            if re.search(pat, up):
                return target

    return None


class MissionBrainNode(Node):
    """
    myAGV primitive mission brain.

    내부 역할:
      1. /place_target 또는 /order_request로 A/B/C 목표를 받는다.
      2. pickup sequence 또는 route를 primitive_route_runner로 실행한다.
      3. to_obj에서는 RACK ArUco target으로 자동 align한다.
      4. object/RACK 도착 시 외부 Orin brain_node로 /nav_status "arrived_objects" 발행.
      5. /arm_status "picked" 수신 후 OBJ -> QR route 실행.
      6. QR 도착 시 외부 Orin brain_node로 /nav_status "arrived" 발행.
      7. /go_parking 수신 후 parking route 실행.
      8. parking route 완료 시 외부 Orin brain_node로 /nav_status "parked" 발행.

    중요 topic:
      - 내부 primitive 상태 구독: /debug/nav_status
      - 외부 Orin brain 알림: /nav_status
      - AGV micro align 허용: /agv_align_enable
    """

    def __init__(self):
        super().__init__('mission_brain_node')

        # ============================================================
        # Topic params
        # ============================================================
        self.declare_parameter('primitive_cmd_topic', '/primitive_route_cmd')

        # 내부 primitive_route_runner 상태 구독용.
        # 절대 Orin brain용 /nav_status와 섞지 않는다.
        self.declare_parameter('nav_status_topic', '/debug/nav_status')

        # Nano mission brain 자체 debug/status
        self.declare_parameter('brain_status_topic', '/brain_status')

        # Orin brain_node로 보내는 AGV/Nav 상태
        # "arrived_objects", "arrived", "parked"
        self.declare_parameter('external_nav_status_topic', '/nav_status')

        # /agv_align_bridge_node enable
        self.declare_parameter('agv_align_enable_topic', '/agv_align_enable')

        self.declare_parameter('place_target_topic', '/place_target')
        self.declare_parameter('order_request_topic', '/order_request')
        self.declare_parameter('arm_status_topic', '/arm_status')
        self.declare_parameter('go_parking_topic', '/go_parking')

        # Legacy compatibility.
        # 기본은 발행하지 않는다. 필요하면 launch에서 publish_legacy_stop_topics:=true.
        self.declare_parameter('stop_obj_topic', '/stop_obj')
        self.declare_parameter('stop_qr_topic', '/stop_qr')
        self.declare_parameter('publish_legacy_stop_topics', False)

        self.declare_parameter('aruco_done_topic', '/aruco_align_done')
        self.declare_parameter('aruco_status_topic', '/aruco_align_status')
        self.declare_parameter('aruco_cmd_topic', '/aruco_align_cmd')
        self.declare_parameter('aruco_target_name_topic', '/marker_align_target')

        # future external align input placeholders
        self.declare_parameter('align_cmd_topic', '/align_cmd')
        self.declare_parameter('align_goal_topic', '/align_goal')

        # ============================================================
        # Mission route params
        # ============================================================
        self.declare_parameter('pickup_goal_sequence', 'way12,to_obj')
        self.declare_parameter('pickup_route_name', '')
        self.declare_parameter('to_obj_goal_name', 'to_obj')

        self.declare_parameter('obj_to_qr_route_template', 'obj_to_qr_{target_lower}')
        self.declare_parameter('qr_to_parking_route_template', 'qr_{from_qr_lower}_to_parking')
        self.declare_parameter('qr_to_obj_route_template', 'qr_{from_qr_lower}_to_obj')

        self.declare_parameter('valid_targets', 'A,B,C')
        self.declare_parameter('default_qr_target', 'B')

        # QR에 서 있는 상태에서 새 place_target을 받으면 QR -> OBJ route로 복귀한다.
        self.declare_parameter('return_to_obj_on_new_target_from_qr', True)

        # ============================================================
        # ArUco target switching params
        # ============================================================
        self.declare_parameter('auto_switch_aruco_target', True)
        self.declare_parameter('rack_aruco_target_name', 'RACK')
        self.declare_parameter('qr_aruco_target_map', 'A:A,B:B,C:C')

        self.declare_parameter('set_rack_target_on_pickup_start', True)
        self.declare_parameter('set_qr_target_before_qr_route', True)
        self.declare_parameter('set_target_again_on_aruco_wait_event', True)

        self.declare_parameter('publish_stop_obj_on_aruco_timeout', True)
        self.declare_parameter('publish_stop_obj_on_manual_aruco_stop', True)
        self.declare_parameter('publish_stop_qr_on_aruco_timeout', True)
        self.declare_parameter('publish_stop_qr_on_manual_aruco_stop', True)

        # ArUco가 오래 걸리면 일단 작업 지점에 왔다고 보고 arrived_objects/arrived를 발행한다.
        self.declare_parameter('force_stop_on_long_aruco_align', True)
        self.declare_parameter('aruco_force_stop_sec', 5.0)
        self.declare_parameter('stop_aruco_on_force_stop', True)
        self.declare_parameter('stop_primitive_on_force_stop', True)

        # ============================================================
        # Timing / behavior params
        # ============================================================
        self.declare_parameter('stop_obj_delay_sec', 1.0)
        self.declare_parameter('stop_qr_delay_sec', 1.0)
        self.declare_parameter('command_timeout_sec', 240.0)

        self.declare_parameter('allow_new_target_when_busy', False)

        # ============================================================
        # State
        # ============================================================
        self.state = 'IDLE'
        self.current_target: Optional[str] = None
        self.current_qr_target: Optional[str] = None

        self.active_command_kind = ''
        self.active_command_name = ''
        self.active_command_sent_time = 0.0

        self.pickup_goals: List[str] = []
        self.pickup_index = 0

        # 이름은 legacy지만, 현재는 arrived_objects / arrived publish 타이밍 관리용으로 사용.
        self.stop_obj_published = False
        self.stop_qr_published = False
        self.stop_obj_due_time = 0.0
        self.stop_qr_due_time = 0.0

        self.last_align_cmd = ''
        self.last_align_goal = None
        self.last_aruco_target = ''

        self.aruco_wait_start_time = 0.0
        self.aruco_wait_kind = ''  # 'obj' or 'qr'
        self.aruco_wait_goal = ''
        self.aruco_force_stop_fired = False

        # ============================================================
        # Publishers
        # ============================================================
        self.primitive_cmd_pub = self.create_publisher(
            String,
            self.get_parameter('primitive_cmd_topic').value,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            self.get_parameter('brain_status_topic').value,
            10
        )

        self.external_nav_status_pub = self.create_publisher(
            String,
            self.get_parameter('external_nav_status_topic').value,
            10
        )

        self.stop_obj_pub = self.create_publisher(
            Empty,
            self.get_parameter('stop_obj_topic').value,
            10
        )

        self.stop_qr_pub = self.create_publisher(
            Empty,
            self.get_parameter('stop_qr_topic').value,
            10
        )

        self.aruco_target_pub = self.create_publisher(
            String,
            self.get_parameter('aruco_target_name_topic').value,
            10
        )

        self.aruco_cmd_pub = self.create_publisher(
            String,
            self.get_parameter('aruco_cmd_topic').value,
            10
        )

        self.agv_align_enable_pub = self.create_publisher(
            Bool,
            self.get_parameter('agv_align_enable_topic').value,
            10
        )

        # ============================================================
        # Subscribers
        # ============================================================
        self.create_subscription(String, self.get_parameter('place_target_topic').value, self.place_target_cb, 10)
        self.create_subscription(String, self.get_parameter('order_request_topic').value, self.order_request_cb, 10)
        self.create_subscription(String, self.get_parameter('arm_status_topic').value, self.arm_status_cb, 10)
        self.create_subscription(Empty, self.get_parameter('go_parking_topic').value, self.go_parking_cb, 10)
        self.create_subscription(String, self.get_parameter('nav_status_topic').value, self.nav_status_cb, 10)
        self.create_subscription(Bool, self.get_parameter('aruco_done_topic').value, self.aruco_done_cb, 10)
        self.create_subscription(String, self.get_parameter('aruco_status_topic').value, self.aruco_status_cb, 10)
        self.create_subscription(String, self.get_parameter('align_cmd_topic').value, self.align_cmd_cb, 10)
        self.create_subscription(PoseStamped, self.get_parameter('align_goal_topic').value, self.align_goal_cb, 10)

        self.timer = self.create_timer(0.10, self.timer_cb)

        self.publish_status(
            'mission_brain_ready | '
            'inputs=/place_target,/order_request,/arm_status,/go_parking | '
            'internal_nav_status=/debug/nav_status | '
            'external_nav_status=/nav_status | '
            'outputs=/primitive_route_cmd,/marker_align_target,/agv_align_enable'
        )

        self.publish_agv_align_enable(False, reason='startup')

    # ============================================================
    # General helpers
    # ============================================================
    def p(self, name):
        return self.get_parameter(name).value

    def publish_status(self, text):
        msg = String()
        msg.data = str(text)
        self.status_pub.publish(msg)
        self.get_logger().info(str(text))

    def publish_external_nav_status(self, status, reason=''):
        msg = String()
        msg.data = str(status)
        self.external_nav_status_pub.publish(msg)

        if reason:
            self.get_logger().info(
                f'external_nav_status_published:{msg.data}:reason={reason}:state={self.state}'
            )
        else:
            self.get_logger().info(
                f'external_nav_status_published:{msg.data}:state={self.state}'
            )

    def publish_agv_align_enable(self, enabled, reason=''):
        msg = Bool()
        msg.data = bool(enabled)
        self.agv_align_enable_pub.publish(msg)
        self.get_logger().info(
            f'agv_align_enable:{msg.data}:reason={reason}:state={self.state}'
        )

    def agv_align_allowed_state(self, state):
        # WAIT_PICKED: object/RACK station에서 pick 시도 중
        # WAIT_NEXT: QR station에서 place 시도 또는 다음 명령 대기 중
        return state in ['WAIT_PICKED', 'WAIT_NEXT']

    def set_state(self, new_state):
        if new_state != self.state:
            old = self.state
            self.state = new_state
            self.publish_status(f'brain_state:{old}->{new_state}')
            self.publish_agv_align_enable(
                self.agv_align_allowed_state(new_state),
                reason=f'state:{old}->{new_state}'
            )

    def valid_target_set(self):
        return {x.upper() for x in split_csv(self.p('valid_targets'))}

    def normalize_target(self, raw):
        return str(raw).strip().upper()

    def format_route_template(self, template, target=None, from_qr=None):
        target_upper = (target or '').upper()
        target_lower = (target or '').lower()
        from_qr_upper = (from_qr or target or '').upper()
        from_qr_lower = (from_qr or target or '').lower()
        return str(template).format(
            target_upper=target_upper,
            target_lower=target_lower,
            from_qr_upper=from_qr_upper,
            from_qr_lower=from_qr_lower,
        )

    def route_name(self, template, target):
        return self.format_route_template(template, target=target)

    def route_from_qr_name(self, template, from_qr, target=None):
        return self.format_route_template(template, target=target, from_qr=from_qr)

    def send_primitive(self, command, kind='', name=''):
        msg = String()
        msg.data = str(command)
        self.primitive_cmd_pub.publish(msg)

        self.active_command_kind = kind
        self.active_command_name = name
        self.active_command_sent_time = time.monotonic()

        self.publish_status(f'primitive_cmd:{command}')

    def send_goal(self, goal_name):
        self.send_primitive(f'goal {goal_name}', kind='goal', name=goal_name)

    def send_route(self, route_name):
        self.send_primitive(f'route {route_name}', kind='route', name=route_name)

    def send_aruco_cmd(self, command):
        msg = String()
        msg.data = str(command)
        self.aruco_cmd_pub.publish(msg)
        self.publish_status(f'aruco_cmd:{command}')

    def single_route_name(self, goal_name):
        return f'single_{goal_name}'

    def active_route_finished_by_status(self, text):
        if not self.active_command_name:
            return False

        if self.active_command_kind == 'route':
            return text.startswith(f'route_finished:{self.active_command_name}')

        if self.active_command_kind == 'goal':
            route_name = self.single_route_name(self.active_command_name)
            return text.startswith(f'route_finished:{route_name}')

        return False

    def active_route_failed_by_status(self, text):
        if not self.active_command_name:
            return False

        if self.active_command_kind == 'route':
            name = self.active_command_name
        elif self.active_command_kind == 'goal':
            name = self.single_route_name(self.active_command_name)
        else:
            return False

        return (
            text.startswith(f'route_cancelled:{name}:') or
            text.startswith(f'route_goal_failed:{name}:')
        )

    # ============================================================
    # ArUco target helpers
    # ============================================================
    def auto_switch_enabled(self):
        return bool(self.p('auto_switch_aruco_target'))

    def qr_aruco_target_name(self, target):
        target = str(target).strip().upper()
        mp = parse_key_value_map(self.p('qr_aruco_target_map'))
        return mp.get(target, target)

    def publish_aruco_target(self, target_name, reason=''):
        if not self.auto_switch_enabled():
            self.publish_status(f'aruco_target_skip:auto_switch_disabled:target={target_name}')
            return

        name = str(target_name).strip().upper()
        if not name:
            self.publish_status('aruco_target_skip:empty_target_name')
            return

        msg = String()
        msg.data = name
        self.aruco_target_pub.publish(msg)
        self.last_aruco_target = name

        if reason:
            self.publish_status(f'aruco_target_set:{name}:reason={reason}')
        else:
            self.publish_status(f'aruco_target_set:{name}')

    def set_rack_aruco_target(self, reason=''):
        self.publish_aruco_target(self.p('rack_aruco_target_name'), reason=reason)

    def set_current_qr_aruco_target(self, reason=''):
        if not self.current_target:
            self.publish_status('aruco_target_qr_skip:no_current_target')
            return
        self.publish_aruco_target(self.qr_aruco_target_name(self.current_target), reason=reason)

    def extract_wait_aruco_goal(self, text):
        prefix = 'route_goal_wait_aruco:'
        if not text.startswith(prefix):
            return ''

        rest = text[len(prefix):]
        parts = rest.split(':')

        # expected: route_name:goal_name:delay=...
        if len(parts) >= 2:
            return parts[1].strip()

        return ''

    def is_to_obj_goal(self, goal_name):
        return str(goal_name).strip() == str(self.p('to_obj_goal_name')).strip()

    def expected_qr_goal_name(self):
        if not self.current_target:
            return ''
        return f'to_qr_{self.current_target.lower()}'

    def is_current_qr_goal(self, goal_name):
        goal = str(goal_name).strip()
        expected = self.expected_qr_goal_name()
        return bool(goal and expected and goal == expected)

    def start_aruco_wait(self, kind, goal_name):
        now = time.monotonic()
        self.aruco_wait_start_time = now
        self.aruco_wait_kind = str(kind)
        self.aruco_wait_goal = str(goal_name)
        self.aruco_force_stop_fired = False
        self.publish_status(
            f'aruco_wait_started:kind={self.aruco_wait_kind}:goal={self.aruco_wait_goal}:'
            f'force_stop_sec={float(self.p("aruco_force_stop_sec")):.1f}'
        )

    def clear_aruco_wait(self):
        self.aruco_wait_start_time = 0.0
        self.aruco_wait_kind = ''
        self.aruco_wait_goal = ''
        self.aruco_force_stop_fired = False

    # ============================================================
    # Input callbacks
    # ============================================================
    def place_target_cb(self, msg):
        target = self.normalize_target(msg.data)
        self.handle_new_target_request(target, source='place_target')

    def order_request_cb(self, msg):
        target = extract_target_token(msg.data, self.valid_target_set())
        if not target:
            self.publish_status(f'order_request_rejected:cannot_parse_target:{msg.data}')
            return
        self.publish_status(f'order_request_received:raw={msg.data}:target={target}')
        self.handle_new_target_request(target, source='order_request')

    def handle_new_target_request(self, target, source='place_target'):
        target = self.normalize_target(target)

        if target not in self.valid_target_set():
            self.publish_status(f'{source}_rejected:invalid_target:{target}')
            return

        if (
            bool(self.p('return_to_obj_on_new_target_from_qr'))
            and self.state in ['WAIT_NEXT']
            and self.current_qr_target in self.valid_target_set()
        ):
            self.start_return_to_obj_mission(target, from_qr=self.current_qr_target, source=source)
            return

        if self.state in ['IDLE', 'PARKED']:
            self.start_pickup_mission(target, source=source)
            return

        if self.state == 'WAIT_PICKED':
            old = self.current_target
            self.current_target = target
            self.publish_status(f'{source}_accepted:update_target_while_wait_picked:{old}->{target}')
            return

        busy_states = {
            'RUN_PICKUP',
            'WAIT_TO_OBJ_ARUCO',
            'WAIT_STOP_OBJ_DELAY',
            'RUN_TO_QR',
            'WAIT_TO_QR_ARUCO',
            'WAIT_STOP_QR_DELAY',
            'RUN_PARKING',
        }

        if self.state in busy_states and not bool(self.p('allow_new_target_when_busy')):
            self.publish_status(f'{source}_rejected:busy:state={self.state}:target={target}')
            return

        self.start_pickup_mission(target, source=source)

    def arm_status_cb(self, msg):
        status = str(msg.data).strip().lower()
        self.publish_status(f'arm_status_received:{status}:state={self.state}')

        if status != 'picked':
            return

        if self.state != 'WAIT_PICKED':
            self.publish_status(f'arm_status_ignored:picked:state={self.state}')
            return

        if not self.current_target:
            self.publish_status('arm_status_error:picked_but_no_current_target')
            return

        route = self.route_name(self.p('obj_to_qr_route_template'), self.current_target)

        if bool(self.p('set_qr_target_before_qr_route')):
            self.set_current_qr_aruco_target(reason=f'before_qr_route:{route}')

        self.set_state('RUN_TO_QR')
        self.stop_qr_published = False
        self.stop_qr_due_time = 0.0
        self.clear_aruco_wait()
        self.send_route(route)
        self.publish_status(f'run_to_qr_started:target={self.current_target}:route={route}')

    def go_parking_cb(self, msg):
        if self.state not in ['WAIT_NEXT', 'PARKED', 'IDLE']:
            self.publish_status(f'go_parking_rejected:state={self.state}')
            return

        from_qr = self.current_qr_target or self.current_target or str(self.p('default_qr_target')).upper()
        route = self.route_from_qr_name(self.p('qr_to_parking_route_template'), from_qr=from_qr)

        self.set_state('RUN_PARKING')
        self.clear_aruco_wait()
        self.send_route(route)
        self.publish_status(f'parking_route_started:from_qr={from_qr}:route={route}')

    def align_cmd_cb(self, msg):
        self.last_align_cmd = str(msg.data).strip()
        self.publish_status(f'align_cmd_received:state={self.state}:cmd={self.last_align_cmd}')

    def align_goal_cb(self, msg):
        self.last_align_goal = msg
        p = msg.pose.position
        self.publish_status(
            f'align_goal_received:state={self.state}:'
            f'x={p.x:.3f}:y={p.y:.3f}:frame={msg.header.frame_id}'
        )

    def aruco_done_cb(self, msg):
        if not msg.data:
            return

        if self.state in ['WAIT_TO_OBJ_ARUCO', 'RUN_PICKUP']:
            self.publish_status(f'aruco_done_true_received:state={self.state}:target={self.last_aruco_target}')
            self.clear_aruco_wait()
            self.schedule_stop_obj('aruco_done_true')
            return

        if self.state in ['WAIT_TO_QR_ARUCO', 'RUN_TO_QR']:
            self.publish_status(f'aruco_done_true_received:state={self.state}:target={self.last_aruco_target}')
            self.clear_aruco_wait()
            self.schedule_stop_qr('aruco_done_true')
            return

    def aruco_status_cb(self, msg):
        text = str(msg.data).strip()
        if not text:
            return

        if (
            self.state == 'WAIT_TO_OBJ_ARUCO'
            and text.startswith('STOP aruco align')
            and bool(self.p('publish_stop_obj_on_manual_aruco_stop'))
        ):
            self.publish_status(f'aruco_status_stop_detected_obj:{text}')
            self.clear_aruco_wait()
            self.schedule_stop_obj('aruco_status_stop')
            return

        if (
            self.state == 'WAIT_TO_QR_ARUCO'
            and text.startswith('STOP aruco align')
            and bool(self.p('publish_stop_qr_on_manual_aruco_stop'))
        ):
            self.publish_status(f'aruco_status_stop_detected_qr:{text}')
            self.clear_aruco_wait()
            self.schedule_stop_qr('aruco_status_stop')
            return

    def nav_status_cb(self, msg):
        text = str(msg.data).strip()
        if not text:
            return

        if any(key in text for key in [
            'route_finished',
            'route_cancelled',
            'route_goal_failed',
            'aruco_align_succeeded',
            'aruco_done_received',
            'route_goal_wait_aruco',
            'aruco_timeout',
        ]):
            self.publish_status(f'nav_event:{text}')

        # ------------------------------------------------------------
        # primitive runner가 ArUco wait에 들어갔을 때 target을 한 번 더 보정
        # ------------------------------------------------------------
        if 'route_goal_wait_aruco:' in text:
            wait_goal = self.extract_wait_aruco_goal(text)

            if self.is_to_obj_goal(wait_goal):
                if bool(self.p('set_target_again_on_aruco_wait_event')):
                    self.set_rack_aruco_target(reason=f'wait_aruco:{wait_goal}')

                if self.state == 'RUN_PICKUP':
                    self.set_state('WAIT_TO_OBJ_ARUCO')
                    self.start_aruco_wait('obj', wait_goal)
                return

            if self.is_current_qr_goal(wait_goal):
                if bool(self.p('set_target_again_on_aruco_wait_event')):
                    self.set_current_qr_aruco_target(reason=f'wait_aruco:{wait_goal}')

                if self.state == 'RUN_TO_QR':
                    self.set_state('WAIT_TO_QR_ARUCO')
                    self.start_aruco_wait('qr', wait_goal)
                return

        # ------------------------------------------------------------
        # ArUco success from primitive runner
        # ------------------------------------------------------------
        if text.startswith('aruco_align_succeeded:'):
            if f':{self.p("to_obj_goal_name")}' in text and self.state in ['WAIT_TO_OBJ_ARUCO', 'RUN_PICKUP']:
                self.clear_aruco_wait()
                self.schedule_stop_obj('aruco_align_succeeded')
                return

            qr_goal = self.expected_qr_goal_name()
            if qr_goal and f':{qr_goal}' in text and self.state in ['WAIT_TO_QR_ARUCO', 'RUN_TO_QR']:
                self.clear_aruco_wait()
                self.schedule_stop_qr('aruco_align_succeeded')
                return

        # ------------------------------------------------------------
        # ArUco timeout/failure from primitive runner
        # ------------------------------------------------------------
        if 'aruco_timeout' in text:
            if self.state in ['WAIT_TO_OBJ_ARUCO', 'RUN_PICKUP'] and bool(self.p('publish_stop_obj_on_aruco_timeout')):
                self.clear_aruco_wait()
                self.schedule_stop_obj('aruco_timeout')
                return

            if self.state in ['WAIT_TO_QR_ARUCO', 'RUN_TO_QR'] and bool(self.p('publish_stop_qr_on_aruco_timeout')):
                self.clear_aruco_wait()
                self.schedule_stop_qr('aruco_timeout')
                return

        # ------------------------------------------------------------
        # Pickup / return-to-object sequence
        # ------------------------------------------------------------
        if self.state == 'RUN_PICKUP':
            self.handle_pickup_nav_status(text)
            return

        # ------------------------------------------------------------
        # To QR route
        # ------------------------------------------------------------
        if self.state in ['RUN_TO_QR', 'WAIT_TO_QR_ARUCO']:
            if self.active_route_finished_by_status(text):
                self.clear_aruco_wait()
                self.schedule_stop_qr('to_qr_route_finished')
                return
            if self.active_route_failed_by_status(text):
                self.clear_aruco_wait()
                self.publish_status(f'to_qr_route_failed:{text}')
                self.set_state('WAIT_NEXT')
                return

        # ------------------------------------------------------------
        # Parking route
        # ------------------------------------------------------------
        if self.state == 'RUN_PARKING':
            if self.active_route_finished_by_status(text):
                self.current_qr_target = None
                self.set_state('PARKED')
                self.publish_external_nav_status('parked', reason='parking_route_finished')
                self.publish_status(f'parking_route_finished:{self.active_command_name}')
                return

            if self.active_route_failed_by_status(text):
                self.publish_status(f'parking_route_failed:{text}')
                self.set_state('WAIT_NEXT')
                return

    # ============================================================
    # Mission actions
    # ============================================================
    def start_pickup_mission(self, target, source='place_target'):
        self.current_target = target
        self.stop_obj_published = False
        self.stop_qr_published = False
        self.stop_obj_due_time = 0.0
        self.stop_qr_due_time = 0.0
        self.clear_aruco_wait()

        if bool(self.p('set_rack_target_on_pickup_start')):
            self.set_rack_aruco_target(reason=f'pickup_start:target={target}:source={source}')

        pickup_route = str(self.p('pickup_route_name')).strip()

        self.set_state('RUN_PICKUP')

        if pickup_route:
            self.pickup_goals = []
            self.pickup_index = 0
            self.send_route(pickup_route)
            self.publish_status(f'pickup_route_started:target={target}:route={pickup_route}:source={source}')
            return

        self.pickup_goals = split_csv(self.p('pickup_goal_sequence'))
        if not self.pickup_goals:
            self.publish_status('pickup_error:empty_pickup_goal_sequence')
            self.set_state('IDLE')
            return

        self.pickup_index = 0
        goal = self.pickup_goals[self.pickup_index]
        self.send_goal(goal)
        self.publish_status(
            f'pickup_sequence_started:target={target}:source={source}:'
            f'goals={self.pickup_goals}:current={goal}'
        )

    def start_return_to_obj_mission(self, new_target, from_qr, source='place_target'):
        self.current_target = new_target
        self.stop_obj_published = False
        self.stop_qr_published = False
        self.stop_obj_due_time = 0.0
        self.stop_qr_due_time = 0.0
        self.clear_aruco_wait()

        if bool(self.p('set_rack_target_on_pickup_start')):
            self.set_rack_aruco_target(reason=f'return_to_obj_start:from_qr={from_qr}:new_target={new_target}')

        route = self.route_from_qr_name(self.p('qr_to_obj_route_template'), from_qr=from_qr, target=new_target)

        self.set_state('RUN_PICKUP')
        self.send_route(route)
        self.publish_status(
            f'return_to_obj_started:source={source}:from_qr={from_qr}:new_target={new_target}:route={route}'
        )

    def handle_pickup_nav_status(self, text):
        if self.active_command_kind == 'route':
            if self.active_route_finished_by_status(text):
                self.clear_aruco_wait()
                self.schedule_stop_obj('pickup_or_return_route_finished')
                return
            if self.active_route_failed_by_status(text):
                self.clear_aruco_wait()
                self.publish_status(f'pickup_route_failed:{text}')
                self.set_state('IDLE')
                return
            return

        if self.active_command_kind != 'goal':
            return

        current_goal = self.active_command_name
        to_obj = str(self.p('to_obj_goal_name'))

        if current_goal == to_obj:
            if self.active_route_finished_by_status(text):
                self.clear_aruco_wait()
                self.schedule_stop_obj('to_obj_goal_finished_no_aruco_or_after_aruco')
                return
            return

        if self.active_route_finished_by_status(text):
            self.pickup_index += 1

            if self.pickup_index >= len(self.pickup_goals):
                self.clear_aruco_wait()
                self.schedule_stop_obj('pickup_sequence_finished')
                return

            next_goal = self.pickup_goals[self.pickup_index]

            if next_goal == to_obj:
                self.set_rack_aruco_target(reason=f'before_goal:{next_goal}')

            self.send_goal(next_goal)
            self.publish_status(
                f'pickup_next_goal:{self.pickup_index + 1}/{len(self.pickup_goals)}:{next_goal}'
            )
            return

        if self.active_route_failed_by_status(text):
            self.clear_aruco_wait()
            self.publish_status(f'pickup_goal_failed:{current_goal}:{text}')
            self.set_state('IDLE')

    def schedule_stop_obj(self, reason):
        if self.stop_obj_published or self.state in ['WAIT_STOP_OBJ_DELAY', 'WAIT_PICKED']:
            return

        self.clear_aruco_wait()
        delay = float(self.p('stop_obj_delay_sec'))
        self.stop_obj_due_time = time.monotonic() + max(0.0, delay)
        self.set_state('WAIT_STOP_OBJ_DELAY')
        self.publish_status(
            f'arrived_objects_scheduled:reason={reason}:delay={delay:.2f}:target={self.current_target}'
        )

    def schedule_stop_qr(self, reason):
        if self.stop_qr_published or self.state in ['WAIT_STOP_QR_DELAY', 'WAIT_NEXT']:
            return

        self.clear_aruco_wait()
        delay = float(self.p('stop_qr_delay_sec'))
        self.stop_qr_due_time = time.monotonic() + max(0.0, delay)
        self.set_state('WAIT_STOP_QR_DELAY')
        self.publish_status(
            f'arrived_scheduled:reason={reason}:delay={delay:.2f}:target={self.current_target}'
        )

    def publish_stop_obj(self):
        # 먼저 WAIT_PICKED로 전환해서 /agv_align_enable=true를 먼저 내보낸다.
        self.stop_obj_published = True
        self.stop_obj_due_time = 0.0
        self.current_qr_target = None
        self.set_state('WAIT_PICKED')

        # Orin brain_node로 object/RACK 도착 알림
        self.publish_external_nav_status('arrived_objects', reason='object_station_ready')

        # 기존 /stop_obj 호환이 필요할 때만 발행
        if bool(self.p('publish_legacy_stop_topics')):
            self.stop_obj_pub.publish(Empty())

        self.publish_status(
            f'arrived_objects_published:target={self.current_target}:waiting_for_arm_status_picked'
        )

    def publish_stop_qr(self):
        self.stop_qr_published = True
        self.stop_qr_due_time = 0.0

        self.current_qr_target = self.current_target

        # 먼저 WAIT_NEXT로 전환해서 /agv_align_enable=true를 먼저 내보낸다.
        self.set_state('WAIT_NEXT')

        # Orin brain_node로 QR 도착 알림
        self.publish_external_nav_status('arrived', reason='qr_station_ready')

        # 기존 /stop_qr 호환이 필요할 때만 발행
        if bool(self.p('publish_legacy_stop_topics')):
            self.stop_qr_pub.publish(Empty())

        self.publish_status(
            f'arrived_published:qr={self.current_qr_target}:'
            f'waiting_for_new_place_target_or_go_parking'
        )

    # ============================================================
    # Timer
    # ============================================================
    def timer_cb(self):
        now = time.monotonic()

        if self.state == 'WAIT_STOP_OBJ_DELAY' and self.stop_obj_due_time > 0.0:
            if now >= self.stop_obj_due_time:
                self.publish_stop_obj()

        if self.state == 'WAIT_STOP_QR_DELAY' and self.stop_qr_due_time > 0.0:
            if now >= self.stop_qr_due_time:
                self.publish_stop_qr()

        # ArUco align이 너무 오래 걸리면 강제로 arrived_objects/arrived 발행.
        if bool(self.p('force_stop_on_long_aruco_align')):
            limit = float(self.p('aruco_force_stop_sec'))
            if limit > 0.0 and self.aruco_wait_start_time > 0.0 and not self.aruco_force_stop_fired:
                elapsed = now - self.aruco_wait_start_time
                if elapsed >= limit:
                    self.aruco_force_stop_fired = True
                    self.publish_status(
                        f'aruco_force_stop:kind={self.aruco_wait_kind}:goal={self.aruco_wait_goal}:'
                        f'elapsed={elapsed:.1f}s:limit={limit:.1f}s'
                    )

                    if bool(self.p('stop_aruco_on_force_stop')):
                        self.send_aruco_cmd('stop')

                    if bool(self.p('stop_primitive_on_force_stop')):
                        self.send_primitive('stop', kind='manual', name='force_stop_aruco')

                    if self.aruco_wait_kind == 'obj' or self.state == 'WAIT_TO_OBJ_ARUCO':
                        self.schedule_stop_obj('aruco_force_stop_5s')
                    elif self.aruco_wait_kind == 'qr' or self.state == 'WAIT_TO_QR_ARUCO':
                        self.schedule_stop_qr('aruco_force_stop_5s')

        timeout = float(self.p('command_timeout_sec'))
        if timeout <= 0.0:
            return

        if self.state in ['RUN_PICKUP', 'WAIT_TO_OBJ_ARUCO', 'RUN_TO_QR', 'WAIT_TO_QR_ARUCO', 'RUN_PARKING']:
            if self.active_command_sent_time > 0.0 and (now - self.active_command_sent_time) > timeout:
                self.publish_status(
                    f'brain_command_timeout:state={self.state}:'
                    f'cmd={self.active_command_kind}:{self.active_command_name}:'
                    f'elapsed={now - self.active_command_sent_time:.1f}'
                )
                self.clear_aruco_wait()
                self.set_state('IDLE')


def main(args=None):
    rclpy.init(args=args)
    node = MissionBrainNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.send_primitive('stop')
            node.send_aruco_cmd('stop')
            node.publish_agv_align_enable(False, reason='shutdown')
            time.sleep(0.05)
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
