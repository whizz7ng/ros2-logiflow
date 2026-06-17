#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
brain_node.py

뷰티 풀필먼트 피킹/소팅 자동화 FSM Brain Node

통신 기준:

[WMS -> Brain]
/order_request : std_msgs/String
    포맷: "OR1:blue", "OR2:red", "OR3:green"
    또는 단순: "OR1" (color 없으면 기본값 green)

[Brain -> WMS]
/wms_update : std_msgs/String

[Brain -> Dashboard]
/brain_state : std_msgs/String

[Brain -> Vision]
/vision_activate : std_msgs/String
    "blue", "red", "green", "stop"

[Vision -> Brain]
/box_pose : std_msgs/Float32MultiArray

[Brain -> Pick]
/pick_command : std_msgs/Float32MultiArray
/place_command : std_msgs/Float32MultiArray

[Pick -> Brain]
/pick_status : std_msgs/String
    "done", "placing_done", "error"

[Brain -> AGV/Nav]
/place_target : std_msgs/String
    "OR1", "OR2", "OR3"

/arm_status : std_msgs/String
    "picked", "placed"

/go_parking : std_msgs/Empty
    모든 주문 완료 후 주차 복귀 명령

[AGV/Nav -> Brain]
/nav_status : std_msgs/String
    "arrived_objects", "arrived_qr", "arrived_qr_red", "arrived_qr_blue", "arrived_qr_green", "parked"

[Keyboard -> Brain/Pick/Nav]
/emergency_stop : std_msgs/String
    "stop", "reset"
"""

from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, Empty


# 포장구역별 로봇팔 플레이싱 좌표
# 실제 실측값으로 교체 필요
ZONE_TO_PLACE = {
    'A': [200.0, 100.0, 80.0, 175.35, -1.1, -89.73],
    'B': [200.0, 150.0, 80.0, 175.35, -1.1, -89.73],
    'C': [200.0, 200.0, 80.0, 175.35, -1.1, -89.73],
}

# 목적지 → 색깔 기본 매핑 (order에 color 없을 때)
# DEST_TO_COLOR = {
#     'A': 'blue',
#     'B': 'red',
#     'C': 'green',
# }


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')

        # =========================
        # Subscribers
        # =========================
        self.create_subscription(String, '/order_request', self._order_callback, 10)
        self.create_subscription(Float32MultiArray, '/box_pose', self._box_pose_callback, 10)
        self.create_subscription(String, '/pick_status', self._pick_status_callback, 10)
        self.create_subscription(String, '/nav_status', self._nav_status_callback, 10)
        self.create_subscription(String, '/emergency_stop', self._emergency_stop_callback, 10)

        # =========================
        # Publishers
        # =========================
        self._vision_activate_pub = self.create_publisher(String, '/vision_activate', 10)

        self._pick_command_pub = self.create_publisher(Float32MultiArray, '/pick_command', 10)
        self._place_command_pub = self.create_publisher(Float32MultiArray, '/place_command', 10)

        self._place_target_pub = self.create_publisher(String, '/place_target', 10)
        self._arm_status_pub = self.create_publisher(String, '/arm_status', 10)
        self._go_parking_pub = self.create_publisher(Empty, '/go_parking', 10)

        self._wms_update_pub = self.create_publisher(String, '/wms_update', 10)
        self._brain_state_pub = self.create_publisher(String, '/brain_state', 10)

        # =========================
        # Internal states
        # =========================
        self.state = 'IDLE'

        self.order_queue = deque()
        self.current_order = None
        self.destination = None
        self.target_color = None  # 집어야 할 블록 색깔

        self.emergency_active = False

        self.get_logger().info('brain_node 시작 - 상태: IDLE')
        self._pub_state()

    # ============================================================
    # Common utils
    # ============================================================
    def _pub_state(self):
        msg = String()
        msg.data = self.state
        self._brain_state_pub.publish(msg)
        self.get_logger().info(f'FSM 상태: {self.state}')

    def _publish_string(self, publisher, data: str):
        msg = String()
        msg.data = data
        publisher.publish(msg)

    def _parse_order(self, order):
        """
        주문 형식: "물품:구역"  예) "red_triangle:A"
        item = 집을 물품 (YOLO 라벨)
        zone = 배송 구역 (A/B/C)
        """
        order = order.strip()
        if ':' in order:
            item, zone = order.split(':', 1)
            item = item.strip()
            zone = zone.upper().strip()
        else:
            # 구역 없으면 기본 A (디버깅용)
            item = order.strip()
            zone = 'A'
        return item, zone

    def _start_next_order(self):
        if self.emergency_active:
            self.get_logger().warn('비상정지 상태이므로 다음 주문 시작 안 함')
            return

        if not self.order_queue:
            self.get_logger().info('대기 주문 없음')
            return

        self.current_order = self.order_queue.popleft()
        self.item, self.zone = self._parse_order(self.current_order)

        self.get_logger().info(
            f'다음 주문 시작: {self.current_order}, destination={self.destination}, color={self.target_color}'
        )

        self.state = 'NAV_TO_RACK'
        self._pub_state()

        self._publish_string(self._place_target_pub, self.item)
        self.get_logger().info(f'/place_target 발행: {self.item}')

    def _finish_current_order(self):
        self.get_logger().info(f'현재 주문 완료: {self.current_order}')

        w_msg = String()
        w_msg.data = self.current_order or 'done'
        self._wms_update_pub.publish(w_msg)
        self.get_logger().info(f'/wms_update 발행: {w_msg.data}')

        self.current_order = None
        self.destination = None
        self.target_color = None

        if self.order_queue:
            self.get_logger().info(
                f'남은 주문 있음: {len(self.order_queue)}개 -> 다음 주문 시작'
            )
            self._start_next_order()
        else:
            self.get_logger().info('남은 주문 없음 -> 주차 복귀 명령 발행')

            self.state = 'GO_PARKING'
            self._pub_state()

            self._go_parking_pub.publish(Empty())
            self.get_logger().info('/go_parking 발행: Empty')

    # ============================================================
    # Callbacks
    # ============================================================
    def _order_callback(self, msg: String):
        if self.emergency_active:
            self.get_logger().warn(f'비상정지 상태라 주문 무시: {msg.data}')
            return

        self.get_logger().info(f'주문 수신: {msg.data}')
        self.order_queue.append(msg.data)

        if self.state == 'IDLE':
            self._start_next_order()
        else:
            self.get_logger().info(
                f'현재 {self.state} 상태라 주문 큐에 저장. 대기 주문 수: {len(self.order_queue)}'
            )

    def _box_pose_callback(self, msg: Float32MultiArray):
        if self.emergency_active:
            self.get_logger().warn('/box_pose 수신했지만 비상정지 상태라 무시')
            return

        self.get_logger().info(f'/box_pose 수신: {list(msg.data)}')

        if self.state != 'VISION':
            self.get_logger().warn(
                f'현재 상태가 VISION이 아니므로 /box_pose 무시. 현재 상태: {self.state}'
            )
            return

        self.state = 'PICKING'
        self._pub_state()

        self._pick_command_pub.publish(msg)
        self.get_logger().info('/pick_command 발행')

    def _pick_status_callback(self, msg: String):
        if self.emergency_active:
            self.get_logger().warn(
                f'/pick_status 수신했지만 비상정지 상태라 무시: {msg.data}'
            )
            return

        self.get_logger().info(f'/pick_status 수신: {msg.data}')

        if msg.data == 'done':
            if self.state != 'PICKING':
                self.get_logger().warn(
                    f'pick done 수신했지만 현재 상태가 PICKING이 아님: {self.state}'
                )
                return

            self.state = 'NAV_TO_DEST'
            self._pub_state()

            self._publish_string(self._arm_status_pub, 'picked')
            self.get_logger().info('/arm_status 발행: picked')

        elif msg.data == 'placing_done':
            if self.state != 'PLACING':
                self.get_logger().warn(
                    f'placing_done 수신했지만 현재 상태가 PLACING이 아님: {self.state}'
                )
                return

            self._publish_string(self._arm_status_pub, 'placed')
            self.get_logger().info('/arm_status 발행: placed')

            self._finish_current_order()

        elif msg.data == 'error':
            self.get_logger().error('pick_node error 수신')
            self.state = 'ERROR'
            self._pub_state()

        else:
            self.get_logger().warn(f'알 수 없는 pick_status: {msg.data}')

    def _nav_status_callback(self, msg: String):
        if self.emergency_active:
            self.get_logger().warn(
                f'/nav_status 수신했지만 비상정지 상태라 무시: {msg.data}'
            )
            return

        self.get_logger().info(f'/nav_status 수신: {msg.data}')

        if msg.data == 'arrived_objects':
            if self.state != 'NAV_TO_RACK':
                self.get_logger().warn(
                    f'arrived_objects 수신했지만 현재 상태가 NAV_TO_RACK이 아님: {self.state}'
                )
                return

            self.state = 'VISION'
            self._pub_state()

            # 색깔 정보와 함께 vision 활성화
            color = self.target_color or 'green'
            self._publish_string(self._vision_activate_pub, color)
            self.get_logger().info(f'/vision_activate 발행: {color}')

        elif msg.data == 'arrived':
            if self.state != 'NAV_TO_DEST':
                self.get_logger().warn(
                    f'arrived_qr 수신했지만 현재 상태가 NAV_TO_DEST가 아님: {self.state}'
                )
                return

            self.state = 'PLACING'
            self._pub_state()

            dest = self.destination if self.destination else 'OR3'

            if dest not in ZONE_TO_PLACE:
                self.get_logger().error(f'PLACE_COORDS에 없는 목적지: {self.zone}')
                self.state = 'ERROR'
                self._pub_state()
                return

            place_msg = Float32MultiArray()
            place_msg.data = PLACE_COORDS[dest]
            self._place_command_pub.publish(place_msg)

            self.get_logger().info(
                f'/place_command 발행: {dest}, {PLACE_COORDS[dest]}'
            )

        elif msg.data == 'parked':
            if self.state != 'GO_PARKING':
                self.get_logger().warn(
                    f'parked 수신했지만 현재 상태가 GO_PARKING이 아님: {self.state}'
                )
                return

            self.get_logger().info('주차 완료 -> IDLE 복귀')

            self.state = 'IDLE'
            self.current_order = None
            self.destination = None
            self.target_color = None
            self._pub_state()

            if self.order_queue:
                self.get_logger().info('주차 중 들어온 주문 있음 -> 다음 주문 시작')
                self._start_next_order()
        else:
            self.get_logger().warn(f'알 수 없는 nav_status: {msg.data}')

    def _emergency_stop_callback(self, msg: String):
        command = msg.data.strip().lower()
        self.get_logger().warn(f'/emergency_stop 수신: {command}')

        if command in ['stop', 'emergency', 'emergency_stop', 'true', '1', 'on']:
            self._enter_emergency_stop()

        elif command in ['reset', 'release', 'clear', 'false', '0', 'off']:
            self._release_emergency_stop()

        else:
            self.get_logger().warn(f'알 수 없는 emergency_stop 명령: {msg.data}')

    # ============================================================
    # Emergency stop
    # ============================================================
    def _enter_emergency_stop(self):
        if self.emergency_active:
            self.get_logger().warn('이미 비상정지 상태')
            return

        self.emergency_active = True
        self.state = 'EMERGENCY_STOP'
        self._pub_state()

        self._publish_string(self._vision_activate_pub, 'stop')

        self.get_logger().error(
            '비상정지 진입. Brain FSM 정지. 실제 모터 정지는 pick_node/nav_node가 /emergency_stop을 직접 처리해야 함.'
        )

    def _release_emergency_stop(self):
        if not self.emergency_active:
            self.get_logger().warn('현재 비상정지 상태가 아님')
            return

        self.emergency_active = False

        self.current_order = None
        self.destination = None
        self.target_color = None

        self.state = 'IDLE'
        self._pub_state()

        self.get_logger().info(
            f'비상정지 해제 -> IDLE 복귀. 대기 주문 수: {len(self.order_queue)}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
