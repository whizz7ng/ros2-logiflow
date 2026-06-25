#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
brain_node.py

뷰티 풀필먼트 피킹/소팅 자동화 FSM Brain Node

통신 기준:

[WMS -> Brain]
/order_request : std_msgs/String
    포맷: "물품:구역"  예) "red_triangle:A"

[Brain -> WMS]
/wms_update : std_msgs/String

[Brain -> Dashboard]
/brain_state : std_msgs/String

[Brain -> Vision]
/vision_activate : std_msgs/String
    물품 라벨 또는 "stop" 또는 "qr_place"

[Vision -> Brain]
/box_pose   : std_msgs/Float32MultiArray   (블록 피킹 좌표)
/place_pose : std_msgs/Float32MultiArray   (QR 기반 플레이싱 좌표)

[Brain -> Pick]
/pick_command : std_msgs/Float32MultiArray
/place_command : std_msgs/Float32MultiArray

[Pick -> Brain]
/pick_status : std_msgs/String
    "done", "placing_done", "error"

[Brain -> AGV/Nav]
/place_target : std_msgs/String
    "A", "B", "C"

/arm_status : std_msgs/String
    "picked", "placed"

/go_parking : std_msgs/Empty

[AGV/Nav -> Brain]
/nav_status : std_msgs/String
    "arrived_objects" | "stop_obj"  (rack 도착)
    "arrived" | "stop_qr"           (QR 도착)
    "parked"                        (주차 완료)

[Keyboard -> Brain/Pick/Nav]
/emergency_stop : std_msgs/String
    "stop", "reset"
"""

from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, Empty


# 포장구역별 로봇팔 플레이싱 좌표 (방법 B에서는 QR 기반 좌표를 우선 사용,
# place_pose 수신 실패 등 fallback 용도로만 유지)
ZONE_TO_PLACE = {
    'A': [200.0, 100.0, 80.0, 175.35, -1.1, -89.73],
    'B': [200.0, 150.0, 80.0, 175.35, -1.1, -89.73],
    'C': [200.0, 200.0, 80.0, 175.35, -1.1, -89.73],
}


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')

        # Subscribers
        self.create_subscription(String, '/order_request', self._order_callback, 10)
        self.create_subscription(Float32MultiArray, '/box_pose', self._box_pose_callback, 10)
        self.create_subscription(Float32MultiArray, '/place_pose', self._place_pose_callback, 10)
        self.create_subscription(String, '/pick_status', self._pick_status_callback, 10)
        self.create_subscription(String, '/nav_status', self._nav_status_callback, 10)
        self.create_subscription(String, '/emergency_stop', self._emergency_stop_callback, 10)

        # Publishers
        self._vision_activate_pub = self.create_publisher(String, '/vision_activate', 10)
        self._pick_command_pub = self.create_publisher(Float32MultiArray, '/pick_command', 10)
        self._place_command_pub = self.create_publisher(Float32MultiArray, '/place_command', 10)
        self._place_target_pub = self.create_publisher(String, '/place_target', 10)
        self._arm_status_pub = self.create_publisher(String, '/arm_status', 10)
        self._go_parking_pub = self.create_publisher(Empty, '/go_parking', 10)
        self._wms_update_pub = self.create_publisher(String, '/wms_update', 10)
        self._brain_state_pub = self.create_publisher(String, '/brain_state', 10)

        # Internal states
        self.state = 'IDLE'
        self.order_queue = deque()
        self.current_order = None
        self.zone = None
        self.item = None
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

    def _publish_string(self, publisher, data):
        msg = String()
        msg.data = data
        publisher.publish(msg)

    def _parse_order(self, order):
        """
        주문 형식: "물품:구역"  예) "red_triangle:A"
        """
        order = order.strip()
        if ':' in order:
            item, zone = order.split(':', 1)
            item = item.strip()
            zone = zone.upper().strip()
        else:
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
            f'다음 주문 시작: {self.current_order}, item={self.item}, zone={self.zone}'
        )

        self.state = 'NAV_TO_RACK'
        self._pub_state()

        self._publish_string(self._place_target_pub, self.zone)
        self.get_logger().info(f'/place_target 발행: {self.zone}')

    def _finish_current_order(self):
        self.get_logger().info(f'현재 주문 완료: {self.current_order}')

        w_msg = String()
        w_msg.data = f'{self.item}:{self.zone}:done'   # ← 3개 (label:zone:done)
        self._wms_update_pub.publish(w_msg)
        self.get_logger().info(f'/wms_update 발행: {w_msg.data}')

        self.current_order = None
        self.zone = None
        self.item = None

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

    def _do_place_with_coords(self, place_data):
        """
        주어진 좌표로 PLACING 진입 + /place_command 발행 (공통 처리).
        place_data: [x, y, z, rx, ry, rz]
        """
        self.state = 'PLACING'
        self._pub_state()

        place_msg = Float32MultiArray()
        place_msg.data = [float(v) for v in place_data]
        self._place_command_pub.publish(place_msg)
        self.get_logger().info(
            f'/place_command 발행: {[round(v, 1) for v in place_data]}'
        )

    # ============================================================
    # Callbacks
    # ============================================================
    def _order_callback(self, msg):
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

    def _box_pose_callback(self, msg):
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

    def _place_pose_callback(self, msg):
        """
        vision_node가 QR을 인식해 계산한 플레이싱 좌표 수신.
        VISION_PLACE 상태에서만 받아 PLACING으로 진입한다.
        """
        if self.emergency_active:
            self.get_logger().warn('/place_pose 수신했지만 비상정지 상태라 무시')
            return

        self.get_logger().info(f'/place_pose 수신: {list(msg.data)}')

        if self.state != 'VISION_PLACE':
            self.get_logger().warn(
                f'현재 상태가 VISION_PLACE가 아니므로 /place_pose 무시. 현재 상태: {self.state}'
            )
            return

        if len(msg.data) != 6:
            self.get_logger().error(
                f'/place_pose 좌표 6개 필요, 받은 개수: {len(msg.data)} -> ZONE_TO_PLACE fallback'
            )
            zone = self.zone if self.zone in ZONE_TO_PLACE else 'A'
            self._do_place_with_coords(ZONE_TO_PLACE[zone])
            return

        # QR 기반 좌표로 플레이싱
        self._do_place_with_coords(list(msg.data))

    def _pick_status_callback(self, msg):
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

    def _nav_status_callback(self, msg):
        if self.emergency_active:
            self.get_logger().warn(
                f'/nav_status 수신했지만 비상정지 상태라 무시: {msg.data}'
            )
            return

        self.get_logger().info(f'/nav_status 수신: {msg.data}')

        # ---- rack(objects) 도착: arrived_objects 또는 stop_obj ----
        if msg.data in ('arrived_objects', 'stop_obj'):
            if self.state != 'NAV_TO_RACK':
                self.get_logger().warn(
                    f'{msg.data} 수신했지만 현재 상태가 NAV_TO_RACK이 아님: {self.state}'
                )
                return

            self.state = 'VISION'
            self._pub_state()

            self._publish_string(self._vision_activate_pub, self.item)
            self.get_logger().info(f'/vision_activate 발행: {self.item}')

        # ---- QR(구역) 도착: arrived 또는 stop_qr ----
        # 방법 B: 고정 좌표로 바로 PLACING 하지 않고,
        #         vision_node에 'qr_place'를 요청해 QR 기반 좌표를 받는다.
        elif msg.data in ('arrived', 'stop_qr'):
            if self.state != 'NAV_TO_DEST':
                self.get_logger().warn(
                    f'{msg.data} 수신했지만 현재 상태가 NAV_TO_DEST가 아님: {self.state}'
                )
                return

            self.state = 'VISION_PLACE'
            self._pub_state()

            self._publish_string(self._vision_activate_pub, 'qr_place')
            self.get_logger().info('/vision_activate 발행: qr_place (QR 기반 플레이싱 좌표 요청)')

        # ---- 주차 완료 ----
        elif msg.data == 'parked':
            if self.state != 'GO_PARKING':
                self.get_logger().warn(
                    f'parked 수신했지만 현재 상태가 GO_PARKING이 아님: {self.state}'
                )
                return

            self.get_logger().info('주차 완료 -> IDLE 복귀')

            self.state = 'IDLE'
            self.current_order = None
            self.zone = None
            self.item = None
            self._pub_state()

            if self.order_queue:
                self.get_logger().info('주차 중 들어온 주문 있음 -> 다음 주문 시작')
                self._start_next_order()

        else:
            self.get_logger().warn(f'알 수 없는 nav_status: {msg.data}')

    def _emergency_stop_callback(self, msg):
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
            '비상정지 진입. Brain FSM 정지. 실제 모터 정지는 pick_node/nav_node가 직접 처리.'
        )

    def _release_emergency_stop(self):
        if not self.emergency_active:
            self.get_logger().warn('현재 비상정지 상태가 아님')
            return

        self.emergency_active = False
        self.current_order = None
        self.zone = None
        self.item = None
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
