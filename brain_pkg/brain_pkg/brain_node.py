#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
brain_node.py  (eye-in-hand 관측 흐름 + AGV align 재관측 + pick_failed 재관측 버전)

[eye-in-hand 변경 요약]
  (1) 주문 포맷에 층(level) 추가: "물품:구역:층"  예) "red_cross:A:1"
      - 층 없으면 기본 1층 (하위호환)
  (2) VISION 진입 시 곧바로 /vision_activate 하지 않고,
      먼저 pick_node에 관측 자세로 가라고 명령(/observe_move) →
      pick_node가 도착 신호(/observe_ready) 보내면 그때 /vision_activate 발행.
      (카메라가 그리퍼에 붙어서, 팔이 관측 자세에 있어야 vision 좌표계산이 맞음)
  (3) /vision_activate 포맷: "item:level" (vision_node가 층별 T_cam2base 선택)

[AGV align 변경 요약]
  vision_node가 depth 없음/too_far/too_close 등으로 /marker_agv_pose를 발행하면,
  agv_align_node가 /agv_align을 짧게 발행해 AGV를 한 번 보정 이동시킨다.
  이동이 끝나면 agv_align_node가 /align_status "step_done"을 발행한다.
  brain_node는 step_done을 받으면 다시 /observe_move부터 시작해서 새 관측을 수행한다.

[pick_failed 변경 요약]
  pick_node가 파지 실패 시 /pick_status "pick_failed"를 발행하면,
  brain_node는 바로 ERROR로 가지 않고 observe_move부터 한 번 더 재관측한다.

새 토픽:
  /observe_move  (String) brain -> pick : 관측할 층 번호 "1"/"2"
  /observe_ready (String) pick -> brain : 관측 자세 도착 완료 "ready"
  /align_status  (String) agv_align -> brain : AGV 보정 1스텝 완료 "step_done"
"""

from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, Empty


# 포장구역별 로봇팔 플레이싱 좌표 (실측값으로 교체 필요)
ZONE_TO_PLACE = {
    'A': [200.0, 100.0, 80.0, 175.35, -1.1, -89.73],
    'B': [200.0, 150.0, 80.0, 175.35, -1.1, -89.73],
    'C': [200.0, 200.0, 80.0, 175.35, -1.1, -89.73],
}

# ===== [신규] 유효 층 목록 (vision_node의 SHELF_POSES 키와 일치해야 함) =====
VALID_LEVELS = {1, 2}
DEFAULT_LEVEL = 1


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')

        # Subscribers
        self.create_subscription(String, '/order_request', self._order_callback, 10)
        self.create_subscription(Float32MultiArray, '/box_pose', self._box_pose_callback, 10)
        self.create_subscription(String, '/pick_status', self._pick_status_callback, 10)
        self.create_subscription(String, '/nav_status', self._nav_status_callback, 10)
        self.create_subscription(String, '/emergency_stop', self._emergency_stop_callback, 10)

        # ===== [신규] pick_node의 관측 자세 도착 신호 =====
        self.create_subscription(String, '/observe_ready', self._observe_ready_callback, 10)

        # ===== [신규] agv_align_node의 차체 보정 1스텝 완료 신호 =====
        self.create_subscription(String, '/align_status', self._align_status_callback, 10)

        # Publishers
        self._vision_activate_pub = self.create_publisher(String, '/vision_activate', 10)
        self._pick_command_pub = self.create_publisher(Float32MultiArray, '/pick_command', 10)
        self._place_command_pub = self.create_publisher(Float32MultiArray, '/place_command', 10)
        self._place_target_pub = self.create_publisher(String, '/place_target', 10)
        self._arm_status_pub = self.create_publisher(String, '/arm_status', 10)
        self._go_parking_pub = self.create_publisher(Empty, '/go_parking', 10)

        # ===== [수정] 기존 self.__pub = create_publisher(String, '/', 10) 는 토픽명이 '/'로 잘못돼 있었음 =====
        # /wms_update 로 명시적으로 발행하도록 수정
        self._wms_update_pub = self.create_publisher(String, '/wms_update', 10)

        self._brain_state_pub = self.create_publisher(String, '/brain_state', 10)

        # ===== [신규] 관측 자세 이동 명령 =====
        self._observe_move_pub = self.create_publisher(String, '/observe_move', 10)

        # Internal states
        self.state = 'IDLE'
        self.order_queue = deque()
        self.current_order = None
        self.zone = None
        self.item = None
        self.level = DEFAULT_LEVEL
        self.emergency_active = False

        # ===== [신규] AGV 차체 보정 재관측 제한 =====
        self.align_retry_count = 0
        self.ALIGN_RETRY_MAX = 8

        # ===== [신규] pick 실패 시 재관측 제한 =====
        self.pick_retry_count = 0
        self.PICK_REOBSERVE_MAX = 1

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
        ===== [변경] 주문 형식: "물품:구역:층"  예) "red_cross:A:1"
        - 층 생략 시 기본 1층 (하위호환): "red_cross:A"
        - 구역/층 모두 생략 시: "red_cross" -> zone='A', level=1
        """
        order = order.strip()
        parts = order.split(':')

        item = parts[0].strip()
        zone = parts[1].upper().strip() if len(parts) >= 2 and parts[1].strip() else 'A'

        level = DEFAULT_LEVEL
        if len(parts) >= 3 and parts[2].strip():
            try:
                level = int(parts[2].strip())
            except ValueError:
                self.get_logger().warn(f'층 파싱 실패("{parts[2]}") -> 기본 {DEFAULT_LEVEL}층')
                level = DEFAULT_LEVEL

        if level not in VALID_LEVELS:
            self.get_logger().warn(f'유효하지 않은 층 {level} -> 기본 {DEFAULT_LEVEL}층')
            level = DEFAULT_LEVEL

        return item, zone, level

    def _start_next_order(self):
        if self.emergency_active:
            self.get_logger().warn('비상정지 상태이므로 다음 주문 시작 안 함')
            return

        if not self.order_queue:
            self.get_logger().info('대기 주문 없음')
            return

        self.current_order = self.order_queue.popleft()

        # ===== [변경] 층까지 파싱 =====
        self.item, self.zone, self.level = self._parse_order(self.current_order)

        # 새 주문 시작 시 보정/재시도 카운터 초기화
        self.align_retry_count = 0
        self.pick_retry_count = 0

        self.get_logger().info(
            f'다음 주문 시작: {self.current_order}, '
            f'item={self.item}, zone={self.zone}, level={self.level}'
        )

        self.state = 'NAV_TO_RACK'
        self._pub_state()

        self._publish_string(self._place_target_pub, self.zone)
        self.get_logger().info(f'/place_target 발행: {self.zone}')

    def _finish_current_order(self):
        self.get_logger().info(f'현재 주문 완료: {self.current_order}')

        # ===== [수정] 올바른 /wms_update 퍼블리셔 사용 =====
        w_msg = String()
        w_msg.data = f'{self.item}:{self.zone}:done'
        self._wms_update_pub.publish(w_msg)
        self.get_logger().info(f'/wms_update 발행: {w_msg.data}')

        self.current_order = None
        self.zone = None
        self.item = None
        self.level = DEFAULT_LEVEL
        self.align_retry_count = 0
        self.pick_retry_count = 0

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
                f'현재 {self.state} 상태라 주문 큐에 저장. '
                f'대기 주문 수: {len(self.order_queue)}'
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

        # 정상 좌표를 받았으므로 align retry는 성공적으로 종료
        self.align_retry_count = 0

        self.state = 'PICKING'
        self._pub_state()

        self._pick_command_pub.publish(msg)
        self.get_logger().info('/pick_command 발행')

    def _pick_status_callback(self, msg):
        if self.emergency_active:
            self.get_logger().warn(
                f'/pick_status 수신했지만 비상정지 상태라 무시: {msg.data}'
            )
            return

        status = msg.data.strip()
        self.get_logger().info(f'/pick_status 수신: {status}')

        if status == 'done':
            if self.state != 'PICKING':
                self.get_logger().warn(
                    f'pick done 수신했지만 현재 상태가 PICKING이 아님: {self.state}'
                )
                return

            # 픽 성공했으므로 재관측 retry 카운터 초기화
            self.pick_retry_count = 0

            self.state = 'NAV_TO_DEST'
            self._pub_state()

            self._publish_string(self._arm_status_pub, 'picked')
            self.get_logger().info('/arm_status 발행: picked')

        elif status == 'placing_done':
            if self.state != 'PLACING':
                self.get_logger().warn(
                    f'placing_done 수신했지만 현재 상태가 PLACING이 아님: {self.state}'
                )
                return

            self._publish_string(self._arm_status_pub, 'placed')
            self.get_logger().info('/arm_status 발행: placed')

            self._finish_current_order()
          
          
        elif status == 'realign_fail':
            self.get_logger().warn('pick_node realign_fail 수신 - AGV 차체 보정 후 재관측 대기')

            # realign_fail은 vision/J1 보정으로 해결 안 되니 AGV 재정차 루프로 넘김
            # 단, 실제 AGV 이동은 /marker_agv_pose -> agv_align_node -> /agv_align 에서 이미 수행됨
            # 여기서는 align_status step_done을 기다리기 위해 VISION 상태를 유지한다.
            if self.state not in ('VISION', 'OBSERVING'):
                self.get_logger().warn(
                    f'realign_fail 수신했지만 현재 상태가 VISION/OBSERVING 아님: {self.state}'
                )
                return

            self.state = 'VISION'
            self._pub_state()
            return

      
        elif status == 'pick_failed':
            if self.state != 'PICKING':
                self.get_logger().warn(
                    f'pick_failed 수신했지만 현재 상태가 PICKING이 아님: {self.state}'
                )
                return

            self.get_logger().warn('pick_failed 수신 - 재관측 retry 판단')

            if self.pick_retry_count < self.PICK_REOBSERVE_MAX:
                self.pick_retry_count += 1

                self.get_logger().warn(
                    f'pick 재관측 재시도 {self.pick_retry_count}/{self.PICK_REOBSERVE_MAX}'
                )

                # 다시 관측 자세부터 시작
                self.state = 'OBSERVING'
                self._pub_state()

                self._publish_string(self._observe_move_pub, str(self.level))
                self.get_logger().info(
                    f'/observe_move 재발행: level={self.level} (pick_failed 재관측)'
                )
                return

            # 재시도 초과
            self.get_logger().error('pick 재관측 재시도 초과 - 픽 실패 처리')
            self.pick_retry_count = 0
            self.state = 'ERROR'
            self._pub_state()
            return

        elif status == 'error':
            self.get_logger().error('pick_node error 수신')
            self.state = 'ERROR'
            self._pub_state()

        else:
            self.get_logger().warn(f'알 수 없는 pick_status: {status}')

    def _nav_status_callback(self, msg):
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

            # ===== [변경] eye-in-hand: 바로 vision 켜지 않고 관측 자세부터 이동 =====
            self.state = 'OBSERVING'
            self._pub_state()

            self._publish_string(self._observe_move_pub, str(self.level))
            self.get_logger().info(
                f'/observe_move 발행: level={self.level} (관측 자세 이동 요청)'
            )

        elif msg.data == 'arrived':
            if self.state != 'NAV_TO_DEST':
                self.get_logger().warn(
                    f'arrived 수신했지만 현재 상태가 NAV_TO_DEST가 아님: {self.state}'
                )
                return

            self.state = 'PLACING'
            self._pub_state()

            zone = self.zone if self.zone else 'A'

            if zone not in ZONE_TO_PLACE:
                self.get_logger().error(f'ZONE_TO_PLACE에 없는 구역: {zone}')
                self.state = 'ERROR'
                self._pub_state()
                return

            place_msg = Float32MultiArray()
            place_msg.data = ZONE_TO_PLACE[zone]
            self._place_command_pub.publish(place_msg)

            self.get_logger().info(f'/place_command 발행: zone={zone}')

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
            self.level = DEFAULT_LEVEL
            self.align_retry_count = 0
            self.pick_retry_count = 0
            self._pub_state()

            if self.order_queue:
                self.get_logger().info('주차 중 들어온 주문 있음 -> 다음 주문 시작')
                self._start_next_order()

        else:
            self.get_logger().warn(f'알 수 없는 nav_status: {msg.data}')

    # ===== [신규] pick_node가 관측 자세에 도착했을 때 =====
    def _observe_ready_callback(self, msg):
        if self.emergency_active:
            self.get_logger().warn('/observe_ready 수신했지만 비상정지 상태라 무시')
            return

        # OBSERVING(최초 관측 / AGV align 후 재관측 / pick_failed 후 재관측)
        # 또는 VISION(J1 보정 재관측) 둘 다 처리
        if self.state not in ('OBSERVING', 'VISION'):
            self.get_logger().warn(
                f'/observe_ready 수신했지만 상태가 OBSERVING/VISION 아님: {self.state}'
            )
            return

        self.get_logger().info(f'/observe_ready 수신: {msg.data} (관측 자세 도착)')

        # 최초 관측이면 VISION으로 전이, 이미 VISION이면(J1 보정 재관측) 유지
        if self.state == 'OBSERVING':
            self.state = 'VISION'
            self._pub_state()

        activate_data = f'{self.item}:{self.level}'
        self._publish_string(self._vision_activate_pub, activate_data)
        self.get_logger().info(f'/vision_activate 발행: {activate_data}')

    # ===== [신규] AGV 차체 보정 1스텝 완료 후 다시 관측 =====
    def _align_status_callback(self, msg):
        if self.emergency_active:
            self.get_logger().warn(
                f'/align_status 수신했지만 비상정지 상태라 무시: {msg.data}'
            )
            return

        status = msg.data.strip()
        self.get_logger().info(f'/align_status 수신: {status}')

        if status != 'step_done':
            self.get_logger().warn(f'알 수 없는 align_status: {status}')
            return

        # 차체 보정은 vision 단계에서만 의미 있음
        if self.state != 'VISION':
            self.get_logger().warn(
                f'align step_done 수신했지만 현재 상태가 VISION이 아님: {self.state}'
            )
            return

        if self.align_retry_count >= self.ALIGN_RETRY_MAX:
            self.get_logger().error('AGV 차체 보정 반복 초과 → ERROR')
            self.state = 'ERROR'
            self._pub_state()
            return

        self.align_retry_count += 1

        self.get_logger().warn(
            f'AGV 차체 보정 후 재관측 {self.align_retry_count}/{self.ALIGN_RETRY_MAX}'
        )

        # 다시 관측 자세부터 시작
        self.state = 'OBSERVING'
        self._pub_state()

        self._publish_string(self._observe_move_pub, str(self.level))
        self.get_logger().info(
            f'/observe_move 재발행: level={self.level} (AGV align 후 재관측)'
        )

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
        self.level = DEFAULT_LEVEL
        self.align_retry_count = 0
        self.pick_retry_count = 0
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
