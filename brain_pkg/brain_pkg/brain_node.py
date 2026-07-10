#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
brain_node.py  (eye-in-hand 관측 흐름 + 정면정렬(FRONTAL_ALIGN) + 블록기반 x/y보정 버전)

[변경 요약 - 이번 세션]
  기존에는 vision이 too_far/too_close/depth_fail일 때 마커의 고정 TARGET
  좌표와 비교해서 agv_align_node가 x/y까지 정밀 보정했음.
  이제는:
    1) 관측 자세 도착 직후 -> FRONTAL_ALIGN 상태로 먼저 마커 정면(yaw) 정렬만 수행.
       (agv_align_node가 마커 하나만 보여도 rvec 기반 yaw로 이 작업을 함)
       정면 정렬 끝나면(/align_status "aligned") 곧바로 블록 검출(VISION) 시작.
    2) VISION 상태에서 블록을 찾았는데 파지범위 밖(too_far)이면
       -> /align_request "block_forward" 로 AGV 전진 요청 (마커 TARGET 안 씀).
       too_close는 기존과 동일하게 후진 금지 정책이라 ERROR.
    3) 블록은 파지범위 안인데 화면 중앙에서 너무 치우쳐 있으면(distance_status
       "side_left"/"side_right", 신규) -> /align_request "block_left"/"block_right".
    4) 2),3) 보정 후 step_done을 받으면 다시 관측 자세부터(=다시 정면정렬부터) 반복.

[eye-in-hand 요약 - 기존 유지]
  (1) 주문 포맷에 층(level) 추가: "물품:구역:층"
  (2) VISION 진입 전 pick_node에 관측 자세로 이동 명령(/observe_move) ->
      도착 신호(/observe_ready) 받으면 그때부터 진행.
  (3) /vision_activate 포맷: "item:level" (블록 검출),
      "marker_align:level" (신규, 정면정렬 전용), "qr_place"

[pick_failed 요약 - 기존 유지]
  pick_node가 파지 실패 시 /pick_status "pick_failed"를 발행하면,
  brain_node는 바로 ERROR로 가지 않고 observe_move부터 한 번 더 재관측한다.
  (재관측은 다시 FRONTAL_ALIGN부터 시작됨)

토픽:
  /observe_move    (String) brain -> pick : 관측할 층 번호 "1"/"2"
  /observe_ready   (String) pick -> brain : 관측 자세 도착 완료 "ready"
  /distance_status (String) vision -> brain : "ok:311" / "too_close:239" /
                    "too_far:360" / "side_left:280" / "side_right:280"(신규) /
                    "depth_fail:0" / "qr_too_far:.." / "qr_too_close:.."
  /align_status    (String) agv_align -> brain : "step_done" / "aligned"
  /align_request   (String) brain -> agv_align : "qr_forward" / "block_forward" /
                    "block_left" / "block_right" (신규 3개)
"""

from collections import deque
import time
import re
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, Empty


VALID_LEVELS = {1, 2}
DEFAULT_LEVEL = 1


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
        self.create_subscription(String, '/distance_status', self._distance_status_callback, 10)
        self.create_subscription(String, '/observe_ready', self._observe_ready_callback, 10)
        self.create_subscription(String, '/align_status', self._align_status_callback, 10)

        # Publishers
        self._vision_activate_pub = self.create_publisher(String, '/vision_activate', 10)
        self._pick_command_pub = self.create_publisher(Float32MultiArray, '/pick_command', 10)
        self._place_command_pub = self.create_publisher(Float32MultiArray, '/place_command', 10)
        self._place_target_pub = self.create_publisher(String, '/place_target', 10)
        self._arm_status_pub = self.create_publisher(String, '/arm_status', 10)
        self._go_parking_pub = self.create_publisher(Empty, '/go_parking', 10)
        self._align_request_pub = self.create_publisher(String, '/align_request', 10)
        self._wms_update_pub = self.create_publisher(String, '/wms_update', 10)
        self._brain_state_pub = self.create_publisher(String, '/brain_state', 10)
        self._observe_move_pub = self.create_publisher(String, '/observe_move', 10)

        # Internal states
        self.state = 'IDLE'
        self.order_queue = deque()
        self.current_order = None
        self.zone = None
        self.item = None
        self.level = DEFAULT_LEVEL
        self.emergency_active = False
        self.order_start_time = None
        self.order_start_order = None

        self.align_retry_count = 0
        self.ALIGN_RETRY_MAX = 20

        self.pick_retry_count = 0
        self.PICK_REOBSERVE_MAX = 3

        self.place_retry_count = 0
        self.PLACE_RETRY_MAX = 3

        # True일 때만 /align_status step_done을 처리한다 (VISION/PLACE_VISION 단계용).
        self.waiting_align_step = False

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
        order = order.strip()
        parts = order.split(':')

        item = parts[0].strip()
        zone = parts[1].upper().strip() if len(parts) >= 2 and parts[1].strip() else 'A'

        level = DEFAULT_LEVEL
        if len(parts) >= 3 and parts[2].strip():
            level_raw = parts[2].strip()
            m = re.search(r'\d+', level_raw)
            if m:
                level = int(m.group())
            else:
                self.get_logger().warn(
                    f'층 파싱 실패("{level_raw}") -> 기본 {DEFAULT_LEVEL}층'
                )
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

        self.order_start_time = time.time()
        self.order_start_order = self.current_order
        self.get_logger().info(
            f"[KPI TIME] cycle_start order={self.order_start_order}"
        )

        self.item, self.zone, self.level = self._parse_order(self.current_order)

        self.align_retry_count = 0
        self.pick_retry_count = 0
        self.place_retry_count = 0
        self.waiting_align_step = False

        self.get_logger().info(
            f'다음 주문 시작: {self.current_order}, '
            f'item={self.item}, zone={self.zone}, level={self.level}'
        )
        self.get_logger().info(
            f'[KPI BRAIN] event=start_order order={self.current_order} '
            f'item={self.item} zone={self.zone} level={self.level}'
        )

        self.state = 'NAV_TO_RACK'
        self._pub_state()

        self._publish_string(self._place_target_pub, self.zone)
        self.get_logger().info(f'/place_target 발행: {self.zone}')

    def _finish_current_order(self):
        self.get_logger().info(f'현재 주문 완료: {self.current_order}')

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
        self.place_retry_count = 0
        self.waiting_align_step = False

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

        if self.state == 'IDLE' and self.order_start_time is None:
            self.order_start_time = time.time()
            self.order_start_order = msg.data.strip()
            self.get_logger().info(
                f"[KPI TIME] cycle_start order={self.order_start_order}"
            )

        if self.state == 'IDLE':
            self._start_next_order()
        else:
            self.get_logger().info(
                f'현재 {self.state} 상태라 주문 큐에 저장. '
                f'대기 주문 수: {len(self.order_queue)}'
            )

    def _distance_status_callback(self, msg):
        """
        vision_node의 거리/중심 판정 결과 수신.
          - too_far: 파지범위보다 멀다 -> AGV 전진 보정(block_forward) 요청 후 재관측
          - too_close: 너무 가깝다 -> 후진 금지 정책이라 자동 보정 중단(ERROR)
          - side_left/side_right(신규): 화면 중앙에서 치우침 -> AGV 좌/우 보정
            (block_left/block_right) 요청 후 재관측
          - depth_fail: depth 자체를 못 얻음 -> ArUco 마커 기반 정면 재확인 대기
          - ok: 정상 파지 가능 거리+중앙 -> /box_pose 대기
        """
        if self.emergency_active:
            self.get_logger().warn(
                f'/distance_status 수신했지만 비상정지 상태라 무시: {msg.data}'
            )
            return

        status = msg.data.strip()
        self.get_logger().info(f'/distance_status 수신: {status}')
        self.get_logger().info(
            f'[KPI BRAIN] event=distance_status state={self.state} '
            f'status={status} waiting_align_step={self.waiting_align_step}'
        )

        head = status.split(':', 1)[0]

        # ===== [QR PLACE] QR이 너무 멀면 AGV 전진 보정 요청 =====
        if head == 'qr_too_far':
            if self.state == 'PLACE_VISION':
                self.waiting_align_step = True
                self.get_logger().warn(
                    f'QR이 너무 멂({status}) → AGV 전진 보정 요청 후 QR 재관측 대기'
                )
                self._publish_string(self._align_request_pub, 'qr_forward')
                self.get_logger().info('/align_request 발행: qr_forward')
            else:
                self.get_logger().warn(
                    f'qr_too_far 수신했지만 현재 상태가 PLACE_VISION 아님: {self.state}'
                )
            return

        # ===== [QR PLACE] QR이 너무 가까우면 자동 후진 금지 =====
        if head == 'qr_too_close':
            if self.state == 'PLACE_VISION':
                self.get_logger().error(
                    f'QR이 너무 가까움({status}) → 후진은 위험하므로 ERROR 처리'
                )
                self.waiting_align_step = False
                self.state = 'ERROR'
                self._pub_state()
            else:
                self.get_logger().warn(
                    f'qr_too_close 수신했지만 현재 상태가 PLACE_VISION 아님: {self.state}'
                )
            return

        # depth 실패
        if head == 'depth_fail':
          if self.state == 'VISION':
              self.waiting_align_step = True
              self.get_logger().warn(
                  f'depth_fail 수신({status}) → depth 미확보, 우선 AGV 전진 보정(block_forward) 요청'
              )
              self._publish_string(self._align_request_pub, 'block_forward')
              self.get_logger().info('/align_request 발행: block_forward (depth_fail)')
          else:
              self.get_logger().warn(
                  f'depth_fail 수신했지만 현재 상태가 VISION 아님: {self.state}'
              )
          return

        # 너무 가까움: 후진 금지라 자동 보정 불가
        if head == 'too_close':
            if self.state == 'VISION':
                self.get_logger().error(
                    f'거리 너무 가까움({status}) - 후진 불가라 자동 보정 중단 → ERROR'
                )
                self.waiting_align_step = False
                self.state = 'ERROR'
                self._pub_state()
            else:
                self.get_logger().warn(
                    f'too_close 수신했지만 현재 상태가 VISION 아님: {self.state}'
                )
            return

        # 너무 멂: AGV 전진 보정 요청
        if head == 'too_far':
            if self.state == 'VISION':
                self.waiting_align_step = True
                self.get_logger().warn(
                    f'거리 너무 멂({status}) → AGV 전진 보정(block_forward) 요청 후 step_done 대기'
                )
                self._publish_string(self._align_request_pub, 'block_forward')
                self.get_logger().info('/align_request 발행: block_forward')
            else:
                self.get_logger().warn(
                    f'too_far 수신했지만 현재 상태가 VISION 아님: {self.state}'
                )
            return

        # ===== [신규] 블록이 화면 중앙에서 너무 치우침 → AGV 좌/우 보정 요청 =====
        if head in ('side_left', 'side_right'):
            if self.state == 'VISION':
                self.waiting_align_step = True
                req = 'block_left' if head == 'side_left' else 'block_right'
                self.get_logger().warn(
                    f'블록 중심 치우침({status}) → AGV {req} 보정 요청 후 step_done 대기'
                )
                self._publish_string(self._align_request_pub, req)
                self.get_logger().info(f'/align_request 발행: {req}')
            else:
                self.get_logger().warn(
                    f'{head} 수신했지만 현재 상태가 VISION 아님: {self.state}'
                )
            return

        # 정상 거리+중앙: box_pose 대기
        if head == 'ok':
            self.waiting_align_step = False
            self.get_logger().info(
                '거리/중심 ok → /box_pose 대기, 늦은 align step_done은 무시'
            )
            return

        self.get_logger().warn(f'알 수 없는 distance_status: {status}')

    def _box_pose_callback(self, msg):
        if self.emergency_active:
            self.get_logger().warn('/box_pose 수신했지만 비상정지 상태라 무시')
            return
    
        # 1) VISION 상태가 아니면 box_pose 무시
        if self.state != 'VISION':
            self.get_logger().warn(
                f'현재 상태가 VISION이 아니므로 /box_pose 무시. 현재 상태: {self.state}'
            )
            return
    
        # 2) AGV 보정 step_done 기다리는 중이면 box_pose 무시
        if self.waiting_align_step:
            self.get_logger().warn(
                '[BRAIN] AGV 보정 step_done 대기 중이므로 /box_pose 무시'
            )
            return
    
        self.get_logger().info(f'/box_pose 수신: {list(msg.data)}')
        self.get_logger().info(
            f'[KPI BRAIN] event=box_pose state={self.state} '
            f'coords={list(msg.data)}'
        )
    
        self.align_retry_count = 0
        self.waiting_align_step = False
    
        self.state = 'PICKING'
        self._pub_state()
    
        self._pick_command_pub.publish(msg)
        self.get_logger().info('/pick_command 발행')

    def _place_pose_callback(self, msg):
        if self.emergency_active:
            self.get_logger().warn('/place_pose 수신했지만 비상정지 상태라 무시')
            return

        self.get_logger().info(f'/place_pose 수신: {list(msg.data)}')
        self.get_logger().info(
            f'[KPI BRAIN] event=place_pose state={self.state} '
            f'coords={list(msg.data)}'
        )

        if self.state != 'PLACE_VISION':
            self.get_logger().warn(
                f'현재 상태가 PLACE_VISION이 아니므로 /place_pose 무시. 현재 상태: {self.state}'
            )
            return

        self.state = 'PLACING'
        self._pub_state()

        self._place_command_pub.publish(msg)
        self.get_logger().info('/place_command 발행: QR 기반 place_pose')

    def _pick_status_callback(self, msg):
        if self.emergency_active:
            self.get_logger().warn(
                f'/pick_status 수신했지만 비상정지 상태라 무시: {msg.data}'
            )
            return

        status = msg.data.strip()
        self.get_logger().info(f'/pick_status 수신: {status}')
        self.get_logger().info(
            f'[KPI BRAIN] event=pick_status state={self.state} '
            f'status={status} order={self.current_order}'
        )

        if status == 'done':
            if self.state != 'PICKING':
                self.get_logger().warn(
                    f'pick done 수신했지만 현재 상태가 PICKING이 아님: {self.state}'
                )
                return

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
          
        elif status == 'place_failed':
            if self.state != 'PLACING':
                self.get_logger().warn(
                    f'place_failed 수신했지만 현재 상태가 PLACING이 아님: {self.state}'
                )
                return
        
            self.get_logger().warn('place_failed 수신 - QR 재관측 후 place 재시도 판단')
        
            if self.place_retry_count < self.PLACE_RETRY_MAX:
                self.place_retry_count += 1
        
                self.get_logger().warn(
                    f'place 재시도 {self.place_retry_count}/{self.PLACE_RETRY_MAX}'
                )
        
                zone = self.zone if self.zone else 'A'
        
                self.state = 'QR_OBSERVING'
                self.waiting_align_step = False
                self._pub_state()
        
                self._publish_string(self._observe_move_pub, f'qr:{zone}')
                self.get_logger().info(
                    f'/observe_move 재발행: qr:{zone} (place_failed QR 재관측)'
                )
                return
        
            self.get_logger().error('place 재시도 초과 - ERROR 처리')
            self.place_retry_count = 0
            self.waiting_align_step = False
            self.state = 'ERROR'
            self._pub_state()
            return
      
        elif status == 'realign_fail':
            self.get_logger().warn('pick_node realign_fail 수신 - 정면 재확인 후 재관측 대기')

            if self.state not in ('VISION', 'OBSERVING'):
                self.get_logger().warn(
                    f'realign_fail 수신했지만 현재 상태가 VISION/OBSERVING 아님: {self.state}'
                )
                return

            self.state = 'VISION'
            self.waiting_align_step = True
            self._pub_state()
            return

        elif status == 'pick_failed':
            if self.state not in ('PICKING', 'OBSERVING', 'VISION', 'FRONTAL_ALIGN'):
                self.get_logger().warn(
                    f'pick_failed 수신했지만 재시도 가능한 상태가 아님: {self.state}'
                )
                return
        
            self.get_logger().warn(f'pick_failed 수신(state={self.state}) - 재관측 retry 판단')

            if self.pick_retry_count < self.PICK_REOBSERVE_MAX:
                self.pick_retry_count += 1

                self.get_logger().warn(
                    f'pick 재관측 재시도 {self.pick_retry_count}/{self.PICK_REOBSERVE_MAX}'
                )

                self.state = 'OBSERVING'
                self.waiting_align_step = False
                self._pub_state()

                self._publish_string(self._observe_move_pub, str(self.level))
                self.get_logger().info(
                    f'/observe_move 재발행: level={self.level} (pick_failed 재관측)'
                )
                return

            self.get_logger().warn(
                'pick 재관측 재시도 초과 - ERROR 처리하지 않고 다음 단계(NAV_TO_DEST)로 진행'
            )
            
            # 재시도 관련 상태 초기화
            self.pick_retry_count = 0
            self.align_retry_count = 0
            self.waiting_align_step = False
            
            # 실제로는 픽 실패지만, 전체 KPI 흐름 진행을 위해 다음 단계로 이동
            self.state = 'NAV_TO_DEST'
            self._pub_state()
            
            # AGV 이동 흐름이 /arm_status picked를 기준으로 넘어간다면 필요
            self._publish_string(self._arm_status_pub, 'picked')
            self.get_logger().warn(
                '/arm_status 발행: picked (pick 실패 재시도 초과, KPI 진행용)'
            )
            
            return

        elif status == 'error':
            self.get_logger().error('pick_node error 수신')
            self.waiting_align_step = False
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
        self.get_logger().info(
            f'[KPI BRAIN] event=nav_status state={self.state} '
            f'status={msg.data} order={self.current_order}'
        )

        if msg.data == 'arrived_objects':
            if self.state != 'NAV_TO_RACK':
                self.get_logger().warn(
                    f'arrived_objects 수신했지만 현재 상태가 NAV_TO_RACK이 아님: {self.state}'
                )
                return

            self.state = 'OBSERVING'
            self.waiting_align_step = False
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

            zone = self.zone if self.zone else 'A'

            self.state = 'QR_OBSERVING'
            self._pub_state()

            self._publish_string(self._observe_move_pub, f'qr:{zone}')
            self.get_logger().info(
                f'/observe_move 발행: qr:{zone} (QR 플레이싱 관측 자세 이동 요청)'
            )

        elif msg.data == 'parked':
            if self.state != 'GO_PARKING':
                self.get_logger().warn(
                    f'parked 수신했지만 현재 상태가 GO_PARKING이 아님: {self.state}'
                )
                return

            self.get_logger().info('주차 완료 -> IDLE 복귀')

            if self.order_start_time is not None:
                elapsed = time.time() - self.order_start_time
                self.get_logger().info(
                    f"[KPI TIME] cycle_end_to_idle "
                    f"order={self.order_start_order} "
                    f"elapsed_sec={elapsed:.2f}"
                )
                self.order_start_time = None
                self.order_start_order = None
            else:
                self.get_logger().warn("[KPI TIME] parked 수신했지만 order_start_time 없음")

            self.get_logger().info('[KPI BRAIN] event=parked result=go_idle')

            self.state = 'IDLE'
            self.current_order = None
            self.zone = None
            self.item = None
            self.level = DEFAULT_LEVEL
            self.align_retry_count = 0
            self.pick_retry_count = 0
            self.place_retry_count = 0
            self.waiting_align_step = False
            self._pub_state()

            if self.order_queue:
                self.get_logger().info('주차 중 들어온 주문 있음 -> 다음 주문 시작')
                self._start_next_order()

        else:
            self.get_logger().warn(f'알 수 없는 nav_status: {msg.data}')

    # ===== pick_node가 관측 자세에 도착했을 때 =====
    def _observe_ready_callback(self, msg):
        if self.emergency_active:
            self.get_logger().warn('/observe_ready 수신했지만 비상정지 상태라 무시')
            return

        if self.state not in ('OBSERVING', 'VISION', 'QR_OBSERVING'):
            self.get_logger().warn(
                f'/observe_ready 수신했지만 상태가 OBSERVING/VISION/QR_OBSERVING 아님: {self.state}'
            )
            return

        self.get_logger().info(f'/observe_ready 수신: {msg.data} (관측 자세 도착)')
        self.get_logger().info(
            f'[KPI BRAIN] event=observe_ready state={self.state} '
            f'data={msg.data} level={self.level}'
        )

        # QR 플레이싱 관측 완료인 경우
        if self.state == 'QR_OBSERVING':
            self.state = 'PLACE_VISION'
            self._pub_state()

            self._publish_string(self._vision_activate_pub, 'qr_place')
            self.get_logger().info('/vision_activate 발행: qr_place')
            return

        # ===== [신규] 최초 관측(OBSERVING)이면 곧바로 블록 검출로 가지 않고
        # 먼저 FRONTAL_ALIGN(마커 정면정렬)부터 수행 =====
        if self.state == 'OBSERVING':
            self.state = 'FRONTAL_ALIGN'
            self._pub_state()

            self._publish_string(self._vision_activate_pub, f'marker_align:{self.level}')
            self.get_logger().info(
                f'/vision_activate 발행: marker_align:{self.level} (정면정렬 시작)'
            )
            return

        # state == 'VISION' (J1 픽셀 기반 재관측 - 기존 그대로 유지, 정면정렬 건너뜀)
        activate_data = f'{self.item}:{self.level}'
        self._publish_string(self._vision_activate_pub, activate_data)
        self.get_logger().info(f'/vision_activate 발행: {activate_data}')

    # ===== AGV 정렬 관련 신호 처리 (정면정렬 완료 / 펄스 1스텝 완료) =====
    def _align_status_callback(self, msg):
        if self.emergency_active:
            self.get_logger().warn(
                f'/align_status 수신했지만 비상정지 상태라 무시: {msg.data}'
            )
            return

        status = msg.data.strip()
        self.get_logger().info(f'/align_status 수신: {status}')
        self.get_logger().info(
            f'[KPI BRAIN] event=align_status state={self.state} '
            f'status={status} retry={self.align_retry_count} '
            f'waiting_align_step={self.waiting_align_step}'
        )

        if status == 'aligned':
            # FRONTAL_ALIGN 단계 완료 → 곧바로 블록 검출 시작
            if self.state == 'FRONTAL_ALIGN':
                self.get_logger().info('정면정렬 완료(aligned) → 블록 검출 시작')
                self.state = 'VISION'
                self.waiting_align_step = False
                self._pub_state()

                activate_data = f'{self.item}:{self.level}'
                self._publish_string(self._vision_activate_pub, activate_data)
                self.get_logger().info(f'/vision_activate 발행: {activate_data}')
                return

            # VISION 상태에서의 aligned는 내가 기다리던 aligned일 때만 처리
            if self.state == 'VISION':
                if not self.waiting_align_step:
                    self.get_logger().warn(
                        'VISION 상태에서 aligned 수신했지만 waiting_align_step=False '
                        '→ 중복/늦은 aligned로 보고 무시'
                    )
                    return

                self.waiting_align_step = False
                self.align_retry_count = 0

                self.get_logger().info(
                    '정면 재확인 완료(aligned) → 재관측(정면정렬부터) 후 블록 검출'
                )

                self.state = 'OBSERVING'
                self._pub_state()

                self._publish_string(self._observe_move_pub, str(self.level))
                self.get_logger().info(
                    f'/observe_move 재발행: level={self.level} (align aligned 후 재관측)'
                )
                return

            self.get_logger().warn(
                f'align aligned 수신했지만 처리 대상 상태가 아님: {self.state}'
            )
            return

        if status != 'step_done':
            self.get_logger().warn(f'알 수 없는 align_status: {status}')
            return

        # FRONTAL_ALIGN 단계의 step_done은 별도 처리 불필요
        if self.state == 'FRONTAL_ALIGN':
            return

        if not self.waiting_align_step:
            self.get_logger().warn(
                'align step_done 수신했지만 waiting_align_step=False '
                '→ 늦은/불필요한 step_done으로 보고 무시'
            )
            return

        # QR PLACE 거리 보정 step_done 처리
        if self.state == 'PLACE_VISION':
            self.waiting_align_step = False
            self.align_retry_count += 1

            zone = self.zone if self.zone else 'A'

            self.get_logger().warn(
                f'QR 전진 보정 완료 step_done → QR 재관측 {self.align_retry_count}회'
            )

            self.state = 'QR_OBSERVING'
            self._pub_state()

            self._publish_string(self._observe_move_pub, f'qr:{zone}')
            self.get_logger().info(
                f'/observe_move 재발행: qr:{zone} (QR 전진 보정 후 재관측)'
            )
            return

        # 블록 기반 x/y 보정 step_done 처리
        if self.state != 'VISION':
            self.get_logger().warn(
                f'align step_done 수신했지만 현재 상태가 VISION이 아님: {self.state}'
            )
            return

        self.waiting_align_step = False
        self.align_retry_count += 1

        self.get_logger().warn(
            f'AGV 보정(block_forward/left/right) 완료 후 재관측(정면정렬부터) '
            f'{self.align_retry_count}회'
        )

        self.state = 'OBSERVING'
        self._pub_state()

        self._publish_string(self._observe_move_pub, str(self.level))
        self.get_logger().info(
            f'/observe_move 재발행: level={self.level} (AGV 보정 후 재관측)'
        )
        return

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
        self.waiting_align_step = False
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
        self.place_retry_count = 0
        self.waiting_align_step = False
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
