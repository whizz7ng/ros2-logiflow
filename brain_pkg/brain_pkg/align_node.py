#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agv_align_node.py  -  AGV 정렬 보정 노드 (정면정렬 전용 + 블록기반 x/y 보정)

[역할]
  1) 정면(yaw) 정렬: vision이 발행한 마커의 yaw(/marker_agv_pose)를 보고
     AGV가 랙을 정면으로 보고 있는지 확인/보정한다. 마커 하나만 보여도
     rvec 기반이라 가능하다. (관측 자세 도착 직후, 블록 검출 시작 전에 수행)
  2) x/y 보정: 더 이상 마커의 고정 TARGET 좌표를 쓰지 않는다. 대신
     block(파지 대상)이 파지범위(GRASP_DEPTH_RANGE) 밖이거나 화면 중앙에서
     너무 치우쳐 있으면, brain_node가 /align_request로 "block_forward" /
     "block_left" / "block_right" 를 요청하고 이 노드는 그 방향으로
     짧게 펄스 이동만 한다 (거리/방향 판단은 vision/brain 쪽에서 이미 끝냄).

[입력]
  /marker_agv_pose  Float32MultiArray
    [level, Lx, Ly, Rx, Ry, Lyaw, Ryaw] (mm/deg, AGV base 기준)
    위치(Lx,Ly,Rx,Ry)는 이제 안 씀(로그용으로만 남겨둠). yaw만 사용.
    안 보이는 마커는 해당 값 NaN.
  /align_request  String  "qr_forward" / "block_forward" / "block_left" / "block_right"

[출력]
  /agv_align     geometry_msgs/Twist (표준 cmd_vel 형식)
  /align_status  String  "step_done" (펄스 1회 종료) / "aligned" (정면정렬 완료, FRONTAL_ALIGN 전용)

[동작 방식]
  기존과 동일하게 "정지 상태에서 측정 → 짧게 이동 → 정지 → 다시 확인" 펄스 방식.
  이동 중 카메라 흔들림으로 값이 안 튀게, 펄스 진행 중에는 새 입력을 무시한다.

[실측 필요]
  YAW_TARGET: 각 층 정상 정차(정면)에서 vision이 계산한 마커별
  yaw(deg, AGV base 기준)를 /marker_agv_pose 에서 echo해서 넣는다.
  (Lyaw는 marker id 0, Ryaw는 marker id 1)
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist


MARKER_ID_LEFT = 0
MARKER_ID_RIGHT = 1

# ===== [정면(yaw) 정렬 기준값 - 실측 필요] =====
# 각 층 정상 정차에서 vision이 계산한 마커별 yaw(deg, AGV base 기준)를
# /marker_agv_pose 에서 echo해서 넣는다.

YAW_TARGET = {
    1: {MARKER_ID_LEFT: -89.5, MARKER_ID_RIGHT: -88.5}, #1층 실측(marker_agv_pose)
    2: {MARKER_ID_LEFT: -86.8, MARKER_ID_RIGHT: -89.4}, #2층 실측(marker_agv_pose)
}

# 정면정렬 허용 오차(deg) / 회전 게인(부호 판단용) / 펄스 시간
TOL_YAW_FRONTAL = 5.0
GAIN_YAW_FRONTAL = 0.01
PULSE_YAW_FRONTAL_SEC = 0.35

# 정면정렬 pulse 이후 안정화 시간
FRONTAL_SETTLE_SEC = 0.8

# yaw가 허용오차 안에 연속으로 몇 번 들어와야 aligned로 볼지
FRONTAL_STABLE_COUNT = 5

# 부호가 반대로 돌면 여기만 -1.0으로
SIGN_YAW = -1.0
ALIGN_WZ = 0.41

# ===== [블록 기반 x/y 보정] =====
# brain이 /align_request로 방향만 요청하면, 여기선 정해진 크기로 짧게 펄스.
BLOCK_FORWARD_VX = 0.08
BLOCK_FORWARD_SEC = 0.35

# ===== [신규] 마커가 둘 다 안 보일 때 - 보일 때까지 무제한 전진 =====
# (관측 거리가 짧게 설계돼 있어서, 마커가 안 보이는 주된 이유는
#  "너무 삐딱함"보다는 "너무 멀어서(nav가 일찍 멈춤)"인 경우가 실제로 있음.
#  횟수/시간 제한 없이 보일 때까지 계속 전진 pulse.)
NO_MARKER_FORWARD_VX = 0.08
NO_MARKER_FORWARD_SEC = 0.35

# 부호가 반대로 움직이면 아래만 바꿀 것.
SIGN_Y = 1.0
BLOCK_LEFT_SIGN  = -1.0   # block_left 요청 시 y 부호
BLOCK_RIGHT_SIGN =  1.0   # block_right 요청 시 y 부호
BLOCK_SIDE_VY = 0.08
BLOCK_SIDE_SEC = 0.5

# QR place 전용 전진 pulse (기존 그대로)
QR_FORWARD_VX = 0.08
QR_FORWARD_SEC = 0.35

STOP_REPEAT = 3
CMD_HZ = 20


class AgvAlignNode(Node):
    def __init__(self):
        super().__init__('agv_align_node')

        self._align_pub = self.create_publisher(Twist, '/agv_align', 10)
        self._align_status_pub = self.create_publisher(String, '/align_status', 10)

        self.create_subscription(
            Float32MultiArray, '/marker_agv_pose', self._marker_callback, 10
        )
        self.create_subscription(
            String, '/align_request', self._align_request_callback, 10
        )
        self.create_subscription(
            String, '/brain_state', self._brain_state_callback, 10
        )

        self._frontal_stop_sent = 0
        self.brain_state = 'IDLE'

        self.frontal_settle_until = 0.0
        self.frontal_stable_count = 0
        self.active_pulse_kind = None
      
        # ===== [펄스 이동 상태] =====
        self.active_cmd = Twist()
        self.cmd_until = 0.0
        self._step_active = False
        self._step_done_sent = False

        self.align_timer = self.create_timer(1.0 / CMD_HZ, self._timer_callback)

        self.get_logger().info(
            'agv_align_node 시작 - 정면정렬(마커yaw) + 블록기반 x/y 보정(align_request)'
        )

    def _timer_callback(self):
        if not self._step_active:
            return

        now = time.time()

        if now < self.cmd_until:
            self._align_pub.publish(self.active_cmd)
            return

        for _ in range(STOP_REPEAT):
            self._align_pub.publish(Twist())
        
        finished_kind = self.active_pulse_kind
        
        self._step_active = False
        self._step_done_sent = True
        self.active_pulse_kind = None
        
        # 정면 yaw 회전 pulse는 step_done을 brain에 보내지 말고,
        # 먼저 안정화 시간을 둔 뒤 marker_callback에서 aligned 여부를 판단하게 한다.
        if finished_kind == "frontal_yaw":
            self.frontal_settle_until = time.time() + FRONTAL_SETTLE_SEC
            self.frontal_stable_count = 0
            self.get_logger().info(
                f'[정렬] yaw pulse 종료 → {FRONTAL_SETTLE_SEC:.1f}s 안정화 대기'
            )
            return
        
        msg = String()
        msg.data = 'step_done'
        self._align_status_pub.publish(msg)
        self.get_logger().info('/align_status 발행: step_done')

    def _start_pulse(self, tw: Twist, pulse_sec: float, log_msg: str, kind: str = "step"):
        self.active_cmd = tw
        self.cmd_until = time.time() + pulse_sec
        self._step_active = True
        self._step_done_sent = False
        self.active_pulse_kind = kind
    
        self.get_logger().info(
            f'{log_msg} → pulse cmd(vx={tw.linear.x:.3f} vy={tw.linear.y:.3f} '
            f'wz={tw.angular.z:.3f}) for {pulse_sec:.2f}s'
        )

    def _brain_state_callback(self, msg: String):
        self.brain_state = msg.data.strip()
    
    # =========================
    # [블록 기반 x/y 보정] /align_request 처리
    # =========================
    def _align_request_callback(self, msg: String):
        data = msg.data.strip()

        if self._step_active:
            self.get_logger().warn(f'[ALIGN REQ] 이미 보정 이동 중이라 {data} 무시')
            return

        cmd = Twist()

        if data == 'qr_forward':
            cmd.linear.x = QR_FORWARD_VX
            self._start_pulse(cmd, QR_FORWARD_SEC, '[QR ALIGN] qr_forward 수신 → 전진 pulse')

        elif data == 'block_forward':
            cmd.linear.x = BLOCK_FORWARD_VX
            self._start_pulse(cmd, BLOCK_FORWARD_SEC, '[BLOCK ALIGN] block_forward 수신 → 전진 pulse')

        elif data == 'block_left':
            cmd.linear.y = SIGN_Y * BLOCK_LEFT_SIGN * BLOCK_SIDE_VY
            self._start_pulse(cmd, BLOCK_SIDE_SEC, '[BLOCK ALIGN] block_left 수신 → 좌측 pulse')

        elif data == 'block_right':
            cmd.linear.y = SIGN_Y * BLOCK_RIGHT_SIGN * BLOCK_SIDE_VY
            self._start_pulse(cmd, BLOCK_SIDE_SEC, '[BLOCK ALIGN] block_right 수신 → 우측 pulse')

        else:
            self.get_logger().warn(f'알 수 없는 /align_request: {data}')

    # =========================
    # [정면정렬] /marker_agv_pose 처리 - STAGE1만 수행
    # =========================
    def _marker_callback(self, msg: Float32MultiArray):
        if self.brain_state != 'FRONTAL_ALIGN':
            self.get_logger().info(
                f'/marker_agv_pose 수신했지만 brain_state={self.brain_state} '
                '→ 정면정렬 단계가 아니므로 무시'
            )
            return
    
        data = list(msg.data)
      
        if len(data) != 7:
            self.get_logger().warn(
                f'/marker_agv_pose 7개 아님(구버전 vision?): {len(data)} - 무시'
            )
            return

        level = int(round(data[0]))
        # 위치(Lx,Ly,Rx,Ry)는 더 이상 정렬 판단에 안 씀 - yaw만 사용
        lyaw, ryaw = data[5], data[6]

        yaw_tgt = YAW_TARGET.get(level)
        if yaw_tgt is None:
            self.get_logger().error(f'층 {level} YAW_TARGET 없음')
            return

        has_left_yaw = not math.isnan(lyaw)
        has_right_yaw = not math.isnan(ryaw)

        if not has_left_yaw and not has_right_yaw:
            # [변경] 마커가 둘 다 안 보이면 정지하지 않고, 보일 때까지
            # 무제한으로 전진 pulse를 반복한다. (관측 거리가 짧게 설계돼
            # 있어서, 안 보이는 주된 이유가 "너무 멀어서"인 경우가 실제로
            # 있다고 판단 - 삐딱함/좌우이탈이면 markers가 아예 안 보이기보다
            # 일부라도 보이는 경우가 많음)
            if self._step_active and time.time() < self.cmd_until:
                return
            tw = Twist()
            tw.linear.x = NO_MARKER_FORWARD_VX
            self._start_pulse(
                tw, NO_MARKER_FORWARD_SEC,
                f'[정렬] L{level} 마커 둘 다 안 보임 → 전진 pulse(무제한 반복)'
            )
            return

        # 이미 펄스 중이면 새 값 무시 (이동 중 흔들림으로 값 튀는 것 방지)
        if self._step_active and time.time() < self.cmd_until:
            return

        yaw_errs = []
        if has_left_yaw:
            yaw_errs.append(lyaw - yaw_tgt[MARKER_ID_LEFT])
        if has_right_yaw:
            yaw_errs.append(ryaw - yaw_tgt[MARKER_ID_RIGHT])
        err_yaw_frontal = sum(yaw_errs) / len(yaw_errs)

        now = time.time()

        # yaw pulse 직후에는 차체가 아직 흔들릴 수 있으므로 aligned 판정 금지
        if now < self.frontal_settle_until:
            remain = self.frontal_settle_until - now
            self.get_logger().info(
                f'[정렬] 안정화 대기 중 {remain:.2f}s - yaw 판정 보류'
            )
            return

        if abs(err_yaw_frontal) >= TOL_YAW_FRONTAL:
            self._frontal_stop_sent = 0
            tw = Twist()
            raw_wz = GAIN_YAW_FRONTAL * err_yaw_frontal
            if raw_wz > 0:
                tw.angular.z = SIGN_YAW * ALIGN_WZ
            else:
                tw.angular.z = -SIGN_YAW * ALIGN_WZ

            self.frontal_stable_count = 0

            self._start_pulse(
                tw, PULSE_YAW_FRONTAL_SEC,
                f'[정렬] L{level} 정면정렬 err_yaw={err_yaw_frontal:.1f} (n={len(yaw_errs)})',
                kind="frontal_yaw"
            )
            return

        # ===== 정면정렬 완료 후보 =====
        self.frontal_stable_count += 1

        self.get_logger().info(
            f'[정렬] L{level} yaw 안정 후보 '
            f'err_yaw={err_yaw_frontal:.1f}, '
            f'stable={self.frontal_stable_count}/{FRONTAL_STABLE_COUNT}'
        )
        
        if self.frontal_stable_count < FRONTAL_STABLE_COUNT:
            return
        
        # 연속으로 충분히 안정됐을 때만 aligned 발행
        for _ in range(STOP_REPEAT):
            self._align_pub.publish(Twist())
        
        msg_out = String()
        msg_out.data = 'aligned'
        self._align_status_pub.publish(msg_out)
        self.get_logger().info('/align_status 발행: aligned (정면정렬 안정 완료)')
        
        self.frontal_stable_count = 0
        self._frontal_stop_sent = 0

    def _publish_stop(self):
        self.active_cmd = Twist()
        self.cmd_until = 0.0
        self._align_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = AgvAlignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
