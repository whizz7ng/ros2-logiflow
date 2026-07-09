#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agv_align_node.py  -  AGV 정렬 보정 노드 (층별 목표)

[역할]
  vision이 발행한 마커의 AGV 기준 좌표(/marker_agv_pose)를 받아,
  해당 층의 정상 정차 목표값과 비교해서 얼마나 벗어났는지 계산하고,
  메카넘 cmd_vel(geometry_msgs/Twist)을 /agv_align 으로 발행한다.
  nav(동료)가 /agv_align 을 받아 AGV를 움직인다.

[입력]  /marker_agv_pose  Float32MultiArray [level, Lx, Ly, Rx, Ry] (mm, AGV base 기준)
                          안 보이는 마커는 NaN. level(1 또는 2)로 층별 목표 선택.
[출력]  /agv_align        geometry_msgs/Twist (표준 cmd_vel 형식)
                          linear.x=앞뒤, linear.y=좌우(메카넘), angular.z=회전

[추가 출력] /align_status  String
                          "step_done" : AGV 보정 이동 1회가 끝났고 정지 명령까지 보냈음.
                          brain_node가 이 신호를 받아 다시 관측 자세부터 재측정한다.

[보정 방식 - 방식 2, 층별]
  각 층 관찰 자세에서 정상 정차일 때 마커 AGV 좌표를 목표(TARGET[level])로 삼고,
  현재 마커가 목표에서 벗어난 만큼 비례해서 cmd_vel.

[중요 변경]
  기존처럼 계속 움직이면서 계속 마커 좌표를 쓰지 않고,
  "정지 상태에서 마커 측정 → 짧게 이동 → 정지 → 다시 관측" 방식으로 동작한다.
  이유: AGV 이동 중 카메라 흔들림/차체 떨림 때문에 ArUco 좌표가 불안정할 수 있기 때문.

[실측 필요]
  TARGET 값은 각 층 관찰 자세 + 정상 정차에서 /marker_agv_pose 값을 echo 해서 입력한다.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist


SINGLE_MARKER_YAW_SEC = 0.25
SINGLE_MARKER_WZ = 0.41

# 방향이 반대로 돌면 이 값만 -1.0으로 바꾸면 됨
SIGN_SINGLE_MARKER_YAW = 1.0

# ===== [층별 목표값 - 실측 후 채울 것] =====
# 각 층 관찰 자세 + 정상 정차(작업 범위 안, 정면)일 때 마커 AGV 좌표 (mm).
# vision /marker_agv_pose 를 각 층 정상 정차에서 echo 해서 넣는다.
# 형식: {층: {'lx':, 'ly':, 'rx':, 'ry':}}
TARGET = {
    1: {'lx': 400.7, 'ly': 113.0, 'rx': 483.9, 'ry': -125.9},   # 1층 (실측)
    2: {'lx': 400.0, 'ly': 121.3, 'rx': 491.0, 'ry': -122.2},   # 2층 (실측)
}

# ===== 제어 게인 =====
# 현재 bridge/safety_filter 구조에서는 속도 크기보다 부호 판단용에 가까움
GAIN_X   = 0.0008
GAIN_Y   = 0.0008
GAIN_YAW = 0.004

# ===== 동료 bridge 제한에 맞춘 출력 속도 =====
# bridge에서 max_vx=0.030, max_vy=0.030, max_wz=0.150으로 잘림
ALIGN_VX = 0.08
ALIGN_VY = 0.08
# safety_filter의 INPLACE_SMALL_TURN 감지 범위가 0.010~0.080이므로
# yaw는 일부러 작게 오래 보냄
ALIGN_WZ = 0.41

# 부호가 반대로 움직이면 여기만 -1.0으로 바꾸면 됨
SIGN_X = 1.0
SIGN_Y = 1.0
SIGN_YAW = 1.0

# ===== 정렬 완료 허용 오차 =====
TOL_XY  = 10.0
TOL_YAW = 8.0

STOP_REPEAT = 3

# ===== 축별 펄스 시간 =====
PULSE_X_SEC   = 0.50
PULSE_Y_SEC   = 0.50
PULSE_YAW_SEC = 0.50

CMD_HZ = 20


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _apply_min_max(v, min_v, max_v):
    """
    계산된 속도 v가 0이 아니면,
    최소 구동 속도 이상 / 최대 제한 이하로 맞춘다.
    """
    if abs(v) < 1e-6:
        return 0.0

    sign = 1.0 if v > 0 else -1.0
    mag = abs(v)

    if mag < min_v:
        mag = min_v

    if mag > max_v:
        mag = max_v

    return sign * mag


class AgvAlignNode(Node):
    def __init__(self):
        super().__init__('agv_align_node')

        self._align_pub = self.create_publisher(Twist, '/agv_align', 10)
        self._align_status_pub = self.create_publisher(String, '/align_status', 10)

        self.create_subscription(
            Float32MultiArray, '/marker_agv_pose', self._marker_callback, 10
        )

        self._aligned_stop_sent = 0

        # ===== [펄스 이동 상태] =====
        self.active_cmd = Twist()
        self.cmd_until = 0.0
        self._step_active = False
        self._step_done_sent = False

        # 20Hz로 active_cmd를 반복 발행하다가 시간이 지나면 정지
        self.align_timer = self.create_timer(1.0 / CMD_HZ, self._timer_callback)

        self.get_logger().info('agv_align_node 시작 - /marker_agv_pose → /agv_align')
        self.get_logger().info(
            '보정 방식: marker 1회 측정 → 짧게 이동 → 정지 → /align_status step_done'
        )

    def _timer_callback(self):
        """
        marker_callback에서 계산된 active_cmd를 pulse_sec 동안 반복 발행한다.
        시간이 끝나면 Twist() 정지를 발행하고, /align_status step_done을 1회 발행한다.
        """
        # 이동 중인 step이 없으면 아무것도 발행하지 않음
        if not self._step_active:
            return

        now = time.time()

        if now < self.cmd_until:
            self._align_pub.publish(self.active_cmd)
            return

        # pulse 시간이 끝났으면 정지 명령
        for _ in range(STOP_REPEAT):
            self._align_pub.publish(Twist())

        self._step_active = False
        self._step_done_sent = True

        msg = String()
        msg.data = 'step_done'
        self._align_status_pub.publish(msg)
        self.get_logger().info('/align_status 발행: step_done')

        # 방금까지 움직이던 step이 끝난 순간에만 step_done 발행
        if self._step_active and not self._step_done_sent:
            self._step_active = False
            self._step_done_sent = True

            msg = String()
            msg.data = 'step_done'
            self._align_status_pub.publish(msg)
            self.get_logger().info('/align_status 발행: step_done')

    def _marker_callback(self, msg: Float32MultiArray):
        data = list(msg.data)
        if len(data) != 5:
            self.get_logger().warn(f'/marker_agv_pose 5개 아님: {len(data)}')
            return

        level = int(round(data[0]))
        lx, ly, rx, ry = data[1], data[2], data[3], data[4]

        tgt = TARGET.get(level)
        if tgt is None:
            self.get_logger().error(f'층 {level} 목표 없음')
            return

        has_left  = not (math.isnan(lx) or math.isnan(ly))
        has_right = not (math.isnan(rx) or math.isnan(ry))

        if not has_left and not has_right:
            self.get_logger().warn('[정렬] 마커 둘 다 없음 - 정지')
            self._publish_stop()
            return

        # ---- 앞뒤(x), 좌우(y) 오차: 보이는 마커 평균 vs 목표 ----
        cur_x, tgt_x, cur_y, tgt_y = [], [], [], []
        if has_left:
            cur_x.append(lx)
            tgt_x.append(tgt['lx'])
            cur_y.append(ly)
            tgt_y.append(tgt['ly'])

        if has_right:
            cur_x.append(rx)
            tgt_x.append(tgt['rx'])
            cur_y.append(ry)
            tgt_y.append(tgt['ry'])

        err_x = (sum(cur_x) - sum(tgt_x)) / len(cur_x)   # +면 마커가 목표보다 멀리
        err_y = (sum(cur_y) - sum(tgt_y)) / len(cur_y)   # +면 마커가 목표보다 오른쪽

        # ---- 각도(yaw) 오차: 양쪽 보일 때만 ----
        err_yaw = 0.0
        if has_left and has_right:
            cur_dy = ry - ly
            tgt_dy = tgt['ry'] - tgt['ly']
            err_yaw = cur_dy - tgt_dy

        # ---- 정렬 완료 판정 ----
        aligned = (
            abs(err_x) < TOL_XY and
            abs(err_y) < TOL_XY and
            abs(err_yaw) < TOL_YAW
        )

        if aligned:
            if self._aligned_stop_sent < STOP_REPEAT:
                self._publish_stop()
                self._aligned_stop_sent += 1
                self.get_logger().info(
                    f'[정렬] 완료 L{level} '
                    f'(ex={err_x:.0f} ey={err_y:.0f} eyaw={err_yaw:.0f}) - 정지'
                )

            # 이미 정렬 완료 상태임을 brain에 알림
            msg = String()
            msg.data = 'aligned'
            self._align_status_pub.publish(msg)
            self.get_logger().info('/align_status 발행: aligned (already aligned)')
            return

        self._aligned_stop_sent = 0

        # ---- 이미 pulse 중이면 새 marker 값은 무시 ----
        # 이동 중 카메라 흔들림으로 값이 계속 갱신되는 걸 막기 위함
        if self._step_active and time.time() < self.cmd_until:
            return
        
        # ---- 현재 bridge/safety_filter에 맞춘 축 분리 제어 ----
        # 중요:
        #   1) yaw는 yaw만 보냄
        #   2) x는 x만 보냄
        #   3) y는 y만 보냄
        #   4) x/y/yaw를 동시에 보내지 않음
        tw = Twist()
        axis = 'none'
        pulse_sec = 0.0
        
        # 0순위: 마커가 하나만 보일 때
        # 한쪽 마커만 보여도 해당 마커의 y 오차는 계산 가능하다.
        # y 오차가 크면 먼저 좌우 보정으로 양쪽 마커가 같이 보이게 만들고,
        # y가 어느 정도 맞았는데도 한쪽만 보이면 yaw 복구를 수행한다.
        if has_right and not has_left:
            if abs(err_y) >= TOL_XY:
                axis = 'single_marker_y'
        
                raw_vy = GAIN_Y * err_y
                if raw_vy > 0:
                    tw.linear.y = SIGN_Y * ALIGN_VY
                else:
                    tw.linear.y = -SIGN_Y * ALIGN_VY
        
                pulse_sec = PULSE_Y_SEC
        
                self.get_logger().warn(
                    f'[정렬] 오른쪽 마커만 보임 + y 오차 큼(err_y={err_y:.0f}) '
                    f'→ y축 보정 먼저'
                )
            else:
                axis = 'single_marker_yaw_left'
                tw.angular.z = SIGN_SINGLE_MARKER_YAW * SINGLE_MARKER_WZ
                pulse_sec = SINGLE_MARKER_YAW_SEC
        
                self.get_logger().warn(
                    '[정렬] 오른쪽 마커만 보임 + y 오차 작음 → 왼쪽 회전으로 yaw 복구'
                )
        
        elif has_left and not has_right:
            if abs(err_y) >= TOL_XY:
                axis = 'single_marker_y'
        
                raw_vy = GAIN_Y * err_y
                if raw_vy > 0:
                    tw.linear.y = SIGN_Y * ALIGN_VY
                else:
                    tw.linear.y = -SIGN_Y * ALIGN_VY
        
                pulse_sec = PULSE_Y_SEC
        
                self.get_logger().warn(
                    f'[정렬] 왼쪽 마커만 보임 + y 오차 큼(err_y={err_y:.0f}) '
                    f'→ y축 보정 먼저'
                )
            else:
                axis = 'single_marker_yaw_right'
                tw.angular.z = -SIGN_SINGLE_MARKER_YAW * SINGLE_MARKER_WZ
                pulse_sec = SINGLE_MARKER_YAW_SEC
        
                self.get_logger().warn(
                    '[정렬] 왼쪽 마커만 보임 + y 오차 작음 → 오른쪽 회전으로 yaw 복구'
                )

        # 1순위: 양쪽 마커가 모두 보일 때 기존 yaw 보정
        elif has_left and has_right and abs(err_yaw) >= TOL_YAW:
            axis = 'yaw'

            raw_wz = GAIN_YAW * err_yaw
            if raw_wz > 0:
                tw.angular.z = SIGN_YAW * ALIGN_WZ
            else:
                tw.angular.z = -SIGN_YAW * ALIGN_WZ

            pulse_sec = PULSE_YAW_SEC

        # 2순위: 앞뒤 x
        # 후진 금지 정책이므로 x축은 전진 방향일 때만 보정한다.
        # 후진이 필요한 상태면 이 align_node로는 해결 불가.
        elif abs(err_x) >= TOL_XY:
            raw_vx = GAIN_X * err_x

            if raw_vx > 0:
                axis = 'x'
                tw.linear.x = SIGN_X * ALIGN_VX
                pulse_sec = PULSE_X_SEC
            else:
                self._publish_stop()

                msg = String()
                msg.data = 'step_done'
                self._align_status_pub.publish(msg)

                self.get_logger().warn(
                    f'[정렬] 후진 필요 err_x={err_x:.0f} 하지만 후진 금지 → x 보정 불가'
                )
                return

        # 3순위: 좌우 y
        elif abs(err_y) >= TOL_XY:
            axis = 'y'

            raw_vy = GAIN_Y * err_y
            if raw_vy > 0:
                tw.linear.y = SIGN_Y * ALIGN_VY
            else:
                tw.linear.y = -SIGN_Y * ALIGN_VY

            pulse_sec = PULSE_Y_SEC

        else:
            # 거의 맞은 상태
            msg = String()
            msg.data = 'aligned'
            self._align_status_pub.publish(msg)
            
            self.get_logger().info(
                f'[정렬] 완료 L{level} '
                f'(ex={err_x:.0f} ey={err_y:.0f} eyaw={err_yaw:.0f}) - 정지'
            )
            self.get_logger().info('/align_status 발행: aligned (already aligned)')
            return
        
        # ---- pulse 시작 ----
        self.active_cmd = tw
        self.cmd_until = time.time() + pulse_sec
        self._step_active = True
        self._step_done_sent = False
        
        self.get_logger().info(
            f'[정렬] L{level} axis={axis} '
            f'err(x={err_x:.0f} y={err_y:.0f} yaw={err_yaw:.0f}) '
            f'→ pulse cmd(vx={tw.linear.x:.3f} vy={tw.linear.y:.3f} wz={tw.angular.z:.3f}) '
            f'for {pulse_sec:.2f}s'
        )

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
