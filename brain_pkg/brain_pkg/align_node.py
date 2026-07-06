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

[보정 방식 - 방식 2, 층별]
  각 층 관찰 자세에서 정상 정차일 때 마커 AGV 좌표를 목표(TARGET[level])로 삼고,
  현재 마커가 목표에서 벗어난 만큼 비례해서 cmd_vel. 정렬되면 0 발행(정지).
  피드백: AGV 움직이면 마커도 움직이고 → vision 재발행 → 다시 계산 (수렴).

[미완] TARGET 값은 실측 후 채울 것.
  각 층 관찰 자세 + 정상 정차에서 /marker_agv_pose 값 재서 아래에 입력.
"""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist


# ===== [층별 목표값 - 실측 후 채울 것] =====
# 각 층 관찰 자세 + 정상 정차(작업 범위 안, 정면)일 때 마커 AGV 좌표 (mm).
# vision /marker_agv_pose 를 각 층 정상 정차에서 echo 해서 넣는다.
# 형식: {층: {'lx':, 'ly':, 'rx':, 'ry':}}
TARGET = {
    1: {'lx': 0.0, 'ly': 0.0, 'rx': 0.0, 'ry': 0.0},   # 1층 - 실측
    2: {'lx': 0.0, 'ly': 0.0, 'rx': 0.0, 'ry': 0.0},   # 2층 - 실측
}

# ===== 제어 게인 (벗어남 mm → 속도, 실측하며 튜닝) =====
GAIN_X   = 0.0008   # 앞뒤: mm당 m/s
GAIN_Y   = 0.0008   # 좌우: mm당 m/s
GAIN_YAW = 0.004    # 회전: (좌우마커 y차이 mm)당 rad/s

# ===== 속도 제한 (m/s, rad/s) =====
MAX_LIN = 0.08      # 앞뒤/좌우 최대 (처음엔 낮게, 부호 확인 후 올리기)
MAX_ANG = 0.20      # 회전 최대

# ===== 정렬 완료 허용 오차 =====
TOL_XY  = 10.0      # mm
TOL_YAW = 8.0       # mm (좌우마커 y차이)

STOP_REPEAT = 3     # 정렬 완료 후 정지 명령 반복


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class AgvAlignNode(Node):
    def __init__(self):
        super().__init__('agv_align_node')

        self._align_pub = self.create_publisher(Twist, '/agv_align', 10)
        self.create_subscription(
            Float32MultiArray, '/marker_agv_pose', self._marker_callback, 10
        )

        self._aligned_stop_sent = 0

        self.get_logger().info('agv_align_node 시작 - /marker_agv_pose → /agv_align')
        self.get_logger().warn(
            'TARGET 목표값이 아직 0. 각 층 정상 정차에서 /marker_agv_pose 재서 채울 것!'
        )

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
            cur_x.append(lx); tgt_x.append(tgt['lx'])
            cur_y.append(ly); tgt_y.append(tgt['ly'])
        if has_right:
            cur_x.append(rx); tgt_x.append(tgt['rx'])
            cur_y.append(ry); tgt_y.append(tgt['ry'])

        err_x = (sum(cur_x) - sum(tgt_x)) / len(cur_x)   # +면 마커가 목표보다 멀리
        err_y = (sum(cur_y) - sum(tgt_y)) / len(cur_y)   # +면 마커가 목표보다 오른쪽

        # ---- 각도(yaw) 오차: 양쪽 보일 때만 ----
        err_yaw = 0.0
        if has_left and has_right:
            cur_dy = ry - ly
            tgt_dy = tgt['ry'] - tgt['ly']
            err_yaw = cur_dy - tgt_dy

        # ---- 정렬 완료 판정 ----
        aligned = (abs(err_x) < TOL_XY and abs(err_y) < TOL_XY and abs(err_yaw) < TOL_YAW)
        if aligned:
            if self._aligned_stop_sent < STOP_REPEAT:
                self._publish_stop()
                self._aligned_stop_sent += 1
                self.get_logger().info(
                    f'[정렬] 완료 L{level} (ex={err_x:.0f} ey={err_y:.0f} eyaw={err_yaw:.0f}) - 정지'
                )
            return
        self._aligned_stop_sent = 0

        # ---- cmd_vel 계산 (부호는 실측하며 맞출 것) ----
        vx = _clamp(GAIN_X * err_x, -MAX_LIN, MAX_LIN)
        vy = _clamp(GAIN_Y * err_y, -MAX_LIN, MAX_LIN)
        wz = _clamp(GAIN_YAW * err_yaw, -MAX_ANG, MAX_ANG)

        tw = Twist()
        tw.linear.x = float(vx)
        tw.linear.y = float(vy)
        tw.angular.z = float(wz)
        self._align_pub.publish(tw)

        self.get_logger().info(
            f'[정렬] L{level} err(x={err_x:.0f} y={err_y:.0f} yaw={err_yaw:.0f}) '
            f'→ cmd(vx={vx:.3f} vy={vy:.3f} wz={wz:.3f})'
        )

    def _publish_stop(self):
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
