#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qr_verify_node.py

뎁스카메라(D435i)로 구역 QR(A/B/C)을 읽어 검증/측정하는 노드.

역할:
- brain의 흐름에는 관여하지 않음 (백업/측정용)
- NAV_TO_DEST 상태일 때만 QR 검출
- 최근 프레임 검출 성공률을 신뢰도로 함께 발행

토픽:
[구독]
  /brain_state : std_msgs/String   (NAV_TO_DEST일 때만 검출 활성화)

[발행]
  /depth_qr : std_msgs/String      "A:0.90" 형식 (구역:검출성공률)
"""

from collections import deque

import numpy as np
import cv2
import pyrealsense2 as rs
from pyzbar import pyzbar

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# 검출 성공률 계산용 최근 프레임 수
WINDOW_SIZE = 10

# 유효한 구역 라벨
VALID_ZONES = {'A', 'B', 'C'}


class QRVerifyNode(Node):
    def __init__(self):
        super().__init__('qr_verify_node')

        # 구독: brain 상태
        self.create_subscription(String, '/brain_state', self._state_callback, 10)

        # 발행: 검증 결과
        self._qr_pub = self.create_publisher(String, '/depth_qr', 10)

        self.active = False
        self.recent = deque(maxlen=WINDOW_SIZE)  # 최근 검출 결과 (구역 or None)

        # RealSense 초기화
        self.get_logger().info('RealSense 초기화 중...')
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.get_logger().info('qr_verify_node 시작 - NAV_TO_DEST 대기중')

        # 주기적으로 프레임 처리 (30Hz 정도)
        self.timer = self.create_timer(0.033, self._process_frame)

    def _state_callback(self, msg: String):
        was_active = self.active
        self.active = (msg.data == 'NAV_TO_DEST')

        if self.active and not was_active:
            self.get_logger().info('NAV_TO_DEST 진입 - QR 검출 시작')
            self.recent.clear()
        elif not self.active and was_active:
            self.get_logger().info('NAV_TO_DEST 종료 - QR 검출 중지')

    def _process_frame(self):
        if not self.active:
            return

        # 프레임 받기
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        img = np.asanyarray(color_frame.get_data())

        # QR 검출
        zone = self._detect_qr(img)
        self.recent.append(zone)  # 성공 시 'A'/'B'/'C', 실패 시 None

        # 가장 많이 잡힌 구역 + 검출 성공률 계산
        if zone is not None:
            # 최근 윈도우에서 None 아닌 것 중 최빈값
            valid = [z for z in self.recent if z is not None]
            if valid:
                # 최빈 구역
                top_zone = max(set(valid), key=valid.count)
                # 검출 성공률 = 그 구역이 잡힌 비율 (전체 프레임 대비)
                rate = self.recent.count(top_zone) / len(self.recent)

                out = String()
                out.data = f'{top_zone}:{rate:.2f}'
                self._qr_pub.publish(out)
                self.get_logger().info(f'/depth_qr 발행: {out.data}')

    def _detect_qr(self, img):
        """
        이미지에서 QR 검출.
        성공 시 구역 문자('A'/'B'/'C'), 실패 시 None 반환.
        """
        decoded = pyzbar.decode(img)
        for obj in decoded:
            try:
                data = obj.data.decode('utf-8').strip().upper()
            except Exception:
                continue
            if data in VALID_ZONES:
                return data
        return None

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = QRVerifyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
