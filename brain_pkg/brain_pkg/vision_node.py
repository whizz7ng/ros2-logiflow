#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_node.py  (YOLO 통합 버전 - 캘리브레이션 전)

D435i 한 대로 두 가지 역할을 모드 전환하며 수행:
- 블록 검출 모드: brain이 /vision_activate로 타겟 클래스명을 보내면
  YOLO로 그 블록을 찾아 카메라 3D 좌표를 계산, /box_pose로 발행
- QR 검증 모드: brain 상태가 NAV_TO_DEST면 구역 QR(A/B/C)을 읽어 /depth_qr로 발행

주의:
- 실행 전 venv 활성화 필요: source ~/yolo_env/bin/activate
- 캘리브레이션 전이라 _cam_to_arm 미구현. 팔 좌표는 더미값 발행 중.
  (카메라 좌표 cam_xyz는 실제 계산됨 - 로그로 검증 가능)

토픽:
[구독]
  /vision_activate : std_msgs/String    클래스명("red_cross" 등) 또는 "stop"
  /brain_state     : std_msgs/String    "NAV_TO_DEST"일 때 QR 모드
[발행]
  /box_pose        : std_msgs/Float32MultiArray   [x,y,z,rx,ry,rz]
  /depth_qr        : std_msgs/String              "A:0.90"
  /camera/image_compressed : sensor_msgs/CompressedImage  원본 영상 jpeg (대시보드용)
  /detected_image  : sensor_msgs/CompressedImage  YOLO 검출 결과 영상 jpeg
"""

from collections import deque

from ultralytics import YOLO

import cv2
import numpy as np
import pyrealsense2 as rs
from pyzbar import pyzbar

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import CompressedImage


WINDOW_SIZE = 10
VALID_ZONES = {'A', 'B', 'C'}

MODE_IDLE  = 'idle'
MODE_BLOCK = 'block'
MODE_QR    = 'qr'

# ===== 설정 =====
MODEL_PATH = '/home/zzz/best.pt'   # Jetson에 scp로 넣은 모델 경로
CONF_THRES = 0.55

# 클래스별 박스 색상 (BGR)
CLASS_COLORS = {
    'blue_pentagon': (255, 100, 0),
    'green_clover':  (0, 200, 0),
    'green_dome':    (0, 255, 150),
    'red_cross':     (0, 0, 255),
    'red_square':    (0, 100, 255),
}


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # 구독
        self.create_subscription(String, '/vision_activate', self._activate_callback, 10)
        self.create_subscription(String, '/brain_state',     self._state_callback,    10)

        # 발행
        self._box_pose_pub       = self.create_publisher(Float32MultiArray, '/box_pose',       10)
        self._qr_pub             = self.create_publisher(String,            '/depth_qr',       10)
        self._raw_image_pub      = self.create_publisher(CompressedImage,   '/camera/image_compressed', 10)
        self._detected_image_pub = self.create_publisher(CompressedImage,   '/detected_image', 10)

        self.mode        = MODE_IDLE
        self.target_item = None
        self.recent_qr   = deque(maxlen=WINDOW_SIZE)

        # YOLO 모델 로드
        self.get_logger().info(f'YOLO 모델 로드 중: {MODEL_PATH}')
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f'YOLO 클래스: {self.model.names}')

        # RealSense 초기화
        self.get_logger().info('RealSense 초기화 중...')
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        self.get_logger().info('vision_node 시작 (YOLO 통합 / 블록·QR)')

        self.timer = self.create_timer(0.033, self._process_frame)

    # ----------------------------------------------------------
    # 콜백
    # ----------------------------------------------------------
    def _activate_callback(self, msg: String):
        data = msg.data.strip()
        if data == 'stop':
            self.mode        = MODE_IDLE
            self.target_item = None
            self.get_logger().info('블록 검출 중지')
        else:
            self.target_item = data
            self.mode        = MODE_BLOCK
            self.get_logger().info(f'블록 검출 모드 - 타겟: {data}')

    def _state_callback(self, msg: String):
        if msg.data == 'NAV_TO_DEST':
            if self.mode != MODE_QR:
                self.mode = MODE_QR
                self.recent_qr.clear()
                self.get_logger().info('QR 검증 모드 진입')
        else:
            if self.mode == MODE_QR:
                self.mode = MODE_IDLE
                self.get_logger().info('QR 검증 모드 종료')

    # ----------------------------------------------------------
    # 프레임 처리
    # ----------------------------------------------------------
    def _process_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=200)
        except Exception:
            return

        aligned     = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            return

        img = np.asanyarray(color_frame.get_data())

        # /camera/image_compressed 항상 발행 (모드 무관)
        self._publish_compressed(self._raw_image_pub, img)

        if self.mode == MODE_IDLE:
            return

        if self.mode == MODE_BLOCK:
            self._detect_block(img, depth_frame)
        elif self.mode == MODE_QR:
            self._detect_qr(img)

    # ----------------------------------------------------------
    # CompressedImage 발행 헬퍼
    # ----------------------------------------------------------
    def _publish_compressed(self, publisher, img):
        try:
            ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                return
            msg = CompressedImage()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_color_optical_frame'
            msg.format = 'jpeg'
            msg.data   = buf.tobytes()
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'CompressedImage 발행 실패: {e}')

    # ----------------------------------------------------------
    # 블록 검출 (YOLO - 캘리브레이션 전 임시)
    # ----------------------------------------------------------
    def _detect_block(self, img, depth_frame):
        results = self.model(img, conf=CONF_THRES, verbose=False)

        # 타겟 클래스 찾기
        target_box = None
        for box in results[0].boxes:
            label = self.model.names[int(box.cls)]
            if label == self.target_item:
                target_box = box
                break

        if target_box is None:
            self.get_logger().warn(f'{self.target_item} 못 찾음, 다음 프레임 재시도')
            return  # mode 유지하고 재시도

        x1, y1, x2, y2 = map(int, target_box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # ===== 잘림 감지 =====
        H, W = img.shape[:2]
        margin = 5
        if x1 <= margin or y1 <= margin or x2 >= W - margin or y2 >= H - margin:
            self.get_logger().warn(
                f'{self.target_item} 잘림 감지(경계 접촉) - 픽업 보류, 재정렬 필요'
            )
            # TODO: brain에 재정렬 신호 보내기
            self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=True)
            return

        # ===== depth 읽기 (5x5 patch median) =====
        dist = self._get_robust_depth(depth_frame, cx, cy)
        if dist <= 0.0:
            self.get_logger().warn(f'{self.target_item} depth 측정 실패(0) - 재시도')
            return

        # ===== 카메라 3D 좌표 =====
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        cam_xyz = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], dist)

        self.get_logger().info(
            f'{self.target_item} 발견 | 픽셀=({cx},{cy}) '
            f'dist={dist:.3f}m cam_xyz={[round(v, 3) for v in cam_xyz]}'
        )

        # ===== 팔 좌표 변환 (캘리브레이션 전 - 더미) =====
        # TODO: arm_xyz = self._cam_to_arm(cam_xyz)
        arm_xyz = [200.0, 150.0, 80.0]
        self.get_logger().warn('  -> 캘리브레이션 전: 더미 팔좌표 발행')

        coords = list(arm_xyz) + [175.35, -1.1, -89.73]
        msg = Float32MultiArray()
        msg.data = [float(v) for v in coords]
        self._box_pose_pub.publish(msg)
        self.get_logger().info(f'/box_pose 발행: {coords}')

        self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=False)
        self.mode = MODE_IDLE

    def _get_robust_depth(self, depth_frame, cx, cy, k=2):
        """중심 주변 (2k+1)x(2k+1) patch에서 0 아닌 값들의 median."""
        vals = []
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                d = depth_frame.get_distance(cx + dx, cy + dy)
                if d > 0.0:
                    vals.append(d)
        if not vals:
            return 0.0
        vals.sort()
        return vals[len(vals) // 2]

    def _draw_and_publish(self, img, x1, y1, x2, y2, label, cut=False):
        color = (0, 0, 255) if cut else CLASS_COLORS.get(label, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tag = f'{label} (CUT)' if cut else label
        cv2.putText(img, tag, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        self._publish_compressed(self._detected_image_pub, img)

    # ----------------------------------------------------------
    # QR 검증
    # ----------------------------------------------------------
    def _detect_qr(self, img):
        zone = None
        decoded = pyzbar.decode(img)
        for obj in decoded:
            try:
                data = obj.data.decode('utf-8').strip().upper()
            except Exception:
                continue
            if data in VALID_ZONES:
                zone = data
                break

        self.recent_qr.append(zone)

        if zone is not None:
            valid = [z for z in self.recent_qr if z is not None]
            if valid:
                top_zone = max(set(valid), key=valid.count)
                rate = self.recent_qr.count(top_zone) / len(self.recent_qr)
                out = String()
                out.data = f'{top_zone}:{rate:.2f}'
                self._qr_pub.publish(out)
                self.get_logger().info(f'/depth_qr 발행: {out.data}')

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
