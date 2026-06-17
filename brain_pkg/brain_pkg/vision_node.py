#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_node.py

D435i 한 대로 두 가지 역할을 모드 전환하며 수행:
- 블록 검출 모드: brain이 /vision_activate로 색을 보내면 그 색 블록의
  3D 좌표를 /box_pose로 발행
- QR 검증 모드: brain 상태가 NAV_TO_DEST면 구역 QR(A/B/C)을 읽어
  /depth_qr로 발행 (백업/측정용, 흐름엔 관여 안 함)

토픽:
[구독]
  /vision_activate : std_msgs/String   "blue"/"red"/"green"/"stop"
  /brain_state     : std_msgs/String   "NAV_TO_DEST"일 때 QR 모드

[발행]
  /box_pose : std_msgs/Float32MultiArray   [x,y,z,rx,ry,rz]
  /depth_qr : std_msgs/String              "A:0.90"
"""

from collections import deque

# from ultralytics import YOLO   # YOLO 적용 시 주석 해제

import numpy as np
import pyrealsense2 as rs
from pyzbar import pyzbar

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray


WINDOW_SIZE = 10
VALID_ZONES = {'A', 'B', 'C'}

# 모드 정의
MODE_IDLE = 'idle'
MODE_BLOCK = 'block'   # 블록 검출
MODE_QR = 'qr'         # QR 검증


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
      
        # 구독
        self.create_subscription(String, '/vision_activate', self._activate_callback, 10)
        self.create_subscription(String, '/brain_state', self._state_callback, 10)

        # 발행
        self._box_pose_pub = self.create_publisher(Float32MultiArray, '/box_pose', 10)
        self._qr_pub = self.create_publisher(String, '/depth_qr', 10)

        self.mode = MODE_IDLE
        self.target_item = None    # target_color → target_item
        self.recent_qr = deque(maxlen=WINDOW_SIZE)

        # RealSense - 노드 살아있는 동안 계속 잡고 있음 (한 노드 독점이라 충돌 없음)
        self.get_logger().info('RealSense 초기화 중...')
        self.pipeline = rs.pipeline()
      
        # ===== YOLO 모델 (적용 시 주석 해제) =====
        # self.model = YOLO('/home/zzz/pj3_ws/src/brain_pkg/brain_pkg/blocks.pt')
      
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)

        # depth -> color 정렬
        self.align = rs.align(rs.stream.color)

        self.get_logger().info('vision_node 시작 (블록/QR 통합)')

        self.timer = self.create_timer(0.033, self._process_frame)

    # def _cam_to_arm(self, cam_xyz):
    #     """
    #     카메라 좌표 -> 로봇팔 좌표 변환.
    #     손-눈 캘리브레이션으로 구한 변환행렬 적용.
    #     TODO: 실측 변환행렬로 교체
    #     """
    #     import numpy as np
    #     # 예시: 단순 오프셋 (실제론 회전+이동 행렬 필요)
    #     R = np.eye(3)              # 회전행렬 (캘리브레이션값)
    #     t = np.array([0, 0, 0])    # 이동벡터 (캘리브레이션값)
    #     cam = np.array(cam_xyz)
    #     arm = R @ cam + t
    #     return arm.tolist()

  
    # ----------------------------------------------------------
    # 콜백
    # ----------------------------------------------------------
    def _activate_callback(self, msg: String):
        data = msg.data.strip()
        if data == 'stop':
            self.mode = MODE_IDLE
            self.target_color = None
            self.get_logger().info('블록 검출 중지')
        else:
            self.target_color = data
            self.mode = MODE_BLOCK
            self.get_logger().info(f'블록 검출 모드 - 타겟: {data}')

    def _state_callback(self, msg: String):
        # NAV_TO_DEST면 QR 모드로
        if msg.data == 'NAV_TO_DEST':
            if self.mode != MODE_QR:
                self.mode = MODE_QR
                self.recent_qr.clear()
                self.get_logger().info('QR 검증 모드 진입')
        else:
            # NAV_TO_DEST 벗어나면 QR 모드 해제 (블록 모드는 vision_activate가 관리)
            if self.mode == MODE_QR:
                self.mode = MODE_IDLE
                self.get_logger().info('QR 검증 모드 종료')

    # ----------------------------------------------------------
    # 프레임 처리
    # ----------------------------------------------------------
    def _process_frame(self):
        if self.mode == MODE_IDLE:
            return

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=200)
        except Exception:
            return

        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            return

        img = np.asanyarray(color_frame.get_data())

        if self.mode == MODE_BLOCK:
            self._detect_block(img, depth_frame)
        elif self.mode == MODE_QR:
            self._detect_qr(img)

    # ----------------------------------------------------------
    # 블록 검출 (TODO: 실제 YOLO/색검출 채우기)
    # ----------------------------------------------------------
    def _detect_block(self, img, depth_frame):
         # TODO: self.target_item 라벨에 맞는 블록을 YOLO로 찾아서 3D 좌표 계산
        # 지금은 디버깅용 고정 더미 좌표
        coords = [200.0, 150.0, 80.0, 175.35, -1.1, -89.73]

        msg = Float32MultiArray()
        msg.data = coords
        self._box_pose_pub.publish(msg)
        self.get_logger().info(f'/box_pose 발행 (더미): {coords}')

        # 한 번 발행하면 블록 모드 종료 (중복 발행 방지)
        self.mode = MODE_IDLE
    
    # YOLO 사용할경우 TODO
    # def _detect_block(self, img, depth_frame):
    #     # 1. YOLO로 블록들 검출
    #     results = self.model(img)
    
    #     # 2. 주문한 물품(self.target_item) 라벨이랑 일치하는 것 찾기
    #     target_box = None
    #     for box in results[0].boxes:
    #         label = self.model.names[int(box.cls)]
    #         if label == self.target_item:
    #             target_box = box
    #             break
    
    #     # 3. 못 찾으면 종료 (다음 프레임에 재시도)
    #     if target_box is None:
    #         self.get_logger().warn(f'{self.target_item} 못 찾음, 재시도')
    #         return  # mode 안 바꿈 → 다음 프레임에 다시 시도
    
    #     # 4. 박스 중심 픽셀 좌표
    #     x1, y1, x2, y2 = target_box.xyxy[0]
    #     cx = int((x1 + x2) / 2)
    #     cy = int((y1 + y2) / 2)
    
    #     # 5. 그 픽셀의 depth(거리) 읽기
    #     dist = depth_frame.get_distance(cx, cy)
    
    #     # 6. 픽셀+거리 → 카메라 3D 좌표
    #     intr = depth_frame.profile.as_video_stream_profile().intrinsics
    #     cam_xyz = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], dist)
    
    #     # 7. 카메라 좌표 → 로봇팔 좌표 (변환행렬)
    #     arm_xyz = self._cam_to_arm(cam_xyz)
    
    #     # 8. 좌표 + 자세 합쳐서 발행
    #     coords = arm_xyz + [175.35, -1.1, -89.73]  # 자세는 고정
    #     msg = Float32MultiArray()
    #     msg.data = [float(v) for v in coords]
    #     self._box_pose_pub.publish(msg)
    #     self.get_logger().info(f'{self.target_item} 발견, /box_pose 발행: {coords}')
    
    #     self.mode = MODE_IDLE

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
