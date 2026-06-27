#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_node.py  (토픽 구독 버전 - 카메라 공유용)

기존: pyrealsense2로 카메라 직접 열기
변경: realsense2_camera 드라이버가 발행하는 토픽을 구독
      → 카메라 한 대를 다른 노드(라인트레이싱 등)와 공유 가능

전제: realsense2_camera 드라이버가 아래 옵션으로 먼저 떠 있어야 함
  ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true enable_depth:=true \
    align_depth.enable:=true \
    rgb_camera.color_profile:=640x480x30

실행 (venv 필요):
  source ~/yolo_env/bin/activate
  python3 vision_node.py

구독 토픽:
  /camera/camera/color/image_raw                  (컬러, YOLO용)
  /camera/camera/aligned_depth_to_color/image_raw (정렬 depth, 거리용)
  /camera/camera/color/camera_info                (intrinsic, deproject용)
  /vision_activate, /brain_state

발행 토픽:
  /box_pose    : 블록 피킹 좌표
  /place_pose  : QR 기반 플레이싱 좌표 (방법 B)
  /depth_qr    : QR 구역 검증
  /detected_image
"""

from collections import deque

from ultralytics import YOLO

import cv2
import numpy as np
from pyzbar import pyzbar
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo, CompressedImage


WINDOW_SIZE = 10
VALID_ZONES = {'A', 'B', 'C'}

MODE_IDLE  = 'idle'
MODE_BLOCK = 'block'
MODE_QR    = 'qr'
MODE_QR_PLACE = 'qr_place'

# place 오프셋 (QR → 실제 놓을 위치, 실측 필요)
PLACE_OFFSET_X = -190.0
PLACE_OFFSET_Y = 0.0
PLACE_OFFSET_Z = -140.0
PLACE_RX = -178.0    # place 자세 (실측 후 수정)
PLACE_RY = 0.0
PLACE_RZ = -90.0

# ===== 설정 =====
MODEL_PATH = '/home/zzz/pj3_ws/src/brain_pkg/brain_pkg/best.pt'
CONF_THRES = 0.55

# 카메라 토픽 (realsense2_camera 드라이버 기준)
TOPIC_COLOR = '/camera/camera/color/image_raw'
TOPIC_DEPTH = '/camera/camera/aligned_depth_to_color/image_raw'
TOPIC_CAMINFO = '/camera/camera/color/camera_info'

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

        self.bridge = CvBridge()

        # 최신 프레임 저장
        self.color_img = None
        self.depth_img = None      # 정렬된 depth (16UC1, mm 단위)
        self.intrinsics = None     # (fx, fy, cx, cy)

        self.mode        = MODE_IDLE
        self.target_item = None
        self.recent_qr   = deque(maxlen=WINDOW_SIZE)

        # YOLO 모델 로드
        self.get_logger().info(f'YOLO 모델 로드 중: {MODEL_PATH}')
        self.model = YOLO(MODEL_PATH)
        self.T_cam2base = np.load("/home/zzz/calibration/T_cam2base.npy")
        self.get_logger().info("캘리브레이션 T 로드 완료")
        self.get_logger().info(f'YOLO 클래스: {self.model.names}')

        # 구독 - 카메라 토픽
        self.create_subscription(Image, TOPIC_COLOR, self._color_callback, 10)
        self.create_subscription(Image, TOPIC_DEPTH, self._depth_callback, 10)
        self.create_subscription(CameraInfo, TOPIC_CAMINFO, self._caminfo_callback, 10)

        # 구독 - brain
        self.create_subscription(String, '/vision_activate', self._activate_callback, 10)
        self.create_subscription(String, '/brain_state',     self._state_callback,    10)

        # 발행
        self._box_pose_pub       = self.create_publisher(Float32MultiArray, '/box_pose',       10)
        self._qr_pub             = self.create_publisher(String,            '/depth_qr',       10)
        self._detected_image_pub = self.create_publisher(CompressedImage,   '/detected_image', 10)
        self._place_pose_pub     = self.create_publisher(Float32MultiArray, '/place_pose',     10)

        self.get_logger().info('vision_node 시작 (토픽 구독 / YOLO 통합)')

        # 처리 타이머 (33ms = 약 30Hz)
        self.timer = self.create_timer(0.033, self._process_frame)

    # ----------------------------------------------------------
    # 카메라 토픽 콜백 - 최신 프레임만 저장
    # ----------------------------------------------------------
    def _color_callback(self, msg: Image):
        # rgb8 -> bgr (OpenCV/YOLO는 bgr 기준)
        self.color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _depth_callback(self, msg: Image):
        # 정렬된 depth, 16UC1 (mm)
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _caminfo_callback(self, msg: CameraInfo):
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        if self.intrinsics is None:
            fx, fy = msg.k[0], msg.k[4]
            cx, cy = msg.k[2], msg.k[5]
            self.intrinsics = (fx, fy, cx, cy)
            self.get_logger().info(
                f'intrinsic 수신: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}'
            )

    # ----------------------------------------------------------
    # brain 콜백
    # ----------------------------------------------------------
    def _activate_callback(self, msg: String):
        data = msg.data.strip()
        if data == 'stop':
            self.mode = MODE_IDLE
            self.target_item = None
            self.get_logger().info('블록 검출 중지')
        elif data == 'qr_place':
            # 방법 B: QR을 인식해 플레이싱 좌표를 계산하는 모드
            self.mode = MODE_QR_PLACE
            self.get_logger().info('QR place 좌표 계산 모드')
        else:
            self.target_item = data
            self.mode = MODE_BLOCK
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
        if self.color_img is None:
            return  # 아직 영상 안 들어옴

        if self.mode == MODE_IDLE:
            return
        if self.mode == MODE_BLOCK:
            self._detect_block()
        elif self.mode == MODE_QR:
            self._detect_qr()
        elif self.mode == MODE_QR_PLACE:
            self._detect_qr_place()

    # ----------------------------------------------------------
    # 블록 검출 (YOLO + 캘리브레이션)
    # ----------------------------------------------------------
    def _detect_block(self):
        if self.depth_img is None or self.intrinsics is None:
            self.get_logger().warn('depth/intrinsic 아직 준비 안 됨')
            return

        img = self.color_img.copy()
        results = self.model(img, conf=CONF_THRES, verbose=False)

        target_box = None
        for box in results[0].boxes:
            label = self.model.names[int(box.cls)]
            if label == self.target_item:
                target_box = box
                break

        if target_box is None:
            self.get_logger().warn(f'{self.target_item} 못 찾음, 재시도')
            return

        x1, y1, x2, y2 = map(int, target_box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 잘림 감지
        H, W = img.shape[:2]
        margin = 5
        if x1 <= margin or y1 <= margin or x2 >= W - margin or y2 >= H - margin:
            self.get_logger().warn(f'{self.target_item} 잘림 감지 - 픽업 보류, 재정렬 필요')
            self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=True)
            return

        # depth 읽기 (정렬된 depth 이미지에서 patch median, mm -> m)
        dist_m = self._get_robust_depth(cx, cy)

        # =========================
        # DEPTH DEBUG: bbox 내부 depth 분포 확인
        # =========================
        roi = self.depth_img[y1:y2, x1:x2]
        valid = roi[(roi > 0) & (roi < 2000)]  # mm 단위, 2m 이하만 확인

         # =========================
        # DEPTH DEBUG: bbox 내부 depth 분포 확인용
        # 실제 depth 선택에는 사용하지 않음
        # =========================
        roi = self.depth_img[y1:y2, x1:x2]
        valid = roi[(roi > 0) & (roi < 2000)]

        if len(valid) > 0:
            self.get_logger().info(
                f"[DEPTH DEBUG] selected={dist_m*1000:.0f}mm | "
                f"bbox min={np.min(valid):.0f}, "
                f"p10={np.percentile(valid, 10):.0f}, "
                f"p30={np.percentile(valid, 30):.0f}, "
                f"median={np.median(valid):.0f}, "
                f"p70={np.percentile(valid, 70):.0f}, "
                f"max={np.max(valid):.0f}, "
                f"count={len(valid)}"
            )
        else:
            self.get_logger().warn("[DEPTH DEBUG] bbox valid depth 없음")

        if dist_m <= 0.0:
            self.get_logger().warn(
                f'{self.target_item} center raw depth 측정 실패(0) - /box_pose 발행 안 함'
            )
            return

        else:
            self.get_logger().warn("[DEPTH DEBUG] bbox valid depth 없음")

        if dist_m <= 0.0:
            self.get_logger().warn(f'{self.target_item} depth 측정 실패(0) - 재시도')
            return

        # 카메라 3D 좌표 (deproject) - intrinsic으로 직접 계산
        fx, fy, ppx, ppy = self.intrinsics
        X = (cx - ppx) / fx * dist_m
        Y = (cy - ppy) / fy * dist_m
        Z = dist_m
        cam_xyz = [X, Y, Z]

        self.get_logger().info(
            f'{self.target_item} 발견 | 픽셀=({cx},{cy}) '
            f'dist={dist_m:.3f}m cam_xyz={[round(v, 3) for v in cam_xyz]}'
        )

        # 캘리브레이션 변환: cam_xyz(m) → base 좌표(mm)
        cam_pt = np.array([cam_xyz[0]*1000.0, cam_xyz[1]*1000.0, cam_xyz[2]*1000.0, 1.0])
        base_pt = (self.T_cam2base @ cam_pt)[:3]
        arm_xyz = [float(base_pt[0]), float(base_pt[1]), float(base_pt[2])]
        self.get_logger().info(f'  변환된 arm_xyz(mm): {[round(v, 1) for v in arm_xyz]}')

        coords = list(arm_xyz) + [-178.06, -0.79, -129.4]

        # /box_pose 발행
        msg = Float32MultiArray()
        msg.data = [float(v) for v in coords]
        self._box_pose_pub.publish(msg)
        self.get_logger().info(f'/box_pose 발행: {[round(v, 1) for v in coords]}')

        self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=False)
        self.mode = MODE_IDLE

    def _get_robust_depth(self, cx, cy, k=12):
        """중심 (cx,cy) 주변 (2k+1)x(2k+1) patch에서 유효 depth를 모아
        가까운 쪽(p30)을 반환. mm -> m.
        - patch를 넓게(15x15) 봐서 중심이 depth 구멍(0)이어도 주변으로 채움
        - median 대신 p30을 써서 배경(먼 값)이 섞여도 블록 표면 거리만 추출"""
        H, W = self.depth_img.shape[:2]
        y0, y1 = max(0, cy - k), min(H, cy + k + 1)
        x0, x1 = max(0, cx - k), min(W, cx + k + 1)
         
        patch = self.depth_img[y0:y1, x0:x1]
        valid = patch[(patch > 0) & (patch < 2000)]  # mm, 2m 이하만
         
        if valid.size < 5:
            return 0.0
              
        depth_mm = float(np.percentile(valid, 30))
        self.get_logger().info(
            f"[DEPTH SELECT] patch k={k}, valid={valid.size}, p5={depth_mm:.0f}mm"
        )
        return depth_mm / 1000.0

    def _draw_and_publish(self, img, x1, y1, x2, y2, label, cut=False):
        color = (0, 0, 255) if cut else CLASS_COLORS.get(label, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tag = f'{label} (CUT)' if cut else label
        cv2.putText(img, tag, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        try:
            ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                out = CompressedImage()
                out.header.stamp = self.get_clock().now().to_msg()
                out.format = 'jpeg'
                out.data = buf.tobytes()
                self._detected_image_pub.publish(out)
        except Exception as e:
            self.get_logger().warn(f'detected_image 발행 실패: {e}')

    # ----------------------------------------------------------
    # QR 검증 (구역 확인용 - 좌표 계산 안 함)
    # ----------------------------------------------------------
    def _detect_qr(self):
        img = self.color_img
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

    # ----------------------------------------------------------
    # QR 기반 플레이싱 좌표 계산 (방법 B)
    #   QR을 인식 → 중심 픽셀 → depth → deproject →
    #   캘리브레이션 변환 → PLACE_OFFSET 적용 → /place_pose 발행
    #   (블록 피킹과 완전히 대칭 구조)
    # ----------------------------------------------------------
    def _detect_qr_place(self):
        if self.depth_img is None or self.intrinsics is None:
            self.get_logger().warn('depth/intrinsic 준비 안 됨')
            return

        decoded = pyzbar.decode(self.color_img)
        if not decoded:
            self.get_logger().warn('QR 못 찾음, 재시도')
            return

        obj = decoded[0]
        try:
            zone = obj.data.decode('utf-8').strip().upper()
        except Exception:
            zone = '?'

        # QR polygon 중심 픽셀
        pts = obj.polygon
        cx = int(sum(p.x for p in pts) / len(pts))
        cy = int(sum(p.y for p in pts) / len(pts))

        dist_m = self._get_robust_depth(cx, cy)
        if dist_m <= 0:
            self.get_logger().warn('QR depth 측정 실패(0) - 재시도')
            return

        fx, fy, ppx, ppy = self.intrinsics
        X = (cx - ppx) / fx * dist_m
        Y = (cy - ppy) / fy * dist_m
        Z = dist_m

        self.get_logger().info(
            f'QR place: zone={zone} 픽셀=({cx},{cy}) dist={dist_m:.3f}m'
        )

        # 캘리브레이션 변환: cam(m) → base(mm)
        cam_pt = np.array([X*1000.0, Y*1000.0, Z*1000.0, 1.0])
        base_pt = (self.T_cam2base @ cam_pt)[:3]

        # QR 위치에서 바구니 안쪽 놓을 위치로 오프셋
        place = [
            float(base_pt[0] + PLACE_OFFSET_X),
            float(base_pt[1] + PLACE_OFFSET_Y),
            float(base_pt[2] + PLACE_OFFSET_Z),
        ]
        coords = place + [PLACE_RX, PLACE_RY, PLACE_RZ]

        msg = Float32MultiArray()
        msg.data = [float(v) for v in coords]
        self._place_pose_pub.publish(msg)
        self.get_logger().info(f'/place_pose 발행: {[round(v, 1) for v in coords]}')
        self.mode = MODE_IDLE


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
