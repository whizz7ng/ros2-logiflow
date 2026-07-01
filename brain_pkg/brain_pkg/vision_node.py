#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_node.py  (eye-in-hand 버전)

[eye-to-hand → eye-in-hand 변경 요약]
  (1) 고정 T_cam2base npy 로드 삭제
      → X_cam2gripper + 층별 관측 포즈(get_coords)로 T_cam2base를 미리 계산
  (2) 회전 변환을 scipy Rotation 'xyz'(=Rz@Ry@Rx)로 통일 (ArUco 실측 검증 완료)
  (3) /vision_activate 포맷: "item:level" (예: "red_cross:1")
  (4) depth 유효범위를 층별 딕셔너리(DEPTH_RANGE)로 분리
  (5) 관측은 팔이 SHELF_ANGLES 자세로 send_angles 이동 후 정지 상태에서만
      (send_coords는 IK 복수해 때문에 관측자세가 A/B로 갈려서 금지 → brain_node에서 처리)

전제: realsense2_camera 드라이버가 아래로 먼저 떠 있어야 함
  ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true enable_depth:=true \
    align_depth.enable:=true \
    rgb_camera.color_profile:=640x480x30
"""

from collections import deque

from ultralytics import YOLO

import cv2
import numpy as np
from pyzbar import pyzbar
from cv_bridge import CvBridge

# ===== [변경] eye-in-hand 좌표변환용 scipy 추가 =====
from scipy.spatial.transform import Rotation as R

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
PLACE_OFFSET_Y = -40.0
PLACE_OFFSET_Z = -140.0
PLACE_RX = -178.0
PLACE_RY = 0.0
PLACE_RZ = -90.0

# ===== 설정 =====
MODEL_PATH = '/home/zzz/pj3_ws/src/brain_pkg/brain_pkg/best.pt'
CONF_THRES = 0.55

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

# ===== [변경] eye-in-hand 설정 =====================================
# 고정 T_cam2base npy 로드를 삭제하고, 아래로 대체.
X_CAM2GRIPPER_PATH = "/home/zzz/calibration/X_cam2gripper.npy"

# 각 층 관측 포즈의 get_coords 실측값 (mm, deg).
# 반복도 테스트(send_angles 이동)로 확정한 "실제 도달값"을 박음.
#   - brain_node는 이동을 send_angles(SHELF_ANGLES[level])로 해야 이 값에 정확히 도달함.
#   - send_coords로 보내면 IK 복수해 때문에 자세가 A/B로 갈려서 이 값과 어긋남.
SHELF_POSES = {
    1: [10.8, -61.6, 228.4, -123.1, -34.2, -66.6],   # 1층 관측 실측 (랙 16.5cm, send_angles+sleep 안정값)
    2: [-46.8, -59.1, 285.1, -89.3, -40.5, -87.1],   # 2층 관측 실측 (랙 12.5cm, send_angles+sleep 안정값)
}

# 층별 depth 유효 범위(mm). 관측 높이가 층마다 달라서 분리.
#   1층 관측 z≈237mm, 2층 관측 z≈278mm → 블록 표면까지 거리도 층마다 다름.
#   실제 픽 로그의 dist_m 값을 보고 좁혀서 조정할 것.
DEPTH_RANGE = {
    1: (150, 320),
    2: (150, 360),
}


def _coords_to_matrix(coords):
    """[변경] myCobot get_coords [x,y,z(mm), rx,ry,rz(deg)] → 4x4 동차변환.
    회전은 scipy 'xyz'(extrinsic) = Rz@Ry@Rx. ArUco 마커 실측으로 검증 완료.
    (기존엔 이 함수 없이 고정 T_cam2base npy를 그대로 썼음)"""
    x, y, z, rx, ry, rz = coords
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.bridge = CvBridge()

        self.color_img = None
        self.depth_img = None
        self.intrinsics = None

        self.mode        = MODE_IDLE
        self.target_item = None
        self.shelf_level = 1              # ===== [변경] 현재 관측 중인 층 (activate로 갱신)
        self.recent_qr   = deque(maxlen=WINDOW_SIZE)

        self.get_logger().info(f'YOLO 모델 로드 중: {MODEL_PATH}')
        self.model = YOLO(MODEL_PATH)

        # ===== [변경] eye-in-hand: 고정 T_cam2base 로드 삭제 =====
        # 기존:
        #   self.T_cam2base = np.load(".../T_cam2base_backup_20260626_233813.npy")
        # 변경: X_cam2gripper + 층별 관측 포즈로 T_cam2base를 층마다 미리 계산
        X_cam2gripper = np.load(X_CAM2GRIPPER_PATH)
        self.T_CAM2BASE = {
            s: _coords_to_matrix(p) @ X_cam2gripper
            for s, p in SHELF_POSES.items()
        }
        self.get_logger().info(
            f'eye-in-hand 캘리브레이션 로드 완료 (층: {list(self.T_CAM2BASE.keys())})'
        )
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

        self.get_logger().info('vision_node 시작 (eye-in-hand / YOLO 통합)')

        self.timer = self.create_timer(0.033, self._process_frame)

    # ----------------------------------------------------------
    # 카메라 토픽 콜백 - 최신 프레임만 저장
    # ----------------------------------------------------------
    def _color_callback(self, msg: Image):
        self.color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _depth_callback(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _caminfo_callback(self, msg: CameraInfo):
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
            self.mode = MODE_QR_PLACE
            self.get_logger().info('QR place 좌표 계산 모드')
        else:
            # ===== [변경] "item:level" 파싱 =====
            # 기존: self.target_item = data  (아이템만 받음)
            # 변경: "red_cross:1" 처럼 층 정보를 함께 받아 shelf_level 갱신
            if ':' in data:
                item, level_str = data.rsplit(':', 1)
                try:
                    level = int(level_str)
                except ValueError:
                    item, level = data, self.shelf_level   # 파싱 실패 시 현재 층 유지
            else:
                item, level = data, self.shelf_level        # 층 없으면 현재 층 유지(하위호환)

            if level not in self.T_CAM2BASE:
                self.get_logger().error(f'알 수 없는 층: {level} - 무시 (티칭 안 됨)')
                return

            self.target_item = item
            self.shelf_level = level
            self.mode = MODE_BLOCK
            self.get_logger().info(f'블록 검출 모드 - 타겟: {item}, 층: {level}')

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
            return
        if self.mode == MODE_IDLE:
            return
        if self.mode == MODE_BLOCK:
            self._detect_block()
        elif self.mode == MODE_QR:
            self._detect_qr()
        elif self.mode == MODE_QR_PLACE:
            self._detect_qr_place()

    # ----------------------------------------------------------
    # 블록 검출 (YOLO + eye-in-hand 변환)
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

        # ===== [변경] depth 범위를 층별로 사용 =====
        # 기존: valid = roi[(roi > 110) & (roi < 250)]  (고정)
        dmin, dmax = DEPTH_RANGE.get(self.shelf_level, (110, 300))
        roi = self.depth_img[y1:y2, x1:x2]
        valid = roi[(roi > dmin) & (roi < dmax)]

        if valid.size < 30:
            self.get_logger().warn('depth 없음, 발행 안 함')
            return

        # 블록 정면 = bbox 내 최소거리 클러스터
        near = np.min(valid)
        block_face = valid[valid < near + 25]

        if block_face.size < 40:
            self.get_logger().warn('블록 정면 픽셀 부족, 발행 안 함')
            return

        dist_m = float(np.median(block_face)) / 1000.0

        # ===== [변경] 층별 거리 sanity check =====
        # 기존: if not (0.165 <= dist_m <= 0.220)  (고정)
        if not (dmin / 1000.0 <= dist_m <= dmax / 1000.0):
            self.get_logger().warn(
                f'dist={dist_m:.3f}m 층{self.shelf_level} 범위 밖 - 발행 안 함'
            )
            return

        # DEPTH DEBUG (dist_m 재계산 안 함)
        self.get_logger().info(
            f"[DEPTH DEBUG] L{self.shelf_level} selected={dist_m*1000:.0f}mm | "
            f"bbox min={np.min(valid):.0f}, "
            f"p30={np.percentile(valid, 30):.0f}, "
            f"median={np.median(valid):.0f}, "
            f"count={len(valid)}"
        )

        if dist_m <= 0.0:
            self.get_logger().warn(
                f'{self.target_item} raw depth 측정 실패(0) - 발행 안 함'
            )
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

        # ===== [변경] eye-in-hand 변환: 현재 층의 T_cam2base 사용 =====
        # 기존: base_pt = (self.T_cam2base @ cam_pt)[:3]
        cam_pt = np.array([cam_xyz[0]*1000.0, cam_xyz[1]*1000.0, cam_xyz[2]*1000.0, 1.0])
        base_pt = (self.T_CAM2BASE[self.shelf_level] @ cam_pt)[:3]
        arm_xyz = [float(base_pt[0]), float(base_pt[1]), float(base_pt[2])]
        self.get_logger().info(
            f'  변환된 arm_xyz(mm) L{self.shelf_level}: {[round(v, 1) for v in arm_xyz]}'
        )

        coords = list(arm_xyz) + [-102.25, -38.21, -82.48]


        msg = Float32MultiArray()
        msg.data = [float(v) for v in coords]
        self._box_pose_pub.publish(msg)
        self.get_logger().info(f'/box_pose 발행: {[round(v, 1) for v in coords]}')

        self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=False)
        self.mode = MODE_IDLE

    def _get_robust_depth(self, cx, cy, k=12):
        """중심 주변 patch에서 유효 depth 모아 p30 반환 (mm→m). QR place용."""
        H, W = self.depth_img.shape[:2]
        y0, y1 = max(0, cy - k), min(H, cy + k + 1)
        x0, x1 = max(0, cx - k), min(W, cx + k + 1)
        patch = self.depth_img[y0:y1, x0:x1]
        valid = patch[(patch > 160) & (patch < 500)]
        if valid.size < 30:
            return 0.0
        depth_mm = float(np.percentile(valid, 30))
        self.get_logger().info(
            f"[DEPTH SELECT] patch k={k}, valid={valid.size}, p30={depth_mm:.0f}mm"
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

        # ===== [변경] eye-in-hand: QR place도 현재 층의 T_cam2base 사용 =====
        # 기존: base_pt = (self.T_cam2base @ cam_pt)[:3]
        cam_pt = np.array([X*1000.0, Y*1000.0, Z*1000.0, 1.0])
        base_pt = (self.T_CAM2BASE[self.shelf_level] @ cam_pt)[:3]

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
