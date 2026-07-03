#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_node.py  (eye-in-hand / 동적 T + 거리판정 버전)

[거리 판정 추가]
  블록 depth(dist_m)가 파지 가능 범위(GRASP_DEPTH_RANGE) 안인지 체크.
  - 범위 안: 정상 파지 진행 (/box_pose 발행)
  - 너무 가까움/멈: /distance_status 발행 ("too_close:195" / "too_far:315"),
    파지 보류. AGV 보정은 별도(nav)에서 이 신호 받아 처리.
  정상일 때도 "ok:250" 발행해서 nav가 현재 거리 알 수 있게 함.

[동적 T]
  pick_node가 관측 자세 도착 후 실제 get_coords를 /observe_pose로 발행.
  vision은 그 실제 자세로 T_cam2base를 매번 새로 계산(current_T_cam2base).
  폴백: /observe_pose 못 받았으면 고정 T(SHELF_POSES) 사용.
  T_cam2base = T_gripper2base(실제 팔 자세) @ X_cam2gripper

[eye-in-hand]
  회전 변환 scipy Rotation 'xyz'(=Rz@Ry@Rx), /vision_activate "item:level",
  depth 유효범위 층별(DEPTH_RANGE), 관측은 pick이 send_angles로 이동.
"""

from collections import deque

from ultralytics import YOLO

import cv2
import numpy as np
from pyzbar import pyzbar
from cv_bridge import CvBridge

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

# ===== eye-in-hand 설정 =====
X_CAM2GRIPPER_PATH = "/home/zzz/calibration/X_cam2gripper.npy"

# 폴백용 관측 포즈 (동적 T 정상 작동 시 안 쓰임)
SHELF_POSES = {
    1: [10.8, -61.6, 228.4, -123.1, -34.2, -66.6],
    2: [-5.0, 79.45, -76.81, -13.71, 5.97, -44.2],
}

# depth 검출 유효 범위(mm) - 노이즈 필터 (넓게)
DEPTH_RANGE = {
    1: (150, 320),
    2: (150, 360),
}

# ===== [거리 판정] 파지 가능 depth 범위(mm) =====
# 블록까지 실제 거리가 이 안이어야 팔이 잡을 수 있음 (실측값).
# 이 범위 밖이면 AGV가 움직여야 함 → /distance_status 발행.
GRASP_DEPTH_RANGE = {
    1: (210, 301),
    2: (210, 301),
}

# ===== [마커 J1 보정] =====
# 블록이 안 보일 때, 좌우 마커(ID0 왼쪽 / ID1 오른쪽)로 AGV 틀어짐을 판단.
# 두 마커 중심 픽셀이 기준(정상 관측 자세)에서 벗어난 만큼 J1 보정량 계산.
MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKER_LENGTH = 0.0365          # 한 변 3.65cm
MARKER_ID_LEFT  = 0
MARKER_ID_RIGHT = 1

# 정상 관측 자세(AGV 정상 정차)일 때 각 마커의 기준 픽셀 x (실측).
# 개별 마커 기준: 보이는 마커가 자기 기준에서 벗어난 만큼 보정 (한쪽만 보여도 됨).
MARKER_REF_PX = {
    1: {MARKER_ID_LEFT: 126, MARKER_ID_RIGHT: 548},   # 1층
    2: {MARKER_ID_LEFT: 190, MARKER_ID_RIGHT: 510},   # 2층
}
# J1 픽셀 환산: J1 10도 → 화면 256픽셀 이동 (실측) → 0.039도/픽셀
# (마커는 블록보다 뒤라 실제론 조금 다르지만, 반복으로 수렴)
PIXEL_TO_J1 = 10.0 / 256.0      # ≈ 0.0391
# 부호: J1+ → 마커 오른쪽 이동. 마커가 기준보다 오른쪽(큰값)이면 J1- 로 되돌림.
J1_SIGN = -1.0
# J1 보정 한계 (1회 보정량이 이 이상이면 팔로 못 잡음 → AGV 재정차)
J1_CORRECTION_MAX = 25.0
# 마커 보정 최대 반복 횟수 (넘으면 AGV 재정차)
MARKER_REALIGN_MAX = 3
# 블록 검출 이 횟수(프레임) 실패하면 마커 보정 트리거
NOT_FOUND_LIMIT = 15


def _coords_to_matrix(coords):
    """myCobot get_coords [x,y,z(mm), rx,ry,rz(deg)] → 4x4 동차변환.
    회전은 scipy 'xyz'(extrinsic) = Rz@Ry@Rx."""
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
        self.shelf_level = 1
        self.recent_qr   = deque(maxlen=WINDOW_SIZE)
        self.not_found_count = 0   # 블록 검출 연속 실패 (마커 보정 트리거용)
        self.realign_count = 0     # 마커 보정 반복 횟수 (한계 넘으면 재정차)

        self.get_logger().info(f'YOLO 모델 로드 중: {MODEL_PATH}')
        self.model = YOLO(MODEL_PATH)

        # X_cam2gripper: 카메라-그리퍼 고정 관계 (안 변함)
        self.X_cam2gripper = np.load(X_CAM2GRIPPER_PATH)

        # 고정 T (폴백용)
        self.T_CAM2BASE = {
            s: _coords_to_matrix(p) @ self.X_cam2gripper
            for s, p in SHELF_POSES.items()
        }
        # 동적 T
        self.current_T_cam2base = None

        # ArUco 검출기 (마커 J1 보정용)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self._aruco_new = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self._aruco_new = False

        self.get_logger().info(
            f'eye-in-hand 캘리브레이션 로드 완료 (폴백 층: {list(self.T_CAM2BASE.keys())})'
        )
        self.get_logger().info(f'YOLO 클래스: {self.model.names}')

        # 구독 - 카메라
        self.create_subscription(Image, TOPIC_COLOR, self._color_callback, 10)
        self.create_subscription(Image, TOPIC_DEPTH, self._depth_callback, 10)
        self.create_subscription(CameraInfo, TOPIC_CAMINFO, self._caminfo_callback, 10)
        # 구독 - brain
        self.create_subscription(String, '/vision_activate', self._activate_callback, 10)
        self.create_subscription(String, '/brain_state',     self._state_callback,    10)
        # 구독 - pick 관측 자세 (동적 T)
        self.create_subscription(
            Float32MultiArray, '/observe_pose', self._observe_pose_callback, 10
        )

        # 발행
        self._box_pose_pub       = self.create_publisher(Float32MultiArray, '/box_pose',       10)
        self._qr_pub             = self.create_publisher(String,            '/depth_qr',       10)
        self._detected_image_pub = self.create_publisher(CompressedImage,   '/detected_image', 10)
        self._place_pose_pub     = self.create_publisher(Float32MultiArray, '/place_pose',     10)
        # [거리 판정] 발행
        self._dist_status_pub    = self.create_publisher(String, '/distance_status', 10)
        # [마커 J1 보정] 발행: "층:J1보정량" (예 "1:8.5"), 또는 "realign_fail"
        self._j1_corr_pub        = self.create_publisher(String, '/j1_correction', 10)

        self.get_logger().info('vision_node 시작 (eye-in-hand / 동적 T / 거리판정)')

        self.timer = self.create_timer(0.033, self._process_frame)

    # ---------- 카메라 콜백 ----------
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

    # ---------- 동적 T ----------
    def _observe_pose_callback(self, msg: Float32MultiArray):
        pose = list(msg.data)
        if len(pose) != 6:
            self.get_logger().warn(f'/observe_pose 6개 아님: {len(pose)}')
            return
        self.current_T_cam2base = _coords_to_matrix(pose) @ self.X_cam2gripper
        self.get_logger().info(
            f'[동적 T] 관측 자세 수신 → T 갱신: {[round(v, 1) for v in pose]}'
        )

    def _current_T(self):
        if self.current_T_cam2base is not None:
            return self.current_T_cam2base
        self.get_logger().warn('[동적 T] 아직 없음 → 고정 T 폴백 사용')
        return self.T_CAM2BASE.get(self.shelf_level)

    def _pub_dist_status(self, status, dist_mm):
        """거리 상태 발행: 'ok:250' / 'too_close:195' / 'too_far:315'"""
        m = String()
        m.data = f'{status}:{dist_mm:.0f}'
        self._dist_status_pub.publish(m)

    def _detect_markers(self):
        """좌우 마커 검출 → {id: 중심픽셀x} 반환. 없으면 빈 dict."""
        if self.color_img is None:
            return {}
        if self._aruco_new:
            corners, ids, _ = self.aruco_detector.detectMarkers(self.color_img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                self.color_img, self.aruco_dict, parameters=self.aruco_params)
        result = {}
        if ids is None:
            return result
        for i, mid in enumerate(ids.flatten()):
            c = corners[i][0]
            px = float(np.mean(c[:, 0]))
            result[int(mid)] = px
        return result

    def _try_marker_realign(self):
        """블록 안 보일 때: 개별 마커 기준으로 J1 보정량 계산해서 /j1_correction 발행.
        - 보이는 마커가 자기 기준 픽셀에서 벗어난 만큼 J1 보정 (한쪽만 보여도 됨).
        - 양쪽 보이면 두 보정량 평균.
        - 최대 MARKER_REALIGN_MAX회 반복, 넘으면 realign_fail (AGV 재정차)."""
        # 반복 한계 체크
        if self.realign_count >= MARKER_REALIGN_MAX:
            self.get_logger().error(
                f'[마커보정] {MARKER_REALIGN_MAX}회 반복해도 못 찾음 → realign_fail (AGV 재정차)'
            )
            self._j1_corr_pub.publish(String(data='realign_fail'))
            self.realign_count = 0
            return

        markers = self._detect_markers()
        ref = MARKER_REF_PX.get(self.shelf_level, {})
        if not ref:
            self.get_logger().error(f'[마커보정] 층 {self.shelf_level} 기준 없음')
            self._j1_corr_pub.publish(String(data='realign_fail'))
            return

        # 보이는 마커 각각의 보정량 계산
        corrections = []
        seen = []
        for mid, ref_px in ref.items():
            if mid in markers:
                cur_px = markers[mid]
                offset_px = cur_px - ref_px
                corr = J1_SIGN * offset_px * PIXEL_TO_J1
                corrections.append(corr)
                seen.append(f'ID{mid}(cur={cur_px:.0f} ref={ref_px} off={offset_px:.0f})')

        if not corrections:
            self.get_logger().warn('[마커보정] 마커 안 보임 → realign_fail (AGV 재정차)')
            self._j1_corr_pub.publish(String(data='realign_fail'))
            self.realign_count = 0
            return

        # 양쪽이면 평균, 한쪽이면 그거
        j1_corr = sum(corrections) / len(corrections)
        self.get_logger().info(
            f'[마커보정] {len(corrections)}개 검출 [{", ".join(seen)}] → J1보정={j1_corr:.1f}도'
        )

        if abs(j1_corr) > J1_CORRECTION_MAX:
            self.get_logger().warn(
                f'[마커보정] 보정량 {j1_corr:.1f}도 > 한계 {J1_CORRECTION_MAX} '
                f'→ realign_fail (AGV 재정차)'
            )
            self._j1_corr_pub.publish(String(data='realign_fail'))
            self.realign_count = 0
            return

        self.realign_count += 1
        self._j1_corr_pub.publish(String(data=f'{self.shelf_level}:{j1_corr:.2f}'))
        self.get_logger().info(
            f'[마커보정] /j1_correction 발행: {self.shelf_level}:{j1_corr:.2f} '
            f'({self.realign_count}/{MARKER_REALIGN_MAX}회째)'
        )
        self.mode = MODE_IDLE

    # ---------- brain 콜백 ----------
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
            if ':' in data:
                item, level_str = data.rsplit(':', 1)
                try:
                    level = int(level_str)
                except ValueError:
                    item, level = data, self.shelf_level
            else:
                item, level = data, self.shelf_level

            if level not in DEPTH_RANGE:
                self.get_logger().error(f'알 수 없는 층: {level} - 무시')
                return

            self.target_item = item
            self.shelf_level = level
            self.mode = MODE_BLOCK
            self.not_found_count = 0
            # 주의: realign_count는 여기서 리셋 안 함.
            # J1 보정 재관측도 observe_ready→vision_activate로 다시 오는데,
            # 여기서 리셋하면 3회 제한이 무효화되어 무한루프. 블록 찾을 때만 리셋.
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

    # ---------- 프레임 처리 ----------
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

    # ---------- 블록 검출 ----------
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
            self.not_found_count += 1
            self.get_logger().warn(
                f'{self.target_item} 못 찾음 ({self.not_found_count}/{NOT_FOUND_LIMIT})'
            )
            if self.not_found_count >= NOT_FOUND_LIMIT:
                # 일정 횟수 실패 → 마커로 J1 보정 시도
                self.get_logger().warn('→ 마커 J1 보정 시도')
                self._try_marker_realign()
                self.not_found_count = 0
            return
        self.not_found_count = 0
        self.realign_count = 0   # 블록 찾음 → 반복 카운터 리셋

        x1, y1, x2, y2 = map(int, target_box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        self.get_logger().info(
            f'[bbox] x1={x1} y1={y1} x2={x2} y2={y2} cx={cx} cy={cy} '
            f'(H={img.shape[0]} W={img.shape[1]})'
        )

        # 잘림 감지 (좌우 + 위. 아래는 관측 자세상 정상이라 제외)
        H, W = img.shape[:2]
        margin = 3
        if x1 <= margin or y1 <= margin or x2 >= W - margin:
            self.get_logger().warn(f'{self.target_item} 잘림 감지 - 픽업 보류, 재정렬 필요')
            self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=True)
            return

        # depth 검출 필터
        dmin, dmax = DEPTH_RANGE.get(self.shelf_level, (110, 300))
        roi = self.depth_img[y1:y2, x1:x2]
        valid = roi[(roi > dmin) & (roi < dmax)]

        if valid.size < 30:
            self.get_logger().warn('depth 없음, 발행 안 함')
            return

        near = np.min(valid)
        block_face = valid[valid < near + 25]

        if block_face.size < 40:
            self.get_logger().warn('블록 정면 픽셀 부족, 발행 안 함')
            return

        dist_m = float(np.median(block_face)) / 1000.0

        if not (dmin / 1000.0 <= dist_m <= dmax / 1000.0):
            self.get_logger().warn(
                f'dist={dist_m:.3f}m 층{self.shelf_level} 검출범위 밖 - 발행 안 함'
            )
            return

        self.get_logger().info(
            f"[DEPTH DEBUG] L{self.shelf_level} selected={dist_m*1000:.0f}mm | "
            f"bbox min={np.min(valid):.0f}, p30={np.percentile(valid, 30):.0f}, "
            f"median={np.median(valid):.0f}, count={len(valid)}"
        )

        if dist_m <= 0.0:
            self.get_logger().warn(f'{self.target_item} raw depth 실패(0) - 발행 안 함')
            return

        # ===== [거리 판정] 파지 가능 범위 체크 =====
        dist_mm = dist_m * 1000
        glo, ghi = GRASP_DEPTH_RANGE.get(self.shelf_level, (210, 301))
        if dist_mm < glo:
            self.get_logger().warn(
                f'[거리] 너무 가까움 {dist_mm:.0f} < {glo} - 파지 보류 (AGV 후진 필요)'
            )
            self._pub_dist_status('too_close', dist_mm)
            self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=True)
            return
        elif dist_mm > ghi:
            self.get_logger().warn(
                f'[거리] 너무 멈 {dist_mm:.0f} > {ghi} - 파지 보류 (AGV 전진 필요)'
            )
            self._pub_dist_status('too_far', dist_mm)
            self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=True)
            return
        else:
            # 작업 가능 거리 → 현재 거리 알림 후 파지 진행
            self._pub_dist_status('ok', dist_mm)

        # 카메라 3D 좌표 (deproject)
        fx, fy, ppx, ppy = self.intrinsics
        X = (cx - ppx) / fx * dist_m
        Y = (cy - ppy) / fy * dist_m
        Z = dist_m
        cam_xyz = [X, Y, Z]

        self.get_logger().info(
            f'{self.target_item} 발견 | 픽셀=({cx},{cy}) '
            f'dist={dist_m:.3f}m cam_xyz={[round(v, 3) for v in cam_xyz]}'
        )

        # ===== [동적 T] 변환 =====
        cam_pt = np.array([cam_xyz[0]*1000.0, cam_xyz[1]*1000.0, cam_xyz[2]*1000.0, 1.0])
        T = self._current_T()
        base_pt = (T @ cam_pt)[:3]
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
        try:
            cv2.imwrite('/home/zzz/pj3_ws/deburg/detect_latest.jpg', img)
        except Exception:
            pass
        self.mode = MODE_IDLE

    def _get_robust_depth(self, cx, cy, k=12):
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

    # ---------- QR 검증 ----------
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

    # ---------- QR place ----------
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

        cam_pt = np.array([X*1000.0, Y*1000.0, Z*1000.0, 1.0])
        T = self._current_T()
        base_pt = (T @ cam_pt)[:3]

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
