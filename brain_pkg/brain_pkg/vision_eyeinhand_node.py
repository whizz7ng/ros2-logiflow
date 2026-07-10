#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_node.py  (eye-in-hand / 동적 T + 거리판정 + 마커 yaw 버전)

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

[신규 - 마커 yaw]
  ArUco 마커의 rvec(회전)까지 solvePnP로 계산해서, 마커 하나만 보여도
  AGV(팔 base) 기준 yaw(수평 회전 오차)를 뽑아낼 수 있게 함.
  /marker_agv_pose 포맷이 [level, Lx, Ly, Rx, Ry] (5개)에서
  [level, Lx, Ly, Rx, Ry, Lyaw, Ryaw] (7개)로 확장됨.
  agv_align_node가 이걸로 "정면 정렬" 단계를 마커 하나만 보여도 수행할 수 있음.

[신규 - 전체 흐름 변경]
  더 이상 마커 기준 고정좌표(TARGET lx/ly/rx/ry)로 정밀 정렬하지 않음.
  대신:
    1) 관측 자세 도착 → MODE_MARKER_ALIGN으로 정면(yaw)만 먼저 맞춤
    2) 정면 끝나면 brain이 곧바로 블록 검출(MODE_BLOCK) activate
    3) 블록이 파지범위(GRASP_DEPTH_RANGE) 안 + 화면 중앙 근처면 바로 /box_pose
    4) 범위 밖(가까움/멂)이면 /distance_status too_close/too_far (기존과 동일)
    5) 화면 중앙에서 너무 치우치면(BLOCK_CENTER_MIN_X~MAX_X 밖)
       /distance_status side_left/side_right (신규) → AGV 좌우 보정 요청
  4),5) 보정 후에는 다시 관측 자세부터(마커 정면정렬 포함) 반복.
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
MODE_MARKER_ALIGN = 'marker_align'   # [신규] 블록 검출 전 마커 정면(yaw) 정렬 전용 모드

# place 오프셋 (QR → 실제 놓을 위치, 실측 필요)
PLACE_OFFSET_X = -150.0
PLACE_OFFSET_Y = 10.0
PLACE_OFFSET_Z = -40.0
PLACE_RX = -178.0
PLACE_RY = 0.0
PLACE_RZ = -90.0

# QR place 거리 기준(mm)
QR_PLACE_MAX_MM = 420.0
QR_PLACE_MIN_MM = 250.0

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

DEPTH_RANGE = {
    1: (150, 450),
    2: (150, 450),
}

GRASP_DEPTH_RANGE = {
    1: (230, 305),
    2: (230, 310),
}

# ===== [마커 J1 보정 - 픽셀 기반, 구버전] =====
MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKER_LENGTH = 0.0365          # 한 변 3.65cm
MARKER_ID_LEFT  = 0
MARKER_ID_RIGHT = 1

MARKER_REF_PX = {
    1: {MARKER_ID_LEFT: 126, MARKER_ID_RIGHT: 548},   # 1층
    2: {MARKER_ID_LEFT: 190, MARKER_ID_RIGHT: 510},   # 2층
}
PIXEL_TO_J1 = 10.0 / 256.0
J1_SIGN = -1.0
J1_CORRECTION_MAX = 25.0
MARKER_REALIGN_MAX = 2
NOT_FOUND_LIMIT = 15

# ===== [AGV 보정용] 팔 g_base → AGV base_link 오프셋 (실측, mm) =====
GBASE_TO_AGV_OFFSET = (75.0, 0.0, 135.0)   # (x, y, z) mm

BLOCK_CENTER_MIN_X = 270 #220
BLOCK_CENTER_MAX_X = 370 #420
BLOCK_CENTER_TARGET_X = 320
BLOCK_PIXEL_TO_J1 = 10.0 / 256.0
BLOCK_J1_SIGN = -1.0
BLOCK_CENTER_CORRECTION_MAX = 20.0
BLOCK_CENTER_REALIGN_MAX = 1


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
        self.not_found_count = 0
        self.cut_count = 0
        self.realign_count = 0
        self.center_miss_count = 0
        self.depth_fail_count = 0

        self.get_logger().info(f'YOLO 모델 로드 중: {MODEL_PATH}')
        self.model = YOLO(MODEL_PATH)

        self.X_cam2gripper = np.load(X_CAM2GRIPPER_PATH)

        self.T_CAM2BASE = {
            s: _coords_to_matrix(p) @ self.X_cam2gripper
            for s, p in SHELF_POSES.items()
        }
        self.current_T_cam2base = None
        self.have_fresh_observe_pose = False

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

        self.create_subscription(Image, TOPIC_COLOR, self._color_callback, 10)
        self.create_subscription(Image, TOPIC_DEPTH, self._depth_callback, 10)
        self.create_subscription(CameraInfo, TOPIC_CAMINFO, self._caminfo_callback, 10)
        self.create_subscription(String, '/vision_activate', self._activate_callback, 10)
        self.create_subscription(String, '/brain_state',     self._state_callback,    10)
        self.create_subscription(
            Float32MultiArray, '/observe_pose', self._observe_pose_callback, 10
        )

        self._box_pose_pub       = self.create_publisher(Float32MultiArray, '/box_pose',       10)
        self._qr_pub             = self.create_publisher(String,            '/depth_qr',       10)
        self._detected_image_pub = self.create_publisher(CompressedImage,   '/detected_image', 10)
        self._place_pose_pub     = self.create_publisher(Float32MultiArray, '/place_pose',     10)
        self._dist_status_pub    = self.create_publisher(String, '/distance_status', 10)
        self._j1_corr_pub        = self.create_publisher(String, '/j1_correction', 10)
        # [AGV 보정] [level, Lx, Ly, Rx, Ry, Lyaw, Ryaw] (mm/deg, AGV base 기준).
        # 안 보이는 마커는 해당 위치/yaw NaN.
        self._marker_agv_pub     = self.create_publisher(Float32MultiArray, '/marker_agv_pose', 10)

        self.get_logger().info('vision_node 시작 (eye-in-hand / 동적 T / 거리판정 / 마커yaw)')

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
        self.have_fresh_observe_pose = True
        self.get_logger().info(
            f'[동적 T] 관측 자세 수신 → T 갱신: {[round(v, 1) for v in pose]}'
        )

    def _current_T(self):
        if self.current_T_cam2base is not None:
            return self.current_T_cam2base
        self.get_logger().warn('[동적 T] 아직 없음 → 고정 T 폴백 사용')
        return self.T_CAM2BASE.get(self.shelf_level)

    def _pub_dist_status(self, status, dist_mm):
        m = String()
        m.data = f'{status}:{dist_mm:.0f}'
        self._dist_status_pub.publish(m)

    def _detect_markers(self):
        """좌우 마커 검출 → {id: 중심픽셀x} 반환. 없으면 빈 dict. (픽셀 J1 보정용)"""
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

    def _detect_markers_pose(self):
        """좌우 마커의 3D pose(카메라 기준) 검출.
        반환: {id: {'tvec': np.array([x,y,z] m), 'rvec': np.array([rx,ry,rz])}}.
        없으면 빈 dict.
        (estimatePoseSingleMarkers가 없는 신버전 OpenCV: solvePnP 사용)"""
        if self.color_img is None or self.intrinsics is None:
            return {}
        fx, fy, ppx, ppy = self.intrinsics
        K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5)
        if self._aruco_new:
            corners, ids, _ = self.aruco_detector.detectMarkers(self.color_img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                self.color_img, self.aruco_dict, parameters=self.aruco_params)
        result = {}
        if ids is None:
            return result

        # 마커 한 변 절반 크기로 3D 코너 좌표 정의 (마커 중심 원점)
        h = MARKER_LENGTH / 2.0
        obj_pts = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0],
        ], dtype=np.float64)

        for i, mid in enumerate(ids.flatten()):
            img_pts = corners[i][0].astype(np.float64)  # 4x2
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok:
                result[int(mid)] = {
                    'tvec': tvec.flatten(),   # [x, y, z] (m), 카메라 기준
                    'rvec': rvec.flatten(),   # 회전벡터, 카메라 기준
                }
        return result

    def _marker_to_agv(self, tvec_m):
        """마커 tvec(카메라 기준, m) → AGV base 기준 좌표(mm).
        동적 T로 팔base 좌표 구한 뒤, static 오프셋(+75,0,+135) 더함 (yaw=0)."""
        T = self._current_T()
        if T is None:
            return None
        cam_pt = np.array([tvec_m[0]*1000.0, tvec_m[1]*1000.0, tvec_m[2]*1000.0, 1.0])
        base_pt = (T @ cam_pt)[:3]   # 팔 g_base 기준 (mm)
        ox, oy, oz = GBASE_TO_AGV_OFFSET
        agv = [float(base_pt[0] + ox), float(base_pt[1] + oy), float(base_pt[2] + oz)]
        return agv

    def _marker_yaw_deg(self, rvec):
        """[신규] 마커 rvec(카메라 기준 회전) → AGV(=팔 base) 기준 yaw(deg).

        R_marker2cam(rvec)을 현재 T_cam2base의 회전 성분으로 base 좌표계로
        변환한 뒤, 'xyz' 오일러(=Rz@Ry@Rx, _coords_to_matrix와 동일 컨벤션)의
        z 성분(=수평 회전, yaw)을 사용한다.
        GBASE_TO_AGV_OFFSET이 회전 없이 오프셋만(yaw=0) 이므로
        base 기준 yaw == AGV 기준 yaw로 취급한다.

        관측 자세(SHELF_ANGLES)는 층별로 고정된 조인트각이라 R_cam2base는
        (미세한 처짐/오차를 빼면) 사실상 상수이므로, 이 값의 변화는
        거의 전부 '마커(고정된 랙)를 보는 AGV의 실제 자세 오차'를 반영한다.

        마커가 완벽히 정면 정렬됐을 때도 이 값이 정확히 0이 된다는 보장은
        없음(랙/마커 설치각에 따라 오프셋이 있을 수 있음) - 정상 정차에서
        한 번 echo해서 agv_align_node의 YAW_TARGET으로 넣어줄 것.
        """
        T = self._current_T()
        if T is None:
            return None
        try:
            R_marker2cam, _ = cv2.Rodrigues(rvec)
        except Exception as e:
            self.get_logger().warn(f'[마커yaw] Rodrigues 변환 실패: {e}')
            return None
        R_cam2base = T[:3, :3]
        R_marker2base = R_cam2base @ R_marker2cam
        euler = R.from_matrix(R_marker2base).as_euler('xyz', degrees=True)
        return float(euler[2])

    def _publish_marker_agv(self):
        """AGV 보정용: 두 마커의 AGV 기준 좌표 + yaw 발행.
        realign_fail / too_far / too_close 등 '팔로 못 잡음' 상황에서 호출.
        형식 [level, Lx, Ly, Rx, Ry, Lyaw, Ryaw] (mm/deg).
        안 보이는 마커는 해당 위치/yaw 값 NaN."""
        poses = self._detect_markers_pose()
        nan = float('nan')
        # level, Lx, Ly, Rx, Ry, Lyaw, Ryaw
        vals = [float(self.shelf_level), nan, nan, nan, nan, nan, nan]

        if MARKER_ID_LEFT in poses:
            p = poses[MARKER_ID_LEFT]
            agv = self._marker_to_agv(p['tvec'])
            if agv is not None:
                vals[1], vals[2] = agv[0], agv[1]
            yaw = self._marker_yaw_deg(p['rvec'])
            if yaw is not None:
                vals[5] = yaw

        if MARKER_ID_RIGHT in poses:
            p = poses[MARKER_ID_RIGHT]
            agv = self._marker_to_agv(p['tvec'])
            if agv is not None:
                vals[3], vals[4] = agv[0], agv[1]
            yaw = self._marker_yaw_deg(p['rvec'])
            if yaw is not None:
                vals[6] = yaw

        m = Float32MultiArray()
        m.data = [float(v) for v in vals]
        self._marker_agv_pub.publish(m)
        self.get_logger().info(
            f'[AGV보정] 마커 AGV 좌표/yaw 발행 (L{self.shelf_level}): '
            f'L=({vals[1]:.0f},{vals[2]:.0f},yaw={vals[5]:.1f}) '
            f'R=({vals[3]:.0f},{vals[4]:.0f},yaw={vals[6]:.1f})'
        )

    def _emit_realign_fail(self):
        """팔로 못 잡음 → realign_fail 발행 + 마커 AGV 좌표 발행 (AGV 보정용)."""
        self._j1_corr_pub.publish(String(data='realign_fail'))
        self._publish_marker_agv()
        self.realign_count = 0

    def _try_marker_realign(self):
        """블록 안 보일 때: 개별 마커 기준으로 J1 보정량 계산해서 /j1_correction 발행.
        (구버전 픽셀 기반 - agv_align_node의 정면정렬 단계와 별개로 유지)"""
        if self.realign_count >= MARKER_REALIGN_MAX:
            self.get_logger().error(
                f'[마커보정] {MARKER_REALIGN_MAX}회 반복해도 못 찾음 → realign_fail (AGV 재정차)'
            )
            self._emit_realign_fail()
            return

        markers = self._detect_markers()
        ref = MARKER_REF_PX.get(self.shelf_level, {})
        if not ref:
            self.get_logger().error(f'[마커보정] 층 {self.shelf_level} 기준 없음')
            self._emit_realign_fail()
            return

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
            self._emit_realign_fail()
            return

        j1_corr = sum(corrections) / len(corrections)
        self.get_logger().info(
            f'[마커보정] {len(corrections)}개 검출 [{", ".join(seen)}] → J1보정={j1_corr:.1f}도'
        )

        if abs(j1_corr) > J1_CORRECTION_MAX:
            self.get_logger().warn(
                f'[마커보정] 보정량 {j1_corr:.1f}도 > 한계 {J1_CORRECTION_MAX} '
                f'→ realign_fail (AGV 재정차)'
            )
            self._emit_realign_fail()
            return

        self.realign_count += 1
        self._j1_corr_pub.publish(String(data=f'{self.shelf_level}:{j1_corr:.2f}'))
        self.get_logger().info(
            f'[마커보정] /j1_correction 발행: {self.shelf_level}:{j1_corr:.2f} '
            f'({self.realign_count}/{MARKER_REALIGN_MAX}회째)'
        )
        self.mode = MODE_IDLE

    def _try_block_center_realign(self, cx):
        if self.realign_count >= BLOCK_CENTER_REALIGN_MAX:
            self.get_logger().error(
                f'[블록중심보정] {BLOCK_CENTER_REALIGN_MAX}회 보정 후에도 중앙 정렬 실패 '
                f'→ realign_fail (AGV 재정차)'
            )
            self._emit_realign_fail()
            return

        offset_px = cx - BLOCK_CENTER_TARGET_X
        j1_corr = BLOCK_J1_SIGN * offset_px * BLOCK_PIXEL_TO_J1

        self.get_logger().info(
            f'[블록중심보정] cx={cx}, target={BLOCK_CENTER_TARGET_X}, '
            f'off={offset_px} → J1보정={j1_corr:.1f}도'
        )

        if abs(j1_corr) > BLOCK_CENTER_CORRECTION_MAX:
            self.get_logger().warn(
                f'[블록중심보정] 보정량 {j1_corr:.1f}도 > 한계 {BLOCK_CENTER_CORRECTION_MAX} '
                f'→ realign_fail (AGV 재정차)'
            )
            self._emit_realign_fail()
            return

        self.realign_count += 1
        self._j1_corr_pub.publish(String(data=f'{self.shelf_level}:{j1_corr:.2f}'))
        self.get_logger().info(
            f'[블록중심보정] /j1_correction 발행: {self.shelf_level}:{j1_corr:.2f} '
            f'({self.realign_count}/{BLOCK_CENTER_REALIGN_MAX}회째)'
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
        elif data.startswith('marker_align:'):
            # [신규] 블록 검출 전, 마커 정면(yaw) 정렬 전용 모드.
            # agv_align_node가 STAGE1(정면정렬)만 수행하고, 끝나면 brain이
            # 곧바로 'item:level'로 다시 activate해서 블록 검출로 넘어간다.
            level_str = data.split(':', 1)[1].strip()
            try:
                level = int(level_str)
            except ValueError:
                self.get_logger().error(f'marker_align 레벨 파싱 실패: {data}')
                return
            if level not in DEPTH_RANGE:
                self.get_logger().error(f'알 수 없는 층: {level} - marker_align 무시')
                return
            self.shelf_level = level
            self.mode = MODE_MARKER_ALIGN
            self.get_logger().info(f'[정면정렬] 마커 정렬 모드 진입 - 층 {level}')
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
            self.cut_count = 0
            self.center_miss_count = 0
            self.depth_fail_count = 0

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
        elif self.mode == MODE_MARKER_ALIGN:
            self._check_marker_align()

    # ---------- 블록 검출 ----------
    def _detect_block(self):
        if self.depth_img is None or self.intrinsics is None:
            self.get_logger().warn('depth/intrinsic 아직 준비 안 됨')
            return

        if not getattr(self, "have_fresh_observe_pose", False):
            self.get_logger().warn("[안전] fresh observe_pose 없음 - 좌표 계산 보류")
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
            self.cut_count = 0

            self.get_logger().warn(
                f'{self.target_item} 못 찾음 ({self.not_found_count}/{NOT_FOUND_LIMIT})'
            )

            if self.not_found_count >= NOT_FOUND_LIMIT:
                self.get_logger().warn('→ 마커 J1 보정 시도')
                self._try_marker_realign()
                self.not_found_count = 0

            return

        self.not_found_count = 0

        x1, y1, x2, y2 = map(int, target_box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        self.get_logger().info(
            f'[bbox] x1={x1} y1={y1} x2={x2} y2={y2} cx={cx} cy={cy} '
            f'(H={img.shape[0]} W={img.shape[1]})'
        )

        H, W = img.shape[:2]
        margin = 5

        if x1 <= margin or y1 <= margin or x2 >= W - margin:
            self.cut_count += 1

            self.get_logger().warn(
                f'{self.target_item} 잘림 감지 ({self.cut_count}/{NOT_FOUND_LIMIT}) - 재정렬 필요'
            )

            self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=True)

            if self.cut_count >= NOT_FOUND_LIMIT:
                self.get_logger().warn('→ 마커 J1 보정 시도 (잘림)')
                self._try_marker_realign()
                self.cut_count = 0

            return

        self.cut_count = 0
        self.realign_count = 0

        dmin, dmax = DEPTH_RANGE.get(self.shelf_level, (110, 300))
        roi = self.depth_img[y1:y2, x1:x2]
        valid = roi[(roi > dmin) & (roi < dmax)]

        if valid.size < 30:
            self.depth_fail_count += 1
            self.get_logger().warn(
                f'depth 없음 ({self.depth_fail_count}/{NOT_FOUND_LIMIT}) - ArUco 기반 차체교정 대기'
            )

            if self.depth_fail_count >= NOT_FOUND_LIMIT:
                self.get_logger().warn('→ depth 연속 실패: ArUco 마커 기반 AGV 자세교정 요청')

                poses = self._detect_markers_pose()

                if not poses:
                    self.get_logger().warn('depth 실패했지만 ArUco 마커도 안 보임 → realign_fail')
                    self._emit_realign_fail()
                else:
                    self._pub_dist_status('depth_fail', 0)
                    self.get_logger().warn('/distance_status 발행: depth_fail:0')

                    self._publish_marker_agv()

                self.mode = MODE_IDLE
                self.depth_fail_count = 0

            return

        self.depth_fail_count = 0

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

        dist_mm = dist_m * 1000
        glo, ghi = GRASP_DEPTH_RANGE.get(self.shelf_level, (210, 301))
        if dist_mm < glo:
            self.get_logger().warn(
                f'[거리] 너무 가까움 {dist_mm:.0f} < {glo} - 파지 보류 (AGV 후진 필요)'
            )
            self._pub_dist_status('too_close', dist_mm)
            self._publish_marker_agv()
            
            # AGV 후진 보정이 들어가면 기존 observe_pose는 더 이상 신뢰하지 않음
            self.have_fresh_observe_pose = False
            
            self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=True)
            self.mode = MODE_IDLE
            return

        elif dist_mm > ghi:
            self.get_logger().warn(
                f'[거리] 너무 멈 {dist_mm:.0f} > {ghi} - 파지 보류 (AGV 전진 필요)'
            )
            self._pub_dist_status('too_far', dist_mm)
            self._publish_marker_agv()
            
            # AGV 전진 보정이 들어가면 기존 observe_pose는 더 이상 신뢰하지 않음
            self.have_fresh_observe_pose = False
            
            self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=True)
            self.mode = MODE_IDLE
            return

        else:
            # ===== 블록 중심이 화면에서 너무 치우쳐 있으면 =====
            # 거리만 ok여도 중심이 틀어졌으면 pick 진행 금지.
            # 중요: side_left/right를 보내기 전에 ok를 보내면 brain이 오판할 수 있으므로
            # ok는 중심까지 통과한 뒤에만 발행한다.
            if cx < BLOCK_CENTER_MIN_X or cx > BLOCK_CENTER_MAX_X:
                side = 'left' if cx < BLOCK_CENTER_MIN_X else 'right'
                self.get_logger().warn(
                    f'[중심보정] 블록 cx={cx} 허용범위=({BLOCK_CENTER_MIN_X}~{BLOCK_CENTER_MAX_X}) '
                    f'밖({side}) → AGV 좌우보정 요청, pick 보류'
                )
                self._pub_dist_status(f'side_{side}', dist_mm)

                # AGV 좌우 보정이 들어가면 기존 observe_pose는 더 이상 신뢰하지 않음
                self.have_fresh_observe_pose = False
                
                self._draw_and_publish(img, x1, y1, x2, y2, self.target_item, cut=True)
                self.mode = MODE_IDLE
                return
        
            # 거리 + 중심 모두 통과했을 때만 ok 발행
            self.get_logger().info(
                f'[거리/중심] 파지 조건 통과 dist={dist_mm:.0f}mm, cx={cx} '
                f'→ /box_pose 진행'
            )
            self._pub_dist_status('ok', dist_mm)

        fx, fy, ppx, ppy = self.intrinsics
        X = (cx - ppx) / fx * dist_m
        Y = (cy - ppy) / fy * dist_m
        Z = dist_m
        cam_xyz = [X, Y, Z]

        self.get_logger().info(
            f'{self.target_item} 발견 | 픽셀=({cx},{cy}) '
            f'dist={dist_m:.3f}m cam_xyz={[round(v, 3) for v in cam_xyz]}'
        )

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

        self.have_fresh_observe_pose = False

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

    def _get_qr_place_depth(self, cx, cy):
        if self.depth_img is None:
            return 0.0

        H, W = self.depth_img.shape[:2]

        for kk in [12, 16, 20, 25, 30]:
            y0, y1 = max(0, cy - kk), min(H, cy + kk + 1)
            x0, x1 = max(0, cx - kk), min(W, cx + kk + 1)

            patch = self.depth_img[y0:y1, x0:x1]

            valid = patch[(patch > 160) & (patch < 600)]

            self.get_logger().info(
                f"[QR DEPTH TRY] patch k={kk}, valid={valid.size}"
            )

            if valid.size < 30:
                continue

            depth_mm = float(np.percentile(valid, 30))

            self.get_logger().info(
                f"[QR DEPTH SELECT] patch k={kk}, valid={valid.size}, p30={depth_mm:.0f}mm"
            )

            return depth_mm / 1000.0

        self.get_logger().warn(
            f"[QR DEPTH FAIL] cx={cx}, cy={cy}, 모든 patch에서 valid depth 부족"
        )
        return 0.0

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

        qr_found = False
        zone = '?'
        cx = cy = None

        if decoded:
            candidates = []

            H, W = self.color_img.shape[:2]

            for obj in decoded:
                try:
                    data = obj.data.decode('utf-8').strip().upper()
                except Exception:
                    continue

                pts = obj.polygon
                if not pts:
                    continue

                qx = sum(p.x for p in pts) / len(pts)
                qy = sum(p.y for p in pts) / len(pts)

                valid_rank = 0 if data in VALID_ZONES else 1
                dist_center = (qx - W / 2) ** 2 + (qy - H / 2) ** 2

                candidates.append((valid_rank, dist_center, data, qx, qy))

            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1]))
                _, _, zone, qx, qy = candidates[0]
                cx = int(qx)
                cy = int(qy)
                qr_found = True
                self.get_logger().info(f'QR pyzbar 인식: zone={zone} cx={cx} cy={cy}')

        if not qr_found:
            gray = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2GRAY)

            qr_detector = cv2.QRCodeDetector()
            data, points, _ = qr_detector.detectAndDecode(gray)

            if data and points is not None:
                zone = data.strip().upper()

                pts = points.reshape(-1, 2)
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))

                qr_found = True
                self.get_logger().info(f'QR OpenCV 인식: zone={zone} cx={cx} cy={cy}')

        if not qr_found:
            self.get_logger().warn('QR 못 찾음, 재시도')
            return

        dist_m = self._get_qr_place_depth(cx, cy)
        if dist_m <= 0:
            self.get_logger().warn('QR depth 측정 실패(0) - 재시도')
            return

        dist_mm = dist_m * 1000.0

        if dist_mm > QR_PLACE_MAX_MM:
            self.get_logger().warn(
                f"[QR PLACE ALIGN] QR이 너무 멂: {dist_mm:.0f}mm > {QR_PLACE_MAX_MM:.0f}mm "
                "→ AGV 전진 보정 필요, /place_pose 발행 안 함"
            )
            self._pub_dist_status('qr_too_far', dist_mm)
            self.mode = MODE_IDLE
            return

        if dist_mm < QR_PLACE_MIN_MM:
            self.get_logger().warn(
                f"[QR PLACE ALIGN] QR이 너무 가까움: {dist_mm:.0f}mm < {QR_PLACE_MIN_MM:.0f}mm "
                "→ /place_pose 발행 안 함"
            )
            self._pub_dist_status('qr_too_close', dist_mm)
            self.mode = MODE_IDLE
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

    # ---------- [신규] 마커 정면(yaw) 정렬 전용 모드 ----------
    def _check_marker_align(self):
        """
        블록 검출을 시작하기 전, 마커가 보이는 동안 매 프레임 마커 AGV
        좌표/yaw를 발행한다. 실제 "정렬됐는지/더 돌려야 하는지" 판단과
        펄스 이동은 agv_align_node가 담당(STAGE1). 여기선 관측만 반복.

        agv_align_node가 정면정렬 완료로 판단하면 /align_status "aligned"를
        보내고, brain_node가 이걸 받아서 이 모드를 끝내고(item:level로
        재activate) 블록 검출로 넘어간다.
        """
        if not getattr(self, "have_fresh_observe_pose", False):
            self.get_logger().warn("[정면정렬] fresh observe_pose 없음 - 대기")
            return
        self._publish_marker_agv()


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
