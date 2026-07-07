#!/usr/bin/env python3

import time
import math
import json

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def normalize_key(text):
    return str(text).strip().upper()


def parse_marker_id_map(text):
    """
    "A:88,B:87,C:73,RACK:89,OBJ:89,TO_OBJ:89" -> dict
    """
    out = {}
    for item in str(text).split(','):
        item = item.strip()
        if not item or ':' not in item:
            continue
        k, v = item.split(':', 1)
        k = normalize_key(k)
        try:
            out[k] = int(v.strip())
        except Exception:
            pass
    return out


def safe_set(obj, name, value):
    """
    OpenCV 버전마다 DetectorParameters 필드가 조금씩 다르므로
    존재하는 필드만 안전하게 세팅한다.
    """
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False
    return False


class ArucoAlignNode(Node):
    """
    myAGV ArUco final align node.

    추가된 핵심:
      - /aruco_align/debug_json std_msgs/String 발행
      - raw detection 전체 id / 크기 / 비율 / valid 여부를 JSON으로 확인 가능
      - target id가 아예 안 잡히는지, 잡혔는데 reject되는지 분리 가능

    Command:
      /aruco_align_cmd std_msgs/String
        "start"
        "stop"
        "reset"

    Target update:
      /marker_align_target std_msgs/String
        "A", "B", "C", "RACK", "OBJ", "TO_OBJ" 또는 숫자 id 문자열.

    Publish:
      /cmd_vel_nav geometry_msgs/Twist
      /aruco_align_status std_msgs/String
      /aruco_align_done std_msgs/Bool
      /aruco_align/debug_image sensor_msgs/Image
      /aruco_align/debug_json std_msgs/String
    """

    def __init__(self):
        super().__init__('aruco_align_node')

        # ============================================================
        # Topics
        # ============================================================
        self.declare_parameter('image_topic', '/myagv_camera/image_raw')
        self.declare_parameter('cmd_topic', '/cmd_vel_nav')
        self.declare_parameter('status_topic', '/aruco_align_status')
        self.declare_parameter('done_topic', '/aruco_align_done')
        self.declare_parameter('debug_image_topic', '/aruco_align/debug_image')
        self.declare_parameter('debug_json_topic', '/aruco_align/debug_json')

        self.declare_parameter('command_topic', '/aruco_align_cmd')
        self.declare_parameter('target_name_topic', '/marker_align_target')
        self.declare_parameter('enable_target_name_sub', True)

        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.done_topic = self.get_parameter('done_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.debug_json_topic = self.get_parameter('debug_json_topic').value

        self.command_topic = self.get_parameter('command_topic').value
        self.target_name_topic = self.get_parameter('target_name_topic').value

        # ============================================================
        # ArUco params
        # ============================================================
        self.declare_parameter('dict_name', 'DICT_4X4_100')
        self.declare_parameter('target_id', -1)
        self.declare_parameter('target_name', 'RACK')
        self.declare_parameter(
            'marker_id_map',
            'C:73,B:87,A:88,RACK:89,OBJ:89,TO_OBJ:89'
        )

        self.declare_parameter('process_every_n_frames', 1)

        # ============================================================
        # Detection quality gate params
        # ============================================================
        self.declare_parameter('min_control_side_px', 60.0)
        self.declare_parameter('max_control_aspect_ratio', 1.50)
        self.declare_parameter('min_control_area_px', 0.0)
        self.declare_parameter('prefer_largest_marker', True)
        self.declare_parameter('clear_marker_on_reject', True)

        self.declare_parameter('print_reject_debug', True)
        self.declare_parameter('reject_debug_period_sec', 0.50)

        # ============================================================
        # JSON debug params
        # ============================================================
        self.declare_parameter('publish_debug_json', True)
        self.declare_parameter('debug_json_period_sec', 0.20)
        self.declare_parameter('debug_json_max_markers', 20)
        self.declare_parameter('debug_json_include_corners', True)

        # ============================================================
        # Align params
        # ============================================================
        self.declare_parameter('target_size_px', 200.0)
        self.declare_parameter('size_tolerance_px', 0.0)

        self.declare_parameter('target_cx_px', -1.0)
        self.declare_parameter('center_tolerance_px', 60.0)

        self.declare_parameter('center_first', True)

        self.declare_parameter('kp_vy', 0.00015)
        self.declare_parameter('kp_vx', 0.00100)

        self.declare_parameter('max_vx', 0.045)
        self.declare_parameter('min_vx', 0.018)
        self.declare_parameter('max_vy', 0.015)
        self.declare_parameter('min_vy', 0.006)

        # image x 오른쪽 증가. ROS base_link +y는 왼쪽.
        # marker가 오른쪽(err_cx +)이면 오른쪽 이동을 위해 vy 음수가 보통 맞음.
        self.declare_parameter('invert_y', False)

        self.declare_parameter('lost_timeout_sec', 1.5)

        # ============================================================
        # Frame-cut / lost-marker recovery params
        # ============================================================
        # marker가 화면 밖으로 잘렸을 때, 마지막으로 보였던 marker의 cx 오차를
        # 기준으로 linear.y만 짧게 내서 다시 frame 안으로 끌어오는 로직.
        self.declare_parameter('enable_lost_recovery', True)

        # 이 시간 이상 marker update가 없으면 stale marker로 보고 recovery를 시작한다.
        # 너무 크게 잡으면 옛 marker로 계속 전진하는 시간이 길어진다.
        self.declare_parameter('lost_recovery_start_sec', 0.25)

        # recovery를 최대 몇 초 동안 지속할지. 이후에는 안전하게 stop.
        self.declare_parameter('lost_recovery_sec', 1.40)

        # recovery 때 내보낼 lateral speed. 너무 크면 반대로 튄다.
        self.declare_parameter('lost_recovery_vy', 0.012)

        # 마지막 marker 오차가 이 값보다 작으면 방향성이 애매하므로 recovery하지 않는다.
        self.declare_parameter('lost_recovery_min_err_px', 25.0)

        # 마지막 유효 marker가 너무 오래된 정보면 recovery하지 않는다.
        self.declare_parameter('lost_recovery_max_age_sec', 3.0)

        self.declare_parameter('lost_recovery_debug_period_sec', 0.30)

        self.declare_parameter('allow_reverse', False)
        self.declare_parameter('max_reverse_vx', 0.025)

        self.declare_parameter('done_required_count', 3)
        self.declare_parameter('marker_smoothing_alpha', 0.35)
        self.declare_parameter('stop_publish_sec', 1.0)

        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('print_debug', True)

        # ============================================================
        # State
        # ============================================================
        self.bridge = CvBridge()

        self.active = False
        self.done = False
        self.done_time = 0.0
        self.done_count = 0

        self.last_seen_time = 0.0
        self.last_marker = None
        self.smoothed_marker = None

        # Lost recovery memory.
        # 마지막으로 marker가 의미 있게 치우쳐 보였던 방향을 기억했다가,
        # frame cut 시 같은 방향으로 vy를 잠깐 더 준다.
        self.last_recovery_err_cx = None
        self.last_recovery_marker_time = 0.0
        self.last_recovery_vy = 0.0
        self.last_recovery_status_time = 0.0

        self.latest_image_width = 640
        self.latest_image_height = 360

        self.frame_count = 0
        self.last_detection_time = 0.0

        # detection debug state
        self.last_raw_markers = []
        self.last_target_candidates = []
        self.last_filtered_markers = []
        self.last_rejected_markers = []
        self.last_raw_ids = []
        self.last_raw_count = 0
        self.last_target_candidate_count = 0
        self.last_accepted_count = 0
        self.last_rejected_count = 0
        self.last_detect_error = ''

        # control debug state
        self.last_control_info = {
            'reason': 'INIT',
            'vx': 0.0,
            'vy': 0.0,
            'wz': 0.0,
            'center_ok': False,
            'size_ok': False,
            'done_count': 0,
            'required_count': 0,
            'err_cx': None,
            'err_size': None,
            'target_cx': None,
            'target_size_px': None,
        }

        self.last_reject_debug_time = 0.0
        self.last_json_publish_time = 0.0

        # ============================================================
        # Detector
        # ============================================================
        self.aruco_dict = self.get_aruco_dictionary()
        self.aruco_params = self.create_detector_params()

        # ============================================================
        # QoS
        # ============================================================
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            image_qos
        )

        self.cmd_sub = self.create_subscription(
            String,
            self.command_topic,
            self.cmd_cb,
            10
        )

        if bool(self.get_parameter('enable_target_name_sub').value):
            self.target_name_sub = self.create_subscription(
                String,
                self.target_name_topic,
                self.target_name_cb,
                10
            )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, cmd_qos)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.done_pub = self.create_publisher(Bool, self.done_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, image_qos)
        self.json_pub = self.create_publisher(String, self.debug_json_topic, 10)

        self.timer = self.create_timer(0.05, self.timer_cb)  # 20 Hz

        # Initial target_name -> target_id
        self.apply_target_name(self.get_parameter('target_name').value, announce=False)

        self.status(
            f'aruco_align_ready | dict={self.get_parameter("dict_name").value} '
            f'target_name={self.get_parameter("target_name").value} '
            f'target_id={self.get_parameter("target_id").value} '
            f'map={self.get_parameter("marker_id_map").value} '
            f'min_control_side={self.get_parameter("min_control_side_px").value} '
            f'max_aspect={self.get_parameter("max_control_aspect_ratio").value} '
            f'debug_json={self.debug_json_topic}'
        )

    # ============================================================
    # Parameter helpers
    # ============================================================
    def p(self, name):
        return self.get_parameter(name).value

    def marker_map(self):
        return parse_marker_id_map(self.p('marker_id_map'))

    def apply_target_name(self, raw_name, announce=True):
        name = normalize_key(raw_name)
        mp = self.marker_map()

        target_id = None

        try:
            target_id = int(name)
        except Exception:
            target_id = None

        if target_id is None:
            if name in mp:
                target_id = int(mp[name])
            else:
                target_id = -1

        self.set_parameters([
            Parameter('target_name', Parameter.Type.STRING, name),
            Parameter('target_id', Parameter.Type.INTEGER, int(target_id)),
        ])

        self.last_marker = None
        self.smoothed_marker = None
        self.done_count = 0
        self.reset_lost_recovery_memory()

        self.last_raw_markers = []
        self.last_target_candidates = []
        self.last_filtered_markers = []
        self.last_rejected_markers = []
        self.last_raw_ids = []
        self.last_raw_count = 0
        self.last_target_candidate_count = 0
        self.last_accepted_count = 0
        self.last_rejected_count = 0
        self.last_detect_error = ''

        self.last_control_info['reason'] = 'TARGET_UPDATED'

        if announce:
            self.status(f'target_updated:name={name}:id={target_id}')
            self.publish_debug_json(event='target_updated', force=True)

    # ============================================================
    # ArUco compatibility
    # ============================================================
    def get_aruco_dictionary(self):
        dict_name = str(self.get_parameter('dict_name').value)

        if not hasattr(cv2, 'aruco'):
            raise RuntimeError('cv2.aruco is not available. Install opencv-contrib.')

        if not hasattr(cv2.aruco, dict_name):
            raise RuntimeError(f'Unknown ArUco dictionary: {dict_name}')

        dict_id = getattr(cv2.aruco, dict_name)

        if hasattr(cv2.aruco, 'getPredefinedDictionary'):
            return cv2.aruco.getPredefinedDictionary(dict_id)

        return cv2.aruco.Dictionary_get(dict_id)

    def create_detector_params(self):
        if hasattr(cv2.aruco, 'DetectorParameters'):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, 'CORNER_REFINE_SUBPIX'):
            safe_set(params, 'cornerRefinementMethod', cv2.aruco.CORNER_REFINE_SUBPIX)
        else:
            safe_set(params, 'cornerRefinementMethod', 1)

        safe_set(params, 'cornerRefinementWinSize', 5)
        safe_set(params, 'cornerRefinementMaxIterations', 30)
        safe_set(params, 'cornerRefinementMinAccuracy', 0.1)

        safe_set(params, 'adaptiveThreshWinSizeMin', 5)
        safe_set(params, 'adaptiveThreshWinSizeMax', 35)
        safe_set(params, 'adaptiveThreshWinSizeStep', 5)

        safe_set(params, 'minDistanceToBorder', 3)
        safe_set(params, 'minMarkerDistanceRate', 0.03)
        safe_set(params, 'minCornerDistanceRate', 0.03)

        # 너무 관대하면 내부 패턴 false positive가 늘고,
        # 너무 빡세면 실제 marker도 놓칠 수 있다.
        safe_set(params, 'errorCorrectionRate', 0.30)
        safe_set(params, 'maxErroneousBitsInBorderRate', 0.30)

        safe_set(params, 'minOtsuStdDev', 5.0)

        return params

    # ============================================================
    # Marker detection / quality gate
    # ============================================================
    def marker_from_pts(self, pts, marker_id):
        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))

        width = x_max - x_min
        height = y_max - y_min

        min_side = min(width, height)
        max_side = max(width, height)

        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))

        area = float(abs(cv2.contourArea(pts.astype(np.float32))))
        aspect_ratio = max_side / max(min_side, 1e-6)

        target_id = int(self.p('target_id'))
        target_match = True if target_id < 0 else (int(marker_id) == target_id)

        return {
            'id': int(marker_id),
            'target_match': bool(target_match),
            'pts': pts,
            'cx': cx,
            'cy': cy,
            'width': width,
            'height': height,
            'min_side': min_side,
            'max_side': max_side,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'size': min_side,
            'valid': False,
            'reject_reason': '',
        }

    def marker_quality_ok(self, marker):
        min_control_side = float(self.p('min_control_side_px'))
        max_aspect_ratio = float(self.p('max_control_aspect_ratio'))
        min_control_area = float(self.p('min_control_area_px'))

        min_side = float(marker.get('min_side', 0.0))
        ratio = float(marker.get('aspect_ratio', 999.0))
        area = float(marker.get('area', 0.0))

        if min_control_side > 0.0 and min_side < min_control_side:
            return False, f'min_side {min_side:.1f} < {min_control_side:.1f}'

        if max_aspect_ratio > 0.0 and ratio > max_aspect_ratio:
            return False, f'aspect {ratio:.2f} > {max_aspect_ratio:.2f}'

        if min_control_area > 0.0 and area < min_control_area:
            return False, f'area {area:.0f} < {min_control_area:.0f}'

        return True, 'ok'

    def print_reject_debug_throttled(self):
        if not bool(self.p('print_reject_debug')):
            return

        if not self.last_rejected_markers:
            return

        now = time.time()
        period = float(self.p('reject_debug_period_sec'))

        if now - self.last_reject_debug_time < max(0.1, period):
            return

        self.last_reject_debug_time = now

        parts = []
        for m in self.last_rejected_markers[:4]:
            parts.append(
                f'id={m["id"]} side={m["min_side"]:.1f} '
                f'ratio={m["aspect_ratio"]:.2f} reason={m["reject_reason"]}'
            )

        text = 'aruco_candidate_rejected | ' + ' | '.join(parts)
        self.status(text)

    def detect_aruco_markers(self, frame):
        self.last_detect_error = ''

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            if hasattr(cv2.aruco, 'ArucoDetector'):
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray,
                    self.aruco_dict,
                    parameters=self.aruco_params
                )
        except Exception as e:
            self.last_detect_error = str(e)
            self.last_raw_markers = []
            self.last_target_candidates = []
            self.last_filtered_markers = []
            self.last_rejected_markers = []
            self.last_raw_ids = []
            self.last_raw_count = 0
            self.last_target_candidate_count = 0
            self.last_accepted_count = 0
            self.last_rejected_count = 0
            return [], None, None

        raw_markers = []
        target_candidates = []
        accepted_markers = []
        rejected_markers = []

        if ids is None or len(ids) == 0:
            self.last_raw_markers = []
            self.last_target_candidates = []
            self.last_filtered_markers = []
            self.last_rejected_markers = []
            self.last_raw_ids = []
            self.last_raw_count = 0
            self.last_target_candidate_count = 0
            self.last_accepted_count = 0
            self.last_rejected_count = 0
            return [], corners, ids

        target_id = int(self.p('target_id'))

        for i, c in enumerate(corners):
            marker_id = int(ids[i][0])
            pts = c.reshape(4, 2).astype(np.float32)
            marker = self.marker_from_pts(pts, marker_id)
            raw_markers.append(marker)

            if target_id >= 0 and marker_id != target_id:
                marker['valid'] = False
                marker['reject_reason'] = f'non_target_id target={target_id}'
                continue

            target_candidates.append(marker)

            ok, reason = self.marker_quality_ok(marker)
            if not ok:
                marker['valid'] = False
                marker['reject_reason'] = reason
                rejected_markers.append(marker)
                continue

            marker['valid'] = True
            marker['reject_reason'] = ''
            accepted_markers.append(marker)

        if bool(self.p('prefer_largest_marker')):
            accepted_markers.sort(key=lambda m: (m['min_side'], m['area']), reverse=True)

        self.last_raw_markers = list(raw_markers)
        self.last_target_candidates = list(target_candidates)
        self.last_filtered_markers = list(accepted_markers)
        self.last_rejected_markers = list(rejected_markers)

        self.last_raw_ids = [int(m['id']) for m in raw_markers]
        self.last_raw_count = len(raw_markers)
        self.last_target_candidate_count = len(target_candidates)
        self.last_accepted_count = len(accepted_markers)
        self.last_rejected_count = len(rejected_markers)

        if rejected_markers:
            self.print_reject_debug_throttled()

        return accepted_markers, corners, ids

    # ============================================================
    # JSON debug
    # ============================================================
    def marker_to_json(self, marker, include_corners=True):
        if marker is None:
            return None

        out = {
            'id': int(marker.get('id', -1)),
            'target_match': bool(marker.get('target_match', False)),
            'valid': bool(marker.get('valid', False)),
            'reject_reason': str(marker.get('reject_reason', '')),
            'cx': round(float(marker.get('cx', 0.0)), 2),
            'cy': round(float(marker.get('cy', 0.0)), 2),
            'width': round(float(marker.get('width', 0.0)), 2),
            'height': round(float(marker.get('height', 0.0)), 2),
            'min_side': round(float(marker.get('min_side', 0.0)), 2),
            'max_side': round(float(marker.get('max_side', 0.0)), 2),
            'aspect_ratio': round(float(marker.get('aspect_ratio', 0.0)), 3),
            'area': round(float(marker.get('area', 0.0)), 1),
        }

        if include_corners and 'pts' in marker and marker['pts'] is not None:
            pts = marker['pts']
            try:
                out['corners'] = [
                    [round(float(p[0]), 2), round(float(p[1]), 2)]
                    for p in pts.reshape(4, 2)
                ]
            except Exception:
                out['corners'] = []

        return out

    def limited_marker_list(self, markers):
        max_n = max(1, int(self.p('debug_json_max_markers')))
        include_corners = bool(self.p('debug_json_include_corners'))
        return [self.marker_to_json(m, include_corners=include_corners) for m in markers[:max_n]]

    def current_target_cx(self):
        target_cx = float(self.p('target_cx_px'))
        if target_cx < 0:
            target_cx = float(self.latest_image_width) / 2.0
        return target_cx

    def publish_debug_json(self, event='frame', force=False):
        if not bool(self.p('publish_debug_json')):
            return

        now = time.time()
        period = float(self.p('debug_json_period_sec'))

        if not force and now - self.last_json_publish_time < max(0.05, period):
            return

        self.last_json_publish_time = now

        target_cx = self.current_target_cx()

        last_seen_age = None
        if self.last_seen_time > 0.0:
            last_seen_age = round(float(now - self.last_seen_time), 3)

        detection_age = None
        if self.last_detection_time > 0.0:
            detection_age = round(float(now - self.last_detection_time), 3)

        selected = self.marker_to_json(
            self.last_marker,
            include_corners=bool(self.p('debug_json_include_corners'))
        )

        data = {
            'stamp_sec': round(float(now), 3),
            'event': str(event),

            'node': {
                'active': bool(self.active),
                'done': bool(self.done),
                'done_count': int(self.done_count),
            },

            'image': {
                'width': int(self.latest_image_width),
                'height': int(self.latest_image_height),
                'frame_count': int(self.frame_count),
            },

            'target': {
                'name': str(self.p('target_name')),
                'id': int(self.p('target_id')),
                'target_cx_px': round(float(target_cx), 2),
                'center_tolerance_px': round(float(self.p('center_tolerance_px')), 2),
                'target_size_px': round(float(self.p('target_size_px')), 2),
                'size_tolerance_px': round(float(self.p('size_tolerance_px')), 2),
            },

            'gate': {
                'min_control_side_px': round(float(self.p('min_control_side_px')), 2),
                'max_control_aspect_ratio': round(float(self.p('max_control_aspect_ratio')), 3),
                'min_control_area_px': round(float(self.p('min_control_area_px')), 1),
                'prefer_largest_marker': bool(self.p('prefer_largest_marker')),
            },

            'detector': {
                'dict_name': str(self.p('dict_name')),
                'raw_count': int(self.last_raw_count),
                'raw_ids': list(self.last_raw_ids),
                'target_candidate_count': int(self.last_target_candidate_count),
                'accepted_count': int(self.last_accepted_count),
                'rejected_count': int(self.last_rejected_count),
                'detect_error': str(self.last_detect_error),
                'detection_age_sec': detection_age,
                'last_seen_age_sec': last_seen_age,
            },

            'raw_markers': self.limited_marker_list(self.last_raw_markers),
            'target_candidates': self.limited_marker_list(self.last_target_candidates),
            'accepted': self.limited_marker_list(self.last_filtered_markers),
            'rejected': self.limited_marker_list(self.last_rejected_markers),

            'selected': selected,
            'valid_marker': selected is not None,

            'control': dict(self.last_control_info),
        }

        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.json_pub.publish(msg)

    # ============================================================
    # Commands
    # ============================================================
    def cmd_cb(self, msg):
        cmd = msg.data.strip().lower()

        if cmd == 'start':
            self.active = True
            self.done = False
            self.done_time = 0.0
            self.done_count = 0
            self.last_marker = None
            self.smoothed_marker = None
            self.reset_lost_recovery_memory()
            self.publish_done(False)

            self.last_control_info = {
                'reason': 'STARTED_WAITING_MARKER',
                'vx': 0.0,
                'vy': 0.0,
                'wz': 0.0,
                'center_ok': False,
                'size_ok': False,
                'done_count': 0,
                'required_count': int(self.p('done_required_count')),
                'err_cx': None,
                'err_size': None,
                'target_cx': self.current_target_cx(),
                'target_size_px': float(self.p('target_size_px')),
            }

            self.status(
                f'START aruco align | target_name={self.p("target_name")} '
                f'target_id={self.p("target_id")} '
                f'min_control_side={float(self.p("min_control_side_px")):.1f} '
                f'max_aspect={float(self.p("max_control_aspect_ratio")):.2f}'
            )
            self.publish_debug_json(event='start', force=True)

        elif cmd == 'stop':
            self.active = False
            self.done = False
            self.done_count = 0
            self.publish_stop()
            self.publish_done(False)
            self.last_control_info['reason'] = 'MANUAL_STOP'
            self.status('STOP aruco align')
            self.publish_debug_json(event='stop', force=True)

        elif cmd == 'reset':
            self.active = False
            self.done = False
            self.done_time = 0.0
            self.done_count = 0
            self.last_marker = None
            self.smoothed_marker = None
            self.reset_lost_recovery_memory()
            self.publish_stop()
            self.publish_done(False)
            self.last_control_info['reason'] = 'RESET'
            self.status('RESET aruco align')
            self.publish_debug_json(event='reset', force=True)

        else:
            self.status(f'unknown command: {cmd}')
            self.last_control_info['reason'] = f'UNKNOWN_CMD:{cmd}'
            self.publish_debug_json(event='unknown_cmd', force=True)

    def target_name_cb(self, msg):
        self.apply_target_name(msg.data, announce=True)

    # ============================================================
    # Smoothing
    # ============================================================
    def smooth_marker(self, marker):
        alpha = float(self.p('marker_smoothing_alpha'))
        alpha = clamp(alpha, 0.0, 1.0)

        if (
            self.smoothed_marker is None or
            self.smoothed_marker.get('id') != marker.get('id')
        ):
            self.smoothed_marker = dict(marker)
            return self.smoothed_marker

        if alpha <= 0.0:
            self.smoothed_marker = dict(marker)
            return self.smoothed_marker

        prev = self.smoothed_marker
        sm = dict(marker)

        for key in [
            'cx',
            'cy',
            'width',
            'height',
            'min_side',
            'max_side',
            'aspect_ratio',
            'area',
            'size'
        ]:
            sm[key] = alpha * float(marker[key]) + (1.0 - alpha) * float(prev[key])

        sm['pts'] = marker['pts']
        sm['id'] = marker['id']
        sm['target_match'] = marker.get('target_match', False)
        sm['valid'] = marker.get('valid', True)
        sm['reject_reason'] = ''

        self.smoothed_marker = sm
        return sm

    # ============================================================
    # Image callback
    # ============================================================
    def image_cb(self, msg):
        self.frame_count += 1

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.status(f'cv_bridge error: {e}')
            self.last_detect_error = f'cv_bridge error: {e}'
            self.publish_debug_json(event='cv_bridge_error', force=True)
            return

        h, w = frame.shape[:2]
        self.latest_image_width = w
        self.latest_image_height = h

        process_n = max(1, int(self.p('process_every_n_frames')))
        do_process = (self.frame_count % process_n) == 0

        aruco_corners = None
        aruco_ids = None

        if do_process:
            self.last_detection_time = time.time()
            candidates, aruco_corners, aruco_ids = self.detect_aruco_markers(frame)

            if candidates:
                marker = candidates[0] if bool(self.p('prefer_largest_marker')) else max(candidates, key=lambda m: m['size'])
                marker = self.smooth_marker(marker)
                self.last_marker = marker
                self.last_seen_time = time.time()
            else:
                if bool(self.p('clear_marker_on_reject')) and self.last_rejected_markers:
                    self.last_marker = None
                    self.smoothed_marker = None
                    self.done_count = 0

        if bool(self.p('publish_debug_image')):
            self.publish_debug(frame, msg.header, aruco_corners, aruco_ids)

        self.publish_debug_json(event='frame')

    # ============================================================
    # Debug image
    # ============================================================
    def draw_marker_poly(self, img, marker, color, thickness=2, label=None):
        try:
            pts = marker['pts'].astype(np.int32)
            cv2.polylines(img, [pts], True, color, thickness)

            cx = int(marker['cx'])
            cy = int(marker['cy'])

            if label:
                cv2.putText(
                    img,
                    label,
                    (max(0, cx - 55), max(15, cy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1
                )
        except Exception:
            pass

    def publish_debug(self, frame, header, aruco_corners, aruco_ids):
        dbg = frame.copy()
        h, w = dbg.shape[:2]

        target_cx = self.current_target_cx()

        center_tol = float(self.p('center_tolerance_px'))
        target_size = float(self.p('target_size_px'))
        size_tol = float(self.p('size_tolerance_px'))

        x1 = int(max(0, target_cx - center_tol))
        x2 = int(min(w - 1, target_cx + center_tol))

        # 중앙 허용 band + center line
        cv2.rectangle(dbg, (x1, 0), (x2, h - 1), (0, 60, 60), 1)
        cv2.line(dbg, (int(target_cx), 0), (int(target_cx), h), (0, 255, 255), 2)

        # raw marker 전체: 파란색
        # target id가 아니어도 보이게 해서 "검출은 되는데 id가 다른지" 확인 가능.
        for m in self.last_raw_markers:
            if m.get('target_match', False):
                continue
            self.draw_marker_poly(
                dbg,
                m,
                (255, 120, 0),
                thickness=1,
                label=f'RAW id={m["id"]} s={m["min_side"]:.0f}'
            )

        # target id였지만 gate에서 reject된 후보: 주황색
        for m in self.last_rejected_markers:
            reason = str(m.get('reject_reason', 'reject'))
            short_reason = reason if len(reason) <= 24 else reason[:24]
            self.draw_marker_poly(
                dbg,
                m,
                (0, 128, 255),
                thickness=2,
                label=f'REJ id={m["id"]} {short_reason}'
            )

        # quality gate 통과 후보: 얇은 초록색
        for m in self.last_filtered_markers:
            self.draw_marker_poly(
                dbg,
                m,
                (0, 255, 0),
                thickness=1,
                label=f'OK id={m["id"]} s={m["min_side"]:.0f}'
            )

        # 실제 control에 쓰는 marker: 두꺼운 초록색 + 빨간 중심점
        if self.last_marker is not None:
            marker = self.last_marker
            pts = marker['pts'].astype(np.int32)
            cv2.polylines(dbg, [pts], True, (0, 255, 0), 3)

            cx = int(marker['cx'])
            cy = int(marker['cy'])
            size = float(marker['size'])
            err_cx = float(marker['cx']) - target_cx

            size_ok = size >= target_size - size_tol
            center_ok = abs(err_cx) <= center_tol

            cv2.circle(dbg, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(
                dbg,
                f'target={self.p("target_name")} id={marker["id"]} '
                f'cx={marker["cx"]:.1f} err={err_cx:.1f} '
                f'minSide={size:.1f}/{target_size:.0f} '
                f'ratio={marker["aspect_ratio"]:.2f} '
                f'C={int(center_ok)} S={int(size_ok)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                dbg,
                f'target={self.p("target_name")} id={self.p("target_id")} '
                f'NO_VALID_MARKER raw={self.last_raw_count} ids={self.last_raw_ids}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 128, 255),
                2
            )

        try:
            dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            dbg_msg.header = header
            self.debug_pub.publish(dbg_msg)
        except Exception:
            pass

    # ============================================================
    # Frame-cut / lost-marker recovery helpers
    # ============================================================
    def reset_lost_recovery_memory(self):
        self.last_recovery_err_cx = None
        self.last_recovery_marker_time = 0.0
        self.last_recovery_vy = 0.0
        self.last_recovery_status_time = 0.0

    def remember_lost_recovery_direction(self, err_cx, vy):
        """Remember the last useful lateral direction for frame-cut recovery."""
        min_err = abs(float(self.p('lost_recovery_min_err_px')))
        if abs(float(err_cx)) < min_err:
            return

        self.last_recovery_err_cx = float(err_cx)
        self.last_recovery_marker_time = time.time()
        self.last_recovery_vy = float(vy)

    def compute_lost_recovery_vy(self):
        if self.last_recovery_err_cx is None:
            return 0.0

        err_cx = float(self.last_recovery_err_cx)
        min_err = abs(float(self.p('lost_recovery_min_err_px')))
        if abs(err_cx) < min_err:
            return 0.0

        # Same sign convention as normal align control:
        # err_cx > 0 means marker is on image-right, so move robot to right => vy negative.
        direction = -1.0 if err_cx > 0.0 else 1.0
        if bool(self.p('invert_y')):
            direction *= -1.0

        vy = direction * abs(float(self.p('lost_recovery_vy')))
        max_vy = abs(float(self.p('max_vy')))
        if max_vy > 0.0:
            vy = clamp(vy, -max_vy, max_vy)
        return float(vy)

    def try_lost_recovery(self, reason, marker_age):
        """Publish lateral-only recovery command when marker is cut from frame.

        Returns True if recovery command was published.
        """
        if not bool(self.p('enable_lost_recovery')):
            return False

        now = time.time()

        if self.last_recovery_err_cx is None or self.last_recovery_marker_time <= 0.0:
            return False

        memory_age = now - self.last_recovery_marker_time
        if memory_age > float(self.p('lost_recovery_max_age_sec')):
            return False

        recovery_start = max(0.0, float(self.p('lost_recovery_start_sec')))
        recovery_sec = max(0.0, float(self.p('lost_recovery_sec')))

        # marker_age is elapsed time from last valid detection.
        # Do not recover before stale threshold, and stop after recovery window.
        if marker_age < recovery_start:
            return False

        if marker_age > recovery_start + recovery_sec:
            return False

        vy = self.compute_lost_recovery_vy()
        if abs(vy) <= 1e-6:
            return False

        self.done_count = 0

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = float(vy)
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        self.update_control_info(
            f'LOST_RECOVERY:{reason}:age={marker_age:.2f}s',
            vx=0.0,
            vy=vy,
            wz=0.0,
            center_ok=False,
            size_ok=False,
            err_cx=self.last_recovery_err_cx,
            err_size=None
        )

        if bool(self.p('print_debug')):
            period = max(0.1, float(self.p('lost_recovery_debug_period_sec')))
            if now - self.last_recovery_status_time >= period:
                self.last_recovery_status_time = now
                self.get_logger().info(
                    f'lost_recovery | target={self.p("target_name")} '
                    f'reason={reason} age={marker_age:.2f}s '
                    f'last_err_cx={self.last_recovery_err_cx:.1f} -> vy={vy:.3f}'
                )

        self.publish_debug_json(event='lost_recovery')
        return True


    # ============================================================
    # Timer control
    # ============================================================
    def update_control_info(self, reason, vx=0.0, vy=0.0, wz=0.0,
                            center_ok=False, size_ok=False,
                            err_cx=None, err_size=None):
        self.last_control_info = {
            'reason': str(reason),
            'vx': round(float(vx), 4),
            'vy': round(float(vy), 4),
            'wz': round(float(wz), 4),
            'center_ok': bool(center_ok),
            'size_ok': bool(size_ok),
            'done_count': int(self.done_count),
            'required_count': int(max(1, int(self.p('done_required_count')))),
            'err_cx': None if err_cx is None else round(float(err_cx), 2),
            'err_size': None if err_size is None else round(float(err_size), 2),
            'target_cx': round(float(self.current_target_cx()), 2),
            'target_size_px': round(float(self.p('target_size_px')), 2),
        }

    def timer_cb(self):
        now = time.time()

        if not self.active:
            return

        if self.done:
            if now - self.done_time < float(self.p('stop_publish_sec')):
                self.publish_stop()
                self.update_control_info('DONE_STOP_HOLD')
            else:
                self.active = False
                self.update_control_info('DONE_INACTIVE')
            self.publish_debug_json(event='control')
            return

        if self.last_marker is None:
            if self.last_raw_count == 0:
                reason = 'NO_ARUCO_DETECTED'
            elif self.last_target_candidate_count == 0:
                reason = 'NO_TARGET_ID_DETECTED'
            elif self.last_rejected_count > 0:
                reason = 'TARGET_REJECTED_BY_GATE'
            else:
                reason = 'NO_VALID_MARKER'

            marker_age = now - self.last_recovery_marker_time if self.last_recovery_marker_time > 0.0 else 999.0
            if self.try_lost_recovery(reason, marker_age):
                return

            self.done_count = 0
            self.publish_stop()
            self.update_control_info(reason)
            self.publish_debug_json(event='control')
            return

        age = now - self.last_seen_time

        # If the selected marker has not been updated recently, do NOT keep using
        # its old vx command. This is the frame-cut case. Move laterally only,
        # using the last remembered marker direction.
        stale_start = max(0.0, float(self.p('lost_recovery_start_sec')))
        if age > stale_start:
            if self.try_lost_recovery('STALE_MARKER', age):
                return

            self.done_count = 0
            self.publish_stop()
            self.update_control_info(f'STALE_MARKER_STOP age={age:.2f}s')
            self.publish_debug_json(event='control')
            return

        if age > float(self.p('lost_timeout_sec')):
            self.done_count = 0
            self.publish_stop()
            self.update_control_info(f'LOST_TIMEOUT age={age:.2f}s')
            self.publish_debug_json(event='control')
            return

        marker = self.last_marker

        # runtime에 gate parameter가 바뀌었을 때도 한 번 더 검사
        ok, reason = self.marker_quality_ok(marker)
        if not ok:
            self.done_count = 0
            self.last_marker = None
            self.smoothed_marker = None
            self.publish_stop()
            self.update_control_info(f'SELECTED_MARKER_REJECTED:{reason}')
            if bool(self.p('print_reject_debug')):
                self.status(
                    f'aruco_selected_marker_rejected | id={marker["id"]} '
                    f'side={marker["min_side"]:.1f} ratio={marker["aspect_ratio"]:.2f} '
                    f'reason={reason}'
                )
            self.publish_debug_json(event='control', force=True)
            return

        target_size = float(self.p('target_size_px'))
        size_tol = float(self.p('size_tolerance_px'))
        center_tol = float(self.p('center_tolerance_px'))

        target_cx = self.current_target_cx()

        cx = float(marker['cx'])
        size = float(marker['size'])

        err_cx = cx - target_cx
        err_size = target_size - size

        size_ok = size >= target_size - size_tol
        center_ok = abs(err_cx) <= center_tol

        if size_ok and center_ok:
            self.done_count += 1
        else:
            self.done_count = 0

        required = max(1, int(self.p('done_required_count')))

        if self.done_count >= required:
            self.done = True
            self.done_time = now
            self.publish_stop()
            self.publish_done(True)
            self.update_control_info(
                'DONE',
                vx=0.0,
                vy=0.0,
                wz=0.0,
                center_ok=center_ok,
                size_ok=size_ok,
                err_cx=err_cx,
                err_size=err_size
            )
            self.status(
                f'DONE aruco align | target={self.p("target_name")} '
                f'id={marker["id"]} cx={cx:.1f}, err_cx={err_cx:.1f}, '
                f'min_side={size:.1f}, target_size={target_size:.1f}, '
                f'center_tol={center_tol:.1f}'
            )
            self.publish_debug_json(event='done', force=True)
            return

        kp_vy = float(self.p('kp_vy'))
        kp_vx = float(self.p('kp_vx'))

        max_vx = float(self.p('max_vx'))
        min_vx = float(self.p('min_vx'))
        max_vy = float(self.p('max_vy'))
        min_vy = float(self.p('min_vy'))

        invert_y = bool(self.p('invert_y'))

        vy = -kp_vy * err_cx
        if invert_y:
            vy = -vy

        vy = clamp(vy, -max_vy, max_vy)

        if center_ok:
            vy = 0.0
        elif abs(vy) < min_vy:
            vy = math.copysign(min_vy, vy)

        # Store the last meaningful lateral direction. If the marker is later
        # cut out of the frame, recovery will keep moving in this direction
        # briefly instead of stopping or continuing stale forward vx.
        self.remember_lost_recovery_direction(err_cx, vy)

        vx = 0.0
        center_first = bool(self.p('center_first'))

        if center_first and not center_ok:
            vx = 0.0
        else:
            if err_size > 0.0:
                vx = kp_vx * err_size
                vx = clamp(vx, 0.0, max_vx)

                if vx > 0.0 and vx < min_vx:
                    vx = min_vx
            else:
                if bool(self.p('allow_reverse')):
                    vx = -float(self.p('max_reverse_vx'))
                else:
                    vx = 0.0

        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.linear.y = float(vy)
        cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

        self.update_control_info(
            'ALIGNING',
            vx=vx,
            vy=vy,
            wz=0.0,
            center_ok=center_ok,
            size_ok=size_ok,
            err_cx=err_cx,
            err_size=err_size
        )

        if bool(self.p('print_debug')):
            self.get_logger().info(
                f'align | target={self.p("target_name")} id={marker["id"]} '
                f'cx={cx:.1f} err={err_cx:.1f} '
                f'min_side={size:.1f}/{target_size:.0f} '
                f'ratio={marker["aspect_ratio"]:.2f} '
                f'center_ok={center_ok} size_ok={size_ok} '
                f'done_count={self.done_count}/{required} '
                f'-> vx={vx:.3f}, vy={vy:.3f}'
            )

        self.publish_debug_json(event='control')

    # ============================================================
    # Publish helpers
    # ============================================================
    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def publish_done(self, value):
        msg = Bool()
        msg.data = bool(value)
        self.done_pub.publish(msg)

    def status(self, text):
        msg = String()
        msg.data = str(text)
        self.status_pub.publish(msg)
        self.get_logger().info(str(text))


def main(args=None):
    rclpy.init(args=args)
    node = ArucoAlignNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_stop()
            time.sleep(0.05)
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
