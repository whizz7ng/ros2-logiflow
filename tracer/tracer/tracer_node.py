#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogiFlow 라인 트레이서 - Layer 2 (v1.2)
=======================================
v1.1(FOLLOW + STOP_END + 빨강 분기 검출) + L코너 처리(CORNER).

CORNER 시퀀스 (odom 없이 vision + 시간만 사용):
  ADVANCE : 코너 감지 후 advance_time(s) 동안 slow_vx 전진 (카메라 사각<14cm 보정)
  ROTATE  : 피드백 없이 회전. 진입 라인이 '연속 N프레임 사라진 뒤'에야 새 라인 중앙 탐색
            -> 진입 라인 오인식 방지 (lose-then-reacquire). + 안전 타임아웃
  STRAFE  : 메카넘 strafe로 잔여 측면오차 제거 -> FOLLOW

색 규약: 흰색=추종 라인 / 빨강=분기(파킹) 마커
분기 종류: 빨강=파킹T(Layer3), 흰색 wide span=L코너(여기), 흰색 2클러스터=QR fork(Layer3)
"""

import math
import time
import threading
import cv2
import numpy as np
from enum import Enum

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Empty
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class NavState(Enum):
    IDLE = 0
    FOLLOW = 1
    CORNER = 2
    JUNCTION = 3
    TURN_180 = 4
    TURN_PAUSE = 5
    STOP_END = 6
    PARK_FORWARD = 7
    PARK_PAUSE = 8
    RETURN_TO_QR_B = 9

class MissionPhase(Enum):
    WAIT_START = 0
    TO_OBJECTS = 1
    WAIT_PICKED = 2
    TO_QR = 3
    WAIT_PLACED = 4
    RETURN_TO_QR_B = 5
    TO_PARKING_RED = 6
    PARKED = 7

class LineTracer(Node):
    def __init__(self):
        super().__init__('line_tracer')

        # ---------------- 파라미터 ----------------
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('use_compressed_input', True)
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('image_timeout_sec', 0.5)
        self.declare_parameter('frame_width', 320)
        self.declare_parameter('frame_height', 240)
        self.declare_parameter('control_rate', 20.0)

        self.declare_parameter('near_top', 0.85)
        self.declare_parameter('near_bot', 0.98)
        self.declare_parameter('far_top', 0.67)
        self.declare_parameter('far_bot', 0.84)

        # 흰색 라인 HSV
        self.declare_parameter('s_max', 70)
        self.declare_parameter('v_min', 175)
        self.declare_parameter('min_area', 250)
        self.declare_parameter('noise_min_area', 160)   # 이보다 작은 흰 덩어리는 노이즈로 버림

        # 빨강 분기 마커 HSV
        self.declare_parameter('red_s_min', 95)
        self.declare_parameter('red_v_min', 65)
        self.declare_parameter('red_area_min', 410)

        # 제어 / 속도
        self.declare_parameter('steer_kp', 0.2)
        self.declare_parameter('steer_kd', 0.0)
        self.declare_parameter('cruise_vx', 0.13)
        self.declare_parameter('slow_vx', 0.05)
        self.declare_parameter('max_wz', 0.3)
        self.declare_parameter('lost_frames_stop', 6)
        

        # 코너(Layer 2) - odom 없이 vision + 시간만 사용
        self.declare_parameter('corner_span_min', 50)   # FAR 흰색 가로폭(px) 이상이면 L코너 (320폭 기준)
        self.declare_parameter('advance_time', 1.0)      # 코너 감지 후 slow_vx 전진 시간(s)
        self.declare_parameter('rot_wz', 0.3)
        self.declare_parameter('rotate_lost_frames', 3)  # 진입 라인이 사라졌다고 볼 연속 프레임수
        self.declare_parameter('rotate_timeout', 8.0)    # 회전 안전 타임아웃(s) - 느려진 만큼 늘림
        self.declare_parameter('reacquire_tol', 0.15)    # 라인 중앙 판정 정규화오차
        self.declare_parameter('strafe_kp', 0.4)
        self.declare_parameter('strafe_max', 0.10)       # strafe 상한 (최대선속도의 ~10%)
        self.declare_parameter('strafe_fix_tol', 0.08)
        self.declare_parameter('strafe_fix_timeout', 2)
        self.declare_parameter('post_corner_grace', 1.8)  # 코너 종료 후 STOP_END 무시 시간(s)
        # 코너(Layer 2) - 시간 기반 사각 통과 + 고정 회전
        self.declare_parameter('corner_advance_dist', 0.29)    # 카메라 전방 사각거리(m)
        self.declare_parameter('corner_advance_vx', 0.13)      # ADVANCE 직진속도(=cruise, 저속보다 반복성 좋음)
        self.declare_parameter('corner_advance_timeout', 6.0)
        self.declare_parameter('rotate_deg', 55.0)             # 코너 회전각(도)
        self.declare_parameter('corner_approach_timeout', 8.0) # 라인 안 잃으면 오검출로 보고 복귀(s)
        self.declare_parameter('near_lost_frames', 3)          # near_cx None 확정 프레임수
        
        self.declare_parameter('rotate_stop_margin_deg', 6.0)       # 90도 목표면 84도쯤부터 정지

        self.declare_parameter('corner_stop_v_thresh', 0.02)
        self.declare_parameter('corner_stop_w_thresh', 0.05)
        self.declare_parameter('corner_stop_settle_timeout', 1.2)

        self.declare_parameter('publish_debug', True)
        self.declare_parameter('enable_drive', False)   # False면 모든 cmd_vel을 0으로 (인지만 테스트)
        self.declare_parameter('telemetry_csv', "/home/er/myagv_ros2/src/tracer/log/telemetry.csv")    # 경로 지정시 매 루프 CSV 1줄 기록 (엑셀 디버깅용)
        
        # ---- 미션 / 분기(JUNCTION) ----
        self.declare_parameter('start_idle', True)        # True면 파킹에서 대기, 시작 메세지로 출발
        self.declare_parameter('default_target', 'B')     # start_idle=False일 때 테스트용 목표
        self.declare_parameter('qr_center_lo', 130)       # 분기 정렬 near_cx 하한
        self.declare_parameter('qr_center_hi', 190)       # 상한 (중앙 160 ±15)
        self.declare_parameter('qr_stop_bbox', 110)       # QR 최소변(px) 이상이면 도착 정지
        self.declare_parameter('qr_min_rate', 0.40)       # 최근 검출률 이상
        self.declare_parameter('qr_rate_window', 15)      # 검출률 평균 윈도우(프레임)
        self.declare_parameter('qr_check_interval', 3)    # FOLLOW 중 QR 검사 주기(프레임,CPU)
        self.declare_parameter('junction_strafe_speed', 0.08)
        self.declare_parameter('junction_strafe_kp', 0.5)
        self.declare_parameter('junction_approach_vx', 0.10)
        self.declare_parameter('junction_align_timeout', 6.0)
        self.declare_parameter('junction_approach_timeout', 8.0)
        
        # ---- mission / object / parking ----
        self.declare_parameter('object_red_turn_dir', -1)       # objects 방향: 로봇 기준 오른쪽이면 보통 -1
        self.declare_parameter('object_red_rotate_deg', 60.0)

        self.declare_parameter('turn_180_wz', 0.30)
        self.declare_parameter('turn_180_deg', 145.0)
        self.declare_parameter('turn_180_cal', 1.0)
        self.declare_parameter('turn_180_timeout', 12.0)
        self.declare_parameter('turn_pause_time', 1.0)

        self.declare_parameter('parking_red_turn_dir', 1)       # 실제 방향 보고 +1/-1 튜닝
        self.declare_parameter('parking_red_rotate_deg', 60.0)

        # 네가 말한 parking 쪽 별도 cal
        self.declare_parameter('parking_advance_cal', 1.0)
        self.declare_parameter('parking_rotate_cal', 1.0)

        self.declare_parameter('parking_forward_time', 4.5)
        self.declare_parameter('parking_pause_time', 1.0)
        
        self.declare_parameter('return_b_strafe_speed', 0.08)
        self.declare_parameter('return_b_timeout', 6.0)
        
        # A/C QR 접근 시, QR 중심이 이 ROI 안에 있고 bbox가 충분히 크면 stop_qr
        # frame 320x240 기준 비율. 네가 그린 노란 사각형 느낌으로 잡은 기본값.
        self.declare_parameter('qr_stop_roi_x1', 0.35)
        self.declare_parameter('qr_stop_roi_x2', 0.90)
        self.declare_parameter('qr_stop_roi_y1', 0.20)
        self.declare_parameter('qr_stop_roi_y2', 0.72)
        

        # ---- line lost search ----
        self.declare_parameter('line_search_wz', 0.12)
        self.declare_parameter('line_search_default_dir', 1)   # +1 좌회전, -1 우회전
        self.declare_parameter('line_search_timeout', 6.0)
        
        # ---- rack white wall depth stop ----
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('use_white_wall_rack_stop', True)
        self.declare_parameter('rack_stop_dist_m', 0.37)
        self.declare_parameter('rack_stop_confirm_frames', 3)

        self.declare_parameter('rack_wall_roi_x1', 0.05)
        self.declare_parameter('rack_wall_roi_x2', 0.95)
        self.declare_parameter('rack_wall_roi_y1', 0.00)
        self.declare_parameter('rack_wall_roi_y2', 0.38)

        self.declare_parameter('rack_wall_white_s_max', 70)
        self.declare_parameter('rack_wall_white_v_min', 170)
        self.declare_parameter('rack_wall_min_valid_px', 80)

        self._load_params()

        # ---------------- 카메라 입력: ROS2 Image 토픽 구독 ----------------
        # 기존 cv2.VideoCapture 대신, Orin Nano/D435에서 publish되는 RGB 토픽을 받는다.
        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.last_grab_time = 0.0

        image_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        if self.use_compressed_input:
            self.image_sub = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.cb_image_compressed,
                image_qos
            )
            self.get_logger().info(f'Compressed image 구독: {self.image_topic}')
        else:
            self.image_sub = self.create_subscription(
                Image,
                self.image_topic,
                self.cb_image,
                image_qos
            )
            self.get_logger().info(f'Raw image 구독: {self.image_topic}')
            
        self.depth_lock = threading.Lock()
        self.latest_depth_m = None
        self.last_depth_time = 0.0
        self.rack_close_count = 0
        self.last_near_cx = None
        self.line_lost_t0 = None
        self.line_search_dir = int(self.line_search_default_dir)
        if self.line_search_dir == 0:
            self.line_search_dir = 1

        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.cb_depth,
            depth_qos
        )
        
        self.get_logger().info(f'Depth 구독: {self.depth_topic}')

        self.return_b_after_phase = None

        # ---------------- 통신 ----------------
        cmd_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', cmd_qos)
        self.status_pub = self.create_publisher(String, '/nav_status', 10)
        self.debug_pub = self.create_publisher(CompressedImage, '/line_tracer/debug/compressed', 2)
        self.stop_obj_pub = self.create_publisher(Empty, '/stop_obj', 10)
        self.stop_qr_pub = self.create_publisher(Empty, '/stop_qr', 10)
        
        

        self.create_subscription(String, '/place_target', self.cb_place_target, 10)
        self.create_subscription(String, '/arm_status', self.cb_arm_status, 10)
        self.create_subscription(Empty, '/go_parking', self.cb_go_parking, 10)
        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 10)
        self.create_subscription(String, '/mission_cmd', self.cb_mission, 10)

        # ---------------- 상태 ----------------
        self.state = NavState.IDLE if self.start_idle else NavState.FOLLOW
        self.target = None if self.start_idle else self.default_target
        self.jphase = None                 # 'ALIGN' | 'APPROACH'
        self.align_left_center = False     # strafe lose-then-reacquire 래치
        self._qr_hist = []                 # 최근 검출 1/0
        self._qr_confirmed = False
        self._frame_i = 0
        self.mission_phase = MissionPhase.WAIT_START
        self.return_b_left_current = False

        self.corner_context = 'normal'       # 'normal' | 'object_red' | 'parking_red'
        self.active_advance_dist = self.corner_advance_dist
        self.active_rotate_deg = self.rotate_deg
        self.active_rotate_cal = 1.0

        self.object_red_done = False
        self.parking_red_done = False

        self.turn_after_phase = None
        self.turn_dir = 1
        
        # QR 디코더 (pyzbar 우선, cv2 fallback)
        try:
            from pyzbar.pyzbar import decode as _zbar
            self._zbar = _zbar
            self._have_zbar = True
        except Exception:
            self._have_zbar = False
        self._qr_det = cv2.QRCodeDetector()
        self.prev_err = 0.0
        self.lost_count = 0
        self.place_target = None
        self.odom = None
        
        # odometry/filtered 적분값
        self.odom_int_dist = 0.0      # ADVANCE 중 적분 거리 [m]
        self.odom_int_yaw = 0.0       # ROTATE 중 적분 yaw [rad]
        self._odom_last_t = None

        # 코너 내부 상태 (odom 미사용)
        self.corner_phase = None       # 'ADVANCE'|'ROTATE'|'STRAFE'
        self.corner_dir = 0            # +1 좌, -1 우
        self.phase_t0 = 0.0
        self.corner_saw_lost = False   # ROTATE 중 진입 라인이 한번 사라졌는지
        self.rotate_lost_count = 0
        self.last_cmd = (0.0, 0.0, 0.0)
        self.post_corner_grace_until = 0.0   # 이 시각까지는 STOP_END 보호
        
        self.far_lost_latched = False     # APPROACH 중 far가 한번 사라졌는지(2차 래치)
        self.approach_near_lost = 0       # near None 연속 카운트(3차)
        self._advance_secs = 0.0          # 진입시 dist/vx*cal로 계산
        self._rotate_secs = 0.0           # 진입시 deg/wz*cal로 계산
        

        # 텔레메트리 CSV (엑셀 디버깅용) — 경로 지정시에만 활성
        self.csv_file = None
        if self.telemetry_csv:
            try:
                self.csv_file = open(self.telemetry_csv, 'w')
                self.csv_file.write('t,state,phase,near_cx,err,far_cx,far_hspan,far_clusters,far_red,vx,vy,wz\n')
                self.get_logger().info(f'텔레메트리 기록 시작: {self.telemetry_csv}')
            except Exception as e:
                self.get_logger().error(f'CSV 열기 실패: {e}')

        self.timer = self.create_timer(1.0 / max(self.control_rate, 1.0), self.loop)
        self.add_on_set_parameters_callback(self._on_param_change)
        self.get_logger().info(
            f'line_tracer 시작 | image_topic={self.image_topic} {self.frame_width}x{self.frame_height} '
            f'| cruise_vx={self.cruise_vx} max_wz={self.max_wz} rot_wz={self.rot_wz} '
            f'steer_kp={self.steer_kp} steer_kd={self.steer_kd} '
            f'| v_min={self.v_min} s_max={self.s_max} corner_span_min={self.corner_span_min} '
            f'| enable_drive={self.enable_drive} publish_debug={self.publish_debug}')

    def _load_params(self):
        g = self.get_parameter
        self.image_topic = g('image_topic').value
        self.use_compressed_input = g('use_compressed_input').value
        self.image_timeout_sec = g('image_timeout_sec').value
        self.frame_width = g('frame_width').value
        self.frame_height = g('frame_height').value
        self.control_rate = g('control_rate').value
        self.near_top = g('near_top').value
        self.near_bot = g('near_bot').value
        self.far_top = g('far_top').value
        self.far_bot = g('far_bot').value
        self.s_max = g('s_max').value
        self.v_min = g('v_min').value
        self.min_area = g('min_area').value
        self.red_s_min = g('red_s_min').value
        self.red_v_min = g('red_v_min').value
        self.red_area_min = g('red_area_min').value
        self.steer_kp = g('steer_kp').value
        self.steer_kd = g('steer_kd').value
        self.cruise_vx = g('cruise_vx').value
        self.slow_vx = g('slow_vx').value
        self.max_wz = g('max_wz').value
        self.lost_frames_stop = g('lost_frames_stop').value
        self.corner_span_min = g('corner_span_min').value
        self.advance_time = g('advance_time').value
        self.rot_wz = g('rot_wz').value
        self.rotate_lost_frames = g('rotate_lost_frames').value
        self.rotate_timeout = g('rotate_timeout').value
        self.reacquire_tol = g('reacquire_tol').value
        self.strafe_kp = g('strafe_kp').value
        self.strafe_max = g('strafe_max').value
        self.strafe_fix_tol = g('strafe_fix_tol').value
        self.strafe_fix_timeout = g('strafe_fix_timeout').value
        self.publish_debug = g('publish_debug').value
        self.enable_drive = g('enable_drive').value
        self.telemetry_csv = g('telemetry_csv').value
        self.post_corner_grace = g('post_corner_grace').value
        self.corner_advance_dist = g('corner_advance_dist').value
        self.corner_advance_vx = g('corner_advance_vx').value
        self.rotate_deg = g('rotate_deg').value
        self.corner_approach_timeout = g('corner_approach_timeout').value
        self.near_lost_frames = g('near_lost_frames').value
        self.noise_min_area = g('noise_min_area').value
        self.start_idle = g('start_idle').value
        self.default_target = g('default_target').value
        self.qr_center_lo = g('qr_center_lo').value
        self.qr_center_hi = g('qr_center_hi').value
        self.qr_stop_bbox = g('qr_stop_bbox').value
        self.qr_min_rate = g('qr_min_rate').value
        self.qr_rate_window = g('qr_rate_window').value
        self.qr_check_interval = g('qr_check_interval').value
        self.junction_strafe_speed = g('junction_strafe_speed').value
        self.junction_strafe_kp = g('junction_strafe_kp').value
        self.junction_approach_vx = g('junction_approach_vx').value
        self.junction_align_timeout = g('junction_align_timeout').value
        self.junction_approach_timeout = g('junction_approach_timeout').value
        self.odom_topic = g('odom_topic').value
        self.corner_advance_timeout = g('corner_advance_timeout').value
        self.rotate_stop_margin_deg = g('rotate_stop_margin_deg').value
        self.corner_stop_v_thresh = g('corner_stop_v_thresh').value
        self.corner_stop_w_thresh = g('corner_stop_w_thresh').value
        self.corner_stop_settle_timeout = g('corner_stop_settle_timeout').value
        self.object_red_turn_dir = g('object_red_turn_dir').value
        self.object_red_rotate_deg = g('object_red_rotate_deg').value

        self.turn_180_wz = g('turn_180_wz').value
        self.turn_180_deg = g('turn_180_deg').value
        self.turn_180_cal = g('turn_180_cal').value
        self.turn_180_timeout = g('turn_180_timeout').value
        self.turn_pause_time = g('turn_pause_time').value

        self.parking_red_turn_dir = g('parking_red_turn_dir').value
        self.parking_red_rotate_deg = g('parking_red_rotate_deg').value
        self.parking_advance_cal = g('parking_advance_cal').value
        self.parking_rotate_cal = g('parking_rotate_cal').value

        self.parking_forward_time = g('parking_forward_time').value
        self.parking_pause_time = g('parking_pause_time').value
        self.return_b_strafe_speed = g('return_b_strafe_speed').value
        self.return_b_timeout = g('return_b_timeout').value
        self.qr_stop_roi_x1 = g('qr_stop_roi_x1').value
        self.qr_stop_roi_x2 = g('qr_stop_roi_x2').value
        self.qr_stop_roi_y1 = g('qr_stop_roi_y1').value
        self.qr_stop_roi_y2 = g('qr_stop_roi_y2').value
        
        self.depth_topic = g('depth_topic').value
        self.use_white_wall_rack_stop = g('use_white_wall_rack_stop').value
        self.rack_stop_dist_m = g('rack_stop_dist_m').value
        self.rack_stop_confirm_frames = g('rack_stop_confirm_frames').value

        self.rack_wall_roi_x1 = g('rack_wall_roi_x1').value
        self.rack_wall_roi_x2 = g('rack_wall_roi_x2').value
        self.rack_wall_roi_y1 = g('rack_wall_roi_y1').value
        self.rack_wall_roi_y2 = g('rack_wall_roi_y2').value

        self.rack_wall_white_s_max = g('rack_wall_white_s_max').value
        self.rack_wall_white_v_min = g('rack_wall_white_v_min').value
        self.rack_wall_min_valid_px = g('rack_wall_min_valid_px').value

        self.line_search_wz = g('line_search_wz').value
        self.line_search_default_dir = int(g('line_search_default_dir').value)
        self.line_search_timeout = g('line_search_timeout').value
        
    def _on_param_change(self, params):
        # 속성명이 파라미터명과 동일하므로 그대로 반영 -> ros2 param set 즉시 적용
        for p in params:
            setattr(self, p.name, p.value)
        return SetParametersResult(successful=True)

    # ==================== 콜백 ====================
    def cb_place_target(self, msg):
        self.place_target = msg.data
        self.get_logger().info(f'place_target 수신: {msg.data}')

    def cb_arm_status(self, msg):
        d = msg.data.strip().lower()
        self.get_logger().info(f'arm_status 수신: {d}')

        if d == 'picked':
            if self.mission_phase != MissionPhase.WAIT_PICKED:
                self.get_logger().warn(
                    f'picked 수신했지만 현재 phase={self.mission_phase.name}, 무시'
                )
                return

            self.mission_phase = MissionPhase.TO_QR

            self.get_logger().info('picked 수신 -> 180도 회전 후 QR 목표로 이동')
            self._enter_turn_180(MissionPhase.TO_QR, reason='picked')
            return

        if d == 'placed':
            if self.mission_phase != MissionPhase.WAIT_PLACED:
                self.get_logger().warn(
                    f'placed 수신했지만 현재 phase={self.mission_phase.name}, 무시'
                )
                return

            # A/C에 도착한 경우: 바로 180도 돌지 말고 QR_B로 lateral 복귀 후 180도
            if self.target in ('A', 'C'):
                self.mission_phase = MissionPhase.RETURN_TO_QR_B
                self.state = NavState.RETURN_TO_QR_B
                self.return_b_after_phase = MissionPhase.TO_OBJECTS
                self.phase_t0 = self._now()
                self.return_b_left_current = False
                self.publish_cmd(0.0, 0.0, 0.0)

                self.get_logger().info(
                    f'placed 수신: QR_{self.target} -> QR_B 복귀 후 180도 회전해서 TO_OBJECTS'
                )
                return

            # B면 기존처럼 바로 180도
            self.mission_phase = MissionPhase.TO_OBJECTS
            self.get_logger().info('placed 수신: QR_B -> 180도 회전 후 objects로 복귀')
            self._enter_turn_180(MissionPhase.TO_OBJECTS, reason='placed_from_B')
            return

    def cb_go_parking(self, msg):
        self.get_logger().info('go_parking 수신')

        if self.mission_phase != MissionPhase.WAIT_PLACED:
            self.get_logger().warn(
                f'go_parking 수신했지만 현재 phase={self.mission_phase.name}. '
                f'WAIT_PLACED가 아니므로 무시'
            )
            return

        self.mission_phase = MissionPhase.RETURN_TO_QR_B

        # 이미 B에 있으면 lateral 이동 필요 없음
        if self.target == 'B':
            self.get_logger().info(
                'go_parking: 현재 target=B라서 QR_B 복귀 이동 생략 -> 180도 회전'
            )
            self._enter_turn_180(MissionPhase.TO_PARKING_RED, reason='go_parking_from_B')
            return

        # A 또는 C에 있으면 먼저 QR_B로 strafe
        self.state = NavState.RETURN_TO_QR_B
        self.return_b_after_phase = MissionPhase.TO_PARKING_RED
        self.phase_t0 = self._now()
        self.return_b_left_current = False
        self.publish_cmd(0.0, 0.0, 0.0)

        self.get_logger().info(
            f'go_parking: target={self.target} 위치에서 QR_B로 lateral 복귀 시작'
        )
        
    def do_return_to_qr_b(self, meas):
        t = self._now() - self.phase_t0

        w = meas['w']
        nt, nb = meas['near_rows']

        # near band에서 중앙에 가장 가까운 흰색 라인 중심 찾기
        ncx = self.near_cx_center(meas['white'], nt, nb, w / 2.0)

        centered = (
            ncx is not None
            and self.qr_center_lo <= ncx <= self.qr_center_hi
        )

        vy = self._vy_from_target_to_b()

        # target이 B면 여기 들어올 필요가 없지만, 방어용
        if abs(vy) < 1e-6:
            self.publish_cmd(0.0, 0.0, 0.0)

            after = self.return_b_after_phase or MissionPhase.TO_PARKING_RED
            self.return_b_after_phase = None

            self.get_logger().info(
                f'RETURN_TO_QR_B: 이미 B로 판단 -> 180도 회전 후 {after.name}'
            )
            self._enter_turn_180(after, reason='return_b_already')
            return

        # 1단계: 현재 A/C 라인이 중앙에서 벗어났는지 확인
        if not self.return_b_left_current:
            if not centered:
                self.return_b_left_current = True
                self.get_logger().info(
                    f'QR {self.target} 라인 이탈 -> QR_B 라인 탐색 시작'
                )

            self.publish_cmd(0.0, vy, 0.0)
            return

        # 2단계: 다음 라인이 중앙에 들어오면 QR_B로 판단
        if centered:
            self.publish_cmd(0.0, 0.0, 0.0)

            after = self.return_b_after_phase or MissionPhase.TO_PARKING_RED
            self.return_b_after_phase = None

            self.get_logger().info(
                f'QR_B 라인 중앙 재획득 ncx={ncx:.1f} -> 180도 회전 후 {after.name}'
            )

            self._enter_turn_180(after, reason=f'return_to_qr_b_done_after_{after.name}')
            return

        if t > self.return_b_timeout:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.state = NavState.STOP_END
            self.publish_status('return_to_qr_b_timeout')
            self.get_logger().error(
                f'QR_B 복귀 timeout target={self.target}, ncx={ncx} -> STOP'
            )
            return

        self.publish_cmd(0.0, vy, 0.0)

    def _vy_from_target_to_b(self):
        if self.target == 'A':
            return +self.return_b_strafe_speed
        if self.target == 'C':
            return -self.return_b_strafe_speed
        return 0.0

    def _odom_stamp_sec(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t <= 0.0:
            return self._now()
        return t


    def _reset_odom_integrator(self):
        self.odom_int_dist = 0.0
        self.odom_int_yaw = 0.0
        self._odom_last_t = None

    def cb_odom(self, msg):
        self.odom = msg

        # __init__ 중 콜백이 먼저 들어오는 경우 보호
        if not hasattr(self, 'state') or not hasattr(self, 'corner_phase'):
            return

        now_t = self._odom_stamp_sec(msg)

        # CORNER의 ADVANCE / ROTATE 중에만 적분
        corner_integrating = (
            self.state == NavState.CORNER
            and self.corner_phase in ('ADVANCE', 'ROTATE')
        )

        turn_integrating = (
            self.state == NavState.TURN_180
        )

        if not (corner_integrating or turn_integrating):
            self._odom_last_t = now_t
            return

        if self._odom_last_t is None:
            self._odom_last_t = now_t
            return

        dt = now_t - self._odom_last_t
        self._odom_last_t = now_t

        # 이상한 dt 방어
        if dt <= 0.0 or dt > 0.5:
            return

        tw = msg.twist.twist

        if self.state == NavState.CORNER and self.corner_phase == 'ADVANCE':
            vx = tw.linear.x
            vy = tw.linear.y
            self.odom_int_dist += math.hypot(vx, vy) * dt

        elif self.state == NavState.CORNER and self.corner_phase == 'ROTATE':
            wz = tw.angular.z
            self.odom_int_yaw += wz * dt

        elif self.state == NavState.TURN_180:
            wz = tw.angular.z
            self.odom_int_yaw += wz * dt
            
    def _odom_speed_xy(self):
        if self.odom is None:
            return 999.0
        tw = self.odom.twist.twist
        return math.hypot(tw.linear.x, tw.linear.y)


    def _odom_abs_wz(self):
        if self.odom is None:
            return 999.0
        return abs(self.odom.twist.twist.angular.z)
    
    def denoise_mask(self, mask, min_comp):
        """작은 연결성분(바닥 알갱이) 제거. 큰 라인 성분만 남김."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, n):                       # 0 = 배경
            if stats[i, cv2.CC_STAT_AREA] >= min_comp:
                out[labels == i] = 255
        return out
    
    def cb_mission(self, msg):
        d = msg.data.strip().upper()
        table = {'A_START': 'A', 'B_START': 'B', 'C_START': 'C'}

        if d not in table:
            self.get_logger().warn(f'알 수 없는 미션 명령: {msg.data}')
            return

        self.target = table[d]

        self.mission_phase = MissionPhase.TO_OBJECTS
        self.state = NavState.FOLLOW

        self.object_red_done = False
        self.parking_red_done = False
        self.lost_count = 0
        self.prev_err = 0.0
        self._qr_hist = []
        self._qr_confirmed = False

        self.get_logger().info(
            f'미션 시작 target={self.target}: parking -> objects 먼저 이동'
        )
            
    def detect_qr_info(self, frame):
        """
        QR 디코드 -> (text or None, min_bbox_px, center_x, center_y)

        min_bbox_px:
        QR bbox의 가로/세로 중 작은 값.
        기존 qr_stop_bbox와 비교하는 값.

        center_x, center_y:
        QR bbox 중심점. ROI 안에 들어왔는지 판단하는 데 사용.
        """
        if self._have_zbar:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = self._zbar(gray)

            if res:
                # 여러 개 잡히면 가장 큰 QR 사용
                r = max(res, key=lambda z: z.rect.width * z.rect.height)

                xs = [p.x for p in r.polygon]
                ys = [p.y for p in r.polygon]

                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                minbb = int(min(x2 - x1, y2 - y1))
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                return r.data.decode('utf-8', 'replace'), minbb, cx, cy

            return None, 0, None, None

        data, pts, _ = self._qr_det.detectAndDecode(frame)

        if pts is not None and data:
            p = pts.reshape(-1, 2)

            x1, x2 = float(p[:, 0].min()), float(p[:, 0].max())
            y1, y2 = float(p[:, 1].min()), float(p[:, 1].max())

            minbb = int(min(x2 - x1, y2 - y1))
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            return data, minbb, cx, cy

        return None, 0, None, None


    def detect_qr(self, frame):
        """
        기존 코드 호환용.
        기존처럼 qtext, minbb만 필요한 곳에서 사용 가능.
        """
        qtext, minbb, _, _ = self.detect_qr_info(frame)
        return qtext, minbb

    def near_cx_center(self, white, top, bot, cx_ref):
        """near 밴드에서 '중앙에 가장 가까운' 연결성분의 cx.
        다른 분기선이 같이 보여도 목표선만 집어냄 (strafe 정렬 안정화)."""
        band = white[top:bot, :]
        n, labels, stats, cents = cv2.connectedComponentsWithStats(band, connectivity=8)
        best, best_d = None, 1e9
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < self.noise_min_area:
                continue
            d = abs(cents[i][0] - cx_ref)
            if d < best_d:
                best_d, best = d, float(cents[i][0])
        return best

    # ==================== 메인 루프 ====================
    def loop(self):
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            grab_age = time.time() - self.last_grab_time
        if frame is None or grab_age > self.image_timeout_sec:
            self.publish_cmd(0.0, 0.0, 0.0)
            return
        if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

        meas = self.perceive(frame)

        if self.state == NavState.FOLLOW:
            self.do_follow(meas, frame)
        elif self.state == NavState.CORNER:
            self.do_corner(meas)
        elif self.state == NavState.JUNCTION:
            self.do_junction(meas, frame)
        elif self.state == NavState.RETURN_TO_QR_B:
            self.do_return_to_qr_b(meas)
        elif self.state == NavState.TURN_180:
            self.do_turn_180()
        elif self.state == NavState.TURN_PAUSE:
            self.do_turn_pause()
        elif self.state == NavState.PARK_FORWARD:
            self.do_parking_forward()
        elif self.state == NavState.PARK_PAUSE:
            self.do_parking_pause()
        elif self.state in (NavState.STOP_END, NavState.IDLE):
            self.publish_cmd(0.0, 0.0, 0.0)


        if self.publish_debug:
            self.publish_debug_image(frame, meas)
            

        self._log_telemetry(meas)
        
    def _qr_in_stop_roi(self, qcx, qcy, w, h):
        if qcx is None or qcy is None:
            return False

        x1 = self.qr_stop_roi_x1 * w
        x2 = self.qr_stop_roi_x2 * w
        y1 = self.qr_stop_roi_y1 * h
        y2 = self.qr_stop_roi_y2 * h

        return x1 <= qcx <= x2 and y1 <= qcy <= y2

    def _log_telemetry(self, meas):
        if self.csv_file is None:
            return
        w = meas['w']
        ncx = meas['near_cx']
        err = '' if ncx is None else round((ncx - w / 2.0) / (w / 2.0), 3)
        ph = self.corner_phase if self.state == NavState.CORNER else ''
        vx, vy, wz = self.last_cmd
        fcx = meas['far_cx']
        try:
            self.csv_file.write(
                f'{self._now():.3f},{self.state.name},{ph},'
                f'{"" if ncx is None else round(ncx, 1)},{err},'
                f'{"" if fcx is None else round(fcx, 1)},'
                f'{meas["far_hspan"]},{meas["far_clusters"]},{meas["far_red_area"]},'
                f'{vx:.3f},{vy:.3f},{wz:.3f}\n')
        except Exception:
            pass

    def cb_image(self, msg):
        """D435 color image_raw 토픽을 OpenCV BGR 프레임으로 변환해 최신 프레임으로 보관한다."""
        try:
            # 기존 처리부(perceive)는 OpenCV BGR frame을 입력으로 받으므로 bgr8로 맞춘다.
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'이미지 변환 실패: {e}', throttle_duration_sec=1.0)
            return

        with self.frame_lock:
            self.latest_frame = frame
            self.last_grab_time = time.time()
            
    def cb_image_compressed(self, msg):
        """CompressedImage 토픽을 OpenCV BGR 프레임으로 디코드."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                self.get_logger().warn('compressed image decode 결과가 None', throttle_duration_sec=1.0)
                return

        except Exception as e:
            self.get_logger().error(f'compressed image decode 실패: {e}', throttle_duration_sec=1.0)
            return

        with self.frame_lock:
            self.latest_frame = frame
            self.last_grab_time = time.time()
            
    def cb_depth(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) * 0.001
            else:
                depth_m = depth.astype(np.float32)

            depth_m[~np.isfinite(depth_m)] = 0.0

            if depth_m.shape[1] != self.frame_width or depth_m.shape[0] != self.frame_height:
                depth_m = cv2.resize(
                    depth_m,
                    (self.frame_width, self.frame_height),
                    interpolation=cv2.INTER_NEAREST
                )

        except Exception as e:
            self.get_logger().warn(f'depth 변환 실패: {e}', throttle_duration_sec=1.0)
            return

        with self.depth_lock:
            self.latest_depth_m = depth_m
            self.last_depth_time = time.time()
            
    def rack_close_by_white_wall_depth(self, frame):
        """
        color frame에서 흰 벽 픽셀만 골라서,
        같은 위치의 aligned depth median으로 rack/wall 도착 판단.
        색깔 물체는 흰색 mask에서 제외됨.
        """
        if not self.use_white_wall_rack_stop:
            return False

        with self.depth_lock:
            depth_m = None if self.latest_depth_m is None else self.latest_depth_m.copy()
            depth_age = time.time() - self.last_depth_time

        if depth_m is None or depth_age > 1.0:
            return False

        h, w = frame.shape[:2]

        x1 = int(np.clip(self.rack_wall_roi_x1, 0.0, 1.0) * w)
        x2 = int(np.clip(self.rack_wall_roi_x2, 0.0, 1.0) * w)
        y1 = int(np.clip(self.rack_wall_roi_y1, 0.0, 1.0) * h)
        y2 = int(np.clip(self.rack_wall_roi_y2, 0.0, 1.0) * h)

        if x2 <= x1 or y2 <= y1:
            return False

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, int(self.rack_wall_white_v_min)], dtype=np.uint8)
        upper = np.array([179, int(self.rack_wall_white_s_max), 255], dtype=np.uint8)

        white = cv2.inRange(hsv, lower, upper)
        white = cv2.morphologyEx(
            white,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        roi_white = white[y1:y2, x1:x2]
        roi_depth = depth_m[y1:y2, x1:x2]

        valid = roi_depth[
            (roi_white > 0) &
            (roi_depth > 0.03) &
            (roi_depth < 2.0)
        ]

        if valid.size < self.rack_wall_min_valid_px:
            self.rack_close_count = 0
            self.get_logger().warn(
                f'white wall depth valid 부족: px={valid.size}',
                throttle_duration_sec=1.0
            )
            return False

        d = float(np.median(valid))

        if d <= self.rack_stop_dist_m:
            self.rack_close_count += 1
        else:
            self.rack_close_count = 0

        self.get_logger().info(
            f'white_wall_depth={d:.3f}m '
            f'px={valid.size} '
            f'confirm={self.rack_close_count}/{self.rack_stop_confirm_frames}',
            throttle_duration_sec=0.5
        )

        if self.rack_close_count >= self.rack_stop_confirm_frames:
            self.get_logger().info(
                f'RACK 도착: white_wall_depth median={d:.3f}m <= {self.rack_stop_dist_m:.3f}m'
            )
            return True

        return False

    # ==================== 인지 ====================
    def white_mask(self, hsv):
        lower = np.array([0, 0, self.v_min], dtype=np.uint8)
        upper = np.array([179, self.s_max, 255], dtype=np.uint8)
        return cv2.morphologyEx(cv2.inRange(hsv, lower, upper),
                                cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def red_mask(self, hsv):
        s, v = self.red_s_min, self.red_v_min
        lo1 = np.array([0, s, v], np.uint8);   up1 = np.array([10, 255, 255], np.uint8)
        lo2 = np.array([170, s, v], np.uint8); up2 = np.array([179, 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lo1, up1) | cv2.inRange(hsv, lo2, up2)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def band_rows(self, h, top_ratio, bot_ratio):
        return int(h * top_ratio), int(h * bot_ratio)

    def band_centroid(self, mask, top, bot):
        band = mask[top:bot, :]
        area = int(cv2.countNonZero(band))
        if area < self.min_area:
            return None, area, 0
        M = cv2.moments(band, binaryImage=True)
        cx = M['m10'] / M['m00']
        col = (band.sum(axis=0) > 0).astype(np.uint8)
        clusters = int(np.sum((col[1:] == 1) & (col[:-1] == 0)) + (1 if col[0] == 1 else 0))
        return cx, area, clusters
    
    def band_centroid_min(self, mask, top, bot, min_area):
        band = mask[top:bot, :]
        area = int(cv2.countNonZero(band))
        if area < min_area:
            return None, area, 0

        M = cv2.moments(band, binaryImage=True)
        if M['m00'] == 0:
            return None, area, 0

        cx = M['m10'] / M['m00']
        col = (band.sum(axis=0) > 0).astype(np.uint8)
        clusters = int(np.sum((col[1:] == 1) & (col[:-1] == 0)) + (1 if col[0] == 1 else 0))
        return cx, area, clusters

    def band_hspan(self, mask, top, bot):
        cols = np.where(mask[top:bot, :].sum(axis=0) > 0)[0]
        return int(cols[-1] - cols[0]) if cols.size else 0

    def perceive(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = self.white_mask(hsv)
        white = self.denoise_mask(white, self.noise_min_area)
        red = self.red_mask(hsv)

        nt, nb = self.band_rows(h, self.near_top, self.near_bot)
        ft, fb = self.band_rows(h, self.far_top, self.far_bot)
        # white 기준: 일반 라인트레이싱 / far guide
        near_cx, near_area, near_clusters = self.band_centroid(white, nt, nb)
        far_cx, far_area, far_clusters = self.band_centroid(white, ft, fb)
        far_hspan = self.band_hspan(white, ft, fb)

        # red 기준: parking/object red corner용
        near_red_cx, near_red_area, near_red_clusters = self.band_centroid_min(
            red, nt, nb, self.red_area_min
        )
        far_red_cx, far_red_area2, far_red_clusters = self.band_centroid_min(
            red, ft, fb, self.red_area_min
        )

        # 기존 호환용: far red area는 count 기반으로 유지
        far_red_area = int(cv2.countNonZero(red[ft:fb, :]))
        near_red_area_count = int(cv2.countNonZero(red[nt:nb, :]))

        return {
            'w': w, 'h': h, 'white': white, 'red': red,
            'near_rows': (nt, nb), 'far_rows': (ft, fb),

            # white
            'near_cx': near_cx, 'near_clusters': near_clusters,
            'far_cx': far_cx, 'far_clusters': far_clusters, 'far_hspan': far_hspan,

            # red
            'near_red_cx': near_red_cx,
            'near_red_area': near_red_area_count,
            'near_red_clusters': near_red_clusters,
            'far_red_cx': far_red_cx,
            'far_red_area': far_red_area,
            'far_red_clusters': far_red_clusters,
        }
    # ==================== FOLLOW ====================
    def do_follow(self, meas, frame):
        w = meas['w']
        near_cx = meas['near_cx']
        far_cx = meas['far_cx']
        
        # TO_OBJECTS에서는 라인끝이 아니라 흰색 벽 depth로 rack 도착 판단
        if self.mission_phase == MissionPhase.TO_OBJECTS:
            if self.rack_close_by_white_wall_depth(frame):
                self._handle_stop_end('rack_depth')
                return
        
        # 분기 진입 트리거: QR이 보이면(=②→③ 회전 끝나고 분기 정면) JUNCTION 진입
        self._frame_i += 1
        if (
            self.mission_phase == MissionPhase.TO_QR
            and self.target is not None
            and self._frame_i % self.qr_check_interval == 0
        ):
            qtext, _ = self.detect_qr(frame)
            if qtext is not None:
                self.state = NavState.JUNCTION
                self.jphase = 'ALIGN'
                self.phase_t0 = self._now()
                self.align_left_center = False
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(f'QR 감지({qtext}) -> JUNCTION ALIGN (목표 {self.target})')
                return

        # mission red corner는 white near_cx 로스트보다 먼저 처리
        if self._junction_ahead(meas):
            if self.mission_phase == MissionPhase.TO_OBJECTS and not self.object_red_done:
                self.get_logger().info('TO_OBJECTS: red_line 감지 -> objects 방향 CORNER 진입')
                self._enter_red_corner('object_red')
                return

            if self.mission_phase == MissionPhase.TO_PARKING_RED and not self.parking_red_done:
                self.get_logger().info('TO_PARKING_RED: red_line 감지 -> parking CORNER 진입')
                self._enter_red_corner('parking_red')
                return

        # 라인 로스트
        if near_cx is None:
            self.lost_count += 1
            now = self._now()
            in_grace = now < self.post_corner_grace_until

            # line lost는 더 이상 rack/QR 도착으로 처리하지 않음.
            # 그냥 제자리에서 약하게 회전하며 near_cx 재획득 시도.
            if self.line_lost_t0 is None:
                self.line_lost_t0 = now

            if not in_grace:
                if now - self.line_lost_t0 > self.line_search_timeout:
                    self.publish_cmd(0.0, 0.0, 0.0)
                    self.get_logger().warn(
                        f'line search timeout {self.line_search_timeout:.1f}s: near_cx 재획득 실패',
                        throttle_duration_sec=1.0
                    )
                    return

                self.publish_cmd(0.0, 0.0, self.line_search_dir * abs(self.line_search_wz))
                self.get_logger().warn(
                    f'line lost -> searching wz={self.line_search_dir * abs(self.line_search_wz):.2f}',
                    throttle_duration_sec=1.0
                )
                return

            self.publish_cmd(0.0, 0.0, 0.0)
            return

        # line reacquired
        self.lost_count = 0
        self.line_lost_t0 = None
        self.last_near_cx = near_cx

        err = (near_cx - w / 2.0) / (w / 2.0)
        
        near_err = (near_cx - w/2) / (w/2) if near_cx else 0
        far_err  = (far_cx  - w/2) / (w/2) if far_cx else 0
        # 조향은 far 중심으로 (lateral 오프셋에 둔감)
        steer_err = 0.8 * far_err + 0.2 * near_err
        
        if abs(steer_err) < 0.08:
            steer_err = 0.0
        self.prev_err = steer_err
        if abs(steer_err) > 0.05:
            # 기존 조향 방향과 같은 방향으로 line search
            self.line_search_dir = 1 if (-self.steer_kp * steer_err) >= 0 else -1
        wz = self._clip_wz(-(self.steer_kp * steer_err))

        # 빨강 분기 감지 -> 감속 + 로그 (회전/주차는 Layer 3)
        if self._junction_ahead(meas):
            self.get_logger().info(
                f'분기(빨강) 감지 far_red_area={meas["far_red_area"]}',
                throttle_duration_sec=1.0
            )
            self.publish_cmd(self.slow_vx, 0.0, wz)
            return
        
        # L코너 감지 -> CORNER 진입 (vision only)
        cdir = self._corner_ahead(meas)
        if cdir != 0:
            self.state = NavState.CORNER
            self.corner_phase = 'APPROACH'      # 기존 'ADVANCE' 아님! 추종 유지
            self.corner_dir = cdir
            self.corner_context = 'normal'
            self.active_advance_dist = self.corner_advance_dist
            self.active_rotate_deg = self.rotate_deg
            self.active_rotate_cal = 1.0
            self.phase_t0 = self._now()
            self.far_lost_latched = False
            self.approach_near_lost = 0
            self.get_logger().info(
                f'L코너 1차 감지 dir={"L" if cdir>0 else "R"} span={meas["far_hspan"]} -> APPROACH')
            return

        turn_ratio = abs(wz) / max(self.max_wz, 1e-6)

        # 회전이 클수록 전진속도 줄임
        vx_scale = 1.0 - 0.6 * turn_ratio
        vx_scale = float(np.clip(vx_scale, 0.4, 1.0))

        vx = self.cruise_vx * vx_scale

        self.publish_cmd(vx, 0.0, wz)

    # ==================== CORNER ====================
    
    
    def _corner_cx_pair(self, meas):
        """
        CORNER context별 near/far cx 선택.

        normal:
        near = white near
        far  = white far

        object_red / parking_red:
        near = red near
        far  = white far

        즉 parking 시작 구간처럼 아래쪽 빨강, 위쪽 흰색일 때
        기존 CORNER APPROACH/ADVANCE/ROTATE 로직을 그대로 재사용한다.
        """
        if self.corner_context in ('object_red', 'parking_red'):
            near_cx = meas.get('near_red_cx')
            far_cx = meas.get('far_cx')
            return near_cx, far_cx

        return meas.get('near_cx'), meas.get('far_cx')

    def do_corner(self, meas):
        w = meas['w']
        # context에 따라 near/far 기준을 바꿈
        # normal: white/white
        # object_red, parking_red: red near / white far
        near_cx, far_cx = self._corner_cx_pair(meas)

        t = self._now() - self.phase_t0

        # --- APPROACH: 라인 추종하며 far→None(2차), near→None(3차) 대기 ---
        if self.corner_phase == 'APPROACH':
            if t > self.corner_approach_timeout:           # 라인 안 잃음 = 오검출
                self._end_corner('approach 타임아웃(오검출)')
                return

            if far_cx is None and not self.far_lost_latched:
                self.far_lost_latched = True
                self.get_logger().info('far_cx 사라짐 (2차) - 코너 임박')

            self.approach_near_lost = self.approach_near_lost + 1 if near_cx is None else 0
            if self.approach_near_lost >= self.near_lost_frames:
                self.corner_phase = 'ADVANCE'
                self.phase_t0 = self._now()
                self._reset_odom_integrator()
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(
                    f'near_cx 사라짐 (3차) -> 정지 후 ADVANCE odom 목표 '
                    f'{self.corner_advance_dist:.3f}m, vx={self.corner_advance_vx:.3f} '
                    f'(far_lost={self.far_lost_latched})')
                return

            # 라인 추종 유지 (far 없으면 near로만)
            if far_cx is not None:
                far_err = (far_cx - w/2.0)/(w/2.0)
                near_err = (near_cx - w/2.0)/(w/2.0) if near_cx is not None else far_err
                steer_err = 0.8*far_err + 0.2*near_err
            elif near_cx is not None:
                steer_err = (near_cx - w/2.0)/(w/2.0)
            else:
                steer_err = 0.0
            self.publish_cmd(self.cruise_vx, 0.0, self._clip_wz(-self.steer_kp * steer_err))
            return

        # --- ADVANCE: odometry/filtered 적분 거리 기반 직진 ---
        if self.corner_phase == 'ADVANCE':
            if self.odom is None:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().warn('/odometry/filtered 미수신 - ADVANCE 대기',
                                    throttle_duration_sec=1.0)
                return

            if self.odom_int_dist >= self.active_advance_dist:
                self.corner_phase = 'ADVANCE_STOP'
                self.phase_t0 = self._now()
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(
                    f'ADVANCE 정지 시작 odom_dist={self.odom_int_dist:.3f}m '
                    f'/ 목표={self.active_advance_dist:.3f}m'
                )
                return

            if t > self.corner_advance_timeout:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.state = NavState.STOP_END
                self.publish_status('corner_advance_timeout')
                self.get_logger().error(
                    f'ADVANCE timeout: odom_dist={self.odom_int_dist:.3f}m '
                    f'목표={self.corner_advance_dist:.3f}m -> STOP_END')
                return

            self.publish_cmd(self.corner_advance_vx, 0.0, 0.0)
            return


        # --- ADVANCE_STOP: 실제 선속도 죽을 때까지 대기 ---
        if self.corner_phase == 'ADVANCE_STOP':
            self.publish_cmd(0.0, 0.0, 0.0)

            speed = self._odom_speed_xy()

            if speed <= self.corner_stop_v_thresh or t > self.corner_stop_settle_timeout:
                self.corner_phase = 'ROTATE'
                self.phase_t0 = self._now()
                self._reset_odom_integrator()
                self.publish_cmd(0.0, 0.0, 0.0)
                
                target_deg = self.active_rotate_deg * self.active_rotate_cal

                self.get_logger().info(
                    f'ADVANCE_STOP 완료 speed={speed:.3f}m/s -> ROTATE 시작 '
                    f'목표={target_deg:.1f}deg, wz={self.rot_wz:.3f}, ctx={self.corner_context}'
                )
                return

            return


        # --- ROTATE: odometry/filtered angular.z 적분 각도 기반 회전 ---
        if self.corner_phase == 'ROTATE':
            if self.odom is None:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().warn('/odometry/filtered 미수신 - ROTATE 대기',
                                    throttle_duration_sec=1.0)
                return

            target_yaw = math.radians(self.active_rotate_deg * self.active_rotate_cal)
            stop_yaw = max(0.0, target_yaw - math.radians(self.rotate_stop_margin_deg))

            yaw_progress = self.corner_dir * self.odom_int_yaw

            if yaw_progress >= stop_yaw:
                self.corner_phase = 'ROTATE_STOP'
                self.phase_t0 = self._now()
                self.publish_cmd(0.0, 0.0, 0.0)

                target_deg = self.active_rotate_deg * self.active_rotate_cal

                self.get_logger().info(
                    f'ROTATE 정지 시작 yaw={math.degrees(yaw_progress):.1f}deg '
                    f'/ 목표={target_deg:.1f}deg, ctx={self.corner_context}'
                )
                return

            if yaw_progress < -0.15:
                self.get_logger().warn(
                    f'회전 odom 부호가 명령 방향과 반대일 수 있음: '
                    f'corner_dir={self.corner_dir}, odom_int_yaw={self.odom_int_yaw:.3f}',
                    throttle_duration_sec=1.0
                )

            if t > self.rotate_timeout:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.state = NavState.STOP_END
                self.publish_status('corner_rotate_timeout')
                target_deg = self.active_rotate_deg * self.active_rotate_cal

                self.get_logger().error(
                    f'ROTATE timeout: yaw_progress={math.degrees(yaw_progress):.1f}deg '
                    f'목표={target_deg:.1f}deg -> STOP_END'
                )
                return

            self.publish_cmd(0.0, 0.0, self.corner_dir * self.rot_wz)
            return


        # --- ROTATE_STOP: 실제 각속도 죽을 때까지 대기 후 FOLLOW 복귀 ---
        if self.corner_phase == 'ROTATE_STOP':
            self.publish_cmd(0.0, 0.0, 0.0)

            abs_wz = self._odom_abs_wz()
            yaw_progress = self.corner_dir * self.odom_int_yaw

            if abs_wz <= self.corner_stop_w_thresh or t > self.corner_stop_settle_timeout:
                target_deg = self.active_rotate_deg * self.active_rotate_cal
                self._end_corner(
                    f'odom 회전 완료 yaw={math.degrees(yaw_progress):.1f}deg '
                    f'/ 목표={target_deg:.1f}deg'
                )
                return

            return
        
    def do_junction(self, meas, frame):
        w = meas['w']
        nt, nb = meas['near_rows']
        t = self._now() - self.phase_t0
        lo, hi = self.qr_center_lo, self.qr_center_hi
        ncx = self.near_cx_center(meas['white'], nt, nb, w / 2.0)
        centered = ncx is not None and lo <= ncx <= hi

        # ---------- ALIGN: 메카넘 strafe로 목표 분기에 정렬 ----------
        if self.jphase == 'ALIGN':
            if t > self.junction_align_timeout:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.state = NavState.STOP_END
                self.publish_status('junction_align_timeout')
                self.get_logger().warn(
                    f'ALIGN 타임아웃: 목표 {self.target} 라인 정렬 실패 -> STOP'
                )
                return

            if self.target == 'B':                          # 중앙: 비례 strafe 센터링
                if centered:
                    self.publish_cmd(0.0, 0.0, 0.0)
                    self.get_logger().info(f'B 정렬 완료 near_cx={ncx:.0f}')
                    self._enter_approach(); return
                err = (ncx - w / 2.0) if ncx is not None else 0.0
                vy = float(np.clip(-self.junction_strafe_kp * err / (w / 2.0),
                                -self.junction_strafe_speed, self.junction_strafe_speed))
                self.publish_cmd(0.0, vy, 0.0); return

            # A: 오른쪽(vy<0), C: 왼쪽(vy>0)으로 한 분기 이동 (lose-then-reacquire)
            vy = -self.junction_strafe_speed if self.target == 'A' else self.junction_strafe_speed
            if not self.align_left_center:
                if not centered:                            # 현재(B)선이 중앙 이탈
                    self.align_left_center = True
                    self.get_logger().info('현재 분기 이탈 - 다음 분기 탐색')
            else:
                if centered:                                # 목표 분기선 중앙 진입
                    self.publish_cmd(0.0, 0.0, 0.0)
                    self.get_logger().info(f'{self.target} 분기 정렬 완료 near_cx={ncx:.0f}')
                    self._enter_approach(); return
            self.publish_cmd(0.0, vy, 0.0); return

        # ---------- APPROACH: 분기선 추종 전진, QR 크기/검출률로 정지 ----------
        if self.jphase == 'APPROACH':
            qtext, minbb, qcx, qcy = self.detect_qr_info(frame)
            qr_in_roi = self._qr_in_stop_roi(qcx, qcy, w, meas['h'])
            self._qr_hist.append(1 if qtext is not None else 0)
            if len(self._qr_hist) > self.qr_rate_window:
                self._qr_hist.pop(0)
            rate = sum(self._qr_hist) / max(len(self._qr_hist), 1)

            if qtext == self.target and not self._qr_confirmed:
                self._qr_confirmed = True
                self.get_logger().info(f'QR 확인 OK: {qtext}')
            elif qtext is not None and qtext != self.target:
                self.get_logger().warn(f'QR 불일치 읽음={qtext} 목표={self.target} (정렬 의심)',
                                    throttle_duration_sec=1.0)

            # A/C는 카메라 각도 때문에 흰 라인이 안 보일 수 있으므로
            # QR 중심이 지정 ROI 안에 있고, QR bbox가 충분히 크면 도착으로 판단
            if self.target in ('A', 'C'):
                if qtext == self.target and qr_in_roi and minbb >= self.qr_stop_bbox:
                    self.get_logger().info(
                        f'A/C QR ROI 도착 target={self.target} '
                        f'q=({qcx:.1f},{qcy:.1f}) bbox={minbb} '
                        f'roi=({self.qr_stop_roi_x1:.2f},{self.qr_stop_roi_y1:.2f})~'
                        f'({self.qr_stop_roi_x2:.2f},{self.qr_stop_roi_y2:.2f}) '
                        f'-> stop_qr'
                    )
                    self._handle_stop_end(f'qr_roi_{self.target}')
                    return
                
            # 도착 정지: 검출률>=임계 & QR 최소변>=임계 & 목표 일치
            if rate >= self.qr_min_rate and minbb >= self.qr_stop_bbox and qtext == self.target:
                self.get_logger().info(
                    f'QR 목표 도착 {self.target} rate={rate:.2f} bbox={minbb} -> stop_qr'
                )
                self._handle_stop_end(f'qr_{self.target}')
                return
            
            if t > self.junction_approach_timeout:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.state = NavState.STOP_END
                self.publish_status('junction_align_timeout')
                self.get_logger().warn(
                    f'ALIGN 타임아웃: 목표 {self.target} 라인 정렬 실패 -> STOP'
                )
                return

            wz = self._clip_wz(-self.steer_kp * (ncx - w / 2.0) / (w / 2.0)) if ncx is not None else 0.0
            self.publish_cmd(self.junction_approach_vx, 0.0, wz)
            return
        
    def do_turn_180(self):
        t = self._now() - self.phase_t0

        if self.odom is None:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.get_logger().warn('/odometry/filtered 미수신 - TURN_180 대기',
                                throttle_duration_sec=1.0)
            return

        target_yaw = math.radians(self.turn_180_deg * self.turn_180_cal)
        yaw_progress = self.turn_dir * self.odom_int_yaw

        if yaw_progress >= target_yaw:
            self.publish_cmd(0.0, 0.0, 0.0)

            self.state = NavState.TURN_PAUSE
            self.phase_t0 = self._now()
            self._reset_odom_integrator()

            self.get_logger().info(
                f'180도 완료 -> {self.turn_pause_time:.1f}s 정지 대기'
            )
            return

        if t > self.turn_180_timeout:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.state = NavState.STOP_END
            self.publish_status('turn_180_timeout')
            self.get_logger().error(
                f'TURN_180 timeout yaw={math.degrees(yaw_progress):.1f}deg '
                f'/ 목표={math.degrees(target_yaw):.1f}deg'
            )
            return

        self.publish_cmd(0.0, 0.0, self.turn_dir * self.turn_180_wz)
        
    def do_turn_pause(self):
        t = self._now() - self.phase_t0
        self.publish_cmd(0.0, 0.0, 0.0)

        if t < self.turn_pause_time:
            return

        next_phase = self.turn_after_phase
        self.turn_after_phase = None

        if next_phase == MissionPhase.TO_QR:
            self.mission_phase = MissionPhase.TO_QR
            self.state = NavState.FOLLOW
            self.lost_count = 0
            self.prev_err = 0.0
            self._qr_hist = []
            self._qr_confirmed = False

            self.get_logger().info(
                f'180도 대기 완료 -> TO_QR target={self.target}'
            )
            return

        if next_phase == MissionPhase.TO_OBJECTS:
            self.mission_phase = MissionPhase.TO_OBJECTS
            self.state = NavState.FOLLOW
            self.object_red_done = False
            self.lost_count = 0
            self.prev_err = 0.0

            self.get_logger().info('180도 대기 완료 -> TO_OBJECTS')
            return

        if next_phase == MissionPhase.TO_PARKING_RED:
            self.mission_phase = MissionPhase.TO_PARKING_RED
            self.state = NavState.FOLLOW
            self.parking_red_done = False
            self.lost_count = 0
            self.prev_err = 0.0

            self.get_logger().info('180도 대기 완료 -> TO_PARKING_RED')
            return

        if next_phase == MissionPhase.PARKED:
            self.mission_phase = MissionPhase.PARKED
            self.state = NavState.IDLE
            self.publish_status('parked')

            self.get_logger().info('parking 최종 180도 대기 완료 -> PARKED')
            return

        self.state = NavState.IDLE
        
    def do_parking_forward(self):
        t = self._now() - self.phase_t0

        if t >= self.parking_forward_time:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.state = NavState.PARK_PAUSE
            self.phase_t0 = self._now()

            self.get_logger().info(
                f'parking forward {self.parking_forward_time:.1f}s 완료 -> pause'
            )
            return

        self.publish_cmd(self.cruise_vx, 0.0, 0.0)


    def do_parking_pause(self):
        t = self._now() - self.phase_t0
        self.publish_cmd(0.0, 0.0, 0.0)

        if t >= self.parking_pause_time:
            self.get_logger().info(
                f'parking pause {self.parking_pause_time:.1f}s 완료 -> 최종 180도 정렬'
            )
            self._enter_turn_180(MissionPhase.PARKED, reason='parking_final_align')
            return
    
    def _enter_turn_180(self, after_phase, reason=''):
        self.state = NavState.TURN_180
        self.turn_after_phase = after_phase
        self.turn_dir = 1

        self.phase_t0 = self._now()
        self._reset_odom_integrator()
        self.publish_cmd(0.0, 0.0, 0.0)

        self.get_logger().info(
            f'180도 회전 시작 reason={reason}, after={after_phase.name}, '
            f'wz={self.turn_180_wz:.2f}, deg={self.turn_180_deg:.1f}, cal={self.turn_180_cal:.3f}'
        )

    def _enter_approach(self):
        self.jphase = 'APPROACH'
        self.phase_t0 = self._now()
        self._qr_hist = []
        self._qr_confirmed = False
        
    def _end_corner(self, why):
        ctx = self.corner_context

        self.corner_phase = None
        self.prev_err = 0.0
        self.lost_count = 0
        self.publish_cmd(0.0, 0.0, 0.0)

        if ctx == 'parking_red':
            self.state = NavState.PARK_FORWARD
            self.phase_t0 = self._now()
            self.corner_context = 'normal'

            self.get_logger().info(
                f'parking_red 코너 완료({why}) -> cruise_vx로 {self.parking_forward_time:.1f}s 직진'
            )
            return

        # object_red 또는 normal은 기존처럼 FOLLOW
        self.state = NavState.FOLLOW
        self.corner_context = 'normal'
        self.post_corner_grace_until = self._now() + self.post_corner_grace

        self.get_logger().info(
            f'코너 완료({why}) -> FOLLOW (grace {self.post_corner_grace}s)'
        )

    def _enter_red_corner(self, context):
        self.state = NavState.CORNER
        self.corner_phase = 'APPROACH'
        self.phase_t0 = self._now()

        self.far_lost_latched = False
        self.approach_near_lost = 0
        self._reset_odom_integrator()

        self.corner_context = context

        if context == 'object_red':
            self.object_red_done = True

            self.corner_dir = int(self.object_red_turn_dir)
            self.active_advance_dist = self.corner_advance_dist
            self.active_rotate_deg = self.object_red_rotate_deg
            self.active_rotate_cal = 1.0

            self.get_logger().info(
                f'object_red CORNER 시작: dir={self.corner_dir}, '
                f'advance={self.active_advance_dist:.3f}m, '
                f'rotate={self.active_rotate_deg:.1f}deg'
            )
            return

        if context == 'parking_red':
            self.parking_red_done = True

            self.corner_dir = int(self.parking_red_turn_dir)
            self.active_advance_dist = self.corner_advance_dist * self.parking_advance_cal
            self.active_rotate_deg = self.parking_red_rotate_deg
            self.active_rotate_cal = self.parking_rotate_cal

            self.get_logger().info(
                f'parking_red CORNER 시작: dir={self.corner_dir}, '
                f'advance={self.active_advance_dist:.3f}m '
                f'(cal={self.parking_advance_cal:.3f}), '
                f'rotate={self.active_rotate_deg:.1f}deg '
                f'(cal={self.parking_rotate_cal:.3f})'
            )
            return
        
    def _handle_stop_end(self, reason='line_end'):
        self.publish_cmd(0.0, 0.0, 0.0)

        if self.mission_phase == MissionPhase.TO_OBJECTS:
            if reason != 'rack_depth':
                self.get_logger().warn(
                    f'TO_OBJECTS에서 {reason} 수신했지만 rack_depth가 아니므로 도착 처리 안 함'
                )
                return

            self.state = NavState.STOP_END
            self.mission_phase = MissionPhase.WAIT_PICKED
            self.publish_stop_obj()
            self.get_logger().info(
                f'OBJECT rack 도착({reason}) -> WAIT_PICKED'
            )
            return

        if self.mission_phase == MissionPhase.TO_QR:
            if not str(reason).startswith('qr_'):
                self.get_logger().warn(
                    f'TO_QR에서 {reason} 수신했지만 QR 확인이 아니므로 도착 처리 안 함'
                )
                return

            self.state = NavState.STOP_END
            self.mission_phase = MissionPhase.WAIT_PLACED
            self.publish_stop_qr()
            self.get_logger().info(
                f'QR 목표 {self.target} 도착({reason}) -> WAIT_PLACED'
            )
            return

        self.state = NavState.STOP_END
        self.publish_status('arrived')
        self.get_logger().info(
            f'STOP_END({reason}) phase={self.mission_phase.name}'
        )

    # ==================== 검출기 ====================
    def _corner_ahead(self, meas):
        """흰 라인이 FAR에서 가로로 넓게 퍼지면 L코너. 반환 +1=좌, -1=우, 0=아님."""
        if meas['far_cx'] is None:
            return 0
        if meas['far_clusters'] >= 2:      # fork는 제외 (Layer 3)
            return 0
        if meas['far_hspan'] < self.corner_span_min:
            return 0
        near_cx = meas['near_cx'] if meas['near_cx'] is not None else meas['w'] / 2.0
        return +1 if meas['far_cx'] < near_cx else -1   # FAR가 왼쪽이면 좌회전

    def _junction_ahead(self, meas):
        return (
            meas.get('far_red_area', 0) >= self.red_area_min
            or meas.get('near_red_area', 0) >= self.red_area_min
        )

    # ==================== 유틸 ====================
    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _clip_wz(self, wz):
        return float(np.clip(wz, -self.max_wz, self.max_wz))

    def publish_cmd(self, vx, vy, wz):
        if not self.enable_drive:
            vx = vy = wz = 0.0

        # ===== combined motion safety limit =====
        # 회전이 클수록 선속도를 줄임
        turn_ratio = abs(wz) / max(self.max_wz, 1e-6)

        if abs(vx) > 1e-6 and abs(wz) > 1e-6:
            vx_scale = 1.0 - 0.6 * turn_ratio
            vx_scale = float(np.clip(vx_scale, 0.4, 1.0))
            vx *= vx_scale

        self.last_cmd = (float(vx), float(vy), float(wz))

        t = Twist()
        t.linear.x = float(vx)
        t.linear.y = float(vy)
        t.angular.z = float(wz)
        self.cmd_pub.publish(t)

    def publish_status(self, text):
        m = String()
        m.data = text
        self.status_pub.publish(m)

    def publish_debug_image(self, frame, meas):
        """
        원본 frame이 아니라 binary white mask를 배경으로 해서
        ROI / 중심점 / red mask / 상태 텍스트를 그린 뒤
        CompressedImage로 publish.
        """

        # white binary mask: 0/255 mono 이미지를 BGR로 변환
        dbg = cv2.cvtColor(meas['white'], cv2.COLOR_GRAY2BGR)

        w = meas['w']
        h = meas['h']

        # red mask는 빨간색으로 overlay
        red = meas.get('red')
        if red is not None:
            dbg[red > 0] = (0, 0, 255)   # BGR 기준 빨강

        # 중앙선
        cv2.line(dbg, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
        
        # QR stop ROI 표시
        x1 = int(self.qr_stop_roi_x1 * w)
        x2 = int(self.qr_stop_roi_x2 * w)
        y1 = int(self.qr_stop_roi_y1 * h)
        y2 = int(self.qr_stop_roi_y2 * h)

        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        # near / far ROI와 중심점
        for (rows, color, cx) in [
            (meas['near_rows'], (0, 200, 0), meas['near_cx']),
            (meas['far_rows'], (200, 120, 0), meas['far_cx']),
        ]:
            t, b = rows
            cv2.rectangle(dbg, (0, t), (w - 1, b), color, 1)

            if cx is not None:
                cv2.circle(dbg, (int(cx), (t + b) // 2), 6, (0, 0, 255), -1)

        for (rows, cx) in [
            (meas['near_rows'], meas.get('near_red_cx')),
            (meas['far_rows'], meas.get('far_red_cx')),
        ]:
            if cx is not None:
                t, b = rows
                cv2.circle(dbg, (int(cx), (t + b) // 2), 5, (255, 0, 255), -1)

        # 상태 텍스트
        ph = self.corner_phase if self.state == NavState.CORNER else ''
        cv2.putText(
            dbg,
            f'{self.state.name} {ph}  span={meas["far_hspan"]} red={meas["far_red_area"]}',
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        # binary 이미지는 jpg보다 png가 깔끔함
        ok, enc = cv2.imencode('.png', dbg)
        if ok:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            msg.format = 'png'
            msg.data = enc.tobytes()
            self.debug_pub.publish(msg)
            
    def publish_stop_obj(self):
        self.stop_obj_pub.publish(Empty())
        self.publish_status('stop_obj')
        self.get_logger().info('/stop_obj publish')


    def publish_stop_qr(self):
        self.stop_qr_pub.publish(Empty())
        self.publish_status('stop_qr')
        self.get_logger().info('/stop_qr publish')

    def destroy_node(self):
        if self.csv_file is not None:
            try:
                self.csv_file.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LineTracer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
