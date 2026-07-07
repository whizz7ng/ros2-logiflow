#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogiFlow 라인 트레이서 - Layer 2 (v1.2)
=======================================
v1.1(FOLLOW + STOP_END + 빨강 분기 검출) + L코너 처리(CORNER).

CORNER 시퀀스 (odom 없이 vision + 시간만 사용):
  ADVANCE : 코너 감지 후 advance_time(s) 동안 slow_vx 전진 (카메라 사각<32cm 보정)
  ROTATE  : 피드백 없이 회전. 진입 라인이 '연속 N프레임 사라진 뒤'에야 새 라인 중앙 탐색
            -> 진입 라인 오인식 방지 (lose-then-reacquire). + 안전 타임아웃
  STRAFE  : 메카넘 strafe로 잔여 측면오차 제거 -> FOLLOW

색 규약: red_line=실제 빨간 주행 라인 / blue=실제 파란 분기·주차 marker / white_wall=실제 흰 벽
분기 종류: 빨강=파킹T(Layer3), 흰색 wide span=L코너(여기), 흰색 2클러스터=QR fork(Layer3)
"""

import math
import time
import threading
import os
import json
import numpy as np
from enum import Enum

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Empty
from nav_msgs.msg import Odometry
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
    PRE_TURN_SETTLE = 10

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
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('use_compressed_input', False)  # control node에서는 미사용(호환용)
        self.declare_parameter('perception_topic', '/line_tracer/perception')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('image_timeout_sec', 0.5)
        self.declare_parameter('frame_width', 320)
        self.declare_parameter('frame_height', 240)
        self.declare_parameter('control_rate', 20.0)

        self.declare_parameter('near_top', 0.80)
        self.declare_parameter('near_bot', 0.98)
        self.declare_parameter('far_top', 0.67)
        self.declare_parameter('far_bot', 0.79)

        # 흰색 라인 HSV
        self.declare_parameter('s_max', 70)
        self.declare_parameter('v_min', 175)
        self.declare_parameter('min_area', 250)
        self.declare_parameter('noise_min_area', 160)   # 이보다 작은 흰 덩어리는 노이즈로 버림

        # 빨강 분기 마커 HSV
        self.declare_parameter('red_s_min', 95)
        self.declare_parameter('red_v_min', 65)
        self.declare_parameter('red_area_min', 350)

        # 제어 / 속도
        self.declare_parameter('steer_kp', 0.2)
        self.declare_parameter('steer_kd', 0.0)
        self.declare_parameter('cruise_vx', 0.22)
        self.declare_parameter('slow_vx', 0.12)
        self.declare_parameter('max_wz', 0.40)
        self.declare_parameter('lost_frames_stop', 6)
        
        # ---- myAGV 2023 angular deadzone 대응 ----
        # 실험 결과: 0 < |angular.z| < 약 0.40 구간이 실제로는 최소 약 0.4rad/s로 튐.
        # 따라서 일반 라인 추종에서는 yaw 대신 mecanum lateral(y) 보정을 우선 사용한다.
        self.declare_parameter('use_lateral_follow', True)
        self.declare_parameter('follow_err_deadband', 0.08)
        self.declare_parameter('follow_big_err', 0.35)
        self.declare_parameter('follow_vy_kp', 0.045)
        self.declare_parameter('follow_vy_max', 0.04)
        self.declare_parameter('follow_turn_vx', 0.02)
        # 큰 오차에서만 쓰는 '효과 목표' yaw. vel_filter_node가 pulse로 안전 변환한다.
        self.declare_parameter('follow_turn_wz', 0.10)
        self.declare_parameter('angular_hw_min_wz', 0.40)
        

        # 코너(Layer 2) - odom 없이 vision + 시간만 사용
        self.declare_parameter('corner_span_min', 100)   # FAR 흰색 가로폭(px) 이상이면 L코너 (320폭 기준)
        self.declare_parameter('advance_time', 1.0)      # 코너 감지 후 slow_vx 전진 시간(s)
        self.declare_parameter('rot_wz', 0.45)
        self.declare_parameter('rotate_lost_frames', 3)  # 진입 라인이 사라졌다고 볼 연속 프레임수
        self.declare_parameter('rotate_timeout', 8.0)    # 회전 안전 타임아웃(s) - 느려진 만큼 늘림
        self.declare_parameter('reacquire_tol', 0.15)    # 라인 중앙 판정 정규화오차
        self.declare_parameter('strafe_kp', 0.4)
        self.declare_parameter('strafe_max', 0.10)       # strafe 상한 (최대선속도의 ~10%)
        self.declare_parameter('strafe_fix_tol', 0.08)
        self.declare_parameter('strafe_fix_timeout', 2)
        self.declare_parameter('post_corner_grace', 0.7)  # 코너 종료 후 STOP_END 무시 시간(s)
        # 코너(Layer 2) - 시간 기반 사각 통과 + 고정 회전
        self.declare_parameter('corner_advance_dist', 0.32)    # 카메라 전방 사각거리(m)
        self.declare_parameter('corner_advance_vx', 0.12)      # ADVANCE 직진속도(=cruise, 저속보다 반복성 좋음)
        self.declare_parameter('corner_advance_timeout', 6.0)
        self.declare_parameter('rotate_deg', 85.0)             # 코너 회전각(도)
        self.declare_parameter('corner_approach_timeout', 8.0) # 라인 안 잃으면 오검출로 보고 복귀(s)
        self.declare_parameter('near_lost_frames', 3)          # near_cx None 확정 프레임수
        
        self.declare_parameter('rotate_stop_margin_deg', 5.0)       # 90도 목표면 85도쯤부터 정지

        # 회전 중 라인이 중앙 윈도우에 재진입하면 각도 목표 전이라도 조기 정지
        # 320px frame 기준 중앙 허용 범위: 90~220px
        self.declare_parameter('rotate_center_lo', 100)
        self.declare_parameter('rotate_center_hi', 220)
        # True: 회전 시작 직후 이미 보이던 기존 라인 오인식을 막기 위해
        #       한 번 중앙 윈도우를 벗어난 뒤 다시 들어올 때만 조기 정지
        self.declare_parameter('rotate_center_require_lost_once', True)

        self.declare_parameter('corner_stop_v_thresh', 0.02)
        self.declare_parameter('corner_stop_w_thresh', 0.05)
        self.declare_parameter('corner_stop_settle_timeout', 1.2)

        self.declare_parameter('publish_debug', True)
        self.declare_parameter('enable_drive', True)   # False면 모든 cmd_vel을 0으로 (인지만 테스트)
        self.declare_parameter('telemetry_csv', "/home/er/myagv_ros2/src/tracer/log/telemetry.csv")    # 경로 지정시 매 루프 CSV 1줄 기록 (엑셀 디버깅용)

        # ---- debug / inspection ----
        # inspect_only=True  : 주행/상태전이 없이 인식 결과만 publish
        # freeze_transition=True : 자동 state/phase 전이를 막고 현재 장면을 멈춰서 관찰
        self.declare_parameter('inspect_only', False)
        self.declare_parameter('freeze_transition', False)
        self.declare_parameter('debug_perception_rate', 5.0)
        self.declare_parameter('debug_snapshot_dir', '/home/er/myagv_ros2/src/tracer/log/snapshots')
        
        # ---- 미션 / 분기(JUNCTION) ----
        self.declare_parameter('start_idle', True)        # True면 파킹에서 대기, 시작 메세지로 출발
        self.declare_parameter('default_target', 'B')     # start_idle=False일 때 테스트용 목표
        self.declare_parameter('qr_center_lo', 130)       # 분기 정렬 near_cx 하한
        self.declare_parameter('qr_center_hi', 190)       # 상한 (중앙 160 ±15) 
        self.declare_parameter('qr_stop_bbox', 50)       # QR 최소변(px) 이상이면 도착 정지
        self.declare_parameter('qr_min_rate', 0.40)       # 최근 검출률 이상
        self.declare_parameter('qr_rate_window', 6)      # 검출률 평균 윈도우(프레임)
        self.declare_parameter('qr_check_interval', 2)    # FOLLOW 중 QR 검사 주기(프레임,CPU)
        
        # QR_B 접근 후 A/C로 y축 이동할 때 timeout
        self.declare_parameter('qr_target_shift_timeout', 6.0)

        # QR 구역에서 빠져나온 뒤 180도 회전 후 추가 정지 시간
        self.declare_parameter('return_turn_pause_time', 2.5)
        
        self.declare_parameter('junction_strafe_speed', 0.08)
        self.declare_parameter('junction_strafe_kp', 0.5)
        self.declare_parameter('junction_approach_vx', 0.10)
        self.declare_parameter('junction_align_timeout', 12.0)
        self.declare_parameter('junction_approach_timeout', 8.0)
        
        # ---- mission / object / parking ----
        self.declare_parameter('object_red_turn_dir', -1)       # objects 방향: 로봇 기준 오른쪽이면 보통 -1
        self.declare_parameter('object_red_rotate_deg', 70.0)

        self.declare_parameter('turn_180_wz', 0.40)
        self.declare_parameter('turn_180_deg', 180.0)
        self.declare_parameter('turn_180_cal', 1.0)
        self.declare_parameter('turn_180_timeout', 12.0)
        self.declare_parameter('turn_pause_time', 1.0)
        
        self.declare_parameter('turn_180_center_stop_enable', True)
        self.declare_parameter('turn_180_center_min_yaw_deg', 120.0)

        # QR A/C -> B 정렬, 혹은 이미 B에서 나갈 때 바로 180도 회전을 시작하지 않고
        # 실제 모터가 멈출 시간을 확보하기 위한 settle 구간
        self.declare_parameter('pre_turn_settle_time', 1.5)
        self.declare_parameter('pre_turn_settle_use_odom', True)

        self.declare_parameter('parking_red_turn_dir', -1)       # 실제 방향 보고 +1/-1 튜닝
        self.declare_parameter('parking_red_rotate_deg', 80.0)

        # 주차 진입은 '파란색이 보임'이 아니라 '빨간 주행 라인과 파란 주차선이 만나는 접점'으로 판단
        self.declare_parameter('parking_joint_required', True)
        self.declare_parameter('parking_joint_min_touch_px', 25)
        self.declare_parameter('parking_joint_confirm_frames', 2)

        # 주차 진입 회전 방향 자동 판단:
        # red+blue 접점(parking_joint_cx)에서 near band의 파란 주차선 중심(near_blue_cx)이
        # 오른쪽에 있으면 우회전(-1), 왼쪽에 있으면 좌회전(+1)로 들어간다.
        self.declare_parameter('parking_dynamic_turn_dir', True)
        self.declare_parameter('parking_turn_dir_deadband_px', 8.0)

        # 주차 진입 회전 종료: 파란 주차구역 테두리(위/아래)가 화면에서 수평으로 잡히면 정지
        # Orin perception의 parking_blue_alignment 값을 사용한다.
        self.declare_parameter('parking_blue_align_stop_enable', True)
        self.declare_parameter('parking_blue_align_confirm_frames', 2)
        self.declare_parameter('parking_blue_align_min_yaw_deg', 40.0)
        self.declare_parameter('parking_blue_align_require_both', True)

        # 네가 말한 parking 쪽 별도 cal
        self.declare_parameter('parking_advance_cal', 1.0)
        self.declare_parameter('parking_rotate_cal', 1.0)

        # parking 진입:
        #   blue marker corner 회전 후, 파란 선이 near/far에서 모두 사라질 때까지 직진
        #   -> odom 기준 10cm 추가 직진
        #   -> 1.5초 정지
        #   -> 최종 180도 회전 후 parked
        self.declare_parameter('parking_forward_time', 4.5)      # legacy: 기존 시간 기반값. 새 로직에서는 timeout 보조용
        self.declare_parameter('parking_forward_timeout', 8.0)   # 파란선 소실 대기 최대 시간
        self.declare_parameter('parking_forward_vx', 0.08)
        self.declare_parameter('parking_extra_dist_m', 0.10)
        self.declare_parameter('parking_pause_time', 1.5)
        self.declare_parameter('parking_blue_lost_area_px', 80)  # near/far blue_px가 이 값 이하이면 '없음'으로 봄
        self.declare_parameter('parking_blue_lost_frames', 3)
        self.declare_parameter('parking_corner_lost_frames', 3)  # parking corner 접근 중 near blue가 사라진 연속 프레임

        self.declare_parameter('return_b_strafe_speed', 0.08)
        self.declare_parameter('return_b_timeout', 6.0)
        
        # A/C QR 접근 시, QR 중심이 이 ROI 안에 있고 bbox가 충분히 크면 stop_qr
        # frame 320x240 기준 비율. 네가 그린 노란 사각형 느낌으로 잡은 기본값.
        self.declare_parameter('qr_stop_roi_x1', 0.25)
        self.declare_parameter('qr_stop_roi_x2', 0.75)
        self.declare_parameter('qr_stop_roi_y1', 0.20)
        self.declare_parameter('qr_stop_roi_y2', 0.70)
        

        # ---- line lost search ----
        self.declare_parameter('line_search_wz', 0.4)
        self.declare_parameter('line_search_default_dir', 1)   # +1 좌회전, -1 우회전
        self.declare_parameter('line_search_timeout', 8.0)
        
        # 추가: line lost 시 좌/우 번갈아 탐색
        self.declare_parameter('line_search_alternate', True)
        self.declare_parameter('line_search_switch_sec', 2.0)
        
        # ---- rack white wall depth stop ----
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('use_white_wall_rack_stop', True)
        self.declare_parameter('rack_stop_dist_m', 0.52)
        
        # 기존 frame confirm은 디버그/호환용으로 남겨둠
        self.declare_parameter('rack_stop_confirm_frames', 3)

        # depth 15fps / 순간 null 대응용
        self.declare_parameter('rack_required_close_sec', 0.25)  # 가까운 상태가 이 시간 이상 유지되면 stop
        self.declare_parameter('rack_depth_hold_sec', 0.35)      # valid 부족/null을 이 시간까지는 리셋 없이 버팀
        self.declare_parameter('rack_depth_max_age_sec', 1.0)    # 최신 depth가 이보다 오래되면 depth 없음으로 판단

        self.declare_parameter('rack_wall_roi_x1', 0.05)
        self.declare_parameter('rack_wall_roi_x2', 0.95)
        self.declare_parameter('rack_wall_roi_y1', 0.00)
        self.declare_parameter('rack_wall_roi_y2', 0.38)

        self.declare_parameter('rack_wall_white_s_max', 60)
        self.declare_parameter('rack_wall_white_v_min', 160)
        self.declare_parameter('rack_wall_min_valid_px', 13000)
        
        self.declare_parameter('stop_obj_publish_delay_sec', 2.5)
        self.declare_parameter('stop_qr_publish_delay_sec', 2.5)
        
        
        # 추가
        self.declare_parameter('rack_approach_dist_m', 1.20)
        self.declare_parameter('rack_approach_min_valid_px', 13000)
        self.declare_parameter('rack_approach_vx', 0.08)
        
        

        self._load_params()

        # ---------------- Perception 입력: Orin Nano가 만든 JSON만 구독 ----------------
        # 이 control node는 image_raw/depth_raw를 절대 구독하지 않는다.
        # raw image/depth는 Orin perception node에서 처리하고, 여기서는 숫자 JSON만 받는다.
        self.meas_lock = threading.Lock()
        self.latest_meas = None
        self.last_perception_time = 0.0
        self.last_qr_info = (None, 0, None, None)
        self.last_rack_wall_depth_m = None
        self.last_rack_wall_valid_px = 0

        # rack depth 안정화용. depth 자체는 안 받지만 Orin JSON의 median 값을 기반으로 시간 confirm한다.
        self.rack_close_count = 0
        self.rack_close_since = None
        self.rack_last_valid_depth_m = None
        self.rack_last_valid_depth_time = 0.0
        self.rack_last_valid_px = 0
        
        self.stop_obj_delay_timer = None
        self.stop_obj_pending = False
        self.stop_qr_delay_timer = None
        self.stop_qr_pending = False

        self.last_near_cx = None
        self.line_lost_t0 = None
        self.line_search_dir = int(self.line_search_default_dir)
        if self.line_search_dir == 0:
            self.line_search_dir = 1

        perception_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.perception_sub = self.create_subscription(
            String,
            self.perception_topic,
            self.cb_perception,
            perception_qos
        )
        self.get_logger().info(f'Perception JSON 구독: {self.perception_topic}')

        self.return_b_after_phase = None
        self.pending_turn_after_phase = None
        self.pending_turn_reason = ''
        # QR 구역에서 빠져나올 때 실제 현재 위치(A/B/C)를 따로 저장한다.
        # 다음 /place_target을 미리 self.target에 넣어도 RETURN_TO_QR_B 방향이 꼬이지 않게 하기 위함.
        self.return_b_from_target = None

        # ---------------- 통신 ----------------
        cmd_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', cmd_qos)
        self.status_pub = self.create_publisher(String, '/nav_status', 10)
        # Orin의 /line_tracer/perception을 다시 publish하면 self-loop가 생기므로 별도 상태 토픽 사용
        self.perception_pub = self.create_publisher(String, '/line_tracer/control_status', 10)
        self.stop_obj_pub = self.create_publisher(Empty, '/stop_obj', 10)
        self.stop_qr_pub = self.create_publisher(Empty, '/stop_qr', 10)
        
        

        self.create_subscription(String, '/place_target', self.cb_place_target, 10)
        self.create_subscription(String, '/arm_status', self.cb_arm_status, 10)
        self.create_subscription(Empty, '/go_parking', self.cb_go_parking, 10)
        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 10)
        # /mission_cmd는 더 이상 시작 트리거로 쓰지 않는다.
        # 시작/재출발은 /place_target 또는 /go_parking 콜백에서 처리한다.
        # self.create_subscription(String, '/mission_cmd', self.cb_mission, 10)
        self.create_subscription(String, '/debug_nav_cmd', self.cb_debug_nav_cmd, 10)

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
        self.rack_depth_armed = False

        self.turn_after_phase = None
        self.turn_dir = 1
        
        self.turn_after_phase = None
        self.turn_dir = 1

        # TURN_180 이후 pause 시간을 상황별로 다르게 쓰기 위한 값
        self.active_turn_pause_time = self.turn_pause_time

        # parking 전용 상태
        self.parking_forward_phase = None       # 'UNTIL_BLUE_LOST' | 'EXTRA'
        self.parking_blue_lost_count = 0
        self.parking_near_blue_seen = False
        self.parking_joint_seen_count = 0
        self.parking_blue_align_seen_count = 0

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
        # ROTATE / TURN_180 조기정지용: 기존 라인 오인식을 막기 위해
        # 중앙 윈도우 이탈 후 재진입했는지 보는 래치
        self.corner_rotate_line_center_armed = False
        self.turn_180_line_center_armed = False
        self.rotate_lost_count = 0
        self.last_cmd = (0.0, 0.0, 0.0)
        self.latest_meas = None
        self.last_qr_info = (None, 0, None, None)
        self.last_rack_wall_depth_m = None
        self.last_rack_wall_valid_px = 0
        self._last_perception_pub_t = 0.0
        self._last_freeze_log_t = 0.0
        self.post_corner_grace_until = 0.0   # 이 시각까지는 STOP_END 보호
        
        self.far_lost_latched = False     # APPROACH 중 far가 한번 사라졌는지(2차 래치)
        self.approach_near_lost = 0       # near None 연속 카운트(3차)
        self._advance_secs = 0.0          # 진입시 dist/vx*cal로 계산
        self._rotate_secs = 0.0           # 진입시 deg/wz*cal로 계산
        
        self.corner_rotate_line_center_armed = False
        self.corner_rotate_blue_center_armed = False
        self.turn_180_line_center_armed = False

        # 텔레메트리 CSV (엑셀 디버깅용) — 경로 지정시에만 활성
        self.csv_file = None
        if self.telemetry_csv:
            try:
                self.csv_file = open(self.telemetry_csv, 'w')
                self.csv_file.write('t,state,phase,near_cx,err,far_cx,far_hspan,far_clusters,blue_far_area,vx,vy,wz\n')
                self.get_logger().info(f'텔레메트리 기록 시작: {self.telemetry_csv}')
            except Exception as e:
                self.get_logger().error(f'CSV 열기 실패: {e}')

        self.timer = self.create_timer(1.0 / max(self.control_rate, 1.0), self.loop)
        self.add_on_set_parameters_callback(self._on_param_change)
        self.get_logger().info(
            f'line_tracer_control 시작 | perception_topic={self.perception_topic} {self.frame_width}x{self.frame_height} '
            f'| cruise_vx={self.cruise_vx} max_wz={self.max_wz} rot_wz={self.rot_wz} '
            f'steer_kp={self.steer_kp} steer_kd={self.steer_kd} '
            f'| v_min={self.v_min} s_max={self.s_max} corner_span_min={self.corner_span_min} '
            f'| enable_drive={self.enable_drive} publish_debug={self.publish_debug} '
            f'| inspect_only={self.inspect_only} freeze_transition={self.freeze_transition}')

    def _load_params(self):
        g = self.get_parameter
        self.image_topic = g('image_topic').value
        self.use_compressed_input = g('use_compressed_input').value
        self.perception_topic = g('perception_topic').value
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
        self.use_lateral_follow = bool(g('use_lateral_follow').value)
        self.follow_err_deadband = float(g('follow_err_deadband').value)
        self.follow_big_err = float(g('follow_big_err').value)
        self.follow_vy_kp = float(g('follow_vy_kp').value)
        self.follow_vy_max = float(g('follow_vy_max').value)
        self.follow_turn_vx = float(g('follow_turn_vx').value)
        self.follow_turn_wz = float(g('follow_turn_wz').value)
        self.angular_hw_min_wz = float(g('angular_hw_min_wz').value)
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
        self.inspect_only = g('inspect_only').value
        self.freeze_transition = g('freeze_transition').value
        self.debug_perception_rate = g('debug_perception_rate').value
        self.debug_snapshot_dir = g('debug_snapshot_dir').value
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
        self.rotate_center_lo = int(g('rotate_center_lo').value)
        self.rotate_center_hi = int(g('rotate_center_hi').value)
        self.rotate_center_require_lost_once = bool(g('rotate_center_require_lost_once').value)
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
        
        self.turn_180_center_stop_enable = bool(g('turn_180_center_stop_enable').value)
        self.turn_180_center_min_yaw_deg = float(g('turn_180_center_min_yaw_deg').value)
        self.pre_turn_settle_time = float(g('pre_turn_settle_time').value)
        self.pre_turn_settle_use_odom = bool(g('pre_turn_settle_use_odom').value)

        self.parking_red_turn_dir = g('parking_red_turn_dir').value
        self.parking_red_rotate_deg = g('parking_red_rotate_deg').value
        self.parking_joint_required = bool(g('parking_joint_required').value)
        self.parking_joint_min_touch_px = int(g('parking_joint_min_touch_px').value)
        self.parking_joint_confirm_frames = int(g('parking_joint_confirm_frames').value)
        self.parking_dynamic_turn_dir = bool(g('parking_dynamic_turn_dir').value)
        self.parking_turn_dir_deadband_px = float(g('parking_turn_dir_deadband_px').value)
        self.parking_blue_align_stop_enable = bool(g('parking_blue_align_stop_enable').value)
        self.parking_blue_align_confirm_frames = int(g('parking_blue_align_confirm_frames').value)
        self.parking_blue_align_min_yaw_deg = float(g('parking_blue_align_min_yaw_deg').value)
        self.parking_blue_align_require_both = bool(g('parking_blue_align_require_both').value)
        self.parking_advance_cal = g('parking_advance_cal').value
        self.parking_rotate_cal = g('parking_rotate_cal').value

        self.parking_forward_time = float(g('parking_forward_time').value)
        self.parking_forward_timeout = float(g('parking_forward_timeout').value)
        self.parking_forward_vx = float(g('parking_forward_vx').value)
        self.parking_extra_dist_m = float(g('parking_extra_dist_m').value)
        self.parking_pause_time = float(g('parking_pause_time').value)
        self.parking_blue_lost_area_px = int(g('parking_blue_lost_area_px').value)
        self.parking_blue_lost_frames = int(g('parking_blue_lost_frames').value)
        self.parking_corner_lost_frames = int(g('parking_corner_lost_frames').value)
        self.return_b_strafe_speed = g('return_b_strafe_speed').value
        self.return_b_timeout = g('return_b_timeout').value
        self.qr_stop_roi_x1 = g('qr_stop_roi_x1').value
        self.qr_stop_roi_x2 = g('qr_stop_roi_x2').value
        self.qr_stop_roi_y1 = g('qr_stop_roi_y1').value
        self.qr_stop_roi_y2 = g('qr_stop_roi_y2').value
        
        self.qr_target_shift_timeout = g('qr_target_shift_timeout').value
        self.return_turn_pause_time = g('return_turn_pause_time').value
        
        self.depth_topic = g('depth_topic').value
        self.use_white_wall_rack_stop = g('use_white_wall_rack_stop').value
        self.rack_stop_dist_m = g('rack_stop_dist_m').value
        self.rack_stop_confirm_frames = g('rack_stop_confirm_frames').value

        self.rack_required_close_sec = g('rack_required_close_sec').value
        self.rack_depth_hold_sec = g('rack_depth_hold_sec').value
        self.rack_depth_max_age_sec = g('rack_depth_max_age_sec').value

        self.rack_wall_roi_x1 = g('rack_wall_roi_x1').value
        self.rack_wall_roi_x2 = g('rack_wall_roi_x2').value
        self.rack_wall_roi_y1 = g('rack_wall_roi_y1').value
        self.rack_wall_roi_y2 = g('rack_wall_roi_y2').value

        self.rack_wall_white_s_max = g('rack_wall_white_s_max').value
        self.rack_wall_white_v_min = g('rack_wall_white_v_min').value
        self.rack_wall_min_valid_px = g('rack_wall_min_valid_px').value
        
        self.stop_obj_publish_delay_sec = float(g('stop_obj_publish_delay_sec').value)
        self.stop_qr_publish_delay_sec = float(g('stop_qr_publish_delay_sec').value)
        # 추가
        self.rack_approach_dist_m = g('rack_approach_dist_m').value
        self.rack_approach_min_valid_px = g('rack_approach_min_valid_px').value
        self.rack_approach_vx = g('rack_approach_vx').value

        self.line_search_wz = g('line_search_wz').value
        self.line_search_default_dir = int(g('line_search_default_dir').value)
        self.line_search_timeout = g('line_search_timeout').value

        self.line_search_alternate = bool(g('line_search_alternate').value)
        self.line_search_switch_sec = float(g('line_search_switch_sec').value)
        
    def _on_param_change(self, params):
        # 속성명이 파라미터명과 동일하므로 그대로 반영 -> ros2 param set 즉시 적용
        for p in params:
            setattr(self, p.name, p.value)
        return SetParametersResult(successful=True)

    # ==================== 콜백 ====================
    def _reset_mission_runtime_flags(self):
        """새 주행 사이클을 시작할 때 쓰는 공통 상태 리셋."""
        self.object_red_done = False
        self.parking_red_done = False
        self.rack_depth_armed = False
        self.rack_close_count = 0
        self.rack_close_since = None
        self.rack_last_valid_depth_m = None
        self.rack_last_valid_depth_time = 0.0
        self.rack_last_valid_px = 0

        self.lost_count = 0
        self.line_lost_t0 = None
        self.prev_err = 0.0
        self._qr_hist = []
        self._qr_confirmed = False
        self.jphase = None
        self.corner_phase = None
        self.corner_context = 'normal'
        self.far_lost_latched = False
        self.approach_near_lost = 0
        self.return_b_left_current = False
        self.pending_turn_after_phase = None
        self.pending_turn_reason = ''

        self.parking_forward_phase = None
        self.parking_blue_lost_count = 0
        self.parking_near_blue_seen = False
        self.parking_joint_seen_count = 0
        self.parking_blue_align_seen_count = 0

    def _start_to_objects(self, target, reason='place_target'):
        """
        parking/start 위치에서 objects 방향으로 출발.
        기존 /mission_cmd A_START/B_START/C_START 역할을 /place_target이 대신한다.
        """
        self.target = target
        self.place_target = target
        self.mission_phase = MissionPhase.TO_OBJECTS
        self.state = NavState.FOLLOW
        self.return_b_after_phase = None
        self.return_b_from_target = None
        self._reset_mission_runtime_flags()
        self.publish_cmd(0.0, 0.0, 0.0)

        self.get_logger().info(
            f'/place_target={target} -> 미션 시작: parking/start -> objects 먼저 이동 '
            f'(reason={reason})'
        )

    def _leave_qr_zone(self, after_phase, from_target=None, reason='leave_qr_zone'):
        """
        QR 도착 후 다음 명령(/place_target 또는 /go_parking)을 받았을 때 QR 구역을 빠져나간다.

        - from_target: 현재 실제 위치한 QR 슬롯(A/B/C). 다음 주문 target과 다를 수 있으므로 분리한다.
        - after_phase: QR_B 복귀 및 180도 후 진행할 phase. TO_OBJECTS 또는 TO_PARKING_RED.
        """
        cur = (from_target or self.return_b_from_target or self.target or 'B')
        cur = str(cur).strip().upper()
        if cur not in ('A', 'B', 'C'):
            self.get_logger().warn(f'알 수 없는 현재 QR 위치={cur}, B로 가정')
            cur = 'B'

        self.return_b_from_target = cur
        self.return_b_after_phase = after_phase
        self.mission_phase = MissionPhase.RETURN_TO_QR_B
        self.return_b_left_current = False
        self.lost_count = 0
        self.line_lost_t0 = None
        self.prev_err = 0.0
        self.publish_cmd(0.0, 0.0, 0.0)

        # 현재가 이미 QR_B이면 y축 복귀 없이 바로 180도
        if cur == 'B':
            self.get_logger().info(
                f'{reason}: 현재 QR_B -> y축 복귀 생략, 180도 후 {after_phase.name}'
            )
            self.return_b_from_target = None
            self._request_turn_180_after_settle(
                after_phase,
                reason=f'{reason}_from_B'
            )
            return

        # A/C에서는 먼저 y축으로 QR_B로 복귀한 뒤 180도
        self.state = NavState.RETURN_TO_QR_B
        self.phase_t0 = self._now()
        self.get_logger().info(
            f'{reason}: 현재 QR_{cur} -> QR_B 복귀 시작, 180도 후 {after_phase.name}, '
            f'next_target={self.target}'
        )

    def cb_place_target(self, msg):
        new_target = msg.data.strip().upper()
        self.get_logger().info(f'place_target 수신: {msg.data}')

        if new_target not in ('A', 'B', 'C'):
            self.get_logger().warn(f'알 수 없는 place_target: {msg.data}')
            return

        # QR 목적지에 도착해 WAIT_PLACED 상태라면,
        # 새 /place_target은 "다음 주문 목표"이면서 동시에 QR 구역 탈출 트리거다.
        if self.mission_phase == MissionPhase.WAIT_PLACED:
            current_qr = self.target  # 지금 실제로 서 있는 QR 슬롯
            self.target = new_target  # 다음 주문에서 사용할 QR 목표
            self.place_target = new_target

            self.get_logger().info(
                f'WAIT_PLACED에서 새 place_target={new_target} 수신: '
                f'현재 QR_{current_qr}에서 빠져나와 objects로 복귀'
            )
            self._leave_qr_zone(
                MissionPhase.TO_OBJECTS,
                from_target=current_qr,
                reason=f'next_place_target_{new_target}'
            )
            return

        # 시작/주차/대기 상태에서는 /mission_cmd 없이 /place_target만으로 objects 쪽 출발
        if (
            self.mission_phase in (MissionPhase.WAIT_START, MissionPhase.PARKED)
            or self.state in (NavState.IDLE,)
        ):
            self._start_to_objects(new_target, reason='place_target')
            return

        # 그 외 주행 중 수신은 target만 갱신하고 즉시 state 전환은 하지 않는다.
        # 예: 이미 objects로 가는 중이거나 QR로 가는 중에 중복 발행된 경우.
        old = self.target
        self.target = new_target
        self.place_target = new_target
        self.get_logger().warn(
            f'주행 중 place_target 갱신만 수행: {old} -> {new_target}, '
            f'state={self.state.name}, phase={self.mission_phase.name}'
        )

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
            # 이제 placed 자체로는 움직이지 않는다.
            # 다음 /place_target 또는 /go_parking을 받으면 그때 QR 구역을 빠져나간다.
            if self.mission_phase == MissionPhase.WAIT_PLACED:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(
                    'placed 수신: 이동하지 않고 대기. 다음 /place_target 또는 /go_parking 대기'
                )
            else:
                self.get_logger().warn(
                    f'placed 수신했지만 현재 phase={self.mission_phase.name}, 이동 없음'
                )
            return

        self.get_logger().warn(f'알 수 없는 arm_status: {msg.data}')

    def cb_go_parking(self, msg):
        self.get_logger().info('go_parking 수신')

        # QR 목적지에서 대기 중일 때만 QR_B 복귀 후 parking 루트로 나간다.
        if self.mission_phase != MissionPhase.WAIT_PLACED:
            self.get_logger().warn(
                f'go_parking 수신했지만 현재 phase={self.mission_phase.name}. '
                f'WAIT_PLACED가 아니므로 무시'
            )
            return

        current_qr = self.target
        self.get_logger().info(
            f'go_parking: 현재 QR_{current_qr}에서 빠져나와 parking route 시작'
        )
        self._leave_qr_zone(
            MissionPhase.TO_PARKING_RED,
            from_target=current_qr,
            reason='go_parking'
        )

   
    def do_return_to_qr_b(self, meas):
        t = self._now() - self.phase_t0

        w = meas['w']

        # 기존 fallback: 빨간 주행 라인 중심
        ncx = meas.get('near_center_cx', meas.get('near_cx'))
        line_centered = (
            ncx is not None
            and self.qr_center_lo <= ncx <= self.qr_center_hi
        )

        # 새 방식: QR 전체 리스트에서 B를 직접 선택
        qr_b = self._select_qr(meas, 'B')
        qr_summary = self._qr_log_summary(meas)

        if qr_b is not None:
            b_bbox = int(qr_b.get('bbox', 0) or 0)
            b_cx = qr_b.get('cx')
            b_cy = qr_b.get('cy')
            b_in_roi = bool(qr_b.get('in_stop_roi', False))
        else:
            b_bbox = 0
            b_cx = None
            b_cy = None
            b_in_roi = False

        # B 복귀 완료 조건:
        # QR_B가 충분히 크고 stop ROI 안에 들어오면 B로 판단.
        # QR이 순간 누락될 경우를 대비해 line_centered도 fallback으로 유지.
        b_qr_centered = (
            qr_b is not None
            and b_bbox >= self.qr_stop_bbox
            and b_in_roi
        )

        from_qr = self.return_b_from_target or self.target
        vy = self._vy_from_target_to_b()

        # 현재 위치가 이미 B이면 y축 복귀 없이 바로 180도
        if abs(vy) < 1e-6:
            self.publish_cmd(0.0, 0.0, 0.0)

            after = self.return_b_after_phase or MissionPhase.TO_PARKING_RED
            self.return_b_after_phase = None
            self.return_b_from_target = None

            self.get_logger().info(
                f'RETURN_TO_QR_B: 이미 QR_B로 판단 -> settle 후 180도 회전, after={after.name}'
            )
            self._request_turn_180_after_settle(
                after,
                reason='return_b_already'
            )
            return

        # 1단계: 현재 A/C 라인에서 벗어났는지 확인
        # 단, QR_B가 이미 stop ROI에 충분히 크게 들어왔다면 바로 B 도착 처리 가능.
        if not self.return_b_left_current:
            if not line_centered:
                self.return_b_left_current = True
                self.get_logger().info(
                    f'QR {from_qr} 라인 이탈 -> QR_B 탐색 시작 '
                    f'line_ncx={ncx}, all_qr=[{qr_summary}]'
                )

            if b_qr_centered:
                self.return_b_left_current = True

            self.publish_cmd(0.0, vy, 0.0)
            return

        # 2단계: QR_B 또는 line_centered로 B 복귀 판단
        if b_qr_centered or line_centered:
            self.publish_cmd(0.0, 0.0, 0.0)

            after = self.return_b_after_phase or MissionPhase.TO_PARKING_RED
            self.return_b_after_phase = None
            self.return_b_from_target = None

            reason = 'qr_B_detected' if b_qr_centered else 'line_centered_fallback'

            self.get_logger().info(
                f'QR_B 복귀 완료 reason={reason}, '
                f'B_bbox={b_bbox}, B_roi={b_in_roi}, B_cx={b_cx}, B_cy={b_cy}, '
                f'line_ncx={ncx}, after={after.name}, all_qr=[{qr_summary}]'
            )

            self._request_turn_180_after_settle(
                after,
                reason=f'return_to_qr_b_done_{reason}_after_{after.name}'
            )
            return

        if t > self.return_b_timeout:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.state = NavState.STOP_END
            self.publish_status('return_to_qr_b_timeout')
            self.get_logger().error(
                f'QR_B 복귀 timeout from={from_qr}, next_target={self.target}, '
                f'line_ncx={ncx}, B_bbox={b_bbox}, B_roi={b_in_roi}, '
                f'B_cx={b_cx}, B_cy={b_cy}, all_qr=[{qr_summary}] -> STOP'
            )
            return

        self.publish_cmd(0.0, vy, 0.0)

    def _vy_from_target_to_b(self):
        """
        QR A/C에서 QR_B로 복귀할 때의 y축 방향.
        self.target은 다음 주문 목표로 이미 바뀌었을 수 있으므로,
        실제 현재 위치는 return_b_from_target을 우선 사용한다.
        """
        cur = self.return_b_from_target or self.target
        if cur == 'A':
            return +self.return_b_strafe_speed
        if cur == 'C':
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

        parking_integrating = (
            self.state == NavState.PARK_FORWARD
            and self.parking_forward_phase == 'EXTRA'
        )

        if not (corner_integrating or turn_integrating or parking_integrating):
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

        elif self.state == NavState.PARK_FORWARD and self.parking_forward_phase == 'EXTRA':
            vx = tw.linear.x
            vy = tw.linear.y
            self.odom_int_dist += math.hypot(vx, vy) * dt
            
    def _odom_speed_xy(self):
        if self.odom is None:
            return 999.0
        tw = self.odom.twist.twist
        return math.hypot(tw.linear.x, tw.linear.y)


    def _odom_abs_wz(self):
        if self.odom is None:
            return 999.0
        return abs(self.odom.twist.twist.angular.z)
    
    
    def _line_search_cmd_wz(self):
        """
        line lost 시 회전 탐색 속도 계산.
        alternate=True이면 1초마다 좌/우 방향을 바꾼다.
        """
        if self.line_lost_t0 is None:
            return self.line_search_dir * abs(self.line_search_wz)

        if not self.line_search_alternate:
            return self.line_search_dir * abs(self.line_search_wz)

        elapsed = self._now() - self.line_lost_t0
        switch_sec = max(float(self.line_search_switch_sec), 0.1)

        phase = int(elapsed / switch_sec)

        # phase 0: default dir
        # phase 1: opposite dir
        # phase 2: default dir
        # ...
        if phase % 2 == 0:
            direction = self.line_search_dir
        else:
            direction = -self.line_search_dir

        return direction * abs(self.line_search_wz)

    def cb_mission(self, msg):
        """
        Deprecated.
        기존 /mission_cmd A_START/B_START/C_START는 더 이상 구독하지 않는다.
        수동으로 이 콜백을 다시 연결하는 경우에만 호환용으로 동작한다.
        """
        d = msg.data.strip().upper()
        table = {'A_START': 'A', 'B_START': 'B', 'C_START': 'C'}

        if d not in table:
            self.get_logger().warn(f'알 수 없는 mission_cmd: {msg.data}')
            return

        self.get_logger().warn(
            f'/mission_cmd={d}는 deprecated. 앞으로는 /place_target={table[d]} 사용 권장'
        )
        self._start_to_objects(table[d], reason='deprecated_mission_cmd')

    # ==================== 디버그 / 관찰 유틸 ====================
    def _opt_float(self, v, ndigits=3):
        if v is None:
            return None
        try:
            return round(float(v), ndigits)
        except Exception:
            return None

    def _perception_pub_due(self):
        rate = float(self.debug_perception_rate)
        if rate <= 0.0:
            return False
        now = self._now()
        period = 1.0 / max(rate, 1e-6)
        if now - self._last_perception_pub_t < period:
            return False
        self._last_perception_pub_t = now
        return True

    def request_transition(self, new_state, reason='', force=False):
        if self.freeze_transition and not force:
            self.publish_cmd(0.0, 0.0, 0.0)
            now = self._now()
            if now - self._last_freeze_log_t > 0.5:
                self.get_logger().warn(
                    f'FREEZE: transition blocked {self.state.name} -> {new_state.name} reason={reason}'
                )
                self.publish_status(f'freeze_block:{self.state.name}->{new_state.name}:{reason}')
                self._last_freeze_log_t = now
            return False
        self.state = new_state
        if reason:
            self.get_logger().info(f'TRANSITION: -> {new_state.name} reason={reason}')
        return True

    def request_jphase(self, new_jphase, reason='', force=False):
        if self.freeze_transition and not force:
            self.publish_cmd(0.0, 0.0, 0.0)
            now = self._now()
            if now - self._last_freeze_log_t > 0.5:
                self.get_logger().warn(
                    f'FREEZE: jphase blocked {self.jphase} -> {new_jphase} reason={reason}'
                )
                self.publish_status(f'freeze_block:jphase:{self.jphase}->{new_jphase}:{reason}')
                self._last_freeze_log_t = now
            return False
        self.jphase = new_jphase
        if reason:
            self.get_logger().info(f'JPHASE: -> {new_jphase} reason={reason}')
        return True

    def cb_debug_nav_cmd(self, msg):
        d = msg.data.strip().upper()

        if d in ('STOP', 'IDLE'):
            self.state = NavState.IDLE
            self.publish_cmd(0.0, 0.0, 0.0)
            self.publish_status('debug_idle')
            self.get_logger().warn('DEBUG: state -> IDLE')
            return
        if d == 'INSPECT_ON':
            self.inspect_only = True
            self.publish_cmd(0.0, 0.0, 0.0)
            self.get_logger().warn('DEBUG: inspect_only=True')
            return
        if d == 'INSPECT_OFF':
            self.inspect_only = False
            self.get_logger().warn('DEBUG: inspect_only=False')
            return
        if d == 'FREEZE_ON':
            self.freeze_transition = True
            self.publish_cmd(0.0, 0.0, 0.0)
            self.get_logger().warn('DEBUG: freeze_transition=True')
            return
        if d == 'FREEZE_OFF':
            self.freeze_transition = False
            self.get_logger().warn('DEBUG: freeze_transition=False')
            return
        if d == 'DRIVE_ON':
            self.enable_drive = True
            self.get_logger().warn('DEBUG: enable_drive=True')
            return
        if d == 'DRIVE_OFF':
            self.enable_drive = False
            self.publish_cmd(0.0, 0.0, 0.0)
            self.get_logger().warn('DEBUG: enable_drive=False')
            return

        if d in ('TARGET_A', 'TARGET A', 'TARGET_A_START'):
            self.target = 'A'
            self.get_logger().warn('DEBUG: target=A')
            return
        if d in ('TARGET_B', 'TARGET B', 'TARGET_B_START'):
            self.target = 'B'
            self.get_logger().warn('DEBUG: target=B')
            return
        if d in ('TARGET_C', 'TARGET C', 'TARGET_C_START'):
            self.target = 'C'
            self.get_logger().warn('DEBUG: target=C')
            return

        if d == 'FOLLOW_OBJECTS':
            self.state = NavState.FOLLOW
            self.mission_phase = MissionPhase.TO_OBJECTS
            self.object_red_done = False
            self.rack_depth_armed = False
            self.rack_close_count = 0
            self.rack_close_since = None
            self.rack_last_valid_depth_m = None
            self.rack_last_valid_depth_time = 0.0
            self.rack_last_valid_px = 0
            self.lost_count = 0
            self.prev_err = 0.0
            self.publish_status('debug_follow_objects')
            self.get_logger().warn('DEBUG: FOLLOW + TO_OBJECTS')
            return
        if d == 'FOLLOW_QR':
            self.state = NavState.FOLLOW
            self.mission_phase = MissionPhase.TO_QR
            if self.target is None:
                self.target = 'B'
            self._qr_hist = []
            self._qr_confirmed = False
            self.publish_status(f'debug_follow_qr_{self.target}')
            self.get_logger().warn(f'DEBUG: FOLLOW + TO_QR target={self.target}')
            return
        if d == 'FOLLOW_PARKING':
            self.state = NavState.FOLLOW
            self.mission_phase = MissionPhase.TO_PARKING_RED
            self.parking_red_done = False
            self.lost_count = 0
            self.publish_status('debug_follow_parking')
            self.get_logger().warn('DEBUG: FOLLOW + TO_PARKING_RED')
            return
        if d == 'JUNCTION_ALIGN':
            self.state = NavState.JUNCTION
            self.jphase = 'ALIGN'
            if self.target is None:
                self.target = 'B'
            self.phase_t0 = self._now()
            self.align_left_center = False
            self.publish_status(f'debug_junction_align_{self.target}')
            self.get_logger().warn(f'DEBUG: JUNCTION ALIGN target={self.target}')
            return
        if d == 'JUNCTION_APPROACH':
            self.state = NavState.JUNCTION
            self.jphase = 'APPROACH'
            if self.target is None:
                self.target = 'B'
            self.phase_t0 = self._now()
            self._qr_hist = []
            self._qr_confirmed = False
            self.publish_status(f'debug_junction_approach_{self.target}')
            self.get_logger().warn(f'DEBUG: JUNCTION APPROACH target={self.target}')
            return
        if d == 'CORNER_OBJECT_RED':
            self._enter_red_corner('object_red', force=True)
            self.publish_status('debug_corner_object_red')
            return
        if d == 'CORNER_PARKING_RED':
            self._enter_red_corner('parking_red', force=True)
            self.publish_status('debug_corner_parking_red')
            return
        if d == 'TURN_180':
            self._enter_turn_180(self.mission_phase, reason='debug', force=True)
            self.publish_status('debug_turn_180')
            return
        if d == 'SNAPSHOT':
            self.save_debug_snapshot()
            return

        self.get_logger().warn(f'Unknown debug cmd: {msg.data}')

    def save_debug_snapshot(self):
        # control node에는 image/mask가 없으므로 JSON meas만 저장한다.
        with self.meas_lock:
            meas = None if self.latest_meas is None else dict(self.latest_meas)
        if meas is None:
            self.get_logger().warn('SNAPSHOT 실패: perception meas 없음')
            return

        os.makedirs(self.debug_snapshot_dir, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        base = os.path.join(self.debug_snapshot_dir, f'line_tracer_control_{stamp}')
        meta = {
            'state': self.state.name,
            'mission_phase': self.mission_phase.name,
            'target': self.target,
            'corner_phase': self.corner_phase,
            'corner_context': self.corner_context,
            'jphase': self.jphase,
            'meas': meas,
            'last_cmd': self.last_cmd,
        }
        with open(base + '_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self.publish_status('debug_snapshot_saved')
        self.get_logger().warn(f'CONTROL SNAPSHOT 저장: {base}_meta.json')

    # ==================== 메인 루프 ====================
    def loop(self):
        with self.meas_lock:
            meas = None if self.latest_meas is None else dict(self.latest_meas)
            perception_age = time.time() - self.last_perception_time

        if meas is None or perception_age > self.image_timeout_sec:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.get_logger().warn(
                f'perception timeout: age={perception_age:.3f}s',
                throttle_duration_sec=1.0
            )
            return

        self.latest_meas = meas

        # inspect_only=True면 주행/상태전이 없이 JSON 인식값만 상태로 재출력한다.
        if self.inspect_only:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.publish_perception_status_from_meas(meas)
            self._log_telemetry(meas)
            return

        if self.state == NavState.FOLLOW:
            self.do_follow(meas)
        elif self.state == NavState.CORNER:
            self.do_corner(meas)
        elif self.state == NavState.JUNCTION:
            self.do_junction(meas)
        elif self.state == NavState.RETURN_TO_QR_B:
            self.do_return_to_qr_b(meas)
        elif self.state == NavState.PRE_TURN_SETTLE:
            self.do_pre_turn_settle()
        elif self.state == NavState.TURN_180:
            self.do_turn_180(meas)
        elif self.state == NavState.TURN_PAUSE:
            self.do_turn_pause()
        elif self.state == NavState.PARK_FORWARD:
            self.do_parking_forward(meas)
        elif self.state == NavState.PARK_PAUSE:
            self.do_parking_pause()
        elif self.state in (NavState.STOP_END, NavState.IDLE):
            self.publish_cmd(0.0, 0.0, 0.0)

        self._log_telemetry(meas)
        
    def _qr_in_stop_roi(self, qcx, qcy, w, h):
        if qcx is None or qcy is None:
            return False

        x1 = self.qr_stop_roi_x1 * w
        x2 = self.qr_stop_roi_x2 * w
        y1 = self.qr_stop_roi_y1 * h
        y2 = self.qr_stop_roi_y2 * h

        return x1 <= qcx <= x2 and y1 <= qcy <= y2


    def _normalize_qr_detections(self, data, w, h):
        """
        Orin JSON의 qr_detections / qr.detections를 tracer 내부에서 쓰기 쉬운 list로 정규화한다.
        각 항목은 text, bbox, cx, cy, in_stop_roi를 가진다.
        """
        raw = None
        if isinstance(data, dict):
            raw = data.get('qr_detections')
            if raw is None and isinstance(data.get('qr'), dict):
                raw = data.get('qr', {}).get('detections')

        out = []
        if isinstance(raw, list):
            for d in raw:
                if not isinstance(d, dict):
                    continue
                text = str(d.get('text', '')).strip().upper()
                if not text:
                    continue
                try:
                    bbox = int(d.get('bbox', 0) or 0)
                except Exception:
                    bbox = 0

                cx = d.get('cx')
                cy = d.get('cy')
                try:
                    cx = None if cx is None else float(cx)
                    cy = None if cy is None else float(cy)
                except Exception:
                    cx, cy = None, None

                in_roi = d.get('in_stop_roi')
                if in_roi is None:
                    in_roi = self._qr_in_stop_roi(cx, cy, w, h)

                out.append({
                    'text': text,
                    'bbox': bbox,
                    'cx': cx,
                    'cy': cy,
                    'in_stop_roi': bool(in_roi),
                    'source': d.get('source'),
                    'area': float(d.get('area', bbox * bbox) or 0.0),
                    'x1': d.get('x1'),
                    'y1': d.get('y1'),
                    'x2': d.get('x2'),
                    'y2': d.get('y2'),
                })

        # 구버전 Orin이 qr_detections를 안 보내는 경우 대표 QR을 fallback으로 넣는다.
        if not out:
            qtext = None
            if isinstance(data.get('qr'), dict):
                qtext = data.get('qr', {}).get('text')
            if qtext is None:
                qtext = data.get('qr_text')

            qtext = None if qtext is None else str(qtext).strip().upper()
            if qtext:
                try:
                    bbox = int((data.get('qr_bbox') if data.get('qr_bbox') is not None else data.get('qr', {}).get('bbox', 0)) or 0)
                except Exception:
                    bbox = 0
                cx = data.get('qr_cx')
                cy = data.get('qr_cy')
                if cx is None and isinstance(data.get('qr'), dict):
                    cx = data.get('qr', {}).get('cx')
                if cy is None and isinstance(data.get('qr'), dict):
                    cy = data.get('qr', {}).get('cy')
                try:
                    cx = None if cx is None else float(cx)
                    cy = None if cy is None else float(cy)
                except Exception:
                    cx, cy = None, None

                in_roi = data.get('qr_in_stop_roi')
                if in_roi is None and isinstance(data.get('qr'), dict):
                    in_roi = data.get('qr', {}).get('in_stop_roi')
                if in_roi is None:
                    in_roi = self._qr_in_stop_roi(cx, cy, w, h)

                out.append({
                    'text': qtext,
                    'bbox': bbox,
                    'cx': cx,
                    'cy': cy,
                    'in_stop_roi': bool(in_roi),
                    'source': 'legacy_single',
                    'area': float(bbox * bbox),
                })

        return out

    def _select_qr(self, meas, text=None):
        """
        여러 QR이 동시에 보일 때 원하는 text(A/B/C)를 직접 선택한다.
        같은 text가 여러 개면 bbox/area가 큰 것을 선택한다.
        """
        detections = meas.get('qr_detections') or []
        wanted = None if text is None else str(text).strip().upper()

        cands = []
        for d in detections:
            if not isinstance(d, dict):
                continue
            dtext = str(d.get('text', '')).strip().upper()
            if wanted is not None and dtext != wanted:
                continue
            cands.append(d)

        if not cands:
            return None

        return max(
            cands,
            key=lambda d: (
                int(d.get('bbox', 0) or 0),
                float(d.get('area', 0.0) or 0.0),
            )
        )

    def _qr_log_summary(self, meas):
        detections = meas.get('qr_detections') or []
        if not detections:
            return 'none'
        parts = []
        for d in detections:
            parts.append(
                f"{d.get('text')}:{int(d.get('bbox', 0) or 0)}"
                f"@({self._opt_float(d.get('cx'), 1)},{self._opt_float(d.get('cy'), 1)})"
            )
        return ','.join(parts)

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
                f'{meas["far_hspan"]},{meas["far_clusters"]},{meas.get("far_blue_area", meas.get("blue_far_area", 0))},'
                f'{vx:.3f},{vy:.3f},{wz:.3f}\n')
        except Exception:
            pass

    def cb_perception(self, msg):
        """Orin perception node가 publish한 JSON을 받아 기존 meas 형태로 flatten한다."""
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'perception JSON parse 실패: {e}', throttle_duration_sec=1.0)
            return

        if not data.get('valid', True):
            self.get_logger().warn(
                f'perception invalid: {data.get("reason", "unknown")}',
                throttle_duration_sec=1.0
            )
            return

        meas = self._normalize_perception(data)

        with self.meas_lock:
            self.latest_meas = meas
            self.last_perception_time = time.time()

        self.last_qr_info = (
            meas.get('qr_text'),
            int(meas.get('qr_bbox', 0) or 0),
            meas.get('qr_cx'),
            meas.get('qr_cy'),
        )
        self.last_rack_wall_depth_m = meas.get('rack_depth_m')
        self.last_rack_wall_valid_px = int(meas.get('rack_valid_px', 0) or 0)

    def _normalize_perception(self, data):
        """
        Orin perception JSON을 기존 state machine이 쓰는 flat meas로 맞춘다.

        새 JSON 이름:
          red_line = 실제 빨간 주행 라인
          blue     = 실제 파란 분기/주차 marker
          white_wall = 실제 흰 벽/rack wall

        내부 state machine에는 아직 legacy 변수명도 남아 있어서,
        near_red_area/far_red_area는 blue marker의 alias로 같이 채워둔다.
        """
        def pick(*keys, default=None):
            for k in keys:
                cur = data
                ok = True
                for part in k.split('.'):
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        ok = False
                        break
                if ok:
                    return cur
            return default

        w = int(pick('w', default=self.frame_width) or self.frame_width)
        h = int(pick('h', default=self.frame_height) or self.frame_height)
        nt, nb = data.get('near_rows', [int(h * self.near_top), int(h * self.near_bot)])
        ft, fb = data.get('far_rows', [int(h * self.far_top), int(h * self.far_bot)])

        # 실제 빨간 주행 라인
        red_line_near_cx = pick('red_line_near_cx', 'red_line.near_cx', 'near_cx', 'white.near_cx')
        red_line_near_center_cx = pick(
            'red_line_near_center_cx', 'red_line.near_center_cx',
            'near_center_cx', 'white.near_center_cx',
            'red_line_near_cx', 'red_line.near_cx', 'near_cx', 'white.near_cx'
        )
        red_line_far_cx = pick('red_line_far_cx', 'red_line.far_cx', 'far_cx', 'white.far_cx')
        red_line_near_clusters = int(pick('red_line_near_clusters', 'red_line.near_clusters', 'near_clusters', 'white.near_clusters', default=0) or 0)
        red_line_far_clusters = int(pick('red_line_far_clusters', 'red_line.far_clusters', 'far_clusters', 'white.far_clusters', default=0) or 0)
        red_line_far_hspan = int(pick('red_line_far_hspan', 'red_line.far_hspan', 'far_hspan', 'white.far_hspan', default=0) or 0)

        # 실제 파란 분기/주차 marker
        blue_near_cx = pick('blue_near_cx', 'blue.near_cx', 'blue_marker.near_cx', 'near_blue_cx', 'near_red_cx', 'red.near_cx')
        blue_near_area = int(pick('blue_near_area', 'blue.near_area', 'blue_marker.near_area', 'near_blue_area', 'near_red_area', 'red.near_area', default=0) or 0)
        blue_near_clusters = int(pick('blue_near_clusters', 'blue.near_clusters', 'blue_marker.near_clusters', 'near_blue_clusters', 'near_red_clusters', 'red.near_clusters', default=0) or 0)
        blue_far_cx = pick('blue_far_cx', 'blue.far_cx', 'blue_marker.far_cx', 'far_blue_cx', 'far_red_cx', 'red.far_cx')
        blue_far_area = int(pick('blue_far_area', 'blue.far_area', 'blue_marker.far_area', 'far_blue_area', 'far_red_area', 'red.far_area', default=0) or 0)
        blue_far_clusters = int(pick('blue_far_clusters', 'blue.far_clusters', 'blue_marker.far_clusters', 'far_blue_clusters', 'far_red_clusters', 'red.far_clusters', default=0) or 0)

        # 빨간 주행 라인 + 파란 주차선 접점
        parking_joint_detected = bool(pick('parking_joint_detected', 'parking_joint.detected', default=False))
        parking_joint_touch_px = int(pick('parking_joint_touch_px', 'parking_joint.touch_px', default=0) or 0)
        parking_joint_cx = pick('parking_joint_cx', 'parking_joint.cx')
        parking_joint_cy = pick('parking_joint_cy', 'parking_joint.cy')

        # 주차 진입 회전 종료용: 파란 주차구역 테두리 수평 검출
        parking_blue_align_detected = bool(pick('parking_blue_align_detected', 'parking_blue_alignment.detected', default=False))
        parking_blue_align_upper_detected = bool(pick('parking_blue_align_upper_detected', 'parking_blue_alignment.upper.detected', default=False))
        parking_blue_align_lower_detected = bool(pick('parking_blue_align_lower_detected', 'parking_blue_alignment.lower.detected', default=False))
        parking_blue_align_upper_angle_deg = pick('parking_blue_align_upper_angle_deg', 'parking_blue_alignment.upper.angle_deg')
        parking_blue_align_lower_angle_deg = pick('parking_blue_align_lower_angle_deg', 'parking_blue_alignment.lower.angle_deg')
        parking_blue_align_upper_len_px = float(pick('parking_blue_align_upper_len_px', 'parking_blue_alignment.upper.length_px', default=0.0) or 0.0)
        parking_blue_align_lower_len_px = float(pick('parking_blue_align_lower_len_px', 'parking_blue_alignment.lower.length_px', default=0.0) or 0.0)

        # QR 전체 리스트. Orin이 qr_detections를 보내면 여기서 A/B/C를 모두 보존한다.
        qr_detections = self._normalize_qr_detections(data, w, h)

        # 실제 흰 벽/rack wall
        rack_depth_m = pick('white_wall_depth_m', 'white_wall.median_m', 'rack_depth_m', 'rack_depth.median_m')
        rack_valid_px = int(pick('white_wall_valid_px', 'white_wall.valid_px', 'rack_valid_px', 'rack_depth.valid_px', default=0) or 0)
        rack_close = bool(pick('white_wall_close', 'white_wall.close', 'rack_close', 'rack_depth.close', default=False))

        meas = {
            'w': w,
            'h': h,
            'near_rows': (int(nt), int(nb)),
            'far_rows': (int(ft), int(fb)),

            # 빨간 주행 라인. 기존 FOLLOW/CORNER 로직은 near_cx/far_cx를 그대로 사용한다.
            'near_cx': red_line_near_cx,
            'near_center_cx': red_line_near_center_cx,
            'near_clusters': red_line_near_clusters,
            'far_cx': red_line_far_cx,
            'far_clusters': red_line_far_clusters,
            'far_hspan': red_line_far_hspan,

            # 파란 marker. 새 이름.
            'near_blue_cx': blue_near_cx,
            'near_blue_area': blue_near_area,
            'near_blue_clusters': blue_near_clusters,
            'far_blue_cx': blue_far_cx,
            'far_blue_area': blue_far_area,
            'far_blue_clusters': blue_far_clusters,

            # 주차 진입용: 빨간 주행 라인과 파란 주차선의 접점
            'parking_joint_detected': parking_joint_detected,
            'parking_joint_touch_px': parking_joint_touch_px,
            'parking_joint_cx': parking_joint_cx,
            'parking_joint_cy': parking_joint_cy,

            # 주차 진입 회전 종료용: 파란 주차구역 테두리 수평 검출
            'parking_blue_align_detected': parking_blue_align_detected,
            'parking_blue_align_upper_detected': parking_blue_align_upper_detected,
            'parking_blue_align_lower_detected': parking_blue_align_lower_detected,
            'parking_blue_align_upper_angle_deg': parking_blue_align_upper_angle_deg,
            'parking_blue_align_lower_angle_deg': parking_blue_align_lower_angle_deg,
            'parking_blue_align_upper_len_px': parking_blue_align_upper_len_px,
            'parking_blue_align_lower_len_px': parking_blue_align_lower_len_px,

            # legacy aliases. 내부 함수명/context명은 안전을 위해 그대로 둔다.
            'near_red_cx': blue_near_cx,
            'near_red_area': blue_near_area,
            'near_red_clusters': blue_near_clusters,
            'far_red_cx': blue_far_cx,
            'far_red_area': blue_far_area,
            'far_red_clusters': blue_far_clusters,

            # legacy 대표 QR 1개. 실제 제어는 qr_detections에서 text별로 선택한다.
            'qr_text': pick('qr_text', 'qr.text'),
            'qr_bbox': int(pick('qr_bbox', 'qr.bbox', default=0) or 0),
            'qr_cx': pick('qr_cx', 'qr.cx'),
            'qr_cy': pick('qr_cy', 'qr.cy'),
            'qr_in_stop_roi': bool(pick('qr_in_stop_roi', 'qr.in_stop_roi', default=False)),
            'qr_detections': qr_detections,

            'rack_depth_m': rack_depth_m,
            'rack_valid_px': rack_valid_px,
            'rack_close': rack_close,
        }
        return meas

    def rack_close_by_white_wall_depth_meas(self, meas):
        """
        Orin에서 계산된 rack_depth_m / rack_valid_px 기반으로 기존 시간 confirm 로직만 수행.
        image/depth raw는 여기서 절대 받지 않는다.
        """
        if not self.use_white_wall_rack_stop:
            self.rack_close_since = None
            self.rack_close_count = 0
            return False

        now = self._now()
        d = meas.get('rack_depth_m')
        valid_px = int(meas.get('rack_valid_px', 0) or 0)
        self.last_rack_wall_depth_m = d
        self.last_rack_wall_valid_px = valid_px
        self.rack_last_valid_px = valid_px

        if d is None or valid_px < self.rack_wall_min_valid_px:
            if (
                self.rack_close_since is not None
                and (now - self.rack_last_valid_depth_time) <= self.rack_depth_hold_sec
            ):
                self.get_logger().warn(
                    f'JSON rack depth valid 부족/None: d={d}, px={valid_px}, close_since 유지',
                    throttle_duration_sec=1.0
                )
                return False

            self.rack_close_since = None
            self.rack_close_count = max(0, self.rack_close_count - 1)
            return False

        d = float(d)
        self.rack_last_valid_depth_m = d
        self.rack_last_valid_depth_time = now

        if d <= self.rack_stop_dist_m:
            if self.rack_close_since is None:
                self.rack_close_since = now
            self.rack_close_count += 1
            close_dt = now - self.rack_close_since

            self.get_logger().info(
                f'JSON white_wall_depth={d:.3f}m px={valid_px} '
                f'close_time={close_dt:.2f}/{self.rack_required_close_sec:.2f}s '
                f'count={self.rack_close_count}',
                throttle_duration_sec=0.5
            )
            return close_dt >= self.rack_required_close_sec

        self.rack_close_since = None
        self.rack_close_count = max(0, self.rack_close_count - 1)
        return False

    def publish_perception_status_from_meas(self, meas):
        data = {
            't': self._opt_float(self._now(), 3),
            'state': self.state.name,
            'mission_phase': self.mission_phase.name,
            'target': self.target,
            'corner_phase': self.corner_phase,
            'corner_context': self.corner_context,
            'jphase': self.jphase,
            'enable_drive': bool(self.enable_drive),
            'inspect_only': bool(self.inspect_only),
            'freeze_transition': bool(self.freeze_transition),
            'red_line': {
                'near_cx': self._opt_float(meas.get('near_cx'), 1),
                'near_center_cx': self._opt_float(meas.get('near_center_cx'), 1),
                'far_cx': self._opt_float(meas.get('far_cx'), 1),
                'near_clusters': int(meas.get('near_clusters', 0)),
                'far_clusters': int(meas.get('far_clusters', 0)),
                'far_hspan': int(meas.get('far_hspan', 0)),
            },
            'blue': {
                'near_cx': self._opt_float(meas.get('near_blue_cx'), 1),
                'near_area': int(meas.get('near_blue_area', 0)),
                'near_clusters': int(meas.get('near_blue_clusters', 0)),
                'far_cx': self._opt_float(meas.get('far_blue_cx'), 1),
                'far_area': int(meas.get('far_blue_area', 0)),
                'far_clusters': int(meas.get('far_blue_clusters', 0)),
            },
            'parking_joint': {
                'detected': bool(meas.get('parking_joint_detected', False)),
                'touch_px': int(meas.get('parking_joint_touch_px', 0) or 0),
                'cx': self._opt_float(meas.get('parking_joint_cx'), 1),
                'cy': self._opt_float(meas.get('parking_joint_cy'), 1),
                'confirm': int(getattr(self, 'parking_joint_seen_count', 0)),
            },
            'qr': {
                'text': meas.get('qr_text'),
                'bbox': int(meas.get('qr_bbox', 0) or 0),
                'cx': self._opt_float(meas.get('qr_cx'), 1),
                'cy': self._opt_float(meas.get('qr_cy'), 1),
                'in_stop_roi': bool(meas.get('qr_in_stop_roi', False)),
                'detections': meas.get('qr_detections', []),
            },
            'white_wall': {
                'median_m': self._opt_float(meas.get('rack_depth_m'), 3),
                'valid_px': int(meas.get('rack_valid_px', 0) or 0),
                'close': bool(meas.get('rack_close', False)),
            },
            # legacy aliases: 필요하면 기존 이름으로도 확인 가능
            'legacy': {
                'white_is_red_line': True,
                'red_is_blue_marker': True,
                'white': {
                    'near_cx': self._opt_float(meas.get('near_cx'), 1),
                    'far_cx': self._opt_float(meas.get('far_cx'), 1),
                    'far_hspan': int(meas.get('far_hspan', 0)),
                },
                'red': {
                    'near_area': int(meas.get('near_blue_area', 0)),
                    'far_area': int(meas.get('far_blue_area', 0)),
                },
            },
            'cmd': {
                'vx': self._opt_float(self.last_cmd[0], 3),
                'vy': self._opt_float(self.last_cmd[1], 3),
                'wz': self._opt_float(self.last_cmd[2], 3),
            },
        }
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.perception_pub.publish(msg)

    # ==================== 인지 ====================

    # ==================== FOLLOW ====================
    def _follow_cmd_from_error(self, steer_err):
        """
        myAGV 2023 angular deadzone 대응 라인 추종 명령 생성.

        실험 결과상 0 < |angular.z| < 약 0.40은 실제 최소 회전속도처럼 동작한다.
        따라서 작은/중간 오차는 angular.z를 쓰지 않고 linear.y로 보정한다.
        큰 오차에서만 vel_filter_node의 pulse 변환을 전제로 작은 effective wz를 요청한다.
        """
        e = float(steer_err)
        ae = abs(e)

        if not self.use_lateral_follow:
            wz = self._clip_wz(-(self.steer_kp * e))
            turn_ratio = abs(wz) / max(self.max_wz, 1e-6)
            vx_scale = 1.0 - 0.6 * turn_ratio
            vx_scale = float(np.clip(vx_scale, 0.4, 1.0))
            return self.cruise_vx * vx_scale, 0.0, wz

        if ae < self.follow_err_deadband:
            return float(self.cruise_vx), 0.0, 0.0

        if ae < self.follow_big_err:
            # line이 오른쪽(+)이면 robot은 오른쪽으로 strafe해야 하므로 vy는 음수.
            vy = float(np.clip(-self.follow_vy_kp * e,
                               -self.follow_vy_max,
                               self.follow_vy_max))
            # 오차가 커질수록 약간 감속. 단, 회전은 하지 않는다.
            ratio = min(ae / max(self.follow_big_err, 1e-6), 1.0)
            vx = float(self.cruise_vx * (1.0 - 0.35 * ratio))
            vx = max(0.04, vx)
            return vx, vy, 0.0

        # 큰 오차: 전진을 거의 죽이고 짧은 yaw pulse를 요청한다.
        # 실제 pulse 변환은 vel_filter_node가 담당한다.
        wz = -math.copysign(abs(self.follow_turn_wz), e)
        return float(self.follow_turn_vx), 0.0, self._clip_wz(wz)


    def do_follow(self, meas):
        w = meas['w']
        near_cx = meas['near_cx']
        far_cx = meas['far_cx']
        
        # TO_OBJECTS라도 object_red CORNER를 정상 완료한 뒤에만 rack depth 사용
        if (
            self.mission_phase == MissionPhase.TO_OBJECTS
            and self.rack_depth_armed
        ):
            if self.rack_close_by_white_wall_depth_meas(meas):
                self._handle_stop_end('rack_depth')
                return
        
        # 분기 진입 트리거: QR이 보이면(=②→③ 회전 끝나고 분기 정면) JUNCTION 진입
        self._frame_i += 1
        if (
            self.mission_phase == MissionPhase.TO_QR
            and self.target is not None
            and self._frame_i % self.qr_check_interval == 0
        ):
            qtext = meas.get('qr_text')
            if qtext is not None:
                if not self.request_transition(
                    NavState.JUNCTION,
                    reason=f'QR 감지({qtext}) -> JUNCTION ALIGN'
                ):
                    return
                self.jphase = 'ALIGN'
                self.phase_t0 = self._now()
                self.align_left_center = False
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(f'QR 감지({qtext}) -> JUNCTION ALIGN (목표 {self.target})')
                return

        # mission corner는 white near_cx 로스트보다 먼저 처리
        # - objects 쪽은 기존처럼 파란 marker가 보이면 진입
        # - parking 쪽은 파란색 단독 검출이 아니라 빨간 라인+파란 라인의 접점이 잡힐 때만 진입
        if self.mission_phase == MissionPhase.TO_OBJECTS and not self.object_red_done:
            if self._junction_ahead(meas):
                self.get_logger().info('TO_OBJECTS: blue marker 감지 -> objects 방향 CORNER 진입')
                self._enter_red_corner('object_red')
                return

        if self.mission_phase == MissionPhase.TO_PARKING_RED and not self.parking_red_done:
            if self._parking_joint_ahead(meas):
                parking_turn_dir = self._parking_entry_turn_dir(meas)
                self.get_logger().info(
                    f'TO_PARKING_RED: red+blue 접점 감지 -> parking CORNER 진입 '
                    f'touch={meas.get("parking_joint_touch_px", 0)}, '
                    f'joint_cx={meas.get("parking_joint_cx")}, '
                    f'near_blue_cx={meas.get("near_blue_cx", meas.get("blue_near_cx"))}, '
                    f'dir={parking_turn_dir}({"L" if parking_turn_dir > 0 else "R"})'
                )
                self._enter_red_corner('parking_red', turn_dir_override=parking_turn_dir)
                return

            if self._junction_ahead(meas):
                self.get_logger().info(
                    f'TO_PARKING_RED: 파란 주차선은 보이지만 red+blue 접점은 아직 아님 '
                    f'blue_far/near={meas.get("far_blue_area", 0)}/{meas.get("near_blue_area", 0)}, '
                    f'touch={meas.get("parking_joint_touch_px", 0)} '
                    f'confirm={self.parking_joint_seen_count}/{self.parking_joint_confirm_frames}',
                    throttle_duration_sec=0.7
                )

        # 라인 로스트
        if near_cx is None:
            now = self._now()

            # =========================================================
            # rack 접근 구간에서는 line lost search 금지
            # 라인이 끊기는 게 정상인 구간:
            #   rack_valid_px 충분히 큼
            #   rack_depth_m <= rack_approach_dist_m
            #   TO_OBJECTS + rack_depth_armed 상태
            # =========================================================
            if self._rack_approach_ahead(meas):
                self.lost_count = 0
                self.line_lost_t0 = None

                # 라인을 찾겠다고 회전하지 말고 rack 쪽으로 천천히 직진
                self.publish_cmd(float(self.rack_approach_vx), 0.0, 0.0)

                self.get_logger().info(
                    f'rack approach: line lost ignored, '
                    f'd={meas.get("rack_depth_m")}, '
                    f'valid_px={meas.get("rack_valid_px")}, '
                    f'vx={self.rack_approach_vx:.3f}',
                    throttle_duration_sec=0.5
                )
                return

            self.lost_count += 1
            in_grace = now < self.post_corner_grace_until

            # 일반 line lost일 때만 search 수행
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

                search_wz = self._line_search_cmd_wz()

                self.publish_cmd(0.0, 0.0, search_wz)
                self.get_logger().warn(
                    f'line lost -> alternating search wz={search_wz:.2f}',
                    throttle_duration_sec=0.5
                )
                return

            self.publish_cmd(0.0, 0.0, 0.0)
            return

        # line reacquired
        self.lost_count = 0
        self.line_lost_t0 = None
        self.last_near_cx = near_cx

        err = (near_cx - w / 2.0) / (w / 2.0)

        near_err = (near_cx - w/2) / (w/2) if near_cx else 0.0
        far_err  = (far_cx  - w/2) / (w/2) if far_cx else near_err

        # far 중심을 주로 쓰되, far가 없으면 near 기준으로 보정한다.
        steer_err = 0.8 * far_err + 0.2 * near_err

        if abs(steer_err) < self.follow_err_deadband:
            steer_err = 0.0

        self.prev_err = steer_err

        if abs(steer_err) > 0.05:
            # 기존 조향 방향과 같은 방향으로 line search
            self.line_search_dir = 1 if (-steer_err) >= 0 else -1

        vx_cmd, vy_cmd, wz_cmd = self._follow_cmd_from_error(steer_err)

        # 빨강 분기 감지 -> 감속 + 로그 (회전/주차는 Layer 3)
        if self._junction_ahead(meas):
            self.get_logger().info(
                f'분기/주차 파란 marker 감지 blue_far_area={meas.get("far_blue_area", meas.get("blue_far_area", 0))}',
                throttle_duration_sec=1.0
            )
            # 분기 접근 중에는 yaw를 쓰지 않고 lateral 보정만 유지한다.
            self.publish_cmd(min(self.slow_vx, vx_cmd), vy_cmd, 0.0)
            return

        # L코너 감지 -> CORNER 진입
        cdir = self._corner_ahead(meas)
        if cdir != 0:
            if not self.request_transition(NavState.CORNER, reason='L corner detected'):
                return

            self.corner_phase = 'APPROACH'
            self.corner_dir = cdir
            self.corner_context = 'normal'
            self.active_advance_dist = self.corner_advance_dist
            self.active_rotate_deg = self.rotate_deg
            self.active_rotate_cal = 1.0
            self.phase_t0 = self._now()
            self.far_lost_latched = False
            self.approach_near_lost = 0

            # 중요: 1차 감지 순간부터 yaw 제거
            # 직전 FOLLOW에서 남아 있던 wz를 끊기 위해 바로 직진 명령 발행
            self.publish_cmd(self.slow_vx, 0.0, 0.0)

            self.get_logger().info(
                f'L코너 1차 감지 dir={"L" if cdir>0 else "R"} '
                f'span={meas["far_hspan"]} -> APPROACH_STRAIGHT'
            )
            return

        self.publish_cmd(vx_cmd, vy_cmd, wz_cmd)

    # ==================== CORNER ====================

    def _cx_in_rotate_center_window(self, cx):
        """회전 조기정지용 cx 윈도우 판정. None이면 False."""
        if cx is None:
            return False
        try:
            x = float(cx)
        except Exception:
            return False
        return self.rotate_center_lo <= x <= self.rotate_center_hi

    def _rotate_line_centered(self, meas, color='red_line'):
        """
        회전 중 라인 중앙 판정.
        color='red_line' : 실제 빨간 주행 라인 near_cx/far_cx 사용
        color='blue'     : 실제 파란 주차/분기선 near_blue_cx/far_blue_cx 사용
        color='blue_near': 실제 파란 주차/분기선 near_blue_cx만 사용
                           parking 진입 회전용. far_blue는 화면 오른쪽에 붙어
                           오검출될 수 있어서 조기정지에 쓰지 않는다.
        """
        if color == 'blue_near':
            near_cx = meas.get('near_blue_cx', meas.get('blue_near_cx'))
            far_cx = None
            centered = self._cx_in_rotate_center_window(near_cx)
            return centered, near_cx, far_cx

        if color == 'blue':
            near_cx = meas.get('near_blue_cx', meas.get('blue_near_cx'))
            far_cx = meas.get('far_blue_cx', meas.get('blue_far_cx'))
        else:
            near_cx = meas.get('near_cx')
            far_cx = meas.get('far_cx')

        centered = (
            self._cx_in_rotate_center_window(near_cx)
            or self._cx_in_rotate_center_window(far_cx)
        )
        return centered, near_cx, far_cx


    def _rotation_center_stop_ready(self, meas, latch_attr, color='red_line'):
        centered, near_cx, far_cx = self._rotate_line_centered(meas, color=color)

        if not centered:
            setattr(self, latch_attr, True)
            return False, near_cx, far_cx

        if (not self.rotate_center_require_lost_once) or getattr(self, latch_attr, False):
            return True, near_cx, far_cx

        return False, near_cx, far_cx

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
            near_cx = meas.get('near_blue_cx', meas.get('blue_near_cx'))
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

        # --- PARKING APPROACH:
        # go_parking 후 파란 주차 marker를 만나면, L코너처럼 far span을 보지 않는다.
        # near 쪽 파란 marker가 한 번 보인 뒤 사라지는 순간을 기준으로 ADVANCE -> ROTATE로 들어간다.
        if self.corner_phase == 'APPROACH' and self.corner_context == 'parking_red':
            if t > self.corner_approach_timeout:
                self.parking_red_done = False
                self._end_corner('parking approach 타임아웃(오검출)')
                return

            near_blue_cx = meas.get('near_blue_cx', meas.get('blue_near_cx'))
            near_blue_area = int(meas.get('near_blue_area', meas.get('blue_near_area', 0)) or 0)

            blue_seen_now = (
                near_blue_cx is not None
                and near_blue_area > self.parking_blue_lost_area_px
            )

            if blue_seen_now:
                self.parking_near_blue_seen = True
                self.approach_near_lost = 0
            elif self.parking_near_blue_seen:
                self.approach_near_lost += 1
            else:
                # 아직 near band까지 파란 marker가 내려오지 않은 상태
                self.approach_near_lost = 0

            if (
                self.parking_near_blue_seen
                and self.approach_near_lost >= self.parking_corner_lost_frames
            ):
                self.corner_phase = 'ADVANCE'
                self.phase_t0 = self._now()
                self._reset_odom_integrator()
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(
                    f'parking near blue lost({self.approach_near_lost} frames) '
                    f'-> ADVANCE odom 목표 {self.active_advance_dist:.3f}m'
                )
                return

            # 파란 marker가 near에 보이는 동안은 파란 marker 중심을 기준으로 천천히 접근
            if near_blue_cx is not None:
                steer_err = (near_blue_cx - w / 2.0) / (w / 2.0)
                wz = self._clip_wz(-self.steer_kp * steer_err)
            else:
                wz = 0.0

            self.publish_cmd(self.slow_vx, 0.0, wz)
            return

        # --- APPROACH: L코너 1차/2차 감지 후 직진만 하며 near lost 대기 ---
        if self.corner_phase == 'APPROACH':
            if t > self.corner_approach_timeout:
                self._end_corner('approach 타임아웃(오검출)')
                return

            # 2차 감지: far가 사라짐
            # 여기서 회전하지 않는다. latch만 걸고 계속 직진.
            if far_cx is None and not self.far_lost_latched:
                self.far_lost_latched = True
                self.get_logger().info('far_cx 사라짐 (2차) - 코너 임박, 직진 유지')

            # 3차 감지: near도 사라짐
            self.approach_near_lost = self.approach_near_lost + 1 if near_cx is None else 0
            if self.approach_near_lost >= self.near_lost_frames:
                self.corner_phase = 'ADVANCE'
                self.phase_t0 = self._now()
                self._reset_odom_integrator()
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(
                    f'near_cx 사라짐 (3차) -> 정지 후 ADVANCE odom 목표 '
                    f'{self.active_advance_dist:.3f}m, vx={self.corner_advance_vx:.3f} '
                    f'(far_lost={self.far_lost_latched})'
                )
                return

            # 중요:
            # 기존에는 여기서 far/near_cx로 steer_err를 계산해서 wz를 넣었음.
            # 이제 L코너 접근 중에는 1차/2차 모두 직진만 한다.
            self.publish_cmd(self.slow_vx, 0.0, 0.0)
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
                self.corner_rotate_line_center_armed = False
                self.corner_rotate_blue_center_armed = False
                self.parking_blue_align_seen_count = 0
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

            if self.corner_context == 'parking_red':
                # 주차 진입 회전에서는 near_blue_cx가 아니라,
                # Orin perception에서 찾은 '파란 주차구역 테두리의 수평 여부'로 조기정지한다.
                align_hit, align_reason = self._parking_blue_align_stop_ready(meas, yaw_progress)
                if align_hit:
                    self.corner_phase = 'ROTATE_STOP'
                    self.phase_t0 = self._now()
                    self.publish_cmd(0.0, 0.0, 0.0)

                    target_deg = self.active_rotate_deg * self.active_rotate_cal
                    self.get_logger().info(
                        f'ROTATE 조기정지: parking blue horizontal align hit '
                        f'{align_reason}, '
                        f'yaw={math.degrees(yaw_progress):.1f}deg / 목표={target_deg:.1f}deg, '
                        f'ctx={self.corner_context}'
                    )
                    return
            else:
                # 일반 L코너 / object 방향 코너는 실제 빨간 주행 라인 기준으로 조기정지
                center_hit, raw_near_cx, raw_far_cx = self._rotation_center_stop_ready(
                    meas,
                    'corner_rotate_line_center_armed',
                    color='red_line'
                )
                if center_hit:
                    self.corner_phase = 'ROTATE_STOP'
                    self.phase_t0 = self._now()
                    self.publish_cmd(0.0, 0.0, 0.0)

                    target_deg = self.active_rotate_deg * self.active_rotate_cal
                    self.get_logger().info(
                        f'ROTATE 조기정지: red_line center hit '
                        f'near_cx={raw_near_cx}, far_cx={raw_far_cx}, '
                        f'window=[{self.rotate_center_lo},{self.rotate_center_hi}], '
                        f'yaw={math.degrees(yaw_progress):.1f}deg / 목표={target_deg:.1f}deg, '
                        f'ctx={self.corner_context}'
                    )
                    return

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
        
    def do_junction(self, meas):
        w = meas['w']
        t = self._now() - self.phase_t0

        # legacy 대표 QR. 로그용으로만 사용한다.
        legacy_qtext = meas.get('qr_text')
        legacy_minbb = int(meas.get('qr_bbox', 0) or 0)
        legacy_qcx = meas.get('qr_cx')
        legacy_qcy = meas.get('qr_cy')
        legacy_qr_in_roi = bool(meas.get('qr_in_stop_roi', False))

        # 새 방식: 화면에 보이는 QR 전체 리스트에서 원하는 QR을 직접 고른다.
        qr_b = self._select_qr(meas, 'B')
        qr_target = self._select_qr(meas, self.target) if self.target in ('A', 'B', 'C') else None
        qr_summary = self._qr_log_summary(meas)

        # ============================================================
        # ALIGN:
        #   target이 A/B/C 무엇이든, 처음에는 QR_B를 우선 lock한다.
        #   A와 B가 동시에 보이더라도 qr_detections에서 B를 직접 선택한다.
        # ============================================================
        if self.jphase == 'ALIGN':
            if qr_b is not None:
                minbb = int(qr_b.get('bbox', 0) or 0)
                qcx = qr_b.get('cx')
                qcy = qr_b.get('cy')

                self.publish_cmd(0.0, 0.0, 0.0)

                self.jphase = 'APPROACH_B'
                self.phase_t0 = self._now()
                self._qr_hist = []
                self._qr_confirmed = False

                self.get_logger().info(
                    f'QR_B lock 완료 -> APPROACH_B 진입 '
                    f'bbox={minbb}, qcx={qcx}, qcy={qcy}, target={self.target}, '
                    f'all_qr=[{qr_summary}], legacy={legacy_qtext}:{legacy_minbb}'
                )
                return

            # 혹시 목표 QR이 바로 크게 보이는 예외 상황이면 바로 도착 처리.
            # 단, B를 먼저 우선시하므로 qr_b가 없을 때만 이 fallback을 사용한다.
            if qr_target is not None:
                target_bbox = int(qr_target.get('bbox', 0) or 0)
                target_roi = bool(qr_target.get('in_stop_roi', False))
                if target_bbox >= self.qr_stop_bbox and target_roi:
                    self.publish_cmd(0.0, 0.0, 0.0)
                    self.get_logger().info(
                        f'ALIGN 중 target QR 직접 도착: '
                        f'target={self.target}, bbox={target_bbox}, roi={target_roi}, '
                        f'all_qr=[{qr_summary}]'
                    )
                    self._handle_stop_end(f'qr_direct_{self.target}')
                    return

            if t > self.junction_align_timeout:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.state = NavState.STOP_END
                self.publish_status('junction_qr_lock_timeout')
                self.get_logger().warn(
                    f'JUNCTION QR_B lock timeout: target={self.target}, '
                    f'legacy={legacy_qtext}:{legacy_minbb}, all_qr=[{qr_summary}] -> STOP'
                )
                return

            self.publish_cmd(0.0, 0.0, 0.0)
            return

        # ============================================================
        # APPROACH_B:
        #   QR_B의 bbox가 충분히 커질 때까지 전진한다.
        #   A/C가 동시에 보여도 B만 골라서 rate/bbox를 계산한다.
        #   target이 B면 여기서 stop_qr.
        #   target이 A/C면 일단 STOP 후 y축 이동 단계로 넘어간다.
        # ============================================================
        if self.jphase == 'APPROACH_B':
            self._qr_hist.append(1 if qr_b is not None else 0)
            if len(self._qr_hist) > self.qr_rate_window:
                self._qr_hist.pop(0)

            rate = sum(self._qr_hist) / max(len(self._qr_hist), 1)

            if qr_b is not None:
                minbb = int(qr_b.get('bbox', 0) or 0)
                qcx = qr_b.get('cx')
                qcy = qr_b.get('cy')
            else:
                minbb = 0
                qcx = None
                qcy = None
                if legacy_qtext is not None:
                    self.get_logger().warn(
                        f'APPROACH_B 중 대표 QR은 {legacy_qtext}이지만 B가 리스트에 없음: '
                        f'all_qr=[{qr_summary}]',
                        throttle_duration_sec=1.0
                    )

            b_ready = (
                qr_b is not None
                and minbb >= self.qr_stop_bbox
                and rate >= self.qr_min_rate
            )

            if b_ready:
                self.publish_cmd(0.0, 0.0, 0.0)

                if self.target == 'B':
                    self.get_logger().info(
                        f'QR_B 목표 도착 rate={rate:.2f}, bbox={minbb} -> stop_qr'
                    )
                    self._handle_stop_end('qr_B')
                    return

                if self.target in ('A', 'C'):
                    self.jphase = 'SHIFT_TARGET'
                    self.phase_t0 = self._now()
                    self._qr_hist = []
                    self._qr_confirmed = False
                    self.align_left_center = False

                    self.get_logger().info(
                        f'QR_B 접근 완료 bbox={minbb}, rate={rate:.2f} '
                        f'-> 잠깐 정지 후 y축 이동으로 QR_{self.target} 탐색, '
                        f'all_qr=[{qr_summary}]'
                    )
                    return

                self.get_logger().warn(
                    f'알 수 없는 target={self.target}, QR_B에서 정지 처리'
                )
                self._handle_stop_end('qr_B_unknown_target')
                return

            if t > self.junction_approach_timeout:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.state = NavState.STOP_END
                self.publish_status('qr_b_approach_timeout')
                self.get_logger().warn(
                    f'QR_B approach timeout: B_bbox={minbb}, rate={rate:.2f}, '
                    f'legacy={legacy_qtext}:{legacy_minbb}, all_qr=[{qr_summary}]'
                )
                return

            # QR_B 중심이 보이면 linear.y로만 약하게 보정한다.
            vy = 0.0
            if qr_b is not None and qcx is not None:
                q_err = (float(qcx) - w / 2.0) / (w / 2.0)
                vy = float(np.clip(
                    -0.04 * q_err,
                    -0.025,
                    0.025
                ))

            self.publish_cmd(self.junction_approach_vx, vy, 0.0)
            return

        # ============================================================
        # SHIFT_TARGET:
        #   QR_B까지 접근한 뒤 A/C로 y축 이동한다.
        #   이 단계에서도 대표 QR이 아니라 qr_detections에서 target QR을 직접 고른다.
        #   target QR 중심이 ROI 안에 들어오면 stop_qr.
        # ============================================================
        if self.jphase == 'SHIFT_TARGET':
            if self.target not in ('A', 'C'):
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().warn(
                    f'SHIFT_TARGET인데 target={self.target}, 잘못된 상태 -> STOP'
                )
                self.state = NavState.STOP_END
                self.publish_status('shift_target_invalid')
                return

            qr_target = self._select_qr(meas, self.target)
            if qr_target is not None:
                qtext = qr_target.get('text')
                minbb = int(qr_target.get('bbox', 0) or 0)
                qcx = qr_target.get('cx')
                qcy = qr_target.get('cy')
                qr_in_roi = bool(qr_target.get('in_stop_roi', False))
            else:
                qtext = None
                minbb = 0
                qcx = None
                qcy = None
                qr_in_roi = False

            if t > self.qr_target_shift_timeout:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.state = NavState.STOP_END
                self.publish_status('qr_target_shift_timeout')
                self.get_logger().warn(
                    f'QR_{self.target} shift timeout: '
                    f'target_q={qtext}, bbox={minbb}, roi={qr_in_roi}, '
                    f'qcx={qcx}, qcy={qcy}, all_qr=[{qr_summary}]'
                )
                return

            # A/C 모두 bbox 크기 조건 없이 ROI 진입만 본다.
            if qr_target is not None and qr_in_roi:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(
                    f'QR_{self.target} ROI 도착: '
                    f'q=({qcx}, {qcy}), bbox={minbb}, roi={qr_in_roi}, '
                    f'all_qr=[{qr_summary}] -> stop_qr'
                )
                self._handle_stop_end(f'qr_roi_{self.target}')
                return

            # 기존 코드 방향 유지:
            # A: vy < 0
            # C: vy > 0
            # 실제 차가 반대로 움직이면 여기 부호만 바꾸면 됨.
            vy = -self.junction_strafe_speed if self.target == 'A' else self.junction_strafe_speed

            self.publish_cmd(0.0, vy, 0.0)

            self.get_logger().info(
                f'QR_{self.target} shift 중: target_q={qtext}, '
                f'bbox={minbb}, roi={qr_in_roi}, vy={vy:.3f}, all_qr=[{qr_summary}]',
                throttle_duration_sec=0.5
            )
            return

        self.publish_cmd(0.0, 0.0, 0.0)
        self.get_logger().warn(f'알 수 없는 jphase={self.jphase} -> STOP')
        self.state = NavState.STOP_END
        self.publish_status('junction_unknown_phase')

    def _request_turn_180_after_settle(self, after_phase, reason='pre_turn_settle'):
        """
        QR A/C -> B y축 정렬 직후 또는 QR_B에서 바로 빠져나가는 상황에서
        곧바로 180도 회전을 시작하지 않고, 정지 안정화 구간을 거친 뒤 TURN_180으로 진입한다.
        """
        self.publish_cmd(0.0, 0.0, 0.0)

        self.pending_turn_after_phase = after_phase
        self.pending_turn_reason = reason
        self.phase_t0 = self._now()

        if not self.request_transition(
            NavState.PRE_TURN_SETTLE,
            reason=f'pre_turn_settle before TURN_180: {reason}'
        ):
            return

        self.get_logger().info(
            f'PRE_TURN_SETTLE 시작: {self.pre_turn_settle_time:.2f}s 후 180도 회전 예정, '
            f'use_odom={self.pre_turn_settle_use_odom}, after={after_phase.name}, reason={reason}'
        )

    def do_pre_turn_settle(self):
        """
        180도 회전 직전 정지 안정화 상태.
        vel_filter가 있어도 vy 정렬 직후 바로 wz를 주면 실제 모터 잔여 속도와 겹칠 수 있으므로,
        일정 시간 또는 odom 정지 조건을 만족할 때까지 0 명령을 유지한다.
        """
        self.publish_cmd(0.0, 0.0, 0.0)

        t = self._now() - self.phase_t0
        speed = self._odom_speed_xy()
        abs_wz = self._odom_abs_wz()

        time_ok = t >= float(self.pre_turn_settle_time)
        odom_ok = (
            speed <= float(self.corner_stop_v_thresh)
            and abs_wz <= float(self.corner_stop_w_thresh)
        )

        # odom이 없으면 _odom_speed_xy/_odom_abs_wz가 999를 반환하므로 시간 조건으로 fallback 된다.
        ready = (time_ok or odom_ok) if self.pre_turn_settle_use_odom else time_ok

        if not ready:
            self.get_logger().info(
                f'pre-turn settle 대기 중 t={t:.2f}/{self.pre_turn_settle_time:.2f}s, '
                f'speed={speed:.3f}, wz={abs_wz:.3f}',
                throttle_duration_sec=0.5
            )
            return

        after = self.pending_turn_after_phase or self.mission_phase
        reason = self.pending_turn_reason or 'pre_turn_settle_done'

        self.pending_turn_after_phase = None
        self.pending_turn_reason = ''

        self.get_logger().info(
            f'pre-turn settle 완료 t={t:.2f}s speed={speed:.3f}, wz={abs_wz:.3f} '
            f'-> TURN_180 시작, after={after.name}, reason={reason}'
        )

        self._enter_turn_180(after, reason=reason)

    def do_turn_180(self, meas):
        t = self._now() - self.phase_t0

        if self.odom is None:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.get_logger().warn('/odometry/filtered 미수신 - TURN_180 대기',
                                throttle_duration_sec=1.0)
            return

        target_yaw = math.radians(self.turn_180_deg * self.turn_180_cal)
        yaw_progress = self.turn_dir * self.odom_int_yaw

        center_hit = False
        raw_near_cx = None
        raw_far_cx = None

        yaw_deg = math.degrees(yaw_progress)

        # 180도 회전은 초반에 기존 라인 far_cx가 중앙에 들어오는 오검출이 많다.
        # 따라서 일정 각도 이상 돈 뒤에만 line-center 조기정지를 허용한다.
        if (
            self.turn_180_center_stop_enable
            and yaw_deg >= float(self.turn_180_center_min_yaw_deg)
        ):
            center_hit, raw_near_cx, raw_far_cx = self._rotation_center_stop_ready(
                meas, 'turn_180_line_center_armed'
            )

        if center_hit:
            self.publish_cmd(0.0, 0.0, 0.0)

            if not self.request_transition(NavState.TURN_PAUSE, reason='turn_180 line center hit'):
                return
            self.phase_t0 = self._now()
            self._reset_odom_integrator()

            self.get_logger().info(
                f'180도 조기정지: line center hit near_cx={raw_near_cx}, far_cx={raw_far_cx}, '
                f'window=[{self.rotate_center_lo},{self.rotate_center_hi}], '
                f'yaw={yaw_deg:.1f}deg / 목표={math.degrees(target_yaw):.1f}deg '
                f'-> {self.active_turn_pause_time:.1f}s 정지 대기'
            )
            return

        if yaw_progress >= target_yaw:
            self.publish_cmd(0.0, 0.0, 0.0)

            if not self.request_transition(NavState.TURN_PAUSE, reason='turn_180 done'):
                return
            self.phase_t0 = self._now()
            self._reset_odom_integrator()

            self.get_logger().info(
                f'180도 완료 -> {self.active_turn_pause_time:.1f}s 정지 대기'
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

        if t < self.active_turn_pause_time:
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

            # QR 구역 탈출 직후 너무 빨리 line lost / corner 판단으로 튀는 것 방지
            self.post_corner_grace_until = self._now() + self.post_corner_grace

            self.get_logger().info(
                f'180도 대기 완료 -> TO_OBJECTS '
                f'(grace {self.post_corner_grace:.1f}s)'
            )
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
        
    def do_parking_forward(self, meas):
        """
        parking_red 회전 완료 후 주차구역으로 직진.

        기존: parking_forward_time 동안 무조건 직진
        변경:
          1) near/far blue_px가 모두 사라질 때까지 직진
             - JSON의 red 필드는 실제 파란 주차 marker
          2) 사라짐이 연속 N프레임 확인되면 odom 기준 parking_extra_dist_m 만큼 추가 직진
          3) PARK_PAUSE로 넘어가 1.5초 정지 후 최종 180도
        """
        t = self._now() - self.phase_t0

        if self.parking_forward_phase is None:
            self.parking_forward_phase = 'UNTIL_BLUE_LOST'
            self.parking_blue_lost_count = 0
            self.phase_t0 = self._now()
            self._reset_odom_integrator()

        near_blue_px = int(meas.get('near_blue_area', meas.get('blue_near_area', 0)) or 0)
        far_blue_px = int(meas.get('far_blue_area', meas.get('blue_far_area', 0)) or 0)

        if self.parking_forward_phase == 'UNTIL_BLUE_LOST':
            blue_gone = (
                near_blue_px <= self.parking_blue_lost_area_px
                and far_blue_px <= self.parking_blue_lost_area_px
            )

            if blue_gone:
                self.parking_blue_lost_count += 1
            else:
                self.parking_blue_lost_count = 0

            if self.parking_blue_lost_count >= self.parking_blue_lost_frames:
                self.parking_forward_phase = 'EXTRA'
                self.phase_t0 = self._now()
                self._reset_odom_integrator()
                self.publish_cmd(0.0, 0.0, 0.0)

                self.get_logger().info(
                    f'parking blue lost confirmed: '
                    f'near_blue={near_blue_px}, far_blue={far_blue_px}, '
                    f'frames={self.parking_blue_lost_count} '
                    f'-> extra {self.parking_extra_dist_m:.2f}m'
                )
                return

            if t > self.parking_forward_timeout:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.state = NavState.STOP_END
                self.publish_status('parking_forward_timeout')
                self.get_logger().error(
                    f'parking forward timeout: near_blue={near_blue_px}, '
                    f'far_blue={far_blue_px}, threshold={self.parking_blue_lost_area_px}'
                )
                return

            self.publish_cmd(self.parking_forward_vx, 0.0, 0.0)
            self.get_logger().info(
                f'parking forward: waiting blue gone '
                f'near/far={near_blue_px}/{far_blue_px}, '
                f'lost_count={self.parking_blue_lost_count}/{self.parking_blue_lost_frames}',
                throttle_duration_sec=0.5
            )
            return

        if self.parking_forward_phase == 'EXTRA':
            if self.odom is None:
                self.publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().warn('/odometry/filtered 미수신 - parking extra 대기',
                                       throttle_duration_sec=1.0)
                return

            if self.odom_int_dist >= self.parking_extra_dist_m:
                self.publish_cmd(0.0, 0.0, 0.0)
                if not self.request_transition(NavState.PARK_PAUSE, reason='parking extra done'):
                    return
                self.phase_t0 = self._now()
                self.parking_forward_phase = None

                self.get_logger().info(
                    f'parking extra 완료 odom_dist={self.odom_int_dist:.3f}m '
                    f'/ 목표={self.parking_extra_dist_m:.3f}m -> pause'
                )
                return

            if t > max(self.parking_forward_timeout, 2.0):
                self.publish_cmd(0.0, 0.0, 0.0)
                self.state = NavState.STOP_END
                self.publish_status('parking_extra_timeout')
                self.get_logger().error(
                    f'parking extra timeout: odom_dist={self.odom_int_dist:.3f}m '
                    f'/ 목표={self.parking_extra_dist_m:.3f}m'
                )
                return

            self.publish_cmd(self.parking_forward_vx, 0.0, 0.0)
            return

        self.publish_cmd(0.0, 0.0, 0.0)
        self.state = NavState.STOP_END
        self.publish_status('parking_forward_bad_phase')
        self.get_logger().error(f'알 수 없는 parking_forward_phase={self.parking_forward_phase}')


    def do_parking_pause(self):
        t = self._now() - self.phase_t0
        self.publish_cmd(0.0, 0.0, 0.0)

        if t >= self.parking_pause_time:
            self.get_logger().info(
                f'parking pause {self.parking_pause_time:.1f}s 완료 -> 최종 180도 정렬'
            )
            self._enter_turn_180(MissionPhase.PARKED, reason='parking_final_align')
            return
    
    def _enter_turn_180(self, after_phase, reason='', force=False):
        if not self.request_transition(NavState.TURN_180, reason=f'TURN_180 start {reason}', force=force):
            return False

        self.turn_after_phase = after_phase

        # QR 구역에서 빠져나와 TO_OBJECTS로 돌아갈 때는 더 오래 정지
        if after_phase == MissionPhase.TO_OBJECTS:
            self.active_turn_pause_time = float(self.return_turn_pause_time)
        else:
            self.active_turn_pause_time = float(self.turn_pause_time)

        self.turn_dir = 1
        self.turn_180_line_center_armed = False

        self.phase_t0 = self._now()
        self._reset_odom_integrator()
        self.publish_cmd(0.0, 0.0, 0.0)

        self.get_logger().info(
            f'180도 회전 시작 reason={reason}, after={after_phase.name}, '
            f'wz={self.turn_180_wz:.2f}, deg={self.turn_180_deg:.1f}, '
            f'cal={self.turn_180_cal:.3f}, pause={self.active_turn_pause_time:.1f}s'
        )
        return True

    def _enter_approach(self, force=False):
        if not self.request_jphase('APPROACH', reason='junction align done', force=force):
            return False
        self.phase_t0 = self._now()
        self._qr_hist = []
        self._qr_confirmed = False
        return True
        
    def _end_corner(self, why):
        if self.freeze_transition:
            self.publish_cmd(0.0, 0.0, 0.0)
            now = self._now()
            if now - self._last_freeze_log_t > 0.5:
                self.get_logger().warn(f'FREEZE: corner end blocked reason={why}')
                self.publish_status(f'freeze_block:corner_end:{why}')
                self._last_freeze_log_t = now
            return

        ctx = self.corner_context

        self.corner_phase = None
        self.prev_err = 0.0
        self.lost_count = 0
        self.publish_cmd(0.0, 0.0, 0.0)

        if ctx == 'parking_red':
            # parking corner가 timeout/오검출이면 PARK_FORWARD로 들어가면 위험하다.
            # 다시 FOLLOW로 복귀해서 blue parking marker를 재시도한다.
            if '타임아웃' in why or '오검출' in why:
                self.parking_red_done = False
                self.corner_context = 'normal'
                self.corner_phase = None
                self.state = NavState.FOLLOW
                self.post_corner_grace_until = self._now() + 0.8

                self.get_logger().warn(
                    f'parking_red CORNER 실패({why}) -> PARK_FORWARD 진입 금지, FOLLOW로 복귀'
                )
                return

            self.state = NavState.PARK_FORWARD
            self.phase_t0 = self._now()
            self.corner_context = 'normal'
            self.parking_forward_phase = 'UNTIL_BLUE_LOST'
            self.parking_blue_lost_count = 0
            self._reset_odom_integrator()

            self.get_logger().info(
                f'parking_red 코너 완료({why}) -> 파란 주차선이 near/far에서 모두 사라질 때까지 직진, '
                f'이후 {self.parking_extra_dist_m:.2f}m 추가 직진'
            )
            return

        if ctx == 'object_red':
            # 오검출/타임아웃이면 depth를 열지 않고, red corner를 다시 시도할 수 있게 함
            if '타임아웃' in why or '오검출' in why:
                self.object_red_done = False
                self.rack_depth_armed = False
                self.rack_close_count = 0

                self.get_logger().warn(
                    f'object_red CORNER 실패({why}) -> rack depth 비활성, red corner 재시도 가능'
                )
            else:
                self.rack_depth_armed = True
                self.rack_close_count = 0

                self.get_logger().info(
                    f'object_red CORNER 정상 완료({why}) -> rack white-wall depth 활성화'
                )

        # object_red 또는 normal은 기존처럼 FOLLOW
        self.state = NavState.FOLLOW
        self.corner_context = 'normal'
        self.post_corner_grace_until = self._now() + self.post_corner_grace

        self.get_logger().info(
            f'코너 완료({why}) -> FOLLOW (grace {self.post_corner_grace}s)'
        )

    def _enter_red_corner(self, context, force=False, turn_dir_override=None):
        if not self.request_transition(NavState.CORNER, reason=f'{context} red corner', force=force):
            return False
        self.corner_phase = 'APPROACH'
        self.phase_t0 = self._now()

        self.far_lost_latched = False
        self.approach_near_lost = 0
        self._reset_odom_integrator()

        self.corner_context = context
        self.corner_rotate_line_center_armed = False
        self.corner_rotate_blue_center_armed = False

        if context == 'object_red':
            # 빨간 분기 코너는 시작했지만, 아직 rack depth는 보면 안 됨
            self.object_red_done = True
            self.rack_depth_armed = False
            self.rack_close_count = 0

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
            self.parking_near_blue_seen = False
            self.parking_blue_lost_count = 0
            self.parking_joint_seen_count = 0
            self.parking_forward_phase = None

            if turn_dir_override is not None:
                self.corner_dir = int(turn_dir_override)
            else:
                self.corner_dir = int(self.parking_red_turn_dir)
            if self.corner_dir == 0:
                self.corner_dir = -1

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

        if self.freeze_transition:
            now = self._now()
            if now - self._last_freeze_log_t > 0.5:
                self.get_logger().warn(f'FREEZE: stop_end handling blocked reason={reason}')
                self.publish_status(f'freeze_block:stop_end:{reason}')
                self._last_freeze_log_t = now
            return

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

    def _parking_blue_align_stop_ready(self, meas, yaw_progress):
        """
        parking_red ROTATE 전용 조기정지.
        near_cx/far_cx 대신 파란 주차구역 테두리가 화면에서 수평으로 잡혔는지 본다.
        """
        if not bool(getattr(self, 'parking_blue_align_stop_enable', True)):
            self.parking_blue_align_seen_count = 0
            return False, 'disabled'

        min_yaw = math.radians(max(0.0, float(getattr(self, 'parking_blue_align_min_yaw_deg', 0.0))))
        if yaw_progress < min_yaw:
            self.parking_blue_align_seen_count = 0
            return False, f'waiting_min_yaw {math.degrees(yaw_progress):.1f}/{math.degrees(min_yaw):.1f}deg'

        upper = bool(meas.get('parking_blue_align_upper_detected', False))
        lower = bool(meas.get('parking_blue_align_lower_detected', False))

        if bool(getattr(self, 'parking_blue_align_require_both', True)):
            hit = upper and lower
            mode = 'both'
        else:
            hit = bool(meas.get('parking_blue_align_detected', False)) or upper or lower
            mode = 'either'

        if hit:
            self.parking_blue_align_seen_count += 1
        else:
            self.parking_blue_align_seen_count = 0

        need = max(1, int(getattr(self, 'parking_blue_align_confirm_frames', 1)))

        up_a = meas.get('parking_blue_align_upper_angle_deg')
        lo_a = meas.get('parking_blue_align_lower_angle_deg')
        up_len = meas.get('parking_blue_align_upper_len_px', 0.0)
        lo_len = meas.get('parking_blue_align_lower_len_px', 0.0)

        reason = (
            f'mode={mode}, upper={int(upper)} angle={up_a} len={up_len}, '
            f'lower={int(lower)} angle={lo_a} len={lo_len}, '
            f'confirm={self.parking_blue_align_seen_count}/{need}'
        )

        if self.parking_blue_align_seen_count > 0 or yaw_progress >= min_yaw:
            self.get_logger().info(
                f'parking blue align check: {reason}',
                throttle_duration_sec=0.5
            )

        return self.parking_blue_align_seen_count >= need, reason

    def _parking_entry_turn_dir(self, meas):
        """
        주차 진입 회전 방향을 영상 인식값으로 결정한다.

        기준:
        - parking_joint_cx : 빨간 주행 라인과 파란 주차선의 접점 x
        - near_blue_cx     : near band에서 보이는 파란 주차선 중심 x

        near_blue_cx가 접점보다 오른쪽이면 화면상 주차선이 오른쪽으로 뻗는 것이므로 우회전(-1),
        왼쪽이면 좌회전(+1)로 본다. 값이 불안정하거나 deadband 안이면 기존
        parking_red_turn_dir를 fallback으로 사용한다.
        """
        fallback = int(self.parking_red_turn_dir)
        if fallback == 0:
            fallback = -1

        if not bool(getattr(self, 'parking_dynamic_turn_dir', True)):
            return fallback

        joint_cx = meas.get('parking_joint_cx')
        blue_cx = meas.get('near_blue_cx', meas.get('blue_near_cx'))

        try:
            joint_x = float(joint_cx)
            blue_x = float(blue_cx)
        except Exception:
            self.get_logger().warn(
                f'parking turn dir fallback: joint_cx={joint_cx}, near_blue_cx={blue_cx}, '
                f'fallback={fallback}',
                throttle_duration_sec=0.7
            )
            return fallback

        dx = blue_x - joint_x
        deadband = max(0.0, float(self.parking_turn_dir_deadband_px))

        if abs(dx) <= deadband:
            self.get_logger().warn(
                f'parking turn dir fallback: dx={dx:.1f}px <= deadband={deadband:.1f}px, '
                f'joint_cx={joint_x:.1f}, near_blue_cx={blue_x:.1f}, fallback={fallback}',
                throttle_duration_sec=0.7
            )
            return fallback

        # 화면 좌표 기준: blue가 접점 오른쪽이면 오른쪽 주차 진입 -> 우회전(-1)
        #                 blue가 접점 왼쪽이면 왼쪽 주차 진입  -> 좌회전(+1)
        turn_dir = -1 if dx > 0.0 else +1

        self.get_logger().info(
            f'parking turn dir dynamic: joint_cx={joint_x:.1f}, near_blue_cx={blue_x:.1f}, '
            f'dx={dx:.1f}px -> dir={turn_dir}({"L" if turn_dir > 0 else "R"})',
            throttle_duration_sec=0.7
        )
        return turn_dir

    def _parking_joint_ahead(self, meas):
        """
        주차 진입 전용 트리거.
        파란 주차선이 화면 오른쪽에 보이는 것만으로는 코너를 시작하지 않고,
        Orin perception이 계산한 red_line + blue 접점이 연속으로 잡힐 때만 True.
        """
        if not self.parking_joint_required:
            return self._junction_ahead(meas)

        detected = bool(meas.get('parking_joint_detected', False))
        touch_px = int(meas.get('parking_joint_touch_px', 0) or 0)

        hit = detected and touch_px >= int(self.parking_joint_min_touch_px)
        if hit:
            self.parking_joint_seen_count += 1
        else:
            self.parking_joint_seen_count = 0

        return self.parking_joint_seen_count >= max(1, int(self.parking_joint_confirm_frames))

    def _junction_ahead(self, meas):
        return (
            meas.get('far_blue_area', meas.get('blue_far_area', 0)) >= self.red_area_min
            or meas.get('near_blue_area', meas.get('blue_near_area', 0)) >= self.red_area_min
        )
        
    def _rack_approach_ahead(self, meas):
        """
        라인이 끊긴 게 아니라 object/rack 쪽 흰 벽에 접근 중인지 판단.

        조건:
        - TO_OBJECTS phase
        - object_red CORNER 이후 rack_depth_armed 상태
        - 흰 벽 valid pixel 충분함
        - rack_depth가 approach 거리 안쪽
        - 아직 stop_dist보다는 멂
        """
        if self.mission_phase != MissionPhase.TO_OBJECTS:
            return False

        if not self.rack_depth_armed:
            return False

        d = meas.get('rack_depth_m')
        valid_px = int(meas.get('rack_valid_px', 0) or 0)

        if d is None:
            return False

        try:
            d = float(d)
        except Exception:
            return False

        if valid_px < int(self.rack_approach_min_valid_px):
            return False

        if d > float(self.rack_approach_dist_m):
            return False

        # 너무 가까운 최종 정지는 do_follow 상단의 rack_close_by_white_wall_depth_meas가 처리
        return True

    # ==================== 유틸 ====================
    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _clip_wz(self, wz):
        return float(np.clip(wz, -self.max_wz, self.max_wz))

    def publish_cmd(self, vx, vy, wz):
        if not self.enable_drive:
            vx = vy = wz = 0.0

        # 이 node는 '의도한 effective cmd'를 /cmd_vel_raw로 낸다.
        # myAGV 2023의 angular deadzone/minimum output 대응은 vel_filter_node에서 수행한다.
        # 단, 실험상 vx+wz 조합에서 yaw가 커지므로 여기서도 1차적으로 감속한다.
        if abs(vx) > 1e-6 and abs(wz) > 1e-6:
            vx = min(abs(vx), float(self.follow_turn_vx)) * (1.0 if vx >= 0.0 else -1.0)

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

            
    def publish_stop_obj(self):
        """
        모터가 완전히 멈출 시간을 준 뒤 /stop_obj 발행.
        직접 sleep하지 않고 one-shot timer를 사용한다.
        """
        delay = max(0.0, float(self.stop_obj_publish_delay_sec))

        # 이미 예약되어 있으면 중복 발행 방지
        if self.stop_obj_pending:
            self.get_logger().warn(
                f'/stop_obj publish already pending, delay={delay:.2f}s',
                throttle_duration_sec=1.0
            )
            return

        self.stop_obj_pending = True

        # 혹시 이전 timer가 남아있으면 정리
        if self.stop_obj_delay_timer is not None:
            try:
                self.stop_obj_delay_timer.cancel()
                self.destroy_timer(self.stop_obj_delay_timer)
            except Exception:
                pass
            self.stop_obj_delay_timer = None

        self.publish_cmd(0.0, 0.0, 0.0)

        self.get_logger().info(
            f'/stop_obj publish 예약: {delay:.2f}s 후 발행'
        )

        if delay <= 0.0:
            self._publish_stop_obj_now()
            return

        self.stop_obj_delay_timer = self.create_timer(
            delay,
            self._publish_stop_obj_now
        )


    def _publish_stop_obj_now(self):
        """
        one-shot timer callback.
        """
        if self.stop_obj_delay_timer is not None:
            try:
                self.stop_obj_delay_timer.cancel()
                self.destroy_timer(self.stop_obj_delay_timer)
            except Exception:
                pass
            self.stop_obj_delay_timer = None

        self.stop_obj_pub.publish(Empty())
        self.publish_status('stop_obj')
        self.stop_obj_pending = False

        self.get_logger().info('/stop_obj publish delayed done')


    def publish_stop_qr(self):
        """
        QR 도착 후 모터가 완전히 멈출 시간을 준 뒤 /stop_qr 발행.
        직접 sleep하지 않고 one-shot timer를 사용한다.
        """
        delay = max(0.0, float(self.stop_qr_publish_delay_sec))

        # 이미 예약되어 있으면 중복 발행 방지
        if self.stop_qr_pending:
            self.get_logger().warn(
                f'/stop_qr publish already pending, delay={delay:.2f}s',
                throttle_duration_sec=1.0
            )
            return

        self.stop_qr_pending = True

        # 혹시 이전 timer가 남아있으면 정리
        if self.stop_qr_delay_timer is not None:
            try:
                self.stop_qr_delay_timer.cancel()
                self.destroy_timer(self.stop_qr_delay_timer)
            except Exception:
                pass
            self.stop_qr_delay_timer = None

        # 먼저 확실히 정지 명령
        self.publish_cmd(0.0, 0.0, 0.0)

        self.get_logger().info(
            f'/stop_qr publish 예약: {delay:.2f}s 후 발행'
        )

        if delay <= 0.0:
            self._publish_stop_qr_now()
            return

        self.stop_qr_delay_timer = self.create_timer(
            delay,
            self._publish_stop_qr_now
        )


    def _publish_stop_qr_now(self):
        """
        one-shot timer callback.
        """
        if self.stop_qr_delay_timer is not None:
            try:
                self.stop_qr_delay_timer.cancel()
                self.destroy_timer(self.stop_qr_delay_timer)
            except Exception:
                pass
            self.stop_qr_delay_timer = None

        self.stop_qr_pub.publish(Empty())
        self.publish_status('stop_qr')
        self.stop_qr_pending = False

        self.get_logger().info('/stop_qr publish delayed done')

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
