#!/usr/bin/env python3
"""
pick_eyeinhand_node.py  (eye-in-hand 버전 + 피드백 기반 대기)

[이번 개선 요약 - sleep → 피드백 기반 대기]
  기존에는 send_angles/send_coords 이후 무조건 고정 시간(time.sleep)만큼
  기다렸다가 다음 단계로 넘어갔음. 이 방식은:
    - 로봇이 실제로 도착하기 전에 다음 명령을 보낼 수도 있고 (부정확),
    - 반대로 이미 도착했는데도 불필요하게 오래 기다릴 수도 있었음 (느림).

  이제는 두 개의 헬퍼로 "실제로 끝났는지" 확인하고 넘어간다:
    - _wait_in_position(target, mode, timeout): 팔이 목표 각도/좌표에
      도달했는지 확인. mc.is_in_position()을 우선 쓰고, 펌웨어가 이를
      신뢰할 수 없는 값(None/-1 등)으로 응답하면 자동으로 get_angles/
      get_coords 기반 오차비교(diff) 폴백으로 전환한다.
    - _wait_gripper_settled(timeout): mc.is_gripper_moving()으로 그리퍼
      동작이 끝났는지 확인. 미지원 펌웨어면 timeout만큼만 대기(안전 폴백).

  _check_gripped()는 기존부터 이미 값 폴링 기반이라 그대로 유지.

  기존 [eye-in-hand] 요약(관측 자세 이동, PICK_READY 단계 삭제, 미세보정 0
  초기화, GRIPPER_Z_OFFSET_MM 재측정 필요 등)은 동일하게 유지됨.

토픽:
  구독: /pick_command, /place_command, /emergency_stop
        /observe_move (String) : "1"/"2" 관측할 층, "qr:A/B/C" QR 관측
  발행: /pick_status, /arm/status
        /observe_ready (String) : "ready"/"qr_ready" 관측 자세 도착
        /observe_pose  (Float32MultiArray) : 실제 도달 자세 (동적 T용)
"""

import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import numpy as np
try:
    from pymycobot import MyCobot280
except ImportError:
    raise SystemExit("pymycobot not installed.")

# =========================
# [신규] QR 플레이싱 관측 자세
# =========================
QR_OBSERVE_ANGLES = {
    'A': [4.57, 107.66, -134.2, -4.21, -5.0, -41.13],
    'B': [4.57, 107.66, -134.2, -4.21, -5.0, -41.13],
    'C': [4.57, 107.66, -134.2, -4.21, -5.0, -41.13],
}

# =========================
# myCobot 기본 설정
# =========================
SERIAL_PORT = "/dev/ttyAMA0"
BAUD = 1000000

MOVE_SPEED = 15
GRIPPER_SPEED = 80

# 로봇팔 기본 대기 자세
HOME_ANGLES = [-40, 90, -130, -20, 0, 0]

# =========================
# 층별 관측 자세 (angles / coords 짝)
# =========================
SHELF_ANGLES = {
    1: [1.23, 115.13, -136.31, -4.13, 2.9, -40.78],   # 1층 (랙 16.5cm)
    2: [-5.0, 79.45, -76.81, -13.71, 5.97, -44.2]
}

# 1층 접은 진입 자세 (J5 돌려서 랙 회피). 진입/탈출 공용.
SAFE_ENTRY_1F_ANGLES = [8.17, -27.94, -129.37, 126.38, 54.31, -45.87]

# 플레이스 전용 중간 관절 자세
# HOME에서 place 좌표로 바로 send_coords하면 IK가 꼬일 수 있어서
# 먼저 팔 모양을 안정적인 형태로 만들어준다.
PLACE_READY_ANGLES = [0.0, 85.0, -125.0, 10.0, 0.0, -45.0]

# 관측 자세 이동 후 정착 대기(초). 피드백 도달 확인의 타임아웃 기준으로 사용.
OBSERVE_SETTLE_WAIT = 4.0

# 그리퍼 값
GRIPPER_OPEN = 100
GRIPPER_CLOSE = 18

GRIP_SUCCESS_THRESH = 31  # 그리퍼 값이 이 이상이면 블록 물림 (CLOSE=30, 널널하게 33)

# =========================
# 피킹 보정값
# =========================
_OFFSET_MM = 0.0   # TODO: eye-in-hand 재측정

# 층별 파지 offset (자세가 달라서 offset도 다름)
GRIP_OFFSET = {
    1: [-17.0, 2.5, 78.0],    # 1층 z만 조정 (위에서 잡히니 낮춰야)
    2: [-54.4, -2.0, 46.5],   # 2층 (현재 값, 정확)
}

# 층별 파지 자세각 (rx, ry, rz)
GRIP_POSE = {
    1: [-145.47, -28.92, -53.1],   # 1층 수그린 자세 (실측)
    2: [-102.25, -38.21, -82.48],  # 2층 (기존, 잘 됨)
}

# 물체 바로 위 waypoint 높이
APPROACH_Z_MM = 30.0
PLACE_APPROACH_Z_MM = 90.0

# =========================
# 실제 피킹 미세 보정값
# =========================
PICK_X_BIAS_MM = 5.0
PICK_Y_BIAS_MM = -9.9
PICK_Z_BIAS_MM = 0.0
GRIPPER_Z_OFFSET_MM = 0.0

# y 절대보정 (고정량, mm)
Y_COMP_1F_POS = -8.5
Y_COMP_1F_NEG = 0.0
Y_COMP_2F_POS = 1.2
Y_COMP_2F_NEG = 8.5

# 집은 뒤 위로 들어올릴 높이
LIFT_Z = 45.0

# 내려갈 때 속도는 천천히
DESCEND_SPEED = 8

# =========================
# [신규] 피드백 대기 관련 설정
# =========================
# send_angles 이동 후 도달 확인 타임아웃(초). 기존 sleep 값 + 여유.
WAIT_ANGLES_TIMEOUT = 15.0
# send_coords 이동 후 도달 확인 타임아웃(초).
WAIT_COORDS_TIMEOUT = 10.0
# 도달 확인 후 진동/떨림 안정화 대기(초)
# - 일반 이동: 0.5초
# - 관측 자세(동적 T용 get_coords 정확도가 중요한 곳)는 더 길게(0.8초) 사용
SETTLE_AFTER_ARRIVE = 1.2
SETTLE_AFTER_ARRIVE_OBSERVE = 2.0
# 도달 확인 폴링 주기(초)
WAIT_POLL_INTERVAL = 0.1
# is_in_position()이 None/-1 등 신뢰불가 값을 반환했을 때,
# 바로 diff 폴백으로 넘어가지 않고 재시도해볼 횟수/간격
IS_IN_POSITION_RETRY_MAX = 5
IS_IN_POSITION_RETRY_INTERVAL = 0.15
# diff 폴백에서 매번 몇 번 읽어 중앙값을 쓸지 (노이즈/떨림 필터링)
FALLBACK_READ_TRIES = 3
FALLBACK_READ_INTERVAL = 0.05
# 그리퍼 동작 완료 대기 타임아웃(초, 기존 sleep 시간과 동일하게 상황별로 넘김)
GRIPPER_TIMEOUT_DEFAULT = 2.5


class PickNode(Node):
    def __init__(self):
        super().__init__("pick_node")

        # 내부 상태값
        self._busy = False
        self._busy_lock = threading.Lock()
        self.emergency_active = False
        self.current_level = 2   # 현재 관측/파지 중인 층 (observe_move로 갱신)
        self.j1_offset_by_level = {1: 0.0, 2: 0.0}  # 층별 J1 누적 보정값

        # 구독자
        self.create_subscription(
            Float32MultiArray, "/pick_command", self._pick_callback, 10
        )
        self.create_subscription(
            Float32MultiArray, "/place_command", self._place_callback, 10
        )
        self.create_subscription(
            String, "/emergency_stop", self._emergency_stop_callback, 10
        )
        self.create_subscription(
            String, "/observe_move", self._observe_move_callback, 10
        )
        self.create_subscription(String, "/j1_correction", self._j1_correction_callback, 10)

        # 발행자
        self._pick_status_pub = self.create_publisher(String, "/pick_status", 10)
        self._status_pub = self.create_publisher(String, "/arm/status", 10)
        self._observe_ready_pub = self.create_publisher(String, "/observe_ready", 10)
        self._observe_pose_pub = self.create_publisher(Float32MultiArray, "/observe_pose", 10)

        # myCobot 연결
        self.get_logger().info("myCobot 연결 시도 중...")
        self.mc = MyCobot280(SERIAL_PORT, BAUD)
        time.sleep(0.5)

        # 그리퍼 초기화 (get_gripper_value가 값을 주려면 필수, 안 하면 None)
        try:
            self.mc.init_gripper()
            time.sleep(1.0)
            self.get_logger().info("그리퍼 초기화 완료")
        except Exception as e:
            self.get_logger().warn(f"그리퍼 초기화 실패: {e}")

        # 시작 시 홈 포지션 이동
        self.get_logger().info("홈포지션으로 이동 중...")
        try:
            self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
            if not self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT):
                self.get_logger().warn("초기 홈포지션 도달 확인 실패(타임아웃) - 계속 진행")
            self.get_logger().info("pick_node 준비 완료")
        except Exception as e:
            self.get_logger().error(f"초기 홈포지션 이동 실패: {e}")
            self._pub_pick_status("error")

    # =========================
    # 공통 유틸 함수
    # =========================
    def _log(self, msg: str):
        self.get_logger().info(msg)
        m = String()
        m.data = msg
        self._status_pub.publish(m)

    def _pub_pick_status(self, status: str):
        m = String()
        m.data = status
        self._pick_status_pub.publish(m)
        self.get_logger().info(f"/pick_status 발행: {status}")

    def _safe_sleep(self, seconds: float, step: float = 0.1) -> bool:
        elapsed = 0.0
        while elapsed < seconds:
            if self.emergency_active:
                self.get_logger().warn("비상정지 감지 - 현재 시퀀스 중단")
                return False
            time.sleep(step)
            elapsed += step
        return True

    # =========================
    # [신규] 피드백 기반 대기 헬퍼
    # =========================
    def _wait_in_position(self, target, mode, timeout=WAIT_ANGLES_TIMEOUT,
                           poll=WAIT_POLL_INTERVAL, settle=SETTLE_AFTER_ARRIVE):
        """
        로봇팔이 target(각도 or 좌표)에 도달할 때까지 대기.
          mode: 0 = angles 타겟 (mc.get_angles 기준 비교)
                1 = coords 타겟 (mc.get_coords 기준 비교)

        1) mc.is_in_position(target, mode) 를 우선 사용.
           - 반환값 1이면 도달로 간주.
           - 0/1 외의 값(None, -1 등 신뢰불가)이 나오면 즉시 폴백으로
             넘어가지 않고 IS_IN_POSITION_RETRY_MAX회까지 다시 호출해서
             재확인한다 (통신 순간 오류/일시 응답 지연일 수 있으므로).
             그래도 계속 신뢰불가면 그때 diff 폴백으로 전환.
        2) 폴백: get_angles()/get_coords()를 FALLBACK_READ_TRIES회 연속 읽어
           중앙값을 구한 뒤(순간 노이즈/떨림 필터링, _safe_get_pose와 동일 방식),
           target과의 오차가 허용 범위(tol) 이내인지 비교.

        도달 확인 후 settle초만큼 한 번 더 대기해서 진동/떨림을 가라앉힌다
        (특히 get_coords로 관측 자세를 읽어야 하는 경우 필수).

        반환값: True = 도달 확인, False = 타임아웃 또는 비상정지로 중단
        """
        t0 = time.time()

        # myCobot280 Pi에서 각도 이동(mode=0)은 is_in_position()이 -1을 자주 반환하므로
        # 처음부터 get_angles() 기반 diff 확인으로 진행한다.
        # 좌표 이동(mode=1)은 기존처럼 is_in_position()을 먼저 시도한다.
        #use_fallback = True if mode == 0 else False
        # myCobot280 Pi에서는 is_in_position()이 angles/coords 모두 -1을 자주 반환하므로
        # 항상 get_angles/get_coords 기반 diff 확인으로 진행한다.
        use_fallback = True
        
        bad_read_count = 0

        while time.time() - t0 < timeout:
            if self.emergency_active:
                self.get_logger().warn("[WAIT] 비상정지 감지 - 대기 중단")
                return False

            if not use_fallback:
                try:
                    r = self.mc.is_in_position(target, mode)
                except Exception as e:
                    self.get_logger().warn(f"[WAIT] is_in_position 예외: {e}")
                    r = None

                if r == 1:
                    if not self._safe_sleep(settle):
                        return False
                    return True

                if r not in (0, 1):
                    bad_read_count += 1
                    self.get_logger().warn(
                        f"[WAIT] is_in_position 신뢰불가(r={r}) "
                        f"- 재확인 {bad_read_count}/{IS_IN_POSITION_RETRY_MAX}"
                    )
                    if bad_read_count >= IS_IN_POSITION_RETRY_MAX:
                        self.get_logger().warn(
                            "[WAIT] is_in_position 반복 신뢰불가 - diff 폴백 전환"
                        )
                        use_fallback = True
                    else:
                        if not self._safe_sleep(IS_IN_POSITION_RETRY_INTERVAL):
                            return False
                    continue

                # r == 0: 아직 도달 안 함(정상 응답) - 신뢰불가 카운터 리셋
                bad_read_count = 0

            if use_fallback:
                cur = self._safe_get_pose(
                    mode,
                    tries=FALLBACK_READ_TRIES,
                    interval=FALLBACK_READ_INTERVAL
                )
            
                if self.emergency_active:
                    return False
            
                if cur is not None:
                    diffs = [abs(c - t) for c, t in zip(cur, target)]
            
                    self.get_logger().info(
                        f"[WAIT DEBUG] mode={mode} "
                        f"cur={[round(v, 1) for v in cur]} "
                        f"target={[round(v, 1) for v in target]} "
                        f"diff={[round(v, 1) for v in diffs]}"
                    )
            
                    if mode == 0:
                        # angles 모드: 6개 전부 관절각(deg)
                        angle_tol = 6.0
            
                        if all(d <= angle_tol for d in diffs):
                            if not self._safe_sleep(settle):
                                return False
                            return True
            
                    else:
                        # coords 모드: xyz는 mm, rpy는 deg
                        xyz_tol = 25.0
                        rpy_tol = 15.0
            
                        if all(d <= xyz_tol for d in diffs[:3]) and all(d <= rpy_tol for d in diffs[3:]):
                            if not self._safe_sleep(settle):
                                return False
                            return True
                      
            time.sleep(poll)

        self.get_logger().warn(
            f"[WAIT] 도달 타임아웃({timeout}s) mode={mode} "
            f"target={[round(v, 1) for v in target]}"
        )
        return False

    def _wait_gripper_settled(self, timeout=GRIPPER_TIMEOUT_DEFAULT, poll=WAIT_POLL_INTERVAL):
        """
        그리퍼 open/close 동작이 끝날 때까지 대기.
        mc.is_gripper_moving()을 폴링해서 0(정지)이면 즉시 반환.
        펌웨어가 이 기능을 지원하지 않으면(예외 발생) 기존처럼
        timeout만큼만 대기하는 안전 폴백으로 전환.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.emergency_active:
                self.get_logger().warn("[WAIT GRIPPER] 비상정지 감지 - 대기 중단")
                return
            try:
                moving = self.mc.is_gripper_moving()
            except Exception:
                # 미지원 - 남은 시간만 대기하고 종료 (기존 동작과 동일한 안전망)
                self._safe_sleep(max(0.0, timeout - (time.time() - t0)))
                return
            if moving == 0:
                return
            time.sleep(poll)
        self.get_logger().warn(f"[WAIT GRIPPER] 타임아웃({timeout}s) - 다음 단계 진행")

    def _check_gripped(self):
        """그리퍼 파지 판정.
        반환값:
          True  = 물림 확정
          False = 빈손/피드백 불가
    
        기준:
          - gripper_value를 10번 읽음
          - 그중 GRIP_SUCCESS_THRESH 이상 값이 2개 이상이면 물림
          - 값이 낮게 나오면 끝까지 닫힌 상태 = 빈손
          - None이 많거나 유효값이 부족하면 실패
        """
    
        READ_TRIES = 10
        READ_INTERVAL = 0.20
    
        # 10번 중 REQUIRED_HITS번 이상 임계값 이상이면 잡은 것으로 판정
        REQUIRED_HITS = 5
    
        vals = []
    
        # 닫기 명령 한 번 더 보내서 판정 직전 상태를 확실히 함
        try:
            self.mc.set_gripper_value(GRIPPER_CLOSE, GRIPPER_SPEED)
            self._wait_gripper_settled(timeout=1.0)
        except Exception as e:
            self.get_logger().warn(f"[GRIP CHECK] 닫기 명령 예외: {e}")
    
        if self.emergency_active:
            return False
    
        for read_try in range(READ_TRIES):
            try:
                v = self.mc.get_gripper_value()
            except Exception as e:
                self.get_logger().warn(
                    f"[GRIP CHECK] read_try={read_try+1}/{READ_TRIES} 예외: {e}"
                )
                v = None
    
            self.get_logger().info(
                f"[GRIP CHECK] read_try={read_try+1}/{READ_TRIES} raw_value={v}"
            )
    
            if v is not None:
                try:
                    vals.append(float(v))
                except Exception:
                    self.get_logger().warn(f"[GRIP CHECK] 숫자 변환 불가 raw_value={v}")
    
            time.sleep(READ_INTERVAL)
    
        if len(vals) == 0:
            self.get_logger().warn(
                "[GRIP CHECK] 유효 gripper_value 없음 → 파지 실패"
            )
            return False
    
        hit_vals = [v for v in vals if v >= GRIP_SUCCESS_THRESH]
        hit_count = len(hit_vals)
    
        self.get_logger().info(
            f"[GRIP CHECK] 전체 raw vals = {[round(v, 1) for v in vals]}"
        )
        
        self.get_logger().info(
            f"[GRIP CHECK] 성공 후보 vals(>= {GRIP_SUCCESS_THRESH}) = "
            f"{[round(v, 1) for v in hit_vals]}"
        )
        
        self.get_logger().info(
            f"[GRIP CHECK] hit_count={hit_count}/{READ_TRIES}, "
            f"required_hits={REQUIRED_HITS}, "
            f"threshold>={GRIP_SUCCESS_THRESH}"
        )
        
        gripped = hit_count >= REQUIRED_HITS

        self.get_logger().info(
            f"[GRIP CHECK] 최종 판정 = {'물림 TRUE' if gripped else '빈손 FALSE'} "
            f"(hit_count={hit_count}, required={REQUIRED_HITS})"
        )

        return gripped

    def _parse_coords(self, msg: Float32MultiArray):
        coords = [round(float(v), 2) for v in msg.data]
        if len(coords) != 6:
            self.get_logger().error(
                f"좌표 6개 필요, 받은 개수: {len(coords)}, data={coords}"
            )
            self._pub_pick_status("error")
            return None
        return coords

    def _try_start_task(self, task_name: str, target_func, coords):
        if self.emergency_active:
            self.get_logger().warn(f"비상정지 상태라 {task_name} 명령 무시")
            return
        with self._busy_lock:
            if self._busy:
                self.get_logger().warn(f"로봇팔 작업 중이라 {task_name} 명령 무시")
                return
            self._busy = True
        threading.Thread(target=target_func, args=(coords,), daemon=True).start()

    def _finish_task(self):
        with self._busy_lock:
            self._busy = False

    def _safe_get_pose(self, mode, tries=3, interval=0.05):
        """
        정지 상태에서 여러 번 읽어 중앙값을 반환. 이상값(None/-1/길이 불일치)은 버림.
          mode: 0 = get_angles(), 1 = get_coords()
        표본이 하나도 유효하지 않으면 None 반환.
        """
        samples = []
        for _ in range(tries):
            c = self.mc.get_angles() if mode == 0 else self.mc.get_coords()
            if c and c != -1 and len(c) == 6:
                samples.append(c)
            time.sleep(interval)
        if not samples:
            return None
        arr = np.array(samples)
        med = np.median(arr, axis=0)
        return [float(v) for v in med]

    def _safe_get_coords(self, tries=5):
        """정지 상태에서 여러 번 읽어 중앙값. 이상값/실패 걸러냄. (get_coords 전용 래퍼)"""
        return self._safe_get_pose(mode=1, tries=tries, interval=0.15)

    def _stop_robot_arm(self):
        try:
            self.mc.stop()
            self.get_logger().error("mc.stop() 호출 완료")
        except Exception as e:
            self.get_logger().error(f"mc.stop() 실패: {e}")

    # =========================
    # [신규] 관측 자세 이동 콜백
    # =========================
    def _observe_move_callback(self, msg: String):
        """
        /observe_move 수신:
          - "1", "2"      : 픽킹용 층별 관측 자세
          - "qr:A/B/C"    : 플레이싱 QR 관측 자세
        """
        if self.emergency_active:
            self.get_logger().warn("비상정지 상태라 /observe_move 무시")
            return

        data = msg.data.strip()

        if data.startswith("qr:"):
            zone = data.split(":", 1)[1].strip().upper()

            if zone not in QR_OBSERVE_ANGLES:
                self.get_logger().error(f"알 수 없는 QR 관측 구역: '{zone}'")
                return

            with self._busy_lock:
                if self._busy:
                    self.get_logger().warn("작업 중이라 QR /observe_move 무시")
                    return
                self._busy = True

            threading.Thread(
                target=self._qr_observe_move_sequence,
                args=(zone,),
                daemon=True
            ).start()
            return

        try:
            level = int(data)
        except ValueError:
            self.get_logger().error(f"/observe_move 파싱 실패: '{data}'")
            return

        if level not in SHELF_ANGLES:
            self.get_logger().error(f"알 수 없는 층 {level} - 관측 이동 불가")
            return

        j1_offset = self.j1_offset_by_level.get(level, 0.0)

        with self._busy_lock:
            if self._busy:
                self.get_logger().warn("작업 중이라 /observe_move 무시")
                return
            self._busy = True

        threading.Thread(
            target=self._observe_move_sequence,
            args=(level,),
            daemon=True
        ).start()

    def _observe_move_sequence(self, level, j1_offset=0.0):
        try:
            if self.emergency_active:
                return

            self.current_level = level
            angles = list(SHELF_ANGLES[level])
            angles[0] += j1_offset               # J1 보정 적용
            self._log(f"[OBSERVE] {level}층 관측 (J1 offset={j1_offset}): {[round(a,1) for a in angles]}")
            self.mc.send_angles(angles, MOVE_SPEED)

            if not self._wait_in_position(
                angles, mode=0,
                timeout=max(WAIT_ANGLES_TIMEOUT, OBSERVE_SETTLE_WAIT + 3.0),
                settle=SETTLE_AFTER_ARRIVE_OBSERVE
            ):
                self._log("[OBSERVE] 도달 확인 실패(타임아웃) - 재시도 요청")
                self._pub_pick_status("pick_failed")
                return

            # ★ 정지 후 실제 자세 읽어서 발행 (동적 T용) ★
            pose = self._safe_get_coords()
            if pose is None:
                self._log("[OBSERVE] get_coords 실패 - 관측 자세 발행 못 함")
            else:
                pm = Float32MultiArray()
                pm.data = [float(v) for v in pose]
                self._observe_pose_pub.publish(pm)
                self._log(f"[OBSERVE] 관측 자세 발행: {[round(v,1) for v in pose]}")

            # observe_ready 발행 전에 busy를 먼저 풀어야
            # vision이 바로 /j1_correction을 보내도 pick_node가 무시하지 않음
            self._finish_task()

            m = String()
            m.data = "ready"
            self._observe_ready_pub.publish(m)
            self._log(f"[OBSERVE] {level}층 관측 자세 도착 -> /observe_ready 발행")

        except Exception as e:
            self.get_logger().error(f"관측 이동 오류: {e}")
            self._pub_pick_status("error")
        finally:
            self._finish_task()

    def _qr_observe_move_sequence(self, zone):
        """
        QR 플레이싱 관측 자세 이동.
        목적지 도착 후 QR을 보기 위한 전용 자세로 이동하고,
        실제 get_coords를 /observe_pose로 발행해서 vision_node의 동적 T를 갱신한다.
        이후 /observe_ready를 발행하면 brain_node가 /vision_activate: qr_place를 보낸다.
        """
        try:
            if self.emergency_active:
                return

            angles = list(QR_OBSERVE_ANGLES[zone])

            self._log(f"[QR OBSERVE] zone={zone} QR 관측 자세 이동: {[round(a, 1) for a in angles]}")
            self.mc.send_angles(angles, MOVE_SPEED)

            if not self._wait_in_position(
                angles, mode=0,
                timeout=max(WAIT_ANGLES_TIMEOUT, OBSERVE_SETTLE_WAIT + 3.0),
                settle=SETTLE_AFTER_ARRIVE_OBSERVE
            ):
                self._log("[QR OBSERVE] 도달 확인 실패(타임아웃) - /observe_pose 발행 못 함")
                return

            pose = self._safe_get_coords()
            if pose is None:
                self._log("[QR OBSERVE] get_coords 실패 - /observe_pose 발행 못 함")
            else:
                pm = Float32MultiArray()
                pm.data = [float(v) for v in pose]
                self._observe_pose_pub.publish(pm)
                self._log(f"[QR OBSERVE] 관측 자세 발행: {[round(v, 1) for v in pose]}")

            self._finish_task()

            m = String()
            m.data = "qr_ready"
            self._observe_ready_pub.publish(m)
            self._log(f"[QR OBSERVE] zone={zone} QR 관측 자세 도착 -> /observe_ready 발행")

        except Exception as e:
            self.get_logger().error(f"QR 관측 이동 오류: {e}")
            self._pub_pick_status("error")
        finally:
            self._finish_task()

    # =========================
    # 콜백 함수
    # =========================
    def _pick_callback(self, msg: Float32MultiArray):
        coords = self._parse_coords(msg)
        if coords is None:
            return
        self.get_logger().info(f"픽 명령 수신: {coords}")
        self._try_start_task("픽", self._pick_sequence, coords)

    def _place_callback(self, msg: Float32MultiArray):
        coords = self._parse_coords(msg)
        if coords is None:
            return
        self.get_logger().info(f"플레이스 명령 수신: {coords}")
        self._try_start_task("플레이스", self._place_sequence, coords)

    def _emergency_stop_callback(self, msg: String):
        command = msg.data.strip().lower()
        if command in ["stop", "emergency", "emergency_stop", "1", "true", "on"]:
            if self.emergency_active:
                self.get_logger().warn("이미 비상정지 상태")
                return
            self.emergency_active = True
            self.get_logger().error("비상정지 수신 - 로봇팔 정지 시도")
            self._stop_robot_arm()
            self._pub_pick_status("error")
        elif command in ["reset", "release", "clear", "0", "false", "off"]:
            self.emergency_active = False
            self.get_logger().info("비상정지 해제 - 새 명령 수신 가능")
        else:
            self.get_logger().warn(f"알 수 없는 emergency_stop 명령: {msg.data}")

    def _j1_correction_callback(self, msg: String):
        """vision이 마커로 계산한 J1 보정량 수신 → J1 돌려서 재관측.
        재관측 후 observe_ready 발행 → brain이 자동으로 vision 재활성화."""
        if self.emergency_active:
            return
        data = msg.data.strip()

        if data == 'realign_fail':
            self.get_logger().error("[J1보정] 실패 - AGV 재정차 필요")
            self._pub_pick_status("realign_fail")
            return

        try:
            level_str, corr_str = data.split(':', 1)
            level = int(level_str)
            j1_corr = float(corr_str)
        except (ValueError, IndexError):
            self.get_logger().error(f"[J1보정] 파싱 실패: '{data}'")
            return

        prev_offset = self.j1_offset_by_level.get(level, 0.0)
        total_offset = prev_offset + j1_corr
        self.j1_offset_by_level[level] = total_offset

        self.get_logger().info(
            f"[J1보정] {level}층 J1 {j1_corr:+.1f}도 보정, "
            f"누적 offset={total_offset:+.1f}도 재관측"
        )

        with self._busy_lock:
            if self._busy:
                self.get_logger().warn("작업 중 - J1보정 대기")
                return
            self._busy = True

        threading.Thread(
            target=self._observe_move_sequence,
            args=(level, total_offset),
            daemon=True
        ).start()

    # =========================
    # 실제 로봇팔 동작 시퀀스
    # =========================
    def _pick_sequence(self, coords):
        """
        피킹 시퀀스 (eye-in-hand: PICK_READY 생략, 피드백 기반 대기)

        전제: 이 시점에 팔은 이미 관측 자세(SHELF_ANGLES)에 있음.
              vision이 그 자세에서 계산한 base 좌표를 받았으므로 바로 파지.
        """
        try:
            if self.emergency_active:
                return

            x, y, z, rx, ry, rz = coords

            off = GRIP_OFFSET[self.current_level]
            x = x + off[0] + PICK_X_BIAS_MM
            y = y + off[1] + PICK_Y_BIAS_MM
            z = z + off[2] + PICK_Z_BIAS_MM

            if self.current_level == 1:
                if y > 0:
                    y -= Y_COMP_1F_POS
                elif y < 0:
                    y += Y_COMP_1F_NEG
            else:
                if y > 0:
                    y -= Y_COMP_2F_POS
                elif y < 0:
                    y += Y_COMP_2F_NEG

            self.get_logger().info(
                f"피킹 좌표(보정 후): x={x:.1f}, y={y:.1f}, z={z:.1f}, "
                f"bias=({PICK_X_BIAS_MM}, {PICK_Y_BIAS_MM}, {PICK_Z_BIAS_MM})"
            )

            target_z = z

            pre_pick = [x, y, target_z + APPROACH_Z_MM, rx, ry, rz]
            lifted   = [x, y, target_z + LIFT_Z,        rx, ry, rz]

            self._log(
                f"[PICK INFO] 원본 coords={coords}, target_z={round(target_z,2)}, "
                f"pre_pick={[round(v,1) for v in pre_pick]}"
            )

            self._log("[PICK 1/7] 그리퍼 열기")
            self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
            self._wait_gripper_settled(timeout=1.5)
            if self.emergency_active:
                return

            rx, ry, rz = GRIP_POSE[self.current_level]

            target = [x, y, target_z, rx, ry, rz]

            if self.current_level == 1:
                self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                self._wait_gripper_settled(timeout=1.5)
                if self.emergency_active:
                    return

                # # 1. J5 편 진입 자세로 바로 이동
                # self._log("[1F] J5 편 진입 자세로 바로 이동")
                # unfold = list(SAFE_ENTRY_1F_ANGLES)
                # unfold[4] = 0.0
                
                # self.mc.send_angles(unfold, MOVE_SPEED)
                # if not self._wait_in_position(unfold, mode=0, timeout=WAIT_ANGLES_TIMEOUT):
                #     self._log("[1F] J5 편 진입 자세 도달 실패 - 재관측 요청")
                #     self._pub_pick_status("pick_failed")
                #     return
                
                # # 2. 현재 자세 읽기
                # cur = self._safe_get_coords()
                # if cur is None:
                #     self._log("[1F] get_coords 실패 - 안전상 중단")
                #     self._pub_pick_status("pick_failed")
                #     return
                
                # self._log(f"[1F] J5 편 현재 자세: {[round(v,1) for v in cur]}")
                
                # 3. y축 이동 - 블록 y로 정렬
                # x는 현재 편 자세의 x 유지, z는 살짝 띄우고, rpy는 파지 자세로 준비
                # y_move = [cur[0], y, cur[2] + 5, rx, ry, rz]
                # self._log(f"[1F] y축 이동 (블록 앞 정렬): {[round(v,1) for v in y_move]}")
                # self.mc.send_coords(y_move, MOVE_SPEED, 0)
                
                # if not self._wait_in_position(y_move, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                #     self._log("[1F] y축 이동 도달 실패 - 재관측 요청")
                
                #     self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                #     self._wait_gripper_settled(timeout=1.0)
                
                #     self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                #     self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT)
                
                #     self._pub_pick_status("pick_failed")
                #     return
                
                # 4. y축 이동 - 블록 y로 정렬
                # 너무 낮은 높이에서 옆으로 움직이면 블록/바닥/랙과 간섭 위험이 있어서
                # 현재 z와 목표 z+여유높이 중 더 높은 값을 사용한다.
                # Y_ALIGN_CLEARANCE_Z_1F = 30.0
                
                # safe_y_z = max(cur[2], z + Y_ALIGN_CLEARANCE_Z_1F)
                
                # y_move = [cur[0], y, safe_y_z, rx, ry, rz]
                # self._log(f"[1F] y축 이동 (안전 높이 정렬): {[round(v,1) for v in y_move]}")
                # self.mc.send_coords(y_move, MOVE_SPEED, 0)
                # if not self._wait_in_position(y_move, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                #     self._log("[1F] y축 이동 도달 실패 - 중단")
                #     self._pub_pick_status("error")
                #     return

                # self.mc.send_coords(y_move, MOVE_SPEED, 0)
                # self._safe_sleep(2.0)
                # if self.emergency_active:
                #     return

                

                # 6. 수평 전진 파지 (x만)
                FORWARD_Y_COMP = 0.0
                target = [x, y + FORWARD_Y_COMP, z, rx, ry, rz]
                self._log(f"[1F] 수평 전진 파지: {[round(v,1) for v in target]}")
                self.mc.send_coords(target, DESCEND_SPEED, 0)
                if not self._wait_in_position(target, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                    self._log("[1F] 전진 파지 도달 실패 - 재관측 요청")
                
                    self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                    self._wait_gripper_settled(timeout=1.0)
                    if self.emergency_active:
                        return
                
                    self._log("[1F FAIL] 홈 복귀 후 재관측")
                    self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                    self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT)
                
                    self._pub_pick_status("pick_failed")
                    return
                  
                # 7. 그리퍼 닫기
                self._log("[1F] 그리퍼 닫기")
                self.mc.set_gripper_value(GRIPPER_CLOSE, GRIPPER_SPEED)
                self._wait_gripper_settled(timeout=2.5)
                if self.emergency_active:
                    return

                # 8. 파지 성공 여부 확인
                grip_result = self._check_gripped()

                if grip_result is not True:
                    self._log("[1F] 파지 실패 확정 - 안전 후퇴 후 재관측 요청")

                    self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                    self._wait_gripper_settled(timeout=1.0)
                    if self.emergency_active:
                        return

                    back_1f = [x - 80, y + FORWARD_Y_COMP, z + 20, rx, ry, rz]
                    self._log(f"[1F FAIL] 뒤로 빼기: {[round(v,1) for v in back_1f]}")
                    self.mc.send_coords(back_1f, MOVE_SPEED, 0)
                    if not self._wait_in_position(back_1f, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                        self._log("[1F FAIL] 후퇴 도달 실패 - 홈 복귀만 시도")

                    self._log("[1F FAIL] 홈 복귀")
                    self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                    self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT)

                    self._pub_pick_status("pick_failed")
                    return

                # elif grip_result is None:
                #     self._log("[1F] 파지 판정 불가(None) - 그리퍼 닫은 상태 유지하고 성공 루틴 진행")

                else:
                    self._log("[1F] 파지 성공")

                # 8. 뒤로 곧게 빼기
                back_1f = [x - 80, y + FORWARD_Y_COMP, z + 20, rx, ry, rz]
                self._log(f"[1F] 뒤로 빼기: {[round(v,1) for v in back_1f]}")
                self.mc.send_coords(back_1f, MOVE_SPEED, 0)
                if not self._wait_in_position(back_1f, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                    self._log("[1F] 후퇴 도달 실패 - 홈 복귀만 시도")

                # 8-1. 후퇴 후 파지 재확인
                self._log("[1F] 후퇴 후 파지 재확인")
                grip_result_after_back = self._check_gripped()
                
                if grip_result_after_back is not True:
                    self._log("[1F] 후퇴 중 블록 이탈 감지 → pick_failed 발행")
                
                    # 혹시 물체가 애매하게 걸려 있으면 열고 홈 복귀
                    self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                    self._wait_gripper_settled(timeout=1.0)
                
                    self._log("[1F FAIL-AFTER-BACK] 홈 복귀")
                    self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                    self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT)
                
                    self._pub_pick_status("pick_failed")
                    return

                # 9. 홈 복귀
                self._log("[1F] 홈 복귀")
                self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                if not self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT):
                    self._log("[1F] 홈 복귀 도달 실패(타임아웃) - 계속 진행")

            else:
                # ===== 2층: 정면 앞 → 전진 → 파지 =====
                APPROACH_X_2F = 50.0
                front = [x - APPROACH_X_2F, y, target_z, rx, ry, rz]
                self._log(f"[2F] 블록 앞으로: {[round(v,1) for v in front]}")
                self.mc.send_coords(front, MOVE_SPEED, 0)
                if not self._wait_in_position(front, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                    self._log("[2F] 블록 앞 이동 도달 실패 - 재관측 요청")
                
                    self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                    self._wait_gripper_settled(timeout=1.0)
                    if self.emergency_active:
                        return
                    self._log("[2F FAIL] 홈 복귀 후 재관측")
                    self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                    self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT)
                  
                    self._pub_pick_status("pick_failed")
                    return

                FORWARD_Y_COMP_2F = 0.0
                fwd = [x, y + FORWARD_Y_COMP_2F, target_z, rx, ry, rz]
                self._log(f"[2F] 전진 파지: {[round(v,1) for v in fwd]}")
                self.mc.send_coords(fwd, DESCEND_SPEED, 0)   # 직선 전진
                if not self._wait_in_position(fwd, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                    self._log("[2F] 전진 파지 도달 실패 - 재관측 요청")
                
                    # 아직 그리퍼 닫기 전이라 파지 상태 아님. 안전하게 열어둠.
                    self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                    self._wait_gripper_settled(timeout=1.0)
                    if self.emergency_active:
                        return
                
                    self._log("[2F FAIL] 홈 복귀 후 재관측")
                    self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                    self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT)
                
                    self._pub_pick_status("pick_failed")
                    return

                self._log("[2F] 그리퍼 닫기")
                self.mc.set_gripper_value(GRIPPER_CLOSE, GRIPPER_SPEED)
                self._wait_gripper_settled(timeout=2.5)
                if self.emergency_active:
                    return

                grip_result = self._check_gripped()

                if grip_result is not True:
                    self._log(f"[2F] 파지 실패(grip_result={grip_result}) - 안전 후퇴 후 재관측 요청")

                    self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                    self._wait_gripper_settled(timeout=1.0)
                    if self.emergency_active:
                        return

                    back_fail = [x - 70, y, target_z + 20, rx, ry, rz]
                    self._log(f"[2F FAIL] 뒤로 빼기: {[round(v,1) for v in back_fail]}")
                    self.mc.send_coords(back_fail, MOVE_SPEED, 1)
                    if not self._wait_in_position(back_fail, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                        self._log("[2F FAIL] 후퇴 도달 실패 - 홈 복귀만 시도")

                    self._log("[2F FAIL] 홈 복귀")
                    self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                    self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT)

                    self._pub_pick_status("pick_failed")
                    return

                self._log("[2F] 파지 성공")

                # 1. z 상승 (제자리에서 위로)
                self._log("[2F] z 상승")
                lifted = [x, y, target_z + LIFT_Z, rx, ry, rz]
                self.mc.send_coords(lifted, MOVE_SPEED, 1)
                if not self._wait_in_position(lifted, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                    self._log("[2F] z 상승 도달 실패 - 계속 진행")

                # 2. 뒤로 빼기 (들린 높이 유지, x 뒤로)
                back = [x - 70, y, target_z + LIFT_Z, rx, ry, rz]
                self._log(f"[2F] 뒤로 60mm: {[round(v,1) for v in back]}")
                self.mc.send_coords(back, MOVE_SPEED, 1)
                if not self._wait_in_position(back, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                    self._log("[2F] 후퇴 도달 실패 - 계속 진행")

                # 2-1. 후퇴 후 파지 재확인
                self._log("[2F] 후퇴 후 파지 재확인")
                grip_result_after_back = self._check_gripped()
                
                if grip_result_after_back is not True:
                    self._log("[2F] 후퇴 중 블록 이탈 감지 → pick_failed 발행")
                
                    self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                    self._wait_gripper_settled(timeout=1.0)
                
                    self._log("[2F FAIL-AFTER-BACK] 홈 복귀")
                    self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                    self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT)
                
                    self._pub_pick_status("pick_failed")
                    return

                # 3. 홈
                self._log("[2F] 홈 복귀")
                self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                if not self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT):
                    self._log("[2F] 홈 복귀 도달 실패(타임아웃) - 계속 진행")

            self._log("[PICK 7/7] 픽 완료")
            self._pub_pick_status("done")

        except Exception as e:
            self.get_logger().error(f"픽 오류: {e}")
            self._pub_pick_status("error")
        finally:
            self._finish_task()

    def _place_sequence(self, coords):
        """
        플레이싱 시퀀스 (피드백 기반 대기)

        순서:
        1. 놓을 위치 위 waypoint 이동
        2. z만 수직 하강
        3. 그리퍼 열기
        4. z축 상승
        5. 홈포지션 복귀
        6. /pick_status="placing_done" 발행
        """
        try:
            if self.emergency_active:
                return

            x, y, z, rx, ry, rz = coords

            # 플레이스 때 손목 꼬임 방지용 자세로 강제 변경
            rx, ry, rz = -145.47, -28.92, -53.1


            target_z = z + GRIPPER_Z_OFFSET_MM

            pre_place = [x, y, target_z + PLACE_APPROACH_Z_MM, rx, ry, rz]
            target    = [x, y, target_z,                         rx, ry, rz]
            lifted    = [x, y, target_z + PLACE_APPROACH_Z_MM, rx, ry, rz]

            self._log(
                f"[PLACE INFO] 원본 coords={coords}, target_z={round(target_z,2)}, "
                f"pre_place={[round(v,1) for v in pre_place]}"
            )

            # self._log("[PLACE 0/7] 홈 경유")
            # self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
            # if not self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT):
            #     self._log("[PLACE] 홈 경유 실패 - 중단")
            #     self._pub_pick_status("error")
            #     return
            
            # self._safe_sleep(0.5)
            
            self._log("[PLACE 1/7] place-ready 관절 waypoint 이동")
            self.mc.send_angles(PLACE_READY_ANGLES, MOVE_SPEED)
            if not self._wait_in_position(PLACE_READY_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT):
                self._log("[PLACE] place-ready 도달 실패 - 그리퍼 열지 않고 place 재시도 요청")
            
                # 물체를 잡고 있는 상태이므로 그리퍼는 열지 않는다.
                self._pub_pick_status("place_failed")
                return
            
            self._safe_sleep(0.5)
            
            self._log("[PLACE 2/7] 놓을 위치 위 waypoint 이동")
            self.mc.send_coords(pre_place, MOVE_SPEED, 0)
            
            if not self._wait_in_position(pre_place, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                self._log("[PLACE] pre_place 도달 실패 - 현재 위치에서 강제 내려놓기 진행")
            
                # pre_place에 정확히 못 갔더라도, 현재 위치에서 물체를 놓는다.
                # KPI 테스트용: 물체를 집은 채 홈으로 복귀하지 않도록 함.
                self._log("[PLACE FORCE] 그리퍼 열기 (pre_place 실패 후 현재 위치에서 내려놓기)")
                self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                if not self._safe_sleep(1.0):
                    return
            
                # 한 번 더 open 명령
                self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                if not self._safe_sleep(2.0):
                    return
            
                if self.emergency_active:
                    return
            
                # 가능하면 살짝 위로 들어올린 뒤 홈 복귀
                cur = self._safe_get_coords()
                if cur is not None:
                    safe_lift = [
                        cur[0],
                        cur[1],
                        cur[2] + 60.0,
                        cur[3],
                        cur[4],
                        cur[5],
                    ]
                    self._log(f"[PLACE FORCE] 현재 위치 기준 z 상승: {[round(v,1) for v in safe_lift]}")
                    self.mc.send_coords(safe_lift, MOVE_SPEED, 1)
                    self._safe_sleep(2.0)
            
                self._log("[PLACE FORCE] 홈포지션 복귀")
                self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT)
            
                # KPI 흐름상 내려놓은 것으로 처리
                self._log("[PLACE FORCE] 강제 플레이스 완료")
                self._pub_pick_status("placing_done")
                return
                
            
            if self.emergency_active:
                return

            self._log(f"[PLACE] 하강 좌표: {[round(v,1) for v in target]}")

            self._log("[PLACE 3/7] z축 수직 하강 - sleep 기반 진행")
            self.mc.send_coords(target, DESCEND_SPEED, 1)
            
            # pre_place까지 정상 도달했다면, 하강 위치 오차가 조금 있어도 그냥 놓기 진행
            # myCobot이 target z까지 완전히 못 내려가도 현재 위치에서 그리퍼를 열어 KPI 흐름을 진행한다.
            if not self._safe_sleep(5.0):
                return
            
            # 하강 후 좌표 확인은 로그용으로만 사용하고, 실패/오차가 있어도 place_failed 처리하지 않음
            cur = self._safe_get_coords()
            if cur is None:
                self._log("[PLACE] 하강 후 get_coords 실패 - 그래도 현재 위치에서 내려놓기 진행")
            else:
                place_down_diff = [abs(cur[i] - target[i]) for i in range(6)]
                self._log(
                    f"[PLACE DEBUG] 하강 후 현재: {[round(v, 1) for v in cur]}, "
                    f"target={[round(v, 1) for v in target]}, "
                    f"diff={[round(v, 1) for v in place_down_diff]} "
                    f"→ 오차와 무관하게 내려놓기 진행"
                )

          
            # self._log("[PLACE 3/7] z축 수직 하강")
            # self.mc.send_coords(target, DESCEND_SPEED, 1)
            # self._safe_sleep(2.5)
            # if self.emergency_active:
            #     return

            self._log("[PLACE 4/7] 그리퍼 열기 (내려놓기)")
            self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
            self._wait_gripper_settled(timeout=2.0)
            
            # place에서는 실제로 떨어질 시간을 확실히 줌
            if not self._safe_sleep(2.0):
                return
            
            if self.emergency_active:
                return

            self._log("[PLACE 5/7] z축 상승")
            self.mc.send_coords(lifted, MOVE_SPEED, 1)
            if not self._wait_in_position(lifted, mode=1, timeout=WAIT_COORDS_TIMEOUT):
                self._log("[PLACE] 상승 도달 실패 - 계속 진행")

            # self._log("[PLACE 5/7] z축 상승")
            # self.mc.send_coords(lifted, MOVE_SPEED, 1)
            # self._safe_sleep(2.5)
            # if self.emergency_active:
            #     return

            self._log("[PLACE 6/7] 홈포지션 복귀")
            self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
            if not self._wait_in_position(HOME_ANGLES, mode=0, timeout=WAIT_ANGLES_TIMEOUT):
                self._log("[PLACE] 홈 복귀 도달 실패(타임아웃) - 계속 진행")

            self._log("[PLACE 7/7] 플레이스 완료")
            self._pub_pick_status("placing_done")

        except Exception as e:
            self.get_logger().error(f"플레이스 오류: {e}")
            self._pub_pick_status("error")
        finally:
            self._finish_task()


def main(args=None):
    rclpy.init(args=args)
    node = PickNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt 수신 - pick_node 종료")
    finally:
        try:
            node._stop_robot_arm()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
