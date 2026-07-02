#!/usr/bin/env python3
"""
pick_node.py  (eye-in-hand 버전)

[eye-in-hand 변경 요약]
  (1) 관측 자세 이동 처리 추가:
      /observe_move (층 번호) 수신 -> send_angles(SHELF_ANGLES[level]) ->
      정착 후 /observe_ready 발행. (카메라가 그리퍼에 붙어서 vision 계산 전에
      팔이 정확한 관측 자세에 있어야 함. IK 복수해 회피 위해 반드시 send_angles 사용)
  (2) 파지 시퀀스에서 PICK_READY_ANGLES 단계 삭제:
      이미 관측 자세에 있으므로 바로 물체 위 waypoint -> step 하강.
  (3) 미세보정(x/y/z bias, y비례보정, ry기울기) 전부 0으로 초기화:
      eye-to-hand 시절 calib 오차 땜빵값이라, 새 eye-in-hand calib에선 무효.
      실측하면서 0에서부터 다시 잡을 것.
  (4) GRIPPER_Z_OFFSET_MM: 카메라 위치가 바뀌었으므로 재측정 필요(주석 참고).

토픽:
  구독: /pick_command, /place_command, /emergency_stop
        /observe_move (String) : "1"/"2" 관측할 층          [신규]
  발행: /pick_status, /arm/status
        /observe_ready (String) : "ready" 관측 자세 도착     [신규]
"""

import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray

try:
    from pymycobot import MyCobot
except ImportError:
    raise SystemExit("pymycobot not installed.")


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
# [신규] 층별 관측 자세 (angles / coords 짝)
# =========================
# SHELF_ANGLES : 실제 이동은 이 관절각으로 (send_angles) -> IK 복수해 없이 항상 같은 자세 도달
# SHELF_POSES  : vision_node가 좌표변환에 쓰는 값 (참고용, pick_node는 angles만 사용)
#   두 값은 "같은 자세의 짝"이므로 vision_node의 SHELF_POSES와 항상 세트로 유지할 것.
#   (send_angles + sleep 후 안정 도달 실측값으로 반복도 검증 완료)
SHELF_ANGLES = {
    1: [1.23, 115.13, -136.31, -4.13, 2.9, -40.78],   # 1층 (랙 16.5cm)
    2: [-1.84, 102.12, -113.11, 12.91, 5.0, -41.39],  # 2층 (랙 12.5cm)
}

# 1층 접은 진입 자세 (J5 돌려서 랙 회피). 진입/탈출 공용.
SAFE_ENTRY_1F_ANGLES = [8.17, -27.94, -129.37, 126.38, 54.31, -45.87]

# 관측 자세 이동 후 정착 대기(초). 반복도 테스트에서 4초로 안정 확인.
OBSERVE_SETTLE_WAIT = 4.0

# 그리퍼 값
GRIPPER_OPEN = 100
GRIPPER_CLOSE = 30


# =========================
# 피킹 보정값
# =========================
# vision_node에서 받은 z는 "물체 위치", send_coords의 z는 그리퍼 끝이 물체에 닿도록
# 플랜지/툴 기준으로 위로 보정해야 함.
#
# [주의] 아래 133.0은 eye-to-hand 시절 값. 카메라를 그리퍼로 옮겼으므로
#        eye-in-hand에서 반드시 재측정할 것.
#        측정법: bias 전부 0인 상태에서 블록 하나 파지 시도 ->
#                그리퍼 끝이 블록보다 얼마나 높이/낮게 멈추는지 보고 조정.
#GRIPPER_Z_OFFSET_MM = .0   # TODO: eye-in-hand 재측정

# 기울어진 파지 자세에서 flange↔그리퍼끝 offset (3축 다)
# 자세각 [-102.25, -38.21, -82.48] 기준 실측
# =========================
# 파지 offset (3축) - 기울어진 그리퍼라 x,y,z 다 필요
# 블록좌표 → flange 목표. 자세각 [-102.25,-38.21,-82.48]에서 실측.
# =========================
# 층별 파지 offset (자세가 달라서 offset도 다름)
GRIP_OFFSET = {
    1: [-54.4, -2.0, 50.0],    # 1층 (현재 값, 정확)
    2: [-54.4, -2.0, 50.0],      # 2층 z만 조정 (위에서 잡히니 낮춰야)
}

# 물체 바로 위 waypoint 높이
APPROACH_Z_MM = 10.0

# =========================
# [변경] 실제 피킹 미세 보정값 — 전부 0으로 초기화
# =========================
# 아래 값들은 이전 eye-to-hand calib의 오차를 땜빵하던 값이었음.
# 새 eye-in-hand calib 기준으로는 0에서 시작해서 실측으로 다시 잡아야 함.
# 순수 calib 정확도부터 확인(vision arm_xyz 로그 vs 실제 블록 위치)한 뒤,
# 빗나가는 만큼만 아래를 채울 것.
PICK_X_BIAS_MM = 13.0     # 이전: 12.0
PICK_Y_BIAS_MM = 0.0     # 이전: -26.0
PICK_Z_BIAS_MM = 0.0     # 이전: -10.0

# 집은 뒤 위로 들어올릴 높이
LIFT_Z = 40.0

# 내려갈 때 속도는 천천히
DESCEND_SPEED = 8


class PickNode(Node):
    def __init__(self):
        super().__init__("pick_node")

        # 내부 상태값
        self._busy = False
        self._busy_lock = threading.Lock()
        self.emergency_active = False
        self.current_level = 2   # 현재 관측/파지 중인 층 (observe_move로 갱신)

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
        # ===== [신규] 관측 자세 이동 명령 =====
        self.create_subscription(
            String, "/observe_move", self._observe_move_callback, 10
        )

        # 발행자
        self._pick_status_pub = self.create_publisher(String, "/pick_status", 10)
        self._status_pub = self.create_publisher(String, "/arm/status", 10)
        # ===== [신규] 관측 자세 도착 신호 =====
        self._observe_ready_pub = self.create_publisher(String, "/observe_ready", 10)

        # myCobot 연결
        self.get_logger().info("myCobot 연결 시도 중...")
        self.mc = MyCobot(SERIAL_PORT, BAUD)
        time.sleep(0.5)

        # 시작 시 홈 포지션 이동
        self.get_logger().info("홈포지션으로 이동 중...")
        try:
            self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
            time.sleep(4.0)
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

    def _safe_sleep(self, seconds: float, step: float = 0.1) -> bool:
        elapsed = 0.0
        while elapsed < seconds:
            if self.emergency_active:
                self.get_logger().warn("비상정지 감지 - 현재 시퀀스 중단")
                return False
            time.sleep(step)
            elapsed += step
        return True

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
        /observe_move 수신: brain이 "관측할 층"을 보내면
        해당 층 관측 자세(SHELF_ANGLES)로 send_angles 이동 후
        /observe_ready 발행.
        - send_coords가 아니라 send_angles를 쓰는 이유: IK 복수해 때문에
          send_coords는 같은 목표에 대해 자세가 갈려서 카메라 위치가 불안정해짐.
        """
        if self.emergency_active:
            self.get_logger().warn("비상정지 상태라 /observe_move 무시")
            return

        level_str = msg.data.strip()
        try:
            level = int(level_str)
        except ValueError:
            self.get_logger().error(f"/observe_move 층 파싱 실패: '{level_str}'")
            return

        if level not in SHELF_ANGLES:
            self.get_logger().error(f"알 수 없는 층 {level} - 관측 이동 불가")
            return

        # 관측 이동도 하나의 작업이므로 busy 처리 (파지와 중복 방지)
        with self._busy_lock:
            if self._busy:
                self.get_logger().warn("작업 중이라 /observe_move 무시")
                return
            self._busy = True

        threading.Thread(
            target=self._observe_move_sequence, args=(level,), daemon=True
        ).start()

    def _observe_move_sequence(self, level):
        try:
            if self.emergency_active:
                return
              
            self.current_level = level
            angles = SHELF_ANGLES[level]
            self._log(f"[OBSERVE] {level}층 관측 자세로 이동: angles={angles}")
            self.mc.send_angles(angles, MOVE_SPEED)
            if not self._safe_sleep(OBSERVE_SETTLE_WAIT):
                return

            # 도착 신호
            m = String()
            m.data = "ready"
            self._observe_ready_pub.publish(m)
            self._log(f"[OBSERVE] {level}층 관측 자세 도착 -> /observe_ready 발행")

        except Exception as e:
            self.get_logger().error(f"관측 이동 오류: {e}")
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

    # =========================
    # 실제 로봇팔 동작 시퀀스
    # =========================
    def _pick_sequence(self, coords):
        """
        [변경] 피킹 시퀀스 (eye-in-hand: PICK_READY 생략)

        전제: 이 시점에 팔은 이미 관측 자세(SHELF_ANGLES)에 있음.
              vision이 그 자세에서 계산한 base 좌표를 받았으므로 바로 파지.

        순서:
        1. 그리퍼 열기
        2. 물체 위 waypoint 이동 (관측 자세에서 바로)
        3. z축 step 수직 하강 (brain이 준 x,y 그대로 사용)
        4. 그리퍼 닫기
        5. z축 상승
        6. 홈포지션 복귀
        7. /pick_status="done" 발행
        """
        try:
            if self.emergency_active:
                return

            x, y, z, rx, ry, rz = coords

            # ===== [변경] 미세보정 — 전부 0 초기화 상태로 단순 적용 =====
            # (기존의 y비례보정 x += abs(y)*0.15, ry += 18 등은 eye-to-hand 땜빵이라 제거)
            # 변경 (층별 offset)
            off = GRIP_OFFSET[self.current_level]
            x = x + off[0] + PICK_X_BIAS_MM
            y = y + off[1] + PICK_Y_BIAS_MM
            z = z + off[2] + PICK_Z_BIAS_MM

            # y 비례보정 (정면 x는 정확, y만 비례 오차). 계수는 튜닝.
            # y < 0(오른쪽)일 때만 비례보정. 정면~왼쪽은 정확해서 건드리지 않음.
            if y > 0:
                y -= y * 0.15
                #x += abs(y) * 0.1     # x는 y 크기에 비례
            elif y < 0:
                y += y * 0.1

            self.get_logger().info(
                f"피킹 좌표(보정 후): x={x:.1f}, y={y:.1f}, z={z:.1f}, "
                f"bias=({PICK_X_BIAS_MM}, {PICK_Y_BIAS_MM}, {PICK_Z_BIAS_MM})"
            )

            if self.current_level == 1:
                ry = ry + 30    # 그리퍼 수그리기 (부호/값 실측)
          
            # 그리퍼 끝이 물체에 닿도록 플랜지 기준 z 보정
            target_z = z

            pre_pick = [x, y, target_z + APPROACH_Z_MM, rx, ry, rz]
            lifted   = [x, y, target_z + LIFT_Z,        rx, ry, rz]

            self._log(
                f"[PICK INFO] 원본 coords={coords}, target_z={round(target_z,2)}, "
                f"pre_pick={[round(v,1) for v in pre_pick]}"
            )

            self._log("[PICK 1/7] 그리퍼 열기")
            self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
            if not self._safe_sleep(1.5):
                return

            # ===== [변경] PICK_READY 단계 삭제. 관측 자세에서 바로 물체 위로 =====
            # self._log("[PICK 2/7] 물체 위 waypoint 이동")
            # self.mc.send_coords(pre_pick, MOVE_SPEED, 0)
            # if not self._safe_sleep(7.0):
            #     return

            # self._log("[PICK 3/7] z축 step 수직 하강 (받은 x,y 그대로)")
            # # get_coords 미사용: myCobot 간헐 오값으로 인한 경로 튐 방지
            # end_z = target_z
            # start_z = target_z + APPROACH_Z_MM
            # step_mm = 20.0
            # z_now = start_z
            # while z_now > end_z:
            #     z_now = max(z_now - step_mm, end_z)
            #     step_target = [x, y, z_now, rx, ry, rz]
            #     self._log(f"[DESCEND STEP] {[round(v,1) for v in step_target]}")
            #     self.mc.send_coords(step_target, DESCEND_SPEED, 1)
            #     if not self._safe_sleep(1.0):
            #         return

            # ===== 관측 자세에서 파지 위치로 바로 (mode 0) =====
            # offset이 반영된 flange 목표 = 블록 딱 잡는 위치. 기울어진 자세라 z 하강 안 함.
            target = [x, y, target_z, rx, ry, rz]

            if self.current_level == 1:
                self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
                if not self._safe_sleep(1.5):
                    return

                # 1. 접은 진입 (랙 회피)
                self._log("[1F] 접은 진입 자세")
                self.mc.send_angles(SAFE_ENTRY_1F_ANGLES, MOVE_SPEED)
                if not self._safe_sleep(4.0):
                    return

                # 2. J5 펴기 (접힘 해제 → 파지 좋은 관절 상태)
                self._log("[1F] J5 펴기")
                unfold = list(SAFE_ENTRY_1F_ANGLES)
                unfold[4] = 0        # ← 대충 편 값. 안 되면 조정
                self.mc.send_angles(unfold, MOVE_SPEED)
                if not self._safe_sleep(3.0):
                    return

                # 3. 블록으로 (vision 자세각으로 최종 정렬)
                self._log(f"[1F] 블록으로: {[round(v,1) for v in target]}")
                self.mc.send_coords(target, MOVE_SPEED, 0)
                if not self._safe_sleep(6.0):
                    return

                # 4. 닫기
                self._log("[1F] 그리퍼 닫기")
                self.mc.set_gripper_value(GRIPPER_CLOSE, GRIPPER_SPEED)
                if not self._safe_sleep(2.5):
                    return

                # 5. 접어서 탈출
                self._log("[1F] 접어서 탈출")
                self.mc.send_angles(SAFE_ENTRY_1F_ANGLES, MOVE_SPEED)
                if not self._safe_sleep(4.0):
                    return

                self._log("[1F] 홈 복귀")
                self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                if not self._safe_sleep(4.0):
                    return

            else:
                # ===== 2층: 기존 로직 (바로 파지) =====
                self._log(f"[2F] 파지 위치로 바로 이동: {[round(v,1) for v in target]}")
                self.mc.send_coords(target, MOVE_SPEED, 0)
                if not self._safe_sleep(7.0):
                    return

                self._log("[2F] 그리퍼 닫기")
                self.mc.set_gripper_value(GRIPPER_CLOSE, GRIPPER_SPEED)
                if not self._safe_sleep(2.5):
                    return

                self._log("[2F] z축 상승")
                self.mc.send_coords(lifted, MOVE_SPEED, 1)
                if not self._safe_sleep(3.0):
                    return

                self._log("[2F] 홈포지션 복귀")
                self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
                if not self._safe_sleep(4.0):
                    return

            self._log("[PICK 7/7] 픽 완료")
            self._pub_pick_status("done")

        except Exception as e:
            self.get_logger().error(f"픽 오류: {e}")
            self._pub_pick_status("error")
        finally:
            self._finish_task()

    def _place_sequence(self, coords):
        """
        플레이싱 시퀀스
        (place는 관측과 무관하므로 기존 구조 유지. 준비자세는 홈 경유로 단순화)

        순서:
        1. 놓을 위치 위 waypoint 이동
        2. get_coords로 실제 자세 읽어 z만 수직 하강
        3. 그리퍼 열기
        4. z축 상승
        5. 홈포지션 복귀
        6. /pick_status="placing_done" 발행
        """
        try:
            if self.emergency_active:
                return

            x, y, z, rx, ry, rz = coords

            # place도 그리퍼 끝 기준 -> 플랜지 기준 z 보정
            target_z = z + GRIPPER_Z_OFFSET_MM

            pre_place = [x, y, target_z + APPROACH_Z_MM, rx, ry, rz]
            target    = [x, y, target_z,                 rx, ry, rz]
            lifted    = [x, y, target_z + LIFT_Z,        rx, ry, rz]

            self._log(
                f"[PLACE INFO] 원본 coords={coords}, target_z={round(target_z,2)}, "
                f"pre_place={[round(v,1) for v in pre_place]}"
            )

            self._log("[PLACE 1/6] 놓을 위치 위 waypoint 이동")
            self.mc.send_coords(pre_place, MOVE_SPEED, 0)
            if not self._safe_sleep(5.0):
                return

            cur = self.mc.get_coords()
            if cur and cur != -1 and len(cur) == 6:
                target = [cur[0], cur[1], target_z, cur[3], cur[4], cur[5]]
                self._log(f"[PLACE] 하강 좌표: {[round(v,1) for v in target]}")
            else:
                self._log(f"[PLACE] get_coords 실패({cur}), 기존 target 사용")

            self._log("[PLACE 2/6] z축 수직 하강")
            self.mc.send_coords(target, DESCEND_SPEED, 1)
            if not self._safe_sleep(4.0):
                return

            self._log("[PLACE 3/6] 그리퍼 열기 (내려놓기)")
            self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
            if not self._safe_sleep(2.0):
                return

            self._log("[PLACE 4/6] z축 상승")
            self.mc.send_coords(lifted, MOVE_SPEED, 1)
            if not self._safe_sleep(3.0):
                return

            self._log("[PLACE 5/6] 홈포지션 복귀")
            self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
            if not self._safe_sleep(4.0):
                return

            self._log("[PLACE 6/6] 플레이스 완료")
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
