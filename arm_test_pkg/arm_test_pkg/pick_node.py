#!/usr/bin/env python3
"""
pick_node.py

역할:
- brain_node로부터 피킹/플레이싱 명령을 받음
- myCobot 280 Pi를 실제로 제어함
- 작업 완료 상태를 /pick_status로 brain_node에 보고함
- /emergency_stop을 받아 로봇팔 비상정지를 처리함

토픽 명세:
1) 구독
   /pick_command    std_msgs/Float32MultiArray
     - brain_node -> pick_node
     - 피킹 명령 좌표
     - data: [x, y, z, rx, ry, rz]

   /place_command   std_msgs/Float32MultiArray
     - brain_node -> pick_node
     - 플레이싱 명령 좌표
     - data: [x, y, z, rx, ry, rz]

   /emergency_stop  std_msgs/String
     - keyboard_estop_node 또는 brain_node -> pick_node
     - data: "stop", "reset"

2) 발행
   /pick_status     std_msgs/String
     - pick_node -> brain_node
     - "done"          : 피킹 완료
     - "placing_done"  : 플레이싱 완료
     - "error"         : 오류 또는 비상정지

   /arm/status      std_msgs/String
     - 디버깅/대시보드용 상태 로그
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
# 실제 환경에서 안전한 자세로 조정 가능
HOME_ANGLES = [-40, 90, -130, -20, 0, 0]

# 블록 피킹 전 준비 자세
# 홈포지션에서 바로 물체 위 waypoint로 가면 관절 이동이 급격할 수 있으므로
# 먼저 이 안전한 중간 자세를 거친 뒤 물체 위로 이동한다.
#
# 중요:
# - 아래 값은 임시 추천값이다.
# - 실제로 팔을 안전한 준비자세로 움직인 뒤 print(mc.get_angles())로 읽어서
#   이 값을 교체하는 것이 가장 좋다.
# - 6번축은 사용자가 확인한 "집게 세로 자세"인 40도로 둔다.
PICK_READY_ANGLES = [0, -40, -60, 20, 0, 40]

# 그리퍼 값
# 사용하는 그리퍼에 따라 열림/닫힘 값은 테스트 후 조정 필요
GRIPPER_OPEN = 100
GRIPPER_CLOSE = 30


# =========================
# 피킹 보정값
# =========================
# vision_node에서 받은 z는 "물체 위치"라고 보고,
# send_coords에 넣을 z는 실제 그리퍼 끝이 물체에 닿도록
# 플랜지/툴 기준 위치를 위로 보정해야 한다.
#
# 사용자가 측정한 두 coords의 z 차이:
# 220.6 - 87.6 = 133.0 mm
GRIPPER_Z_OFFSET_MM = 133.0

# 물체 바로 위 waypoint 높이
# 처음 테스트는 50mm 정도가 적당하다.
# 너무 높으면 40, 30으로 줄여가면 된다.
APPROACH_Z_MM = 100.0

# =========================
# 실제 피킹 미세 보정값
# =========================
# 목표보다 항상 한쪽으로 빗나갈 때 적용하는 보정값
# 단위: mm
PICK_X_BIAS_MM = -20.0

# 로봇 기준 좌우 보정
# mycobot 기준 -는 오른쪽  +는 왼쪽
PICK_Y_BIAS_MM = -40.0

# 더 내려가야 하면 음수
PICK_Z_BIAS_MM = 0

# 집은 뒤 위로 들어올릴 높이
LIFT_Z = 40.0

# 6번축이 40도일 때 집게가 세로로 맞는다고 했으므로 고정값으로 사용
GRIPPER_VERTICAL_J6 = 20.0

# 내려갈 때 속도는 천천히
DESCEND_SPEED = 8

# 6번축만 보정할 때 속도
J6_ALIGN_SPEED = 10

# 피킹 준비자세로 이동할 때 대기 시간
PICK_READY_WAIT = 3.0


class PickNode(Node):
    def __init__(self):
        super().__init__("pick_node")

        # =========================
        # 내부 상태값
        # =========================
        self._busy = False
        self._busy_lock = threading.Lock()
        self.emergency_active = False

        # =========================
        # 구독자
        # =========================
        self.create_subscription(
            Float32MultiArray,
            "/pick_command",
            self._pick_callback,
            10
        )

        self.create_subscription(
            Float32MultiArray,
            "/place_command",
            self._place_callback,
            10
        )

        self.create_subscription(
            String,
            "/emergency_stop",
            self._emergency_stop_callback,
            10
        )

        # =========================
        # 발행자
        # =========================
        self._pick_status_pub = self.create_publisher(
            String,
            "/pick_status",
            10
        )

        # 디버깅/대시보드용 상태 문자열
        self._status_pub = self.create_publisher(
            String,
            "/arm/status",
            10
        )

        # =========================
        # myCobot 연결
        # =========================
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
        """
        터미널 로그 + /arm/status 발행
        """
        self.get_logger().info(msg)

        m = String()
        m.data = msg
        self._status_pub.publish(m)

    def _pub_pick_status(self, status: str):
        """
        brain_node로 작업 상태 발행
        """
        m = String()
        m.data = status
        self._pick_status_pub.publish(m)
        self.get_logger().info(f"/pick_status 발행: {status}")

    def _parse_coords(self, msg: Float32MultiArray):
        """
        Float32MultiArray를 좌표 리스트로 변환
        기대 형식:
        [x, y, z, rx, ry, rz]
        """
        coords = [round(float(v), 2) for v in msg.data]

        if len(coords) != 6:
            self.get_logger().error(
                f"좌표 6개 필요, 받은 개수: {len(coords)}, data={coords}"
            )
            self._pub_pick_status("error")
            return None

        return coords

    def _try_start_task(self, task_name: str, target_func, coords):
        """
        피킹/플레이싱 작업 시작 공통 처리

        - 비상정지 상태면 명령 무시
        - 이미 작업 중이면 명령 무시
        - 작업 시작 전 _busy=True로 설정해서 중복 실행 방지
        """
        if self.emergency_active:
            self.get_logger().warn(f"비상정지 상태라 {task_name} 명령 무시")
            return

        with self._busy_lock:
            if self._busy:
                self.get_logger().warn(f"로봇팔 작업 중이라 {task_name} 명령 무시")
                return

            self._busy = True

        threading.Thread(
            target=target_func,
            args=(coords,),
            daemon=True
        ).start()

    def _finish_task(self):
        """
        작업 종료 시 busy 해제
        """
        with self._busy_lock:
            self._busy = False

    def _safe_sleep(self, seconds: float, step: float = 0.1) -> bool:
        """
        sleep 중간에도 emergency_active를 확인하기 위한 함수

        반환값:
        True  = 정상 대기 완료
        False = 대기 중 비상정지 발생
        """
        elapsed = 0.0

        while elapsed < seconds:
            if self.emergency_active:
                self.get_logger().warn("비상정지 감지 - 현재 시퀀스 중단")
                return False

            time.sleep(step)
            elapsed += step

        return True

    def _stop_robot_arm(self):
        """
        myCobot 정지 시도

        pymycobot 버전/펌웨어에 따라 stop() 동작이 다를 수 있으므로
        예외 처리를 해둔다.
        """
        try:
            self.mc.stop()
            self.get_logger().error("mc.stop() 호출 완료")
        except Exception as e:
            self.get_logger().error(f"mc.stop() 실패: {e}")

    def _align_gripper_vertical(self) -> bool:
        """
        6번축을 세로 집기 자세로 보정한다.

        send_coords는 RPY 기반 IK로 동작하기 때문에,
        이동 후 6번축이 원하는 각도와 달라질 수 있다.
        따라서 waypoint 이동 후 또는 하강 후에 6번축만 다시 보정한다.
        """
        try:
            angles = self.mc.get_angles()

            if angles == -1 or angles is None:
                self.get_logger().warn("6번축 보정 실패: get_angles() = -1")
                return False

            if len(angles) < 6:
                self.get_logger().warn(f"6번축 보정 실패: angles 길이 이상함: {angles}")
                return False

            self.get_logger().info(f"현재 관절각: {[round(a, 2) for a in angles]}")

            angles[5] = GRIPPER_VERTICAL_J6
            self.get_logger().info(f"6번축 세로 보정: J6 -> {GRIPPER_VERTICAL_J6}")

            self.mc.send_angles(angles, J6_ALIGN_SPEED)
            return True

        except Exception as e:
            self.get_logger().warn(f"6번축 보정 중 예외 발생: {e}")
            return False

    # =========================
    # 콜백 함수
    # =========================
    def _pick_callback(self, msg: Float32MultiArray):
        """
        /pick_command 수신 콜백

        brain_node가 물체 좌표를 보내면
        해당 좌표로 이동해서 물체를 집는다.
        """
        coords = self._parse_coords(msg)
        if coords is None:
            return

        self.get_logger().info(f"픽 명령 수신: {coords}")
        self._try_start_task("픽", self._pick_sequence, coords)

    def _place_callback(self, msg: Float32MultiArray):
        """
        /place_command 수신 콜백

        brain_node가 포장구역 내려놓기 좌표를 보내면
        해당 좌표로 이동해서 물체를 내려놓는다.
        """
        coords = self._parse_coords(msg)
        if coords is None:
            return

        self.get_logger().info(f"플레이스 명령 수신: {coords}")
        self._try_start_task("플레이스", self._place_sequence, coords)

    def _emergency_stop_callback(self, msg: String):
        """
        /emergency_stop 수신 콜백

        "stop"  계열 문자열 수신 시:
          - emergency_active=True
          - 로봇팔 정지 시도
          - /pick_status="error" 발행

        "reset" 계열 문자열 수신 시:
          - emergency_active=False
          - 이후 새 명령 수신 가능
        """
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
        피킹 시퀀스 (피킹 준비자세 + 물체 위 waypoint + z축 수직 하강)

        기존 방식:
        - 블록 앞에서 x축 정면 접근

        변경 방식:
        - 홈포지션에서 바로 물체 위로 가지 않고
          먼저 PICK_READY_ANGLES 준비자세를 거친다.
        - 그 다음 물체 바로 위 waypoint로 이동한다.
        - 6번축을 세로 집기 자세로 보정한다.
        - z축으로 천천히 수직 하강한다.
        - 다시 6번축을 40도로 보정한다.
        - 그리퍼를 닫고 z축으로 상승한다.
        - 홈포지션으로 복귀한다.

        순서:
        1. 그리퍼 열기
        2. 피킹 준비자세 이동
        3. 물체 위 waypoint 이동
        4. 집게 세로 정렬
        5. z축 수직 하강
        6. 하강 후 집게 세로 재정렬
        7. 그리퍼 닫기
        8. z축 상승
        9. 홈포지션 복귀
        10. /pick_status="done" 발행
        """
        try:
            if self.emergency_active:
                return

            x, y, z, rx, ry, rz = coords

            # 오른쪽 블록 y 보정 (캘리브레이션 y 오차)
            # if y < 0:
            #     y = y * 1.5   # 계수는 테스트로 조정
            
            # 살짝 든 자세로 (ry 조정해서 그리퍼 기울임)
            ry = ry + 15   # 15도 들기 (값은 테스트, 부호도 확인)

            # 실제 피킹 위치 미세 보정
            x += PICK_X_BIAS_MM
            y += PICK_Y_BIAS_MM
            z += PICK_Z_BIAS_MM

            self.get_logger().info(
                f"피킹 보정 적용 후 좌표: "
                f"x={x:.1f}, y={y:.1f}, z={z:.1f}, "
                f"bias=({PICK_X_BIAS_MM}, {PICK_Y_BIAS_MM}, {PICK_Z_BIAS_MM})"
            )


            # vision_node에서 받은 z는 물체 위치라고 보고,
            # 실제 send_coords에는 그리퍼 끝 길이만큼 z를 더해서
            # 플랜지/툴 기준 좌표로 보정한다.
            target_z = z + GRIPPER_Z_OFFSET_MM

            pre_pick = [x, y, target_z + APPROACH_Z_MM, rx, ry, rz]  # 물체 위 waypoint
            target   = [x, y, target_z,                 rx, ry, rz]  # 실제 집는 위치
            lifted   = [x, y, target_z + LIFT_Z,         rx, ry, rz]  # 집은 뒤 위로 상승

            self._log(
                f"[PICK INFO] 원본 coords={coords}, "
                f"보정 target_z={round(target_z, 2)}, "
                f"pre_pick={pre_pick}, target={target}, lifted={lifted}, "
                f"PICK_READY_ANGLES={PICK_READY_ANGLES}"
            )

            self._log("[PICK 1/10] 그리퍼 열기")
            self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
            if not self._safe_sleep(1.5):
                return

            self._log("[PICK 2/10] 피킹 준비자세 이동")
            self.mc.send_angles(PICK_READY_ANGLES, MOVE_SPEED)
            if not self._safe_sleep(PICK_READY_WAIT):
                return

            self._log("[PICK 3/10] 물체 위 waypoint 이동")
            self.mc.send_coords(pre_pick, MOVE_SPEED, 1)
            if not self._safe_sleep(5.0):
                return

            # self._log("[PICK 4/10] 집게 세로 정렬 J6=40")
            # self._align_gripper_vertical()
            # if not self._safe_sleep(1.5):
            #     return

            # J6 정렬 끝난 후의 실제 자세 읽기 (세로 정렬 반영됨)
            cur = self.mc.get_coords()
            if cur and cur != -1 and len(cur) == 6:
                target = [cur[0], cur[1], target_z, cur[3], cur[4], cur[5]]
                self._log(f"[PICK] 하강 좌표: {[round(v,1) for v in target]}")
            else:
                self._log(f"[PICK] get_coords 실패({cur}), 기존 target 사용")

            self._log("[PICK 5/10] z축 수직 하강")
            self.mc.send_coords(target, DESCEND_SPEED, 1)
            if not self._safe_sleep(4.0):
                return

            # self._log("[PICK 6/10] 하강 후 집게 세로 재정렬 J6=40")
            # #self._align_gripper_vertical()
            # if not self._safe_sleep(1.0):
            #     return

            self._log("[PICK 7/10] 그리퍼 닫기")
            self.mc.set_gripper_value(GRIPPER_CLOSE, GRIPPER_SPEED)
            if not self._safe_sleep(2.5):
                return

            self._log("[PICK 8/10] z축 상승")
            self.mc.send_coords(lifted, MOVE_SPEED, 1)
            if not self._safe_sleep(3.0):
                return

            self._log("[PICK 9/10] 홈포지션 복귀")
            self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
            if not self._safe_sleep(4.0):
                return

            self._log("[PICK 10/10] 픽 완료")
            self._pub_pick_status("done")

        except Exception as e:
            self.get_logger().error(f"픽 오류: {e}")
            self._pub_pick_status("error")

        finally:
            self._finish_task()

    def _place_sequence(self, coords):
       """
       플레이싱 시퀀스 (픽과 대칭 구조)
   
       순서:
       1. 피킹 준비자세 이동 (급격한 이동 방지)
       2. 놓을 위치 위 waypoint 이동
       3. get_coords로 실제 자세 읽어 z만 수직 하강
       4. 그리퍼 열기 (내려놓기)
       5. z축 상승
       6. 홈포지션 복귀
       7. /pick_status="placing_done" 발행
       """
       try:
           if self.emergency_active:
               return
   
           x, y, z, rx, ry, rz = coords
   
           # 픽과 동일하게 살짝 기울인 자세로 접근
           ry = ry + 15
   
           # place도 그리퍼 끝 기준 → 플랜지 기준 z 보정
           target_z = z + GRIPPER_Z_OFFSET_MM
   
           pre_place = [x, y, target_z + APPROACH_Z_MM, rx, ry, rz]  # 놓을 위치 위
           target    = [x, y, target_z,                 rx, ry, rz]  # 실제 놓는 위치
           lifted    = [x, y, target_z + LIFT_Z,        rx, ry, rz]  # 놓고 상승
   
           self._log(
               f"[PLACE INFO] 원본 coords={coords}, "
               f"보정 target_z={round(target_z, 2)}, "
               f"pre_place={pre_place}, target={target}, lifted={lifted}"
           )
   
           self._log("[PLACE 1/7] 준비자세 이동")
           self.mc.send_angles(PICK_READY_ANGLES, MOVE_SPEED)
           if not self._safe_sleep(PICK_READY_WAIT):
               return
   
           self._log("[PLACE 2/7] 놓을 위치 위 waypoint 이동")
           self.mc.send_coords(pre_place, MOVE_SPEED, 1)
           if not self._safe_sleep(5.0):
               return
   
           # pre_place 도착 후 실제 자세 읽어서 그 자세로 수직 하강
           cur = self.mc.get_coords()
           if cur and cur != -1 and len(cur) == 6:
               target = [cur[0], cur[1], target_z, cur[3], cur[4], cur[5]]
               self._log(f"[PLACE] 하강 좌표: {[round(v,1) for v in target]}")
           else:
               self._log(f"[PLACE] get_coords 실패({cur}), 기존 target 사용")
   
           self._log("[PLACE 3/7] z축 수직 하강")
           self.mc.send_coords(target, DESCEND_SPEED, 1)
           if not self._safe_sleep(4.0):
               return
   
           self._log("[PLACE 4/7] 그리퍼 열기 (내려놓기)")
           self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
           if not self._safe_sleep(2.0):
               return
   
           self._log("[PLACE 5/7] z축 상승")
           self.mc.send_coords(lifted, MOVE_SPEED, 1)
           if not self._safe_sleep(3.0):
               return
   
           self._log("[PLACE 6/7] 홈포지션 복귀")
           self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
           if not self._safe_sleep(4.0):
               return
   
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
