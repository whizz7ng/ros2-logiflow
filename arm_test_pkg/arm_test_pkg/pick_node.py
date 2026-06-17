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
HOME_ANGLES = [0, -30, -30, 0, 0, 0]

# 그리퍼 값
# 사용하는 그리퍼에 따라 열림/닫힘 값은 테스트 후 조정 필요
GRIPPER_OPEN = 100
GRIPPER_CLOSE = 55


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
        피킹 시퀀스

        순서:
        1. 그리퍼 열기
        2. 목표 물체 좌표로 이동
        3. 그리퍼 닫기
        4. 홈포지션 복귀
        5. /pick_status="done" 발행
        """
        try:
            if self.emergency_active:
                return

            self._log("[PICK 1/5] 그리퍼 열기")
            self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
            if not self._safe_sleep(1.5):
                return

            self._log("[PICK 2/5] 목표 위치로 이동")
            self.mc.send_coords(coords, MOVE_SPEED, 1)
            if not self._safe_sleep(6.0):
                return

            self._log("[PICK 3/5] 그리퍼 닫기")
            self.mc.set_gripper_value(GRIPPER_CLOSE, GRIPPER_SPEED)
            if not self._safe_sleep(2.5):
                return

            self._log("[PICK 4/5] 홈포지션 복귀")
            self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
            if not self._safe_sleep(4.0):
                return

            self._log("[PICK 5/5] 픽 완료")
            self._pub_pick_status("done")

        except Exception as e:
            self.get_logger().error(f"픽 오류: {e}")
            self._pub_pick_status("error")

        finally:
            self._finish_task()

    def _place_sequence(self, coords):
        """
        플레이싱 시퀀스

        순서:
        1. 내려놓기 좌표로 이동
        2. 그리퍼 열기
        3. 홈포지션 복귀
        4. /pick_status="placing_done" 발행
        """
        try:
            if self.emergency_active:
                return

            self._log("[PLACE 1/4] 내려놓기 위치로 이동")
            self.mc.send_coords(coords, MOVE_SPEED, 1)
            if not self._safe_sleep(6.0):
                return

            self._log("[PLACE 2/4] 그리퍼 열기")
            self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
            if not self._safe_sleep(1.5):
                return

            self._log("[PLACE 3/4] 홈포지션 복귀")
            self.mc.send_angles(HOME_ANGLES, MOVE_SPEED)
            if not self._safe_sleep(4.0):
                return

            self._log("[PLACE 4/4] 플레이스 완료")
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
