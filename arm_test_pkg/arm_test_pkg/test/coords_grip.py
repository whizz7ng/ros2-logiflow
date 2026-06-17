from pymycobot.mycobot import MyCobot
import time

mc = MyCobot('/dev/ttyAMA0', 1000000)
mc.focus_all_servos()
time.sleep(1)

print("홈으로 이동...")
mc.send_angles([0, -30, -60, 0, 0, 0], 30)
time.sleep(3)

print("블록 위쪽 접근...")
mc.send_coords([15.3, -293.7, 280.0, -165.29, 13.59, -167.77], 15, 1)
time.sleep(3)

print("그리퍼 열기...")
mc.set_gripper_value(100, 80)
time.sleep(1)

print("수직으로 내려오기...")
mc.send_coords([15.3, -293.7, 160.0, -165.29, 13.59, -167.77], 15, 1)
time.sleep(4)

print("집기...")
mc.set_gripper_value(55, 30)
time.sleep(2)

print("들어올리기...")
mc.send_coords([15.3, -293.7, 280.0, -165.29, 13.59, -167.77], 15, 1)
time.sleep(3)

print("홈으로 이동...")
mc.send_angles([0, -30, -60, 0, 0, 0], 15)
time.sleep(3)

print("원래 위치 위쪽으로...")
mc.send_coords([15.3, -293.7, 280.0, -165.29, 13.59, -167.77], 15, 1)
time.sleep(3)

print("내려놓기 위치로...")
mc.send_coords([15.3, -293.7, 160.0, -165.29, 13.59, -167.77], 15, 1)
time.sleep(3)

print("그리퍼 열기...")
mc.set_gripper_value(0, 80)
time.sleep(1)

print("올라오기...")
mc.send_coords([15.3, -293.7, 280.0, -165.29, 13.59, -167.77], 15, 1)
time.sleep(3)

print("홈으로 복귀...")
mc.send_angles([0, -30, -60, 0, 0, 0], 15)
time.sleep(3)

print("완료!")
