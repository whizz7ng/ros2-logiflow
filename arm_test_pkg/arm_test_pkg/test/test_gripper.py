from pymycobot.mycobot import MyCobot
mc = MyCobot('/dev/ttyAMA0', 1000000)

# 현재 각도 확인
print(mc.get_angles())

# 홈 포지션으로 이동
mc.send_angles([0, 0, 0, 0, 0, 0], 50)

# 그리퍼 열기
mc.set_gripper_state(0, 80)

# 그리퍼 닫기
mc.set_gripper_state(1, 80)
