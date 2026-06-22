from pymycobot.mycobot import MyCobot
import time

mc = MyCobot('/dev/ttyAMA0', 1000000)

# 1. 현재 위치 확인
print("현재 각도:", mc.get_angles())
print("현재 좌표:", mc.get_coords())
