
from pymycobot.mycobot import MyCobot

mc = MyCobot('/dev/ttyAMA0', 1000000)

mc.release_all_servos()

print("준비완료. 각도:", mc.get_angles())

