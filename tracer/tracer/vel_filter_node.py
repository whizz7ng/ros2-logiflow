#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class CmdVelSafetyFilter(Node):
    def __init__(self):
        super().__init__('cmd_vel_safety_filter')

        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub = self.create_subscription(
            Twist,
            '/cmd_vel_raw',
            self.cmd_cb,
            cmd_qos
        )

        self.pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            cmd_qos
        )

        self.target = Twist()
        self.current = Twist()

        self.last_cmd_time = time.time()
        self.last_update_time = time.time()

        # ===== safety params =====
        self.max_vx = 0.13
        self.max_vy = 0.08
        self.max_wz = 0.30

        self.max_acc_vx = 0.10
        self.max_acc_vy = 0.08
        self.max_acc_wz = 0.25

        self.cmd_timeout = 0.30
        self.timer_period = 0.05  # 20 Hz

        self.timer = self.create_timer(self.timer_period, self.update)

    def clamp(self, x, lo, hi):
        return max(lo, min(hi, x))

    def limit_rate(self, current, target, max_delta):
        diff = target - current

        if diff > max_delta:
            return current + max_delta
        elif diff < -max_delta:
            return current - max_delta
        else:
            return target

    def cmd_cb(self, msg):
        self.last_cmd_time = time.time()

        self.target.linear.x = self.clamp(
            msg.linear.x,
            -self.max_vx,
            self.max_vx
        )

        self.target.linear.y = self.clamp(
            msg.linear.y,
            -self.max_vy,
            self.max_vy
        )

        self.target.linear.z = 0.0

        self.target.angular.x = 0.0
        self.target.angular.y = 0.0

        self.target.angular.z = self.clamp(
            msg.angular.z,
            -self.max_wz,
            self.max_wz
        )

    def update(self):
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now

        if now - self.last_cmd_time > self.cmd_timeout:
            self.target.linear.x = 0.0
            self.target.linear.y = 0.0
            self.target.angular.z = 0.0

        max_dvx = self.max_acc_vx * dt
        max_dvy = self.max_acc_vy * dt
        max_dw = self.max_acc_wz * dt

        self.current.linear.x = self.limit_rate(
            self.current.linear.x,
            self.target.linear.x,
            max_dvx
        )

        self.current.linear.y = self.limit_rate(
            self.current.linear.y,
            self.target.linear.y,
            max_dvy
        )

        self.current.angular.z = self.limit_rate(
            self.current.angular.z,
            self.target.angular.z,
            max_dw
        )

        self.pub.publish(self.current)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSafetyFilter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
