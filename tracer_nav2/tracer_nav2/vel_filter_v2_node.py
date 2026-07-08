#!/usr/bin/env python3
"""
Nav2-oriented cmd_vel safety filter for myAGV.

Design intent after surge-probe experiments:
  - Do NOT aggressively rewrite every Nav2 curve command.
  - Keep hard safety clamps.
  - Protect only risky transitions:
      1) turn exit: command becomes straight while robot may still be yawing
      2) large sign flip: +wz -> -wz without a zero-cross hold
      3) high |wz|: reduce vx gradually, but do not kill all curved motion
  - Disable old small-wz pulse/deadzone boost by default.

Recommended chain:
  Nav2 /cmd_vel remapped to /cmd_vel_nav
      -> vel_filter_v2_node
      -> /cmd_vel
      -> myAGV driver
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import String


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sign(x, eps=1e-9):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


class VelFilterV2Node(Node):
    def __init__(self):
        super().__init__('cmd_vel_safety_filter')

        # ============================================================
        # Topics / timing
        # ============================================================
        self.declare_parameter('input_topic', '/cmd_vel_nav')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('state_topic', '/cmd_vel_safety_filter/state')

        # publish_hz=20 is enough to stay below myAGV driver's ~0.3s timeout,
        # while avoiding old aggressive 40Hz republish load.
        self.declare_parameter('publish_hz', 20.0)
        self.declare_parameter('cmd_timeout_sec', 0.35)
        self.declare_parameter('publish_zero_when_idle', True)

        # ============================================================
        # Hard limits
        # ============================================================
        self.declare_parameter('max_vx', 0.20)
        self.declare_parameter('max_vy', 0.06)
        self.declare_parameter('max_wz', 0.60)
        self.declare_parameter('block_reverse', True)

        # Acceleration limiting. These are deliberately gentler than v1.
        self.declare_parameter('enable_acc_limit', True)
        self.declare_parameter('max_acc_vx', 0.20)
        self.declare_parameter('max_acc_vy', 0.15)
        self.declare_parameter('max_acc_wz', 0.80)

        # ============================================================
        # High turn vx soft-limit
        # ============================================================
        # Below high_wz_start, Nav2 curved motion is preserved.
        # Above high_wz_start, vx is gradually limited.
        self.declare_parameter('enable_high_wz_vx_limit', True)
        self.declare_parameter('high_wz_start', 0.35)
        self.declare_parameter('high_wz_full', 0.55)
        self.declare_parameter('vx_limit_at_high_wz_start', 0.12)
        self.declare_parameter('vx_limit_at_high_wz_full', 0.04)

        # Compatibility parameter used by old launch patches. Not used as a mode switch.
        self.declare_parameter('straight_vx_on', 0.03)
        self.declare_parameter('kill_vy_in_straight', False)
        self.declare_parameter('kill_vy_in_turn', False)

        # ============================================================
        # Sign flip zero-cross guard
        # ============================================================
        self.declare_parameter('enable_zero_cross_guard', True)
        self.declare_parameter('zero_cross_threshold', 0.30)
        self.declare_parameter('zero_cross_hold_sec', 0.25)
        self.declare_parameter('zero_cross_vx_limit', 0.00)
        self.declare_parameter('zero_cross_vy_limit', 0.00)

        # ============================================================
        # Turn-exit hold
        # ============================================================
        # Main finding from floor/air tests: after turning, wz may remain in odom
        # even when cmd wz becomes 0 and vx resumes. During this window, reduce vx.
        self.declare_parameter('enable_turn_exit_hold', True)
        self.declare_parameter('turn_exit_hold_sec', 0.45)
        self.declare_parameter('turn_exit_min_prev_wz', 0.35)
        self.declare_parameter('turn_exit_target_wz', 0.08)
        self.declare_parameter('turn_exit_max_vx', 0.03)
        self.declare_parameter('turn_exit_max_vy', 0.02)
        self.declare_parameter('allow_turn_during_exit_hold', True)

        # ============================================================
        # Small wz behavior
        # ============================================================
        # Old v1 used pulse/deadzone boosting. Experiments suggested that can
        # over-modify Nav2. Keep it off by default. Only deadband tiny noise.
        self.declare_parameter('enable_small_wz_deadband', True)
        self.declare_parameter('small_wz_deadband', 0.025)
        self.declare_parameter('enable_wz_deadzone_adapter', False)
        self.declare_parameter('use_pulse_adapter', False)
        self.declare_parameter('hw_min_wz', 0.40)
        self.declare_parameter('min_pulse_duty', 0.35)
        self.declare_parameter('max_pulse_duty', 0.60)
        self.declare_parameter('pulse_period', 0.45)

        # ============================================================
        # Debug
        # ============================================================
        self.declare_parameter('print_mode_change', True)
        self.declare_parameter('debug_period_sec', 0.50)

        self.input_topic = self.p('input_topic')
        self.output_topic = self.p('output_topic')
        self.state_topic = self.p('state_topic')

        # Input: BEST_EFFORT subscriber can receive both reliable Nav2 and
        # best-effort manual/bridge publishers. Output: RELIABLE for driver.
        input_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        output_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.cmd_sub = self.create_subscription(Twist, self.input_topic, self.cmd_cb, input_qos)
        self.cmd_pub = self.create_publisher(Twist, self.output_topic, output_qos)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        self.raw_cmd = Twist()
        self.last_cmd_time = 0.0
        self.have_cmd = False

        self.out_cmd = Twist()
        self.last_filter_time = time.monotonic()
        self.last_output_wz = 0.0
        self.last_nonzero_wz = 0.0

        self.zero_cross_until = 0.0
        self.turn_exit_until = 0.0
        self.mode = 'INIT'
        self.last_mode = ''
        self.last_debug_time = 0.0

        period = 1.0 / max(float(self.p('publish_hz')), 1.0)
        self.timer = self.create_timer(period, self.timer_cb)

        self.publish_state(
            f'vel_filter_v2_ready input={self.input_topic} output={self.output_topic} '
            f'max_vx={self.p("max_vx")} max_vy={self.p("max_vy")} max_wz={self.p("max_wz")} '
            f'publish_hz={self.p("publish_hz")}'
        )

    def p(self, name):
        return self.get_parameter(name).value

    def cmd_cb(self, msg):
        self.raw_cmd = msg
        self.last_cmd_time = time.monotonic()
        self.have_cmd = True

    def make_stop(self):
        return Twist()

    def copy_twist(self, msg):
        out = Twist()
        out.linear.x = float(msg.linear.x)
        out.linear.y = float(msg.linear.y)
        out.linear.z = 0.0
        out.angular.x = 0.0
        out.angular.y = 0.0
        out.angular.z = float(msg.angular.z)
        return out

    def publish_state(self, text):
        msg = String()
        msg.data = str(text)
        self.state_pub.publish(msg)
        self.get_logger().info(str(text))

    def set_mode(self, mode, extra=''):
        self.mode = mode
        if bool(self.p('print_mode_change')) and mode != self.last_mode:
            self.last_mode = mode
            msg = f'mode:{mode}'
            if extra:
                msg += f' | {extra}'
            self.publish_state(msg)

    def hard_clamp(self, cmd):
        out = self.copy_twist(cmd)

        max_vx = abs(float(self.p('max_vx')))
        max_vy = abs(float(self.p('max_vy')))
        max_wz = abs(float(self.p('max_wz')))

        if bool(self.p('block_reverse')):
            out.linear.x = clamp(out.linear.x, 0.0, max_vx)
        else:
            out.linear.x = clamp(out.linear.x, -max_vx, max_vx)

        out.linear.y = clamp(out.linear.y, -max_vy, max_vy)
        out.angular.z = clamp(out.angular.z, -max_wz, max_wz)

        if bool(self.p('enable_small_wz_deadband')):
            dz = abs(float(self.p('small_wz_deadband')))
            if abs(out.angular.z) < dz:
                out.angular.z = 0.0

        return out

    def apply_high_wz_vx_limit(self, cmd):
        if not bool(self.p('enable_high_wz_vx_limit')):
            return cmd

        wz_abs = abs(cmd.angular.z)
        start = abs(float(self.p('high_wz_start')))
        full = abs(float(self.p('high_wz_full')))

        if wz_abs < start:
            return cmd

        if full <= start:
            ratio = 1.0
        else:
            ratio = clamp((wz_abs - start) / (full - start), 0.0, 1.0)

        vx_start = abs(float(self.p('vx_limit_at_high_wz_start')))
        vx_full = abs(float(self.p('vx_limit_at_high_wz_full')))
        vx_lim = vx_start + ratio * (vx_full - vx_start)
        vx_lim = max(0.0, vx_lim)

        if abs(cmd.linear.x) > vx_lim:
            cmd.linear.x = math.copysign(vx_lim, cmd.linear.x)

        return cmd

    def detect_zero_cross(self, target, now):
        if not bool(self.p('enable_zero_cross_guard')):
            return

        th = abs(float(self.p('zero_cross_threshold')))
        prev = self.last_output_wz
        cur = target.angular.z

        if abs(prev) >= th and abs(cur) >= th and sign(prev) != sign(cur):
            hold = max(0.0, float(self.p('zero_cross_hold_sec')))
            self.zero_cross_until = max(self.zero_cross_until, now + hold)
            self.set_mode('ZERO_CROSS_HOLD', f'prev_wz={prev:.3f} target_wz={cur:.3f} hold={hold:.2f}')

    def detect_turn_exit(self, target, now):
        if not bool(self.p('enable_turn_exit_hold')):
            return

        prev_th = abs(float(self.p('turn_exit_min_prev_wz')))
        target_th = abs(float(self.p('turn_exit_target_wz')))

        prev = self.last_output_wz
        cur = target.angular.z

        # If a new meaningful turn command arrives, do not hold it back.
        if abs(cur) > target_th:
            if bool(self.p('allow_turn_during_exit_hold')):
                self.turn_exit_until = 0.0
            return

        if abs(prev) >= prev_th and abs(cur) <= target_th:
            hold = max(0.0, float(self.p('turn_exit_hold_sec')))
            self.turn_exit_until = max(self.turn_exit_until, now + hold)
            self.set_mode('TURN_EXIT_HOLD', f'prev_wz={prev:.3f} target_wz={cur:.3f} hold={hold:.2f}')

    def apply_transition_holds(self, cmd, now):
        # zero-cross hold has priority.
        if now < self.zero_cross_until:
            cmd.angular.z = 0.0
            vx_lim = abs(float(self.p('zero_cross_vx_limit')))
            vy_lim = abs(float(self.p('zero_cross_vy_limit')))
            cmd.linear.x = clamp(cmd.linear.x, -vx_lim, vx_lim)
            cmd.linear.y = clamp(cmd.linear.y, -vy_lim, vy_lim)
            self.set_mode('ZERO_CROSS_HOLD')
            return cmd

        if now < self.turn_exit_until and abs(cmd.angular.z) <= abs(float(self.p('turn_exit_target_wz'))):
            vx_lim = abs(float(self.p('turn_exit_max_vx')))
            vy_lim = abs(float(self.p('turn_exit_max_vy')))
            cmd.linear.x = clamp(cmd.linear.x, -vx_lim, vx_lim)
            cmd.linear.y = clamp(cmd.linear.y, -vy_lim, vy_lim)
            cmd.angular.z = 0.0
            self.set_mode('TURN_EXIT_HOLD')
            return cmd

        return cmd

    def apply_acc_limit(self, cmd, dt):
        if not bool(self.p('enable_acc_limit')):
            return cmd

        dt = max(0.001, min(float(dt), 0.20))
        out = self.copy_twist(cmd)

        ax = abs(float(self.p('max_acc_vx'))) * dt
        ay = abs(float(self.p('max_acc_vy'))) * dt
        aw = abs(float(self.p('max_acc_wz'))) * dt

        out.linear.x = clamp(out.linear.x, self.out_cmd.linear.x - ax, self.out_cmd.linear.x + ax)
        out.linear.y = clamp(out.linear.y, self.out_cmd.linear.y - ay, self.out_cmd.linear.y + ay)
        out.angular.z = clamp(out.angular.z, self.out_cmd.angular.z - aw, self.out_cmd.angular.z + aw)

        return out

    def timer_cb(self):
        now = time.monotonic()
        dt = now - self.last_filter_time
        self.last_filter_time = now

        if (not self.have_cmd) or ((now - self.last_cmd_time) > float(self.p('cmd_timeout_sec'))):
            if not bool(self.p('publish_zero_when_idle')):
                return
            target = self.make_stop()
            self.set_mode('TIMEOUT_STOP')
        else:
            target = self.hard_clamp(self.raw_cmd)
            self.detect_zero_cross(target, now)
            self.detect_turn_exit(target, now)
            target = self.apply_transition_holds(target, now)
            target = self.apply_high_wz_vx_limit(target)

            if self.mode not in ['ZERO_CROSS_HOLD', 'TURN_EXIT_HOLD']:
                if abs(target.angular.z) >= abs(float(self.p('high_wz_start'))):
                    self.set_mode('HIGH_WZ_SOFT_LIMIT')
                else:
                    self.set_mode('PASS_THROUGH')

        out = self.apply_acc_limit(target, dt)

        self.cmd_pub.publish(out)
        self.out_cmd = out

        if abs(out.angular.z) > 1e-4:
            self.last_nonzero_wz = out.angular.z
        self.last_output_wz = out.angular.z

        dbg_period = max(0.05, float(self.p('debug_period_sec')))
        if now - self.last_debug_time > dbg_period:
            self.last_debug_time = now
            raw = self.raw_cmd
            self.get_logger().debug(
                f'filter_v2 mode={self.mode} '
                f'raw=({raw.linear.x:.3f},{raw.linear.y:.3f},{raw.angular.z:.3f}) '
                f'out=({out.linear.x:.3f},{out.linear.y:.3f},{out.angular.z:.3f})'
            )


def main(args=None):
    rclpy.init(args=args)
    node = VelFilterV2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
