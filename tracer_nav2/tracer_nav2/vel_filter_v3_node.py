#!/usr/bin/env python3
"""
Vel Filter V3 for myAGV + Nav2 Galactic.

Design principle
----------------
myagv.yaml / Nav2 owns normal motion planning:
  - normal vx / wz limits
  - acceleration / deceleration limits
  - curved-path tracking and obstacle avoidance

This filter only handles hardware-facing safety exceptions:
  1) stale or invalid command -> immediate stop
  2) absolute hard clamp
  3) in-place angular dead-zone compensation
  4) odometry-based angular zero-cross protection
  5) odometry-based turn-exit protection
  6) odometry overspeed monitoring (WARN by default)

Recommended chain
-----------------
Nav2 / recovery / ArUco command arbitration
    -> /cmd_vel_nav
    -> vel_filter_v3_node
    -> /cmd_vel
    -> myAGV driver

Important
---------
- V3 intentionally does NOT apply a general acceleration limiter.
- V3 intentionally does NOT reduce vx merely because |wz| is high.
- V3 intentionally does NOT apply a global small-wz deadband.
- ArUco linear.y is preserved except for the absolute hard clamp.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def sign(value: float, eps: float = 1.0e-9) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


class VelFilterV3Node(Node):
    def __init__(self) -> None:
        super().__init__('cmd_vel_safety_filter')

        self._declare_parameters()

        self.input_topic = str(self.p('input_topic'))
        self.output_topic = str(self.p('output_topic'))
        self.odom_topic = str(self.p('odom_topic'))
        self.state_topic = str(self.p('state_topic'))
        self.emergency_stop_topic = str(self.p('emergency_stop_topic'))
        self.emergency_reset_topic = str(self.p('emergency_reset_topic'))
        self.go_home_topic = str(self.p('go_home_topic'))
        self.go_home_release_topic = str(self.p('go_home_release_topic'))

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

        self.cmd_sub = self.create_subscription(
            Twist,
            self.input_topic,
            self.cmd_cb,
            input_qos,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_cb,
            input_qos,
        )
        self.estop_sub = self.create_subscription(
            String,
            self.emergency_stop_topic,
            self.emergency_stop_cb,
            10,
        )
        self.estop_reset_sub = self.create_subscription(
            String,
            self.emergency_reset_topic,
            self.emergency_reset_cb,
            10,
        )
        self.go_home_sub = self.create_subscription(
            String,
            self.go_home_topic,
            self.go_home_cb,
            10,
        )
        self.go_home_release_sub = self.create_subscription(
            String,
            self.go_home_release_topic,
            self.go_home_release_cb,
            10,
        )
        self.cmd_pub = self.create_publisher(Twist, self.output_topic, output_qos)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        # External safety latches.
        # E-stop remains latched until /emergency_reset is received.
        # Go-home hold is released by mission_brain after the robot is stationary.
        self.estop_latched = False
        self.go_home_hold = False

        # Latest command input.
        self.raw_cmd = Twist()
        self.have_cmd = False
        self.last_cmd_time = 0.0

        # Latest odometry velocity.
        self.have_odom = False
        self.last_odom_time = 0.0
        self.odom_vx = 0.0
        self.odom_vy = 0.0
        self.odom_wz = 0.0

        # Last command actually published to the driver.
        self.last_output_cmd = Twist()

        # Guard state.
        self.zero_cross_active = False
        self.zero_cross_started = 0.0
        self.zero_cross_bypass_until = 0.0

        self.turn_exit_active = False
        self.turn_exit_started = 0.0
        self.turn_exit_bypass_until = 0.0

        # Overspeed monitor state.
        self.overspeed_since: Optional[float] = None
        self.overspeed_reported = False
        self.overspeed_stop_until = 0.0

        # Logging / state publication.
        self.mode = 'INIT'
        self.last_mode = ''
        self.last_debug_time = 0.0
        self.last_warning_times: Dict[str, float] = {}

        self._validate_parameters()

        publish_hz = max(float(self.p('publish_hz')), 1.0)
        self.timer = self.create_timer(1.0 / publish_hz, self.timer_cb)

        self.publish_state(
            'vel_filter_v3_ready '
            f'input={self.input_topic} output={self.output_topic} '
            f'odom={self.odom_topic} publish_hz={publish_hz:.1f} '
            f'estop={self.emergency_stop_topic} reset={self.emergency_reset_topic} '
            f'go_home={self.go_home_topic} go_home_release={self.go_home_release_topic} '
            f'hard_max=(vx:{float(self.p("hard_max_vx")):.3f}, '
            f'vy:{float(self.p("hard_max_vy")):.3f}, '
            f'wz:{float(self.p("hard_max_wz")):.3f}) '
            f'hw_min_wz={float(self.p("hw_min_wz")):.3f}'
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        # Topics / timing
        self.declare_parameter('input_topic', '/cmd_vel_nav')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('state_topic', '/cmd_vel_safety_filter/state')
        self.declare_parameter('emergency_stop_topic', '/emergency_stop')
        self.declare_parameter('emergency_reset_topic', '/emergency_reset')
        self.declare_parameter('go_home_topic', '/go_home')
        self.declare_parameter('go_home_release_topic', '/go_home_release')

        self.declare_parameter('publish_hz', 20.0)
        self.declare_parameter('cmd_timeout_sec', 0.25)
        self.declare_parameter('odom_timeout_sec', 0.30)
        self.declare_parameter('publish_zero_when_idle', True)

        # Absolute hardware-facing limits.
        # Normal Nav2 limits should remain in myagv.yaml.
        self.declare_parameter('hard_max_vx', 0.22)
        self.declare_parameter('hard_max_reverse_vx', 0.05)
        self.declare_parameter('hard_max_vy', 0.08)
        # Matches recoveries_server.max_rotational_vel in the current YAML.
        self.declare_parameter('hard_max_wz', 0.75)

        # reverse_policy: allow | limited | block
        self.declare_parameter('reverse_policy', 'limited')

        # In-place angular dead-zone compensation.
        self.declare_parameter('enable_in_place_wz_adapter', True)
        self.declare_parameter('in_place_linear_threshold', 0.02)
        self.declare_parameter('in_place_wz_request_threshold', 0.005)
        self.declare_parameter('hw_min_wz', 0.40)

        # Odom-based angular sign-flip protection.
        self.declare_parameter('enable_zero_cross_guard', True)
        self.declare_parameter('zero_cross_request_threshold', 0.30)
        self.declare_parameter('zero_cross_odom_trigger_wz', 0.10)
        self.declare_parameter('zero_cross_odom_release_wz', 0.08)
        self.declare_parameter('zero_cross_max_hold_sec', 0.60)
        self.declare_parameter('zero_cross_rearm_sec', 0.20)

        # Odom-based turn-exit protection.
        self.declare_parameter('enable_turn_exit_guard', True)
        self.declare_parameter('turn_exit_linear_request_threshold', 0.02)
        self.declare_parameter('turn_exit_target_wz_threshold', 0.05)
        self.declare_parameter('turn_exit_odom_trigger_wz', 0.10)
        self.declare_parameter('turn_exit_odom_release_wz', 0.08)
        self.declare_parameter('turn_exit_max_hold_sec', 0.70)
        self.declare_parameter('turn_exit_rearm_sec', 0.20)
        self.declare_parameter('turn_exit_max_vx', 0.00)
        self.declare_parameter('turn_exit_max_vy', 0.00)

        # Odom overspeed monitor.
        # action: off | warn | stop
        self.declare_parameter('overspeed_action', 'warn')
        self.declare_parameter('overspeed_confirm_sec', 0.10)
        self.declare_parameter('overspeed_stop_hold_sec', 0.50)
        self.declare_parameter('overspeed_ratio', 2.0)
        self.declare_parameter('overspeed_min_cmd_vx', 0.03)
        self.declare_parameter('overspeed_min_cmd_vy', 0.02)
        self.declare_parameter('overspeed_min_cmd_wz', 0.05)
        self.declare_parameter('overspeed_vx_margin', 0.08)
        self.declare_parameter('overspeed_vy_margin', 0.08)
        self.declare_parameter('overspeed_wz_margin', 0.20)
        self.declare_parameter('odom_abs_vx_limit', 0.35)
        self.declare_parameter('odom_abs_vy_limit', 0.20)
        self.declare_parameter('odom_abs_wz_limit', 0.90)

        # Debug
        self.declare_parameter('print_mode_change', True)
        self.declare_parameter('debug_period_sec', 0.50)

    def p(self, name: str):
        return self.get_parameter(name).value

    def _validate_parameters(self) -> None:
        reverse_policy = str(self.p('reverse_policy')).lower()
        if reverse_policy not in {'allow', 'limited', 'block'}:
            self.get_logger().warning(
                f'Invalid reverse_policy={reverse_policy!r}; using limited behavior.'
            )

        overspeed_action = str(self.p('overspeed_action')).lower()
        if overspeed_action not in {'off', 'warn', 'stop'}:
            self.get_logger().warning(
                f'Invalid overspeed_action={overspeed_action!r}; monitor will act as warn.'
            )

        hard_max_wz = abs(float(self.p('hard_max_wz')))
        hw_min_wz = abs(float(self.p('hw_min_wz')))
        if hw_min_wz > hard_max_wz:
            self.get_logger().warning(
                'hw_min_wz is greater than hard_max_wz. '
                'The in-place adapter will be clipped by the hard clamp.'
            )

        zc_trigger = abs(float(self.p('zero_cross_odom_trigger_wz')))
        zc_release = abs(float(self.p('zero_cross_odom_release_wz')))
        if zc_release > zc_trigger:
            self.get_logger().warning(
                'zero_cross_odom_release_wz is greater than trigger. '
                'Hysteresis is reversed.'
            )

        te_trigger = abs(float(self.p('turn_exit_odom_trigger_wz')))
        te_release = abs(float(self.p('turn_exit_odom_release_wz')))
        if te_release > te_trigger:
            self.get_logger().warning(
                'turn_exit_odom_release_wz is greater than trigger. '
                'Hysteresis is reversed.'
            )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def cmd_cb(self, msg: Twist) -> None:
        self.raw_cmd = self.copy_twist(msg)
        self.last_cmd_time = time.monotonic()
        self.have_cmd = True

    def odom_cb(self, msg: Odometry) -> None:
        twist = msg.twist.twist
        values = (twist.linear.x, twist.linear.y, twist.angular.z)
        if not all(math.isfinite(float(value)) for value in values):
            self.warn_throttled(
                'invalid_odom',
                'Received NaN/Inf in odometry twist; ignoring this odom sample.',
                1.0,
            )
            return

        self.odom_vx = float(twist.linear.x)
        self.odom_vy = float(twist.linear.y)
        self.odom_wz = float(twist.angular.z)
        self.last_odom_time = time.monotonic()
        self.have_odom = True

    def emergency_stop_cb(self, msg: String) -> None:
        # Any received String payload triggers the latch.
        del msg
        first_latch = not self.estop_latched
        self.estop_latched = True
        self.go_home_hold = False
        self.have_cmd = False
        self.publish_immediate_stop(
            'EMERGENCY_STOP_LATCHED',
            'reset_required=/emergency_reset',
        )
        if first_latch:
            self.publish_state('event:EMERGENCY_STOP_LATCHED')

    def emergency_reset_cb(self, msg: String) -> None:
        # Any received String payload releases the E-stop latch.
        del msg
        was_latched = self.estop_latched
        self.estop_latched = False
        self.have_cmd = False
        self.last_cmd_time = 0.0
        self.zero_cross_active = False
        self.turn_exit_active = False
        self.publish_immediate_stop('EMERGENCY_STOP_RESET')
        if was_latched:
            self.publish_state('event:EMERGENCY_STOP_RESET')

    def go_home_cb(self, msg: String) -> None:
        # Any received String payload pauses motion immediately.
        del msg
        if self.estop_latched:
            self.publish_state('event:GO_HOME_REJECTED | reason=estop_latched')
            return
        self.go_home_hold = True
        self.have_cmd = False
        self.publish_immediate_stop('GO_HOME_HOLD')
        self.publish_state('event:GO_HOME_HOLD_STARTED')

    def go_home_release_cb(self, msg: String) -> None:
        # Mission brain releases this hold only after stop confirmation/timeout.
        del msg
        if self.estop_latched:
            self.publish_state('event:GO_HOME_RELEASE_IGNORED | reason=estop_latched')
            return
        was_active = self.go_home_hold
        self.go_home_hold = False
        self.have_cmd = False
        self.last_cmd_time = 0.0
        self.publish_immediate_stop('GO_HOME_RELEASED')
        if was_active:
            self.publish_state('event:GO_HOME_HOLD_RELEASED')

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------
    @staticmethod
    def make_stop() -> Twist:
        return Twist()

    @staticmethod
    def copy_twist(msg: Twist) -> Twist:
        out = Twist()
        out.linear.x = float(msg.linear.x)
        out.linear.y = float(msg.linear.y)
        out.linear.z = 0.0
        out.angular.x = 0.0
        out.angular.y = 0.0
        out.angular.z = float(msg.angular.z)
        return out

    @staticmethod
    def twist_is_finite(msg: Twist) -> bool:
        values = (
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        )
        return all(math.isfinite(float(value)) for value in values)

    @staticmethod
    def planar_speed(msg: Twist) -> float:
        return math.hypot(float(msg.linear.x), float(msg.linear.y))

    def odom_is_fresh(self, now: float) -> bool:
        if not self.have_odom:
            return False
        timeout = max(0.01, float(self.p('odom_timeout_sec')))
        return (now - self.last_odom_time) <= timeout

    # ------------------------------------------------------------------
    # Logging / state
    # ------------------------------------------------------------------
    def publish_state(self, text: str) -> None:
        msg = String()
        msg.data = str(text)
        self.state_pub.publish(msg)
        self.get_logger().info(str(text))

    def set_mode(self, mode: str, extra: str = '') -> None:
        self.mode = mode
        if bool(self.p('print_mode_change')) and mode != self.last_mode:
            self.last_mode = mode
            text = f'mode:{mode}'
            if extra:
                text += f' | {extra}'
            self.publish_state(text)

    def warn_throttled(self, key: str, text: str, period_sec: float) -> None:
        now = time.monotonic()
        last = self.last_warning_times.get(key, 0.0)
        if (now - last) >= max(0.0, period_sec):
            self.last_warning_times[key] = now
            self.get_logger().warning(text)

    # ------------------------------------------------------------------
    # Core filtering
    # ------------------------------------------------------------------
    def hard_clamp(self, cmd: Twist) -> Twist:
        out = self.copy_twist(cmd)

        max_vx = abs(float(self.p('hard_max_vx')))
        max_reverse_vx = abs(float(self.p('hard_max_reverse_vx')))
        max_vy = abs(float(self.p('hard_max_vy')))
        max_wz = abs(float(self.p('hard_max_wz')))

        reverse_policy = str(self.p('reverse_policy')).lower()
        if reverse_policy == 'allow':
            out.linear.x = clamp(out.linear.x, -max_vx, max_vx)
        elif reverse_policy == 'block':
            out.linear.x = clamp(out.linear.x, 0.0, max_vx)
        else:
            # Default and invalid-value fallback: limited reverse.
            out.linear.x = clamp(out.linear.x, -max_reverse_vx, max_vx)

        out.linear.y = clamp(out.linear.y, -max_vy, max_vy)
        out.angular.z = clamp(out.angular.z, -max_wz, max_wz)
        return out

    def apply_in_place_wz_adapter(self, cmd: Twist) -> Tuple[Twist, bool]:
        if not bool(self.p('enable_in_place_wz_adapter')):
            return cmd, False

        linear_threshold = abs(float(self.p('in_place_linear_threshold')))
        request_threshold = abs(float(self.p('in_place_wz_request_threshold')))
        hw_min_wz = abs(float(self.p('hw_min_wz')))

        linear_speed = self.planar_speed(cmd)
        wz_abs = abs(float(cmd.angular.z))

        if (
            linear_speed <= linear_threshold
            and request_threshold <= wz_abs < hw_min_wz
        ):
            out = self.copy_twist(cmd)
            out.angular.z = math.copysign(hw_min_wz, cmd.angular.z)
            return out, True

        return cmd, False

    def apply_zero_cross_guard(
        self,
        cmd: Twist,
        now: float,
        odom_fresh: bool,
    ) -> Tuple[Twist, Optional[str], str]:
        if not bool(self.p('enable_zero_cross_guard')):
            self.zero_cross_active = False
            return cmd, None, ''

        if not odom_fresh:
            self.zero_cross_active = False
            return cmd, None, ''

        request_threshold = abs(float(self.p('zero_cross_request_threshold')))
        trigger_wz = abs(float(self.p('zero_cross_odom_trigger_wz')))
        release_wz = abs(float(self.p('zero_cross_odom_release_wz')))
        max_hold = max(0.0, float(self.p('zero_cross_max_hold_sec')))
        rearm = max(0.0, float(self.p('zero_cross_rearm_sec')))

        target_wz = float(cmd.angular.z)
        odom_wz = float(self.odom_wz)

        target_sign = sign(target_wz)
        odom_sign = sign(odom_wz)
        opposite_sign_request = (
            abs(target_wz) >= request_threshold
            and target_sign != 0
            and odom_sign != 0
            and target_sign != odom_sign
        )
        opposite_request = opposite_sign_request and abs(odom_wz) >= trigger_wz

        if self.zero_cross_active:
            # New target no longer requests a meaningful opposite rotation.
            if not opposite_sign_request:
                self.zero_cross_active = False
                return cmd, None, ''

            # Actual angular velocity has sufficiently decayed.
            if abs(odom_wz) <= release_wz:
                self.zero_cross_active = False
                return cmd, None, ''

            elapsed = now - self.zero_cross_started
            if max_hold > 0.0 and elapsed >= max_hold:
                self.zero_cross_active = False
                self.zero_cross_bypass_until = now + rearm
                self.warn_throttled(
                    'zero_cross_timeout',
                    'ZERO_CROSS_HOLD reached max hold time; releasing guard. '
                    f'odom_wz={odom_wz:.3f}, target_wz={target_wz:.3f}',
                    0.5,
                )
                return cmd, 'ZERO_CROSS_TIMEOUT_RELEASE', (
                    f'odom_wz={odom_wz:.3f} target_wz={target_wz:.3f}'
                )

            return self.make_stop(), 'ZERO_CROSS_HOLD', (
                f'odom_wz={odom_wz:.3f} target_wz={target_wz:.3f} '
                f'elapsed={elapsed:.2f}'
            )

        if now < self.zero_cross_bypass_until:
            return cmd, None, ''

        if opposite_request:
            self.zero_cross_active = True
            self.zero_cross_started = now
            return self.make_stop(), 'ZERO_CROSS_HOLD', (
                f'odom_wz={odom_wz:.3f} target_wz={target_wz:.3f}'
            )

        return cmd, None, ''

    def apply_turn_exit_guard(
        self,
        cmd: Twist,
        now: float,
        odom_fresh: bool,
    ) -> Tuple[Twist, Optional[str], str]:
        if not bool(self.p('enable_turn_exit_guard')):
            self.turn_exit_active = False
            return cmd, None, ''

        if not odom_fresh:
            self.turn_exit_active = False
            return cmd, None, ''

        linear_request_threshold = abs(
            float(self.p('turn_exit_linear_request_threshold'))
        )
        target_wz_threshold = abs(float(self.p('turn_exit_target_wz_threshold')))
        trigger_wz = abs(float(self.p('turn_exit_odom_trigger_wz')))
        release_wz = abs(float(self.p('turn_exit_odom_release_wz')))
        max_hold = max(0.0, float(self.p('turn_exit_max_hold_sec')))
        rearm = max(0.0, float(self.p('turn_exit_rearm_sec')))

        linear_request = self.planar_speed(cmd)
        target_wz = float(cmd.angular.z)
        odom_wz = float(self.odom_wz)

        straight_request = (
            linear_request >= linear_request_threshold
            and abs(target_wz) <= target_wz_threshold
        )
        residual_turn = abs(odom_wz) >= trigger_wz

        if self.turn_exit_active:
            # A new meaningful turning command should not be blocked.
            if not straight_request:
                self.turn_exit_active = False
                return cmd, None, ''

            if abs(odom_wz) <= release_wz:
                self.turn_exit_active = False
                return cmd, None, ''

            elapsed = now - self.turn_exit_started
            if max_hold > 0.0 and elapsed >= max_hold:
                self.turn_exit_active = False
                self.turn_exit_bypass_until = now + rearm
                self.warn_throttled(
                    'turn_exit_timeout',
                    'TURN_EXIT_HOLD reached max hold time; releasing guard. '
                    f'odom_wz={odom_wz:.3f}, target=('
                    f'{cmd.linear.x:.3f},{cmd.linear.y:.3f},{target_wz:.3f})',
                    0.5,
                )
                return cmd, 'TURN_EXIT_TIMEOUT_RELEASE', (
                    f'odom_wz={odom_wz:.3f} elapsed={elapsed:.2f}'
                )

            guarded = self.copy_twist(cmd)
            max_vx = abs(float(self.p('turn_exit_max_vx')))
            max_vy = abs(float(self.p('turn_exit_max_vy')))
            guarded.linear.x = clamp(guarded.linear.x, -max_vx, max_vx)
            guarded.linear.y = clamp(guarded.linear.y, -max_vy, max_vy)
            guarded.angular.z = 0.0
            return guarded, 'TURN_EXIT_HOLD', (
                f'odom_wz={odom_wz:.3f} elapsed={elapsed:.2f}'
            )

        if now < self.turn_exit_bypass_until:
            return cmd, None, ''

        if straight_request and residual_turn:
            self.turn_exit_active = True
            self.turn_exit_started = now

            guarded = self.copy_twist(cmd)
            max_vx = abs(float(self.p('turn_exit_max_vx')))
            max_vy = abs(float(self.p('turn_exit_max_vy')))
            guarded.linear.x = clamp(guarded.linear.x, -max_vx, max_vx)
            guarded.linear.y = clamp(guarded.linear.y, -max_vy, max_vy)
            guarded.angular.z = 0.0
            return guarded, 'TURN_EXIT_HOLD', f'odom_wz={odom_wz:.3f}'

        return cmd, None, ''

    # ------------------------------------------------------------------
    # Overspeed monitor
    # ------------------------------------------------------------------
    def overspeed_action(self) -> str:
        action = str(self.p('overspeed_action')).lower()
        if action not in {'off', 'warn', 'stop'}:
            return 'warn'
        return action

    def overspeed_reasons(self, odom_fresh: bool) -> List[str]:
        if not odom_fresh or self.overspeed_action() == 'off':
            return []

        ratio = max(1.0, float(self.p('overspeed_ratio')))
        cmd = self.last_output_cmd
        reasons: List[str] = []

        axis_values = (
            (
                'vx',
                abs(self.odom_vx),
                abs(float(cmd.linear.x)),
                abs(float(self.p('overspeed_min_cmd_vx'))),
                abs(float(self.p('overspeed_vx_margin'))),
                abs(float(self.p('odom_abs_vx_limit'))),
            ),
            (
                'vy',
                abs(self.odom_vy),
                abs(float(cmd.linear.y)),
                abs(float(self.p('overspeed_min_cmd_vy'))),
                abs(float(self.p('overspeed_vy_margin'))),
                abs(float(self.p('odom_abs_vy_limit'))),
            ),
            (
                'wz',
                abs(self.odom_wz),
                abs(float(cmd.angular.z)),
                abs(float(self.p('overspeed_min_cmd_wz'))),
                abs(float(self.p('overspeed_wz_margin'))),
                abs(float(self.p('odom_abs_wz_limit'))),
            ),
        )

        for name, odom_value, cmd_value, min_cmd, margin, absolute_limit in axis_values:
            absolute_exceeded = absolute_limit > 0.0 and odom_value > absolute_limit
            relative_exceeded = (
                cmd_value >= min_cmd
                and odom_value > (ratio * cmd_value + margin)
            )

            if absolute_exceeded or relative_exceeded:
                reasons.append(
                    f'{name}:odom={odom_value:.3f},cmd={cmd_value:.3f},'
                    f'abs_lim={absolute_limit:.3f},rel_lim={ratio * cmd_value + margin:.3f}'
                )

        return reasons

    def update_overspeed_monitor(self, now: float, odom_fresh: bool) -> bool:
        action = self.overspeed_action()
        if action == 'off':
            self.overspeed_since = None
            self.overspeed_reported = False
            return False

        reasons = self.overspeed_reasons(odom_fresh)
        if not reasons:
            self.overspeed_since = None
            self.overspeed_reported = False
            return now < self.overspeed_stop_until

        if self.overspeed_since is None:
            self.overspeed_since = now

        confirm_sec = max(0.0, float(self.p('overspeed_confirm_sec')))
        confirmed = (now - self.overspeed_since) >= confirm_sec
        if not confirmed:
            return now < self.overspeed_stop_until

        reason_text = ' | '.join(reasons)
        if not self.overspeed_reported:
            self.overspeed_reported = True
            self.get_logger().warning(f'ODOM_OVERSPEED confirmed | {reason_text}')

            state_msg = String()
            state_msg.data = f'event:ODOM_OVERSPEED | action={action} | {reason_text}'
            self.state_pub.publish(state_msg)

        if action == 'stop':
            hold = max(0.0, float(self.p('overspeed_stop_hold_sec')))
            self.overspeed_stop_until = max(self.overspeed_stop_until, now + hold)

        return now < self.overspeed_stop_until

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish_output(self, cmd: Twist) -> None:
        out = self.hard_clamp(cmd)
        self.cmd_pub.publish(out)
        self.last_output_cmd = self.copy_twist(out)

    def publish_immediate_stop(self, mode: str, extra: str = '') -> None:
        self.zero_cross_active = False
        self.turn_exit_active = False
        stop = self.make_stop()
        self.cmd_pub.publish(stop)
        self.last_output_cmd = stop
        self.set_mode(mode, extra)

    # ------------------------------------------------------------------
    # Main timer
    # ------------------------------------------------------------------
    def timer_cb(self) -> None:
        now = time.monotonic()
        odom_fresh = self.odom_is_fresh(now)

        # 0) External safety latches have absolute priority.
        if self.estop_latched:
            self.publish_immediate_stop(
                'EMERGENCY_STOP_LATCHED',
                'reset_required=/emergency_reset',
            )
            return

        if self.go_home_hold:
            self.publish_immediate_stop('GO_HOME_HOLD')
            return

        # 1) Stale command -> immediate zero. No acceleration ramp.
        command_timed_out = (
            (not self.have_cmd)
            or ((now - self.last_cmd_time) > max(0.01, float(self.p('cmd_timeout_sec'))))
        )
        if command_timed_out:
            if bool(self.p('publish_zero_when_idle')):
                self.publish_immediate_stop('TIMEOUT_STOP')
            return

        # 2) Invalid command -> immediate zero.
        if not self.twist_is_finite(self.raw_cmd):
            self.publish_immediate_stop('INVALID_COMMAND_STOP', 'NaN/Inf input')
            return

        # 3) Optional odometry overspeed stop latch.
        if self.update_overspeed_monitor(now, odom_fresh):
            self.publish_immediate_stop('OVERSPEED_STOP')
            return

        # 4) Absolute clamp only. Normal DWB shaping remains untouched.
        target = self.hard_clamp(self.raw_cmd)
        mode = 'PASS_THROUGH'
        extra = ''

        # 5) Only in-place small-wz commands get hardware dead-zone compensation.
        target, in_place_boosted = self.apply_in_place_wz_adapter(target)
        if in_place_boosted:
            mode = 'IN_PLACE_WZ_BOOST'
            extra = (
                f'raw_wz={self.raw_cmd.angular.z:.3f} '
                f'boosted_wz={target.angular.z:.3f}'
            )

        # 6) Actual odometry-based opposite-direction protection.
        target, guard_mode, guard_extra = self.apply_zero_cross_guard(
            target,
            now,
            odom_fresh,
        )
        if guard_mode is not None:
            mode = guard_mode
            extra = guard_extra
        else:
            # 7) Actual odometry-based turn-to-straight protection.
            target, guard_mode, guard_extra = self.apply_turn_exit_guard(
                target,
                now,
                odom_fresh,
            )
            if guard_mode is not None:
                mode = guard_mode
                extra = guard_extra

        if not odom_fresh and (
            bool(self.p('enable_zero_cross_guard'))
            or bool(self.p('enable_turn_exit_guard'))
            or self.overspeed_action() != 'off'
        ):
            self.warn_throttled(
                'odom_stale',
                'Odometry is missing/stale; odom-based guards are temporarily bypassed.',
                1.0,
            )
            if mode == 'PASS_THROUGH':
                mode = 'PASS_THROUGH_ODOM_STALE'

        # 8) Final hard clamp and publish. No general acceleration limiter.
        self.publish_output(target)
        self.set_mode(mode, extra)

        # Periodic DEBUG-level trace.
        debug_period = max(0.05, float(self.p('debug_period_sec')))
        if (now - self.last_debug_time) >= debug_period:
            self.last_debug_time = now
            out = self.last_output_cmd
            self.get_logger().debug(
                f'filter_v3 mode={self.mode} '
                f'raw=({self.raw_cmd.linear.x:.3f},'
                f'{self.raw_cmd.linear.y:.3f},'
                f'{self.raw_cmd.angular.z:.3f}) '
                f'out=({out.linear.x:.3f},'
                f'{out.linear.y:.3f},'
                f'{out.angular.z:.3f}) '
                f'odom=({self.odom_vx:.3f},'
                f'{self.odom_vy:.3f},'
                f'{self.odom_wz:.3f}) '
                f'odom_fresh={odom_fresh}'
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VelFilterV3Node()
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
