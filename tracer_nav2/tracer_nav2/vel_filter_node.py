#!/usr/bin/env python3
"""
Tracer Nav2 comparison filter (V2 algorithm).

Placement:
  ~/myagv_ros2/src/tracer_nav2/tracer_nav2/vel_filter_node.py

This file intentionally preserves the uploaded vel_filter_node.py behavior:
  - STRAIGHT / TURN / LATERAL mode separation
  - acceleration limiting
  - pulse/dead-zone adapter
  - command-state based zero-cross / turn-exit handling
  - emergency-stop and go-home safety latches

It is provided inside tracer_nav2 so it can be compared, one run at a time,
against tracer_nav2.vel_filter_v3_node.
"""

import time
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sgn(x, eps=1e-9):
    if x > eps:
        return 1.0
    if x < -eps:
        return -1.0
    return 0.0


def approach(current, target, step):
    if current < target:
        return min(current + step, target)
    if current > target:
        return max(current - step, target)
    return current


class CmdVelSafetyFilter(Node):
    """
    myAGV용 cmd_vel safety filter.

    역할:
      Nav2의 /cmd_vel_nav를 myAGV가 수행 가능한 motion primitive로 변환한다.

    Mode:
      STOP
      STRAIGHT
      TURN
      LATERAL
      INPLACE_SMALL_TURN
      TURN_EXIT_HOLD

    핵심:
      - 직진 중 angular.z 제거
      - 회전 중 linear.x 제거
      - vx 거의 0 + 작은 wz 지속 = 제자리 회전 의도라고 판단하고 boost
      - TURN 종료 후 완전 정지 hold
      - zero-cross 시 0을 먼저 거침
      - 후진 명령은 기본 차단
    """

    def __init__(self):
        super().__init__('cmd_vel_safety_filter')

        # ============================================================
        # Topic params
        # ============================================================
        self.declare_parameter('input_topic', '/cmd_vel_nav')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('state_topic', '/cmd_vel_safety_filter/state')
        self.declare_parameter('emergency_stop_topic', '/emergency_stop')
        self.declare_parameter('emergency_reset_topic', '/emergency_reset')
        self.declare_parameter('go_home_topic', '/go_home')
        self.declare_parameter('go_home_release_topic', '/go_home_release')

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.state_topic = self.get_parameter('state_topic').value
        self.emergency_stop_topic = self.get_parameter('emergency_stop_topic').value
        self.emergency_reset_topic = self.get_parameter('emergency_reset_topic').value
        self.go_home_topic = self.get_parameter('go_home_topic').value
        self.go_home_release_topic = self.get_parameter('go_home_release_topic').value

        # ============================================================
        # Runtime params
        # ============================================================
        self.declare_parameter('publish_hz', 40.0)
        self.declare_parameter('cmd_timeout', 0.30)

        # ============================================================
        # Basic velocity limits
        # ============================================================
        self.declare_parameter('max_vx', 0.20)
        self.declare_parameter('max_vy', 0.06)
        self.declare_parameter('max_wz', 0.40)

        self.declare_parameter('block_reverse', True)

        # Acceleration limits
        self.declare_parameter('max_acc_vx', 0.12)
        self.declare_parameter('max_acc_vy', 0.15)
        self.declare_parameter('max_acc_wz', 0.35)

        # Deadband
        self.declare_parameter('linear_deadband', 0.005)
        self.declare_parameter('angular_deadband', 0.03)

        # ============================================================
        # Motion separator params
        # ============================================================
        self.declare_parameter('straight_vx_on', 0.0)
        self.declare_parameter('straight_wz_kill', 0.35)

        self.declare_parameter('turn_wz_on', 0.40)
        self.declare_parameter('turn_wz_off', 0.30)
        self.declare_parameter('turn_max_vx', 0.00)

        self.declare_parameter('kill_vy_in_straight', False)
        self.declare_parameter('kill_vy_in_turn', True)

        # ============================================================
        # In-place small turn params
        # ============================================================
        # Nav2가 180도 회전해야 하는데도 wz=-0.02 같은 작은 값을 지속적으로 내는 경우 처리.
        self.declare_parameter('enable_inplace_small_turn', True)

        # vx가 이 값 이하이면 "거의 제자리 상태"로 판단
        self.declare_parameter('inplace_vx_max', 0.03)

        # 이 범위의 작은 wz가 지속되면 제자리 회전 의도라고 판단
        self.declare_parameter('inplace_wz_min', 0.010)
        self.declare_parameter('inplace_wz_max', 0.080)

        # 같은 방향 작은 wz가 이 시간 이상 지속되어야 boost
        self.declare_parameter('inplace_hold_sec', 0.30)

        # boost할 회전 속도. 보통 hw_min_wz와 같게 둔다.
        self.declare_parameter('inplace_boost_wz', 0.40)

        # 작은 wz 감지 중 허용할 target sign reset 시간
        self.declare_parameter('inplace_reset_sec', 0.60)

        # ============================================================
        # Deadzone adapter / pulse params
        # ============================================================
        # STRAIGHT에서는 사용하지 않고 TURN/INPLACE_SMALL_TURN에서만 사용.
        self.declare_parameter('enable_wz_deadzone_adapter', True)
        self.declare_parameter('hw_min_wz', 0.40)

        # TURN 모드에서 작은 wz를 pulse로 보정할지.
        self.declare_parameter('use_pulse_adapter', True)
        self.declare_parameter('min_pulse_duty', 0.35)
        self.declare_parameter('max_pulse_duty', 0.60)
        self.declare_parameter('pulse_period', 0.45)

        # INPLACE_SMALL_TURN에서도 pulse를 쓸지.
        # 180도 회전이 안 먹으면 false로 두고 연속 boost가 낫다.
        self.declare_parameter('inplace_use_pulse_adapter', False)

        # ============================================================
        # Zero-cross guard params
        # ============================================================
        self.declare_parameter('enable_zero_cross_guard', True)
        self.declare_parameter('zero_cross_threshold', 0.05)
        self.declare_parameter('zero_cross_hold_sec', 0.35)

        # ============================================================
        # Turn-exit hold params
        # ============================================================
        self.declare_parameter('enable_turn_exit_hold', True)
        self.declare_parameter('turn_exit_hold_sec', 0.70)
        self.declare_parameter('turn_exit_min_prev_wz', 0.10)
        self.declare_parameter('allow_turn_during_exit_hold', False)

        # ============================================================
        # Debug params
        # ============================================================
        self.declare_parameter('print_mode_change', True)

        # ============================================================
        # State
        # ============================================================
        self.target = Twist()
        self.current = Twist()

        self.last_cmd_time = time.time()
        self.last_update_time = time.time()

        self.mode = 'STOP'
        self.prev_mode = 'STOP'

        self.last_nonzero_wz_sign = 0.0
        self.zero_hold_until = 0.0
        self.turn_exit_hold_until = 0.0

        self.pulse_start_time = time.time()
        self.last_pulse_sign = 0.0

        self.inplace_candidate_sign = 0.0
        self.inplace_candidate_start = 0.0
        self.inplace_candidate_last = 0.0

        # ============================================================
        # External safety latch state
        # ============================================================
        # E-stop remains active until /emergency_reset.
        # go_home hold remains active until mission_brain publishes
        # /go_home_release after odometry stop confirmation.
        self.estop_latched = False
        self.go_home_hold = False

        # ============================================================
        # QoS
        # ============================================================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(
            Twist,
            self.input_topic,
            self.cmd_cb,
            qos
        )

        self.estop_sub = self.create_subscription(
            String,
            self.emergency_stop_topic,
            self.emergency_stop_cb,
            10
        )
        self.estop_reset_sub = self.create_subscription(
            String,
            self.emergency_reset_topic,
            self.emergency_reset_cb,
            10
        )
        self.go_home_sub = self.create_subscription(
            String,
            self.go_home_topic,
            self.go_home_cb,
            10
        )
        self.go_home_release_sub = self.create_subscription(
            String,
            self.go_home_release_topic,
            self.go_home_release_cb,
            10
        )

        self.pub = self.create_publisher(
            Twist,
            self.output_topic,
            qos
        )
        self.state_pub = self.create_publisher(
            String,
            self.state_topic,
            10
        )

        publish_hz = float(self.get_parameter('publish_hz').value)
        self.timer = self.create_timer(1.0 / publish_hz, self.update)

        self.get_logger().info(
            f'cmd_vel_safety_filter started | input={self.input_topic}, '
            f'output={self.output_topic}, publish_hz={publish_hz:.1f}'
        )

        self.get_logger().info(
            'motion separator enabled | STRAIGHT / TURN / INPLACE_SMALL_TURN / TURN_EXIT_HOLD'
        )
        self.publish_state(
            'primitive_vel_filter_safety_ready | '
            f'estop={self.emergency_stop_topic} reset={self.emergency_reset_topic} '
            f'go_home={self.go_home_topic} go_home_release={self.go_home_release_topic}'
        )

    # ============================================================
    # Callback
    # ============================================================
    def cmd_cb(self, msg):
        self.target = msg
        self.last_cmd_time = time.time()

    def publish_state(self, text):
        msg = String()
        msg.data = str(text)
        self.state_pub.publish(msg)
        self.get_logger().info(str(text))

    def reset_motion_memory(self):
        self.target = Twist()
        self.current = Twist()
        self.last_cmd_time = 0.0
        self.zero_hold_until = 0.0
        self.turn_exit_hold_until = 0.0
        self.last_nonzero_wz_sign = 0.0
        self.last_pulse_sign = 0.0
        self.inplace_candidate_sign = 0.0
        self.inplace_candidate_start = 0.0
        self.inplace_candidate_last = 0.0

    def emergency_stop_cb(self, msg):
        # Any received String payload triggers the latch.
        del msg
        first_latch = not self.estop_latched
        self.estop_latched = True
        self.go_home_hold = False
        self.reset_motion_memory()
        self.set_mode('EMERGENCY_STOP_LATCHED')
        self.publish_forced_stop()
        if first_latch:
            self.publish_state('event:EMERGENCY_STOP_LATCHED | reset_required=/emergency_reset')

    def emergency_reset_cb(self, msg):
        # Any received String payload releases the E-stop latch.
        del msg
        was_latched = self.estop_latched
        self.estop_latched = False
        self.reset_motion_memory()
        self.set_mode('STOP')
        self.publish_forced_stop()
        if was_latched:
            self.publish_state('event:EMERGENCY_STOP_RESET')

    def go_home_cb(self, msg):
        # Stop immediately while mission_brain cancels the current primitive route.
        del msg
        if self.estop_latched:
            self.publish_state('event:GO_HOME_REJECTED | reason=estop_latched')
            return
        self.go_home_hold = True
        self.reset_motion_memory()
        self.set_mode('GO_HOME_HOLD')
        self.publish_forced_stop()
        self.publish_state('event:GO_HOME_HOLD_STARTED')

    def go_home_release_cb(self, msg):
        # Released only after mission_brain confirms stop or reaches max wait.
        del msg
        if self.estop_latched:
            self.publish_state('event:GO_HOME_RELEASE_IGNORED | reason=estop_latched')
            return
        was_active = self.go_home_hold
        self.go_home_hold = False
        self.reset_motion_memory()
        self.set_mode('STOP')
        self.publish_forced_stop()
        if was_active:
            self.publish_state('event:GO_HOME_HOLD_RELEASED')

    def p(self, name):
        return self.get_parameter(name).value

    def set_mode(self, new_mode):
        if new_mode != self.mode:
            if bool(self.p('print_mode_change')):
                self.get_logger().info(f'mode change: {self.mode} -> {new_mode}')
            self.prev_mode = self.mode
            self.mode = new_mode

    # ============================================================
    # In-place small turn detection
    # ============================================================
    def update_inplace_candidate(self, vx, wz, now):
        if not bool(self.p('enable_inplace_small_turn')):
            self.inplace_candidate_sign = 0.0
            self.inplace_candidate_start = 0.0
            self.inplace_candidate_last = 0.0
            return False

        avx = abs(vx)
        awz = abs(wz)

        inplace_vx_max = float(self.p('inplace_vx_max'))
        wz_min = float(self.p('inplace_wz_min'))
        wz_max = float(self.p('inplace_wz_max'))
        hold_sec = float(self.p('inplace_hold_sec'))
        reset_sec = float(self.p('inplace_reset_sec'))

        valid = (avx <= inplace_vx_max and wz_min <= awz <= wz_max)

        if not valid:
            # 너무 오래 끊기면 candidate reset
            if self.inplace_candidate_last > 0.0 and now - self.inplace_candidate_last > reset_sec:
                self.inplace_candidate_sign = 0.0
                self.inplace_candidate_start = 0.0
                self.inplace_candidate_last = 0.0
            return False

        sign = sgn(wz)

        if sign == 0.0:
            return False

        # 새 방향이면 candidate 새로 시작
        if sign != self.inplace_candidate_sign:
            self.inplace_candidate_sign = sign
            self.inplace_candidate_start = now
            self.inplace_candidate_last = now
            return False

        self.inplace_candidate_last = now

        if now - self.inplace_candidate_start >= hold_sec:
            return True

        return False

    # ============================================================
    # Mode decision
    # ============================================================
    def decide_mode(self, vx, vy, wz, now):
        avx = abs(vx)
        avy = abs(vy)
        awz = abs(wz)

        straight_vx_on = float(self.p('straight_vx_on'))
        straight_wz_kill = float(self.p('straight_wz_kill'))
        turn_wz_on = float(self.p('turn_wz_on'))
        turn_wz_off = float(self.p('turn_wz_off'))
        angular_deadband = float(self.p('angular_deadband'))
        linear_deadband = float(self.p('linear_deadband'))

        # 1. 작은 wz가 제자리 상태에서 지속되면 180도 회전 같은 in-place turn 의도로 해석
        if self.update_inplace_candidate(vx, wz, now):
            return 'INPLACE_SMALL_TURN'

        # 2. 명확한 회전 의도
        if awz >= turn_wz_on:
            return 'TURN'

        # 3. 이전 TURN/INPLACE면 hysteresis로 유지
        if self.mode == 'TURN' and awz >= turn_wz_off:
            return 'TURN'

        if self.mode == 'INPLACE_SMALL_TURN':
            # candidate가 유지되는 동안은 계속 INPLACE 유지
            if self.inplace_candidate_sign != 0.0 and awz >= float(self.p('inplace_wz_min')):
                return 'INPLACE_SMALL_TURN'

        # 4. 순수 lateral align 의도.
        #    ArUco final align에서 vx=0, wz=0, vy!=0인 명령을 안전하게 통과시킨다.
        if avx <= linear_deadband and awz <= angular_deadband and avy > linear_deadband:
            return 'LATERAL'

        # 5. 전진 명령이 충분하고 회전 명령이 kill 영역이면 직진
        if avx >= straight_vx_on and awz <= straight_wz_kill:
            return 'STRAIGHT'

        # 5. 전진 명령이 있으면 애매한 구간도 직진 우선
        if avx >= straight_vx_on:
            return 'STRAIGHT'

        # 6. 전진은 거의 없고 회전이 deadband보다 크면 TURN
        if awz > angular_deadband:
            return 'TURN'

        return 'STOP'

    # ============================================================
    # Pulse helper
    # ============================================================
    def pulse_wz(self, wz, now):
        hw_min_wz = float(self.p('hw_min_wz'))
        min_duty = float(self.p('min_pulse_duty'))
        max_duty = float(self.p('max_pulse_duty'))
        period = float(self.p('pulse_period'))

        if period <= 1e-6:
            return wz

        sign = sgn(wz)
        if sign == 0.0:
            return 0.0

        if sign != self.last_pulse_sign:
            self.pulse_start_time = now
            self.last_pulse_sign = sign

        awz = abs(wz)

        if awz >= hw_min_wz:
            return wz

        duty = clamp(awz / hw_min_wz, min_duty, max_duty)

        phase = (now - self.pulse_start_time) % period
        on_time = period * duty

        if phase < on_time:
            return sign * hw_min_wz

        return 0.0

    def publish_forced_stop(self):
        out = Twist()
        self.current = out
        self.pub.publish(out)

    # ============================================================
    # Main update
    # ============================================================
    def update(self):
        now = time.time()

        # External safety commands have absolute priority over every primitive mode,
        # acceleration ramp, pulse adapter, and turn-exit state.
        if self.estop_latched:
            self.set_mode('EMERGENCY_STOP_LATCHED')
            self.publish_forced_stop()
            return

        if self.go_home_hold:
            self.set_mode('GO_HOME_HOLD')
            self.publish_forced_stop()
            return

        dt = now - self.last_update_time
        self.last_update_time = now

        if dt <= 0.0 or dt > 1.0:
            dt = 1.0 / float(self.p('publish_hz'))

        cmd_timeout = float(self.p('cmd_timeout'))

        # ------------------------------------------------------------
        # Timeout
        # ------------------------------------------------------------
        if now - self.last_cmd_time > cmd_timeout:
            self.set_mode('STOP')
            self.turn_exit_hold_until = 0.0
            self.zero_hold_until = 0.0
            self.last_nonzero_wz_sign = 0.0
            self.publish_forced_stop()
            return

        # ------------------------------------------------------------
        # Read raw target + clamp
        # ------------------------------------------------------------
        max_vx = float(self.p('max_vx'))
        max_vy = float(self.p('max_vy'))
        max_wz = float(self.p('max_wz'))

        vx = clamp(float(self.target.linear.x), -max_vx, max_vx)
        vy = clamp(float(self.target.linear.y), -max_vy, max_vy)
        wz = clamp(float(self.target.angular.z), -max_wz, max_wz)

        # 후진 차단
        if bool(self.p('block_reverse')) and vx < 0.0:
            vx = 0.0

        # ------------------------------------------------------------
        # Deadband
        # ------------------------------------------------------------
        if abs(vx) < float(self.p('linear_deadband')):
            vx = 0.0

        if abs(vy) < float(self.p('linear_deadband')):
            vy = 0.0

        # 중요:
        # inplace small turn을 잡아야 하므로, angular_deadband보다 작은 wz를 바로 죽이지 않는다.
        # 단, inplace_wz_min보다도 작은 건 제거한다.
        if abs(wz) < float(self.p('inplace_wz_min')):
            wz = 0.0

        # ------------------------------------------------------------
        # Decide requested mode
        # ------------------------------------------------------------
        old_mode = self.mode
        requested_mode = self.decide_mode(vx, vy, wz, now)

        # ------------------------------------------------------------
        # Turn-exit hold trigger
        # ------------------------------------------------------------
        if bool(self.p('enable_turn_exit_hold')):
            was_turn_like = (old_mode == 'TURN' or old_mode == 'INPLACE_SMALL_TURN')
            exiting_turn = (was_turn_like and requested_mode not in ['TURN', 'INPLACE_SMALL_TURN'])
            prev_wz_big = abs(self.current.angular.z) >= float(self.p('turn_exit_min_prev_wz'))

            if exiting_turn and prev_wz_big:
                self.turn_exit_hold_until = now + float(self.p('turn_exit_hold_sec'))
                self.zero_hold_until = 0.0
                self.last_nonzero_wz_sign = 0.0
                self.last_pulse_sign = 0.0
                self.inplace_candidate_sign = 0.0
                self.inplace_candidate_start = 0.0
                self.inplace_candidate_last = 0.0

                self.set_mode('TURN_EXIT_HOLD')
                self.get_logger().info(
                    f'TURN exit hold start: {float(self.p("turn_exit_hold_sec")):.2f}s'
                )
                self.publish_forced_stop()
                return

        # ------------------------------------------------------------
        # Turn-exit hold active
        # ------------------------------------------------------------
        if now < self.turn_exit_hold_until:
            if bool(self.p('allow_turn_during_exit_hold')):
                if requested_mode in ['TURN', 'INPLACE_SMALL_TURN']:
                    self.turn_exit_hold_until = 0.0
                    self.set_mode(requested_mode)
                else:
                    self.set_mode('TURN_EXIT_HOLD')
                    self.publish_forced_stop()
                    return
            else:
                self.set_mode('TURN_EXIT_HOLD')
                self.publish_forced_stop()
                return

        if self.mode == 'TURN_EXIT_HOLD' and now >= self.turn_exit_hold_until:
            self.turn_exit_hold_until = 0.0
            self.set_mode(requested_mode)
        else:
            self.set_mode(requested_mode)

        # ------------------------------------------------------------
        # Mode policy
        # ------------------------------------------------------------
        if self.mode == 'STRAIGHT':
            wz = 0.0

            if bool(self.p('kill_vy_in_straight')):
                vy = 0.0

        elif self.mode == 'LATERAL':
            # Pure lateral alignment mode for mecanum final alignment.
            # Keep lateral y only, remove forward and yaw.
            vx = 0.0
            wz = 0.0

        elif self.mode == 'TURN':
            turn_max_vx = float(self.p('turn_max_vx'))

            if abs(vx) > turn_max_vx:
                vx = sgn(vx) * turn_max_vx

            if bool(self.p('kill_vy_in_turn')):
                vy = 0.0

            if bool(self.p('enable_wz_deadzone_adapter')):
                hw_min_wz = float(self.p('hw_min_wz'))
                if wz != 0.0 and abs(wz) < hw_min_wz:
                    if bool(self.p('use_pulse_adapter')):
                        wz = self.pulse_wz(wz, now)
                    else:
                        wz = sgn(wz) * hw_min_wz

            wz = clamp(wz, -max_wz, max_wz)

        elif self.mode == 'INPLACE_SMALL_TURN':
            vx = 0.0
            vy = 0.0

            boost = float(self.p('inplace_boost_wz'))
            sign = self.inplace_candidate_sign if self.inplace_candidate_sign != 0.0 else sgn(wz)

            wz = sign * boost

            if bool(self.p('inplace_use_pulse_adapter')):
                wz = self.pulse_wz(wz, now)

            wz = clamp(wz, -max_wz, max_wz)

        else:
            vx = 0.0
            vy = 0.0
            wz = 0.0

        # ------------------------------------------------------------
        # Zero-cross guard
        # ------------------------------------------------------------
        if bool(self.p('enable_zero_cross_guard')):
            zthr = float(self.p('zero_cross_threshold'))
            hold_sec = float(self.p('zero_cross_hold_sec'))

            new_sign = sgn(wz, zthr)

            if new_sign != 0.0:
                if self.last_nonzero_wz_sign != 0.0 and new_sign != self.last_nonzero_wz_sign:
                    self.zero_hold_until = now + hold_sec
                    self.last_nonzero_wz_sign = 0.0

                    self.current.angular.z = 0.0
                    wz = 0.0

                    if self.mode in ['TURN', 'INPLACE_SMALL_TURN']:
                        vx = 0.0
                        vy = 0.0

                    self.get_logger().info(
                        f'zero-cross hold start: {hold_sec:.2f}s'
                    )
                else:
                    self.last_nonzero_wz_sign = new_sign

            if now < self.zero_hold_until:
                wz = 0.0
                if self.mode in ['TURN', 'INPLACE_SMALL_TURN']:
                    vx = 0.0
                    vy = 0.0

        # ------------------------------------------------------------
        # Desired output
        # ------------------------------------------------------------
        desired = Twist()

        desired.linear.x = vx
        desired.linear.y = vy
        desired.linear.z = 0.0

        desired.angular.x = 0.0
        desired.angular.y = 0.0
        desired.angular.z = wz

        # ------------------------------------------------------------
        # Acceleration limiting
        # ------------------------------------------------------------
        max_acc_vx = float(self.p('max_acc_vx'))
        max_acc_vy = float(self.p('max_acc_vy'))
        max_acc_wz = float(self.p('max_acc_wz'))

        out = Twist()

        out.linear.x = approach(
            self.current.linear.x,
            desired.linear.x,
            max_acc_vx * dt
        )

        out.linear.y = approach(
            self.current.linear.y,
            desired.linear.y,
            max_acc_vy * dt
        )

        out.angular.z = approach(
            self.current.angular.z,
            desired.angular.z,
            max_acc_wz * dt
        )

        if now < self.zero_hold_until:
            out.angular.z = 0.0

        out.linear.z = 0.0
        out.angular.x = 0.0
        out.angular.y = 0.0

        self.current = out
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSafetyFilter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        try:
            node.pub.publish(stop)
            time.sleep(0.05)
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
