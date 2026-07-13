#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class AgvAlignBridgeNode(Node):
    """
    /agv_align geometry_msgs/msg/Twist -> /cmd_vel_nav bridge.

    목적:
      외부 brain_node가 로봇팔 작업 중 AGV 미세 보정이 필요할 때
      /cmd_vel에 직접 쓰지 않고 /agv_align으로 요청하게 한다.
      이 bridge는 허용 상태에서만 /cmd_vel_nav로 전달한다.

    권장 체인:
      external brain_node -> /agv_align -> agv_align_bridge_node -> /cmd_vel_nav
      -> cmd_vel_safety_filter -> /cmd_vel -> myAGV driver

    Safety:
      - /agv_align_enable true일 때만 전달
      - command timeout이 지나면 0 publish
      - vx/vy/wz clamp
      - optional brain_status fallback parsing
    """

    def __init__(self):
        super().__init__('agv_align_bridge_node')

        # ============================================================
        # Topic params
        # ============================================================
        self.declare_parameter('input_topic', '/agv_align')
        self.declare_parameter('output_topic', '/cmd_vel_nav')
        self.declare_parameter('enable_topic', '/agv_align_enable')
        self.declare_parameter('brain_status_topic', '/brain_status')
        self.declare_parameter('status_topic', '/agv_align_bridge/status')

        # ============================================================
        # Runtime params
        # ============================================================
        self.declare_parameter('publish_hz', 20.0)
        self.declare_parameter('cmd_timeout_sec', 0.35)
        self.declare_parameter('require_enable', True)
        self.declare_parameter('allow_brain_status_fallback', True)

        # low-speed clamp for arm-assist alignment
        self.declare_parameter('max_vx', 0.100)
        self.declare_parameter('max_vy', 0.100)
        self.declare_parameter('max_wz', 0.400)

        # block unused axes for safety
        self.declare_parameter('block_z_axes', True)
        self.declare_parameter('print_debug', True)
        self.declare_parameter('debug_period_sec', 0.50)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.enable_topic = self.get_parameter('enable_topic').value
        self.brain_status_topic = self.get_parameter('brain_status_topic').value
        self.status_topic = self.get_parameter('status_topic').value

        # ============================================================
        # State
        # ============================================================
        self.enabled = False
        self.last_cmd = Twist()
        self.last_cmd_time = 0.0
        self.last_pub_was_stop = True
        self.last_debug_time = 0.0
        self.last_enable_reason = 'init_false'

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.cmd_sub = self.create_subscription(
            Twist,
            self.input_topic,
            self.cmd_cb,
            qos,
        )

        self.enable_sub = self.create_subscription(
            Bool,
            self.enable_topic,
            self.enable_cb,
            10,
        )

        self.brain_status_sub = self.create_subscription(
            String,
            self.brain_status_topic,
            self.brain_status_cb,
            10,
        )

        self.cmd_pub = self.create_publisher(Twist, self.output_topic, qos)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        publish_hz = float(self.get_parameter('publish_hz').value)
        self.timer = self.create_timer(1.0 / max(publish_hz, 1.0), self.timer_cb)

        self.publish_status(
            f'agv_align_bridge_ready | input={self.input_topic} output={self.output_topic} '
            f'enable={self.enable_topic} require_enable={bool(self.p("require_enable"))}'
        )

    def p(self, name):
        return self.get_parameter(name).value

    def publish_status(self, text):
        msg = String()
        msg.data = str(text)
        self.status_pub.publish(msg)
        self.get_logger().info(str(text))

    def set_enabled(self, value, reason=''):
        value = bool(value)
        if value != self.enabled:
            self.enabled = value
            self.last_enable_reason = reason or 'unknown'
            self.publish_status(f'enable_changed:{self.enabled}:reason={self.last_enable_reason}')
            if not self.enabled:
                self.publish_stop()
        else:
            self.last_enable_reason = reason or self.last_enable_reason

    def enable_cb(self, msg):
        self.set_enabled(bool(msg.data), reason='enable_topic')

    def brain_status_cb(self, msg):
        if not bool(self.p('allow_brain_status_fallback')):
            return

        text = str(msg.data)
        # mission_brain_node publishes e.g. brain_state:WAIT_STOP_OBJ_DELAY->WAIT_PICKED
        if text.startswith('brain_state:') and '->' in text:
            state = text.split('->', 1)[1].strip()
            # WAIT_PICKED: object station, picked retry window.
            # WAIT_NEXT: QR/placed retry window in current mission_brain design.
            if state in ['WAIT_PICKED', 'WAIT_NEXT']:
                self.set_enabled(True, reason=f'brain_status:{state}')
            else:
                self.set_enabled(False, reason=f'brain_status:{state}')

    def cmd_cb(self, msg):
        self.last_cmd = msg
        self.last_cmd_time = time.monotonic()

    def filtered_cmd(self, msg):
        out = Twist()

        max_vx = abs(float(self.p('max_vx')))
        max_vy = abs(float(self.p('max_vy')))
        max_wz = abs(float(self.p('max_wz')))

        out.linear.x = clamp(float(msg.linear.x), -max_vx, max_vx)
        out.linear.y = clamp(float(msg.linear.y), -max_vy, max_vy)
        out.angular.z = clamp(float(msg.angular.z), -max_wz, max_wz)

        if bool(self.p('block_z_axes')):
            out.linear.z = 0.0
            out.angular.x = 0.0
            out.angular.y = 0.0
        else:
            out.linear.z = float(msg.linear.z)
            out.angular.x = float(msg.angular.x)
            out.angular.y = float(msg.angular.y)

        return out

    def publish_stop(self):
        msg = Twist()
        self.cmd_pub.publish(msg)
        self.last_pub_was_stop = True

    def timer_cb(self):
        now = time.monotonic()
        timeout = float(self.p('cmd_timeout_sec'))
        require_enable = bool(self.p('require_enable'))

        allowed = (self.enabled or not require_enable)
        fresh = self.last_cmd_time > 0.0 and (now - self.last_cmd_time) <= timeout

        if allowed and fresh:
            out = self.filtered_cmd(self.last_cmd)
            self.cmd_pub.publish(out)
            self.last_pub_was_stop = False

            if bool(self.p('print_debug')) and (now - self.last_debug_time) >= float(self.p('debug_period_sec')):
                self.last_debug_time = now
                self.get_logger().info(
                    f'forward /agv_align -> {self.output_topic} | '
                    f'vx={out.linear.x:.4f}, vy={out.linear.y:.4f}, wz={out.angular.z:.4f}'
                )
            return

        # If disabled or stale, publish one stop then stay quiet.
        if not self.last_pub_was_stop:
            self.publish_stop()
            if not allowed:
                self.publish_status(f'stop:disabled:reason={self.last_enable_reason}')
            elif not fresh:
                self.publish_status(f'stop:timeout:{timeout:.2f}s')


def main(args=None):
    rclpy.init(args=args)
    node = AgvAlignBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_stop()
            time.sleep(0.05)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
