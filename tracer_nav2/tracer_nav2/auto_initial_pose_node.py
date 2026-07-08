#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


class AutoInitialPoseNode(Node):
    """
    Auto initial pose + optional 360deg localization spin.

    목적:
      RViz 2D Pose Estimate를 매번 수동으로 누르지 않기 위해,
      시작 pose를 /initialpose로 자동 발행하고,
      필요하면 제자리 회전으로 AMCL 수렴을 돕는다.

    기본 가정:
      로봇은 대부분 parking_region 근처에서 시작한다.
    """

    def __init__(self):
        super().__init__('auto_initial_pose_node')

        # ============================================================
        # Topic params
        # ============================================================
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('amcl_topic', '/amcl_pose')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_nav')

        self.declare_parameter('cmd_topic', '/auto_localize_cmd')
        self.declare_parameter('status_topic', '/auto_localize_status')
        self.declare_parameter('ready_topic', '/localization_ready')

        # ============================================================
        # Start pose params
        # parking_region from your latest pose
        # ============================================================
        self.declare_parameter('start_x', 0.28207093477249146)
        self.declare_parameter('start_y', 0.02868373692035675)
        self.declare_parameter('start_yaw', 0.005267)

        # covariance: 너무 작으면 AMCL이 틀린 초기값에 과신할 수 있음
        self.declare_parameter('cov_xx', 0.05)
        self.declare_parameter('cov_yy', 0.05)
        self.declare_parameter('cov_yaw', 0.25)

        # ============================================================
        # Behavior params
        # ============================================================
        self.declare_parameter('auto_on_start', True)
        self.declare_parameter('republish_initialpose_count', 10)
        self.declare_parameter('initialpose_publish_hz', 5.0)

        # AMCL 확인
        self.declare_parameter('amcl_valid_timeout_sec', 2.0)
        self.declare_parameter('amcl_wait_after_initialpose_sec', 2.0)

        # covariance threshold
        # 너무 빡세게 잡으면 계속 실패할 수 있어서 초반엔 넉넉하게 둠
        self.declare_parameter('max_cov_xx', 1.0)
        self.declare_parameter('max_cov_yy', 1.0)
        self.declare_parameter('max_cov_yaw', 1.0)

        # 360 spin
        self.declare_parameter('enable_spin_scan', True)
        self.declare_parameter('spin_wz', 0.40)
        self.declare_parameter('spin_angle_rad', 6.28318530718)
        self.declare_parameter('spin_min_sec', 3.0)
        self.declare_parameter('spin_max_sec', 25.0)
        self.declare_parameter('stop_when_amcl_good_after_min_spin', True)

        # global localization service는 있으면 사용, 없으면 무시
        self.declare_parameter('use_global_localization_service', False)
        self.declare_parameter('global_localization_service', '/reinitialize_global_localization')

        self.initialpose_topic = self.get_parameter('initialpose_topic').value
        self.amcl_topic = self.get_parameter('amcl_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.ready_topic = self.get_parameter('ready_topic').value

        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            self.initialpose_topic,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, cmd_qos)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.ready_pub = self.create_publisher(Bool, self.ready_topic, 10)

        self.create_subscription(String, self.cmd_topic, self.cmd_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, self.amcl_topic, self.amcl_cb, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)

        self.global_loc_client = self.create_client(
            Empty,
            self.get_parameter('global_localization_service').value
        )

        self.state = 'IDLE'
        self.ready = False

        self.last_amcl_time = 0.0
        self.last_amcl_msg = None

        self.odom_yaw = None
        self.prev_odom_yaw_for_spin = None
        self.spin_accum = 0.0
        self.spin_start_time = 0.0

        self.initialpose_sent_count = 0
        self.last_initialpose_pub_time = 0.0
        self.wait_start_time = 0.0

        self.timer = self.create_timer(0.05, self.timer_cb)

        self.publish_ready(False)
        self.status('auto_initial_pose_ready')

        if bool(self.get_parameter('auto_on_start').value):
            self.start_localization('auto_on_start')

    # ============================================================
    # Callbacks
    # ============================================================
    def cmd_cb(self, msg):
        cmd = msg.data.strip().lower()
        if cmd in ['start', 'localize', 'reset']:
            self.start_localization(f'cmd:{cmd}')
        elif cmd == 'stop':
            self.stop_motion()
            self.state = 'IDLE'
            self.status('auto_localize_stopped')
        else:
            self.status(f'unknown_auto_localize_cmd:{cmd}')

    def amcl_cb(self, msg):
        self.last_amcl_msg = msg
        self.last_amcl_time = time.monotonic()

    def odom_cb(self, msg):
        yaw = normalize_angle(yaw_from_quat(msg.pose.pose.orientation))
        self.odom_yaw = yaw

        if self.state == 'SPIN_SCAN':
            if self.prev_odom_yaw_for_spin is None:
                self.prev_odom_yaw_for_spin = yaw
            else:
                dyaw = normalize_angle(yaw - self.prev_odom_yaw_for_spin)
                self.spin_accum += abs(dyaw)
                self.prev_odom_yaw_for_spin = yaw

    # ============================================================
    # Helpers
    # ============================================================
    def status(self, text):
        msg = String()
        msg.data = str(text)
        self.status_pub.publish(msg)
        self.get_logger().info(str(text))

    def publish_ready(self, value):
        self.ready = bool(value)
        msg = Bool()
        msg.data = self.ready
        self.ready_pub.publish(msg)

    def stop_motion(self):
        self.cmd_vel_pub.publish(Twist())

    def start_localization(self, reason):
        self.stop_motion()
        self.publish_ready(False)

        self.state = 'PUBLISH_INITIAL_POSE'
        self.initialpose_sent_count = 0
        self.last_initialpose_pub_time = 0.0
        self.wait_start_time = time.monotonic()

        self.spin_accum = 0.0
        self.prev_odom_yaw_for_spin = None
        self.spin_start_time = 0.0

        self.status(f'auto_localize_start:{reason}')

        if bool(self.get_parameter('use_global_localization_service').value):
            self.call_global_localization_if_available()

    def call_global_localization_if_available(self):
        if not self.global_loc_client.service_is_ready():
            self.status('global_localization_service_not_ready_skip')
            return

        req = Empty.Request()
        self.global_loc_client.call_async(req)
        self.status('global_localization_service_called')

    def make_initialpose_msg(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        x = float(self.get_parameter('start_x').value)
        y = float(self.get_parameter('start_y').value)
        yaw = float(self.get_parameter('start_yaw').value)

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        qx, qy, qz, qw = quat_from_yaw(yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        cov = [0.0] * 36
        cov[0] = float(self.get_parameter('cov_xx').value)
        cov[7] = float(self.get_parameter('cov_yy').value)
        cov[35] = float(self.get_parameter('cov_yaw').value)
        msg.pose.covariance = cov

        return msg

    def publish_initialpose_once(self):
        msg = self.make_initialpose_msg()
        self.initialpose_pub.publish(msg)
        self.initialpose_sent_count += 1

        self.status(
            'initialpose_published:'
            f'x={msg.pose.pose.position.x:.3f}:'
            f'y={msg.pose.pose.position.y:.3f}:'
            f'count={self.initialpose_sent_count}'
        )

    def amcl_is_good(self):
        if self.last_amcl_msg is None:
            return False

        now = time.monotonic()
        timeout = float(self.get_parameter('amcl_valid_timeout_sec').value)
        if now - self.last_amcl_time > timeout:
            return False

        cov = self.last_amcl_msg.pose.covariance
        cov_xx = float(cov[0])
        cov_yy = float(cov[7])
        cov_yaw = float(cov[35])

        if cov_xx > float(self.get_parameter('max_cov_xx').value):
            return False
        if cov_yy > float(self.get_parameter('max_cov_yy').value):
            return False
        if cov_yaw > float(self.get_parameter('max_cov_yaw').value):
            return False

        return True

    def finish_success(self, reason):
        self.stop_motion()
        self.publish_ready(True)
        self.state = 'IDLE'
        self.status(f'auto_localize_done:{reason}')

    # ============================================================
    # Timer state machine
    # ============================================================
    def timer_cb(self):
        now = time.monotonic()

        if self.state == 'IDLE':
            return

        if self.state == 'PUBLISH_INITIAL_POSE':
            count = int(self.get_parameter('republish_initialpose_count').value)
            hz = float(self.get_parameter('initialpose_publish_hz').value)
            period = 1.0 / max(hz, 0.1)

            if self.initialpose_sent_count < count:
                if now - self.last_initialpose_pub_time >= period:
                    self.last_initialpose_pub_time = now
                    self.publish_initialpose_once()
                return

            self.wait_start_time = now
            self.state = 'WAIT_AMCL_AFTER_INITIALPOSE'
            self.status('wait_amcl_after_initialpose')
            return

        if self.state == 'WAIT_AMCL_AFTER_INITIALPOSE':
            wait_sec = float(self.get_parameter('amcl_wait_after_initialpose_sec').value)

            if self.amcl_is_good() and not bool(self.get_parameter('enable_spin_scan').value):
                self.finish_success('amcl_good_without_spin')
                return

            if now - self.wait_start_time < wait_sec:
                self.stop_motion()
                return

            if bool(self.get_parameter('enable_spin_scan').value):
                self.state = 'SPIN_SCAN'
                self.spin_start_time = now
                self.spin_accum = 0.0
                self.prev_odom_yaw_for_spin = self.odom_yaw
                self.status('spin_scan_start')
                return

            if self.amcl_is_good():
                self.finish_success('amcl_good_after_wait')
            else:
                self.status('amcl_not_good_but_spin_disabled')
                self.finish_success('forced_ready_after_initialpose')
            return

        if self.state == 'SPIN_SCAN':
            spin_wz = abs(float(self.get_parameter('spin_wz').value))
            spin_angle = abs(float(self.get_parameter('spin_angle_rad').value))
            spin_min_sec = float(self.get_parameter('spin_min_sec').value)
            spin_max_sec = float(self.get_parameter('spin_max_sec').value)

            elapsed = now - self.spin_start_time

            if (
                bool(self.get_parameter('stop_when_amcl_good_after_min_spin').value)
                and elapsed >= spin_min_sec
                and self.amcl_is_good()
            ):
                self.finish_success(
                    f'amcl_good_after_spin:elapsed={elapsed:.1f}:angle={self.spin_accum:.2f}'
                )
                return

            if self.spin_accum >= spin_angle:
                self.finish_success(
                    f'spin_angle_done:elapsed={elapsed:.1f}:angle={self.spin_accum:.2f}'
                )
                return

            if elapsed >= spin_max_sec:
                if self.amcl_is_good():
                    self.finish_success(f'spin_max_done_amcl_good:elapsed={elapsed:.1f}')
                else:
                    self.status(f'spin_max_done_amcl_not_good:elapsed={elapsed:.1f}')
                    self.finish_success('forced_ready_after_spin_timeout')
                return

            cmd = Twist()
            cmd.angular.z = spin_wz
            self.cmd_vel_pub.publish(cmd)
            return


def main(args=None):
    rclpy.init(args=args)
    node = AutoInitialPoseNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_motion()
            time.sleep(0.05)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
