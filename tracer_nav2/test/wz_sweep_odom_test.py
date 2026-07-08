#!/usr/bin/env python3

import argparse
import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


ANGULAR_SPEEDS = [
    0.41,
    0.42,
    0.43,
    0.44,
    0.45,
    0.48,
    0.50,
    0.52,
    0.55,
    0.60,
    0.65,
    0.38,
    0.40,
]


def make_twist(vx: float, wz: float) -> Twist:
    msg = Twist()
    msg.linear.x = float(vx)
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = float(wz)
    return msg


def yaw_from_quaternion(q) -> float:
    # ROS quaternion -> yaw
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class WzSweepOdomTest(Node):
    def __init__(self, args):
        super().__init__('wz_sweep_odom_test')

        self.args = args

        self.pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.sub_odom = self.create_subscription(
            Odometry,
            args.odom_topic,
            self.odom_cb,
            20
        )

        self.last_odom = None
        self.last_odom_recv_time = None

        self.sequence = self.build_sequence()
        self.start_time = time.time()

        csv_path = os.path.expanduser(args.csv)
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        self.csv_file = open(csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            'elapsed_sec',
            'phase',
            'trial_id',
            'vx_group',
            'speed_abs',
            'direction',
            'cmd_topic',
            'cmd_vx',
            'cmd_wz',

            'odom_stamp_sec',
            'odom_age_sec',
            'odom_pos_x',
            'odom_pos_y',
            'odom_yaw_rad',
            'odom_vx',
            'odom_vy',
            'odom_wz',
        ])

        self.timer = self.create_timer(1.0 / args.rate, self.timer_cb)

        total_run_sec = sum(item['duration'] for item in self.sequence)
        total_trial_count = sum(1 for item in self.sequence if item['phase'] == 'run')

        self.get_logger().info('====================================')
        self.get_logger().info('wz sweep odom test started')
        self.get_logger().info(f'cmd_topic: {args.cmd_topic}')
        self.get_logger().info(f'odom_topic: {args.odom_topic}')
        self.get_logger().info(f'csv: {csv_path}')
        self.get_logger().info(f'trial count: {total_trial_count}')
        self.get_logger().info(f'estimated duration: {total_run_sec:.1f} sec')
        self.get_logger().info('====================================')

    def build_sequence(self):
        seq = []
        trial_id = 0

        # linear.x = 0.0 그룹, linear.x = 0.08 그룹
        vx_groups = [0.08]

        for vx in vx_groups:
            # 그룹 시작 전 정지
            seq.append({
                'phase': 'stop',
                'duration': self.args.stop_sec,
                'trial_id': -1,
                'vx': 0.0,
                'wz': 0.0,
                'vx_group': vx,
                'speed_abs': 0.0,
                'direction': 0,
            })

            for speed in ANGULAR_SPEEDS:
                for direction in [1, -1]:
                    trial_id += 1
                    wz = direction * speed

                    seq.append({
                        'phase': 'run',
                        'duration': self.args.run_sec,
                        'trial_id': trial_id,
                        'vx': vx,
                        'wz': wz,
                        'vx_group': vx,
                        'speed_abs': speed,
                        'direction': direction,
                    })

                    # 각 trial 사이 1.5초 정지
                    seq.append({
                        'phase': 'stop',
                        'duration': self.args.stop_sec,
                        'trial_id': trial_id,
                        'vx': 0.0,
                        'wz': 0.0,
                        'vx_group': vx,
                        'speed_abs': speed,
                        'direction': direction,
                    })

        # 마지막 안전 정지
        seq.append({
            'phase': 'final_stop',
            'duration': 2.0,
            'trial_id': -1,
            'vx': 0.0,
            'wz': 0.0,
            'vx_group': 0.0,
            'speed_abs': 0.0,
            'direction': 0,
        })

        return seq

    def odom_cb(self, msg: Odometry):
        self.last_odom = msg
        self.last_odom_recv_time = time.time()

    def get_current_step(self, elapsed):
        acc = 0.0
        for item in self.sequence:
            acc += item['duration']
            if elapsed <= acc:
                return item
        return None

    def write_row(self, elapsed, step):
        odom_stamp_sec = ''
        odom_age_sec = ''
        odom_pos_x = ''
        odom_pos_y = ''
        odom_yaw_rad = ''
        odom_vx = ''
        odom_vy = ''
        odom_wz = ''

        if self.last_odom is not None:
            odom = self.last_odom

            odom_stamp_sec = (
                float(odom.header.stamp.sec)
                + float(odom.header.stamp.nanosec) * 1e-9
            )

            if self.last_odom_recv_time is not None:
                odom_age_sec = time.time() - self.last_odom_recv_time

            odom_pos_x = odom.pose.pose.position.x
            odom_pos_y = odom.pose.pose.position.y
            odom_yaw_rad = yaw_from_quaternion(odom.pose.pose.orientation)

            odom_vx = odom.twist.twist.linear.x
            odom_vy = odom.twist.twist.linear.y
            odom_wz = odom.twist.twist.angular.z

        self.writer.writerow([
            elapsed,
            step['phase'],
            step['trial_id'],
            step['vx_group'],
            step['speed_abs'],
            step['direction'],
            self.args.cmd_topic,
            step['vx'],
            step['wz'],

            odom_stamp_sec,
            odom_age_sec,
            odom_pos_x,
            odom_pos_y,
            odom_yaw_rad,
            odom_vx,
            odom_vy,
            odom_wz,
        ])

        self.csv_file.flush()

    def publish_stop(self):
        self.pub.publish(make_twist(0.0, 0.0))

    def stop_and_close(self):
        self.get_logger().info('Publishing stop command...')
        stop = make_twist(0.0, 0.0)

        for _ in range(10):
            self.pub.publish(stop)
            time.sleep(0.05)

        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass

    def timer_cb(self):
        elapsed = time.time() - self.start_time
        step = self.get_current_step(elapsed)

        if step is None:
            self.stop_and_close()
            self.get_logger().info('Test finished.')
            rclpy.shutdown()
            return

        cmd = make_twist(step['vx'], step['wz'])
        self.pub.publish(cmd)
        self.write_row(elapsed, step)

        # trial 시작 로그를 너무 많이 찍지 않기 위해 대략 1초마다만 표시
        if int(elapsed * 10) % int(self.args.rate) == 0:
            self.get_logger().info(
                f"phase={step['phase']} "
                f"trial={step['trial_id']} "
                f"vx={step['vx']:.2f} "
                f"wz={step['wz']:.2f}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cmd-topic',
        default='/cmd_vel',
        help='Command topic. Direct firmware test: /cmd_vel, filter test: /cmd_vel_raw'
    )
    parser.add_argument(
        '--odom-topic',
        default='/odom',
        help='Odometry topic to record'
    )
    parser.add_argument(
        '--csv',
        default='~/wz_sweep_odom.csv',
        help='CSV output path'
    )
    parser.add_argument(
        '--rate',
        type=float,
        default=10.0,
        help='publish/log rate Hz'
    )
    parser.add_argument(
        '--run-sec',
        type=float,
        default=7.0,
        help='duration for each velocity command'
    )
    parser.add_argument(
        '--stop-sec',
        type=float,
        default=1.5,
        help='stop duration between trials'
    )

    args = parser.parse_args()

    rclpy.init()
    node = WzSweepOdomTest(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn('KeyboardInterrupt received.')
    finally:
        node.stop_and_close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
