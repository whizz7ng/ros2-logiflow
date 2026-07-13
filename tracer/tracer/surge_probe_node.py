#!/usr/bin/env python3

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class SurgeProbeNode(Node):
    """
    myAGV 급발진/과응답 재현용 cmd_vel probe.

    목적:
      safety filter가 막고 있던 조건의 반대 조건을 직접 만들어서
      어떤 조합에서 odom overshoot / abnormal turn / surge가 나오는지 확인한다.

    권장 실행 조건:
      1) 처음에는 바퀴 공중
      2) Nav2 / safety_filter 모두 OFF
      3) myAGV driver만 ON
      4) rosbag으로 /cmd_vel, /odom, /odometry/filtered 기록
      5) emergency stop 가능 상태
    """

    def __init__(self, args):
        super().__init__('surge_probe_node')

        self.args = args
        self.pub = self.create_publisher(Twist, args.topic, 10)
        self.event_pub = self.create_publisher(String, '/surge_probe_event', 10)

        self.get_logger().warn('============================================================')
        self.get_logger().warn('SURGE PROBE NODE STARTED')
        self.get_logger().warn(f'topic={args.topic}')
        self.get_logger().warn(f'pattern={args.pattern}')
        self.get_logger().warn(f'rate={args.rate_hz} Hz')
        self.get_logger().warn(f'armed={args.armed}')
        self.get_logger().warn('처음에는 반드시 바퀴를 공중에 띄우고 테스트하세요.')
        self.get_logger().warn('============================================================')

        if not args.armed:
            self.get_logger().error('Not armed. Add --armed to actually publish nonzero cmd_vel.')
            self.publish_stop_for(1.0)
            raise SystemExit(1)

    def make_twist(self, vx=0.0, vy=0.0, wz=0.0):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz)
        return msg

    def mark(self, name):
        msg = String()
        msg.data = name
        self.event_pub.publish(msg)
        self.get_logger().warn(f'[EVENT] {name}')

    def publish_cmd_for(self, duration, vx=0.0, vy=0.0, wz=0.0, label='cmd'):
        period = 1.0 / max(0.1, self.args.rate_hz)
        n = max(1, int(duration / period))

        self.mark(f'{label}_start vx={vx:.3f} vy={vy:.3f} wz={wz:.3f} duration={duration:.2f}')

        msg = self.make_twist(vx, vy, wz)
        t_end = time.time() + duration

        while rclpy.ok() and time.time() < t_end:
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        self.mark(f'{label}_end')

    def publish_stop_for(self, duration=1.0):
        self.publish_cmd_for(duration, 0.0, 0.0, 0.0, label='stop')

    def publish_once_then_gap(self, vx, vy, wz, gap, label):
        self.mark(f'{label}_single_publish_then_gap vx={vx:.3f} wz={wz:.3f} gap={gap:.2f}')
        self.pub.publish(self.make_twist(vx, vy, wz))
        time.sleep(gap)

    def run(self):
        p = self.args.pattern

        # 시작 안정화
        self.publish_stop_for(1.0)

        if p == 'vx_only':
            # max_vx 제한 필요성 확인
            self.publish_cmd_for(2.0, vx=self.args.vx, wz=0.0, label='vx_only')
            self.publish_stop_for(2.0)

        elif p == 'wz_only':
            # max_wz / recovery spin 제한 필요성 확인
            self.publish_cmd_for(2.0, vx=0.0, wz=self.args.wz, label='wz_only')
            self.publish_stop_for(2.0)

        elif p == 'small_wz':
            # 작은 angular.z가 실제로는 어느 정도 회전으로 튀는지 확인
            # 예전 실험에서 wz=0.12 명령이 실제 약 0.40 rad/s로 보였던 구간 검증
            self.publish_cmd_for(3.0, vx=0.0, wz=self.args.small_wz, label='small_wz_positive')
            self.publish_stop_for(2.0)
            self.publish_cmd_for(3.0, vx=0.0, wz=-self.args.small_wz, label='small_wz_negative')
            self.publish_stop_for(2.0)

        elif p == 'mixed_step':
            # 가장 의심되는 조합: linear.x + angular.z 동시 step
            self.publish_cmd_for(2.0, vx=self.args.vx, wz=self.args.wz, label='mixed_step_positive')
            self.publish_stop_for(2.0)
            self.publish_cmd_for(2.0, vx=self.args.vx, wz=-self.args.wz, label='mixed_step_negative')
            self.publish_stop_for(2.0)

        elif p == 'mixed_ramp':
            # 계단 입력이 문제인지, 같은 최댓값이라도 ramp는 괜찮은지 비교
            self.mark('mixed_ramp_start')
            steps = max(2, int(self.args.ramp_time * self.args.rate_hz))
            period = 1.0 / self.args.rate_hz
            for i in range(steps):
                a = (i + 1) / steps
                vx = self.args.vx * a
                wz = self.args.wz * a
                self.pub.publish(self.make_twist(vx, 0.0, wz))
                time.sleep(period)
            self.publish_cmd_for(1.0, vx=self.args.vx, wz=self.args.wz, label='mixed_ramp_hold')
            self.publish_stop_for(2.0)

        elif p == 'sign_flip':
            # zero-cross hold 없이 +wz -> -wz 바로 전환
            self.publish_cmd_for(1.5, vx=0.0, wz=self.args.wz, label='sign_flip_positive')
            self.publish_cmd_for(1.5, vx=0.0, wz=-self.args.wz, label='sign_flip_negative_no_zero')
            self.publish_stop_for(2.0)

        elif p == 'mixed_sign_flip':
            # 더 위험한 조합: vx 유지 + wz 부호만 바로 반전
            self.publish_cmd_for(1.5, vx=self.args.vx, wz=self.args.wz, label='mixed_sign_flip_positive')
            self.publish_cmd_for(1.5, vx=self.args.vx, wz=-self.args.wz, label='mixed_sign_flip_negative_no_zero')
            self.publish_stop_for(2.0)

        elif p == 'turn_exit':
            # 회전 직후 hold 없이 바로 직진
            self.publish_cmd_for(1.5, vx=0.0, wz=self.args.wz, label='turn_phase')
            self.publish_cmd_for(1.5, vx=self.args.vx, wz=0.0, label='straight_immediate_after_turn')
            self.publish_stop_for(2.0)

        elif p == 'sparse_gap':
            # driver timeout / command gap 영향 확인
            # 0.3초 timeout보다 긴 gap을 일부러 넣음
            for i in range(5):
                self.publish_once_then_gap(self.args.vx, 0.0, self.args.wz, self.args.gap, f'sparse_gap_{i}')
            self.publish_stop_for(2.0)

        elif p == 'full_suspect_suite':
            # 한 번에 다 하지 말고, 바퀴 공중에서만 사용 권장
            self.publish_cmd_for(2.0, vx=0.0, wz=self.args.small_wz, label='suite_small_wz')
            self.publish_stop_for(2.0)

            self.publish_cmd_for(2.0, vx=self.args.vx, wz=self.args.wz, label='suite_mixed_step')
            self.publish_stop_for(2.0)

            self.publish_cmd_for(1.5, vx=0.0, wz=self.args.wz, label='suite_sign_positive')
            self.publish_cmd_for(1.5, vx=0.0, wz=-self.args.wz, label='suite_sign_negative_no_zero')
            self.publish_stop_for(2.0)

            self.publish_cmd_for(1.5, vx=0.0, wz=self.args.wz, label='suite_turn')
            self.publish_cmd_for(1.5, vx=self.args.vx, wz=0.0, label='suite_turn_exit_no_hold')
            self.publish_stop_for(2.0)

        else:
            self.get_logger().error(f'Unknown pattern: {p}')

        self.publish_stop_for(1.0)
        self.mark('surge_probe_done')
        self.get_logger().warn('DONE')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--armed', action='store_true')
    parser.add_argument('--topic', default='/cmd_vel')

    parser.add_argument(
        '--pattern',
        default='mixed_step',
        choices=[
            'vx_only',
            'wz_only',
            'small_wz',
            'mixed_step',
            'mixed_ramp',
            'sign_flip',
            'mixed_sign_flip',
            'turn_exit',
            'sparse_gap',
            'full_suspect_suite',
        ]
    )

    parser.add_argument('--rate-hz', type=float, default=10.0)

    # 기본값은 일부러 너무 세게 잡지 않음.
    # 바퀴 공중 -> 바닥 저속 -> 바닥 점진 증가 순서로만 올릴 것.
    parser.add_argument('--vx', type=float, default=0.13)
    parser.add_argument('--wz', type=float, default=0.30)
    parser.add_argument('--small-wz', type=float, default=0.12)

    parser.add_argument('--ramp-time', type=float, default=2.0)
    parser.add_argument('--gap', type=float, default=0.45)

    args = parser.parse_args()

    rclpy.init()
    node = SurgeProbeNode(args)

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warn('KeyboardInterrupt: publishing stop')
        node.publish_stop_for(1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
