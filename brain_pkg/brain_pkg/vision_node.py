#!/usr/bin/env python3
"""vision_node.py — YOLO + D435i 블록 인식 (ROS2 Humble)"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import numpy as np

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # 구독
        self.create_subscription(String, '/vision_activate', self._activate_callback, 10)

        # 발행
        self._box_pose_pub = self.create_publisher(Float32MultiArray, '/box_pose', 10)

        self.active = False
        self.get_logger().info('vision_node 시작 - /vision_activate 대기중')

    def _activate_callback(self, msg):
        if msg.data == 'stop':
            self.active = False
            self.get_logger().info('vision_node 비활성화')
        else:
            # blue / red / green 등 색깔이 들어옴
            self.target_color = msg.data
            self.active = True
            self.get_logger().info(f'vision_node 활성화 - 타겟 색깔: {msg.data}')
            self._detect_block()

    def _detect_block(self):
        # TODO: 실제 D435i + 변환행렬 + YOLO 코드로 교체
        self.get_logger().info('블록 감지중...')
        # 더미 좌표
        coords = [200.0, 150.0, 80.0, 175.35, -1.1, -89.73]
        msg = Float32MultiArray()
        msg.data = coords
        self._box_pose_pub.publish(msg)
        self.get_logger().info(f'box_pose 발행: {coords}')

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
