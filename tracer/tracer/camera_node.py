#!/usr/bin/env python3

import time
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class MyAGVCameraNode(Node):
    def __init__(self):
        super().__init__('myagv_camera_node')

        # ============================================================
        # Parameters: backend / device
        # ============================================================
        self.declare_parameter('backend', 'argus')   # argus or v4l2
        self.declare_parameter('device_id', 0)
        self.declare_parameter('sensor_id', 0)

        # Argus sensor mode.
        # Jetson log 기준:
        #   mode 2: 1920x1080 @ 30fps
        #   mode 5: 1280x720 @ 120fps
        #
        # CPU 100% 방지를 위해 120fps mode를 피하고 30fps mode를 고정하는 목적.
        self.declare_parameter('sensor_mode', 2)

        # ============================================================
        # Parameters: capture stream
        # ============================================================
        # 실제 nvarguscamerasrc에서 받는 입력 해상도/fps.
        # publish 해상도와 분리한다.
        self.declare_parameter('capture_width', 1920)
        self.declare_parameter('capture_height', 1080)
        self.declare_parameter('capture_fps', 30.0)

        # ============================================================
        # Parameters: ROS publish image
        # ============================================================
        # ROS topic으로 publish할 최종 이미지 크기/fps.
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 360)
        self.declare_parameter('fps', 10.0)

        self.declare_parameter('frame_id', 'myagv_camera_frame')
        self.declare_parameter('image_topic', '/myagv_camera/image_raw')
        self.declare_parameter('camera_info_topic', '/myagv_camera/camera_info')
        self.declare_parameter('publish_camera_info', True)

        # 0: none
        # 1: 90 CCW
        # 2: 180
        # 3: 90 CW
        self.declare_parameter('flip_method', 2)

        # Debug window는 Jetson에서 CPU/GUI 문제 만들 수 있으므로 기본 false.
        self.declare_parameter('show_debug_window', False)

        # ============================================================
        # Parameters: robustness
        # ============================================================
        self.declare_parameter('warn_period_sec', 1.0)
        self.declare_parameter('reopen_on_fail', True)
        self.declare_parameter('max_consecutive_failures', 30)
        self.declare_parameter('reopen_cooldown_sec', 2.0)

        # ============================================================
        # Load parameters
        # ============================================================
        self.backend = str(self.get_parameter('backend').value)
        self.device_id = int(self.get_parameter('device_id').value)
        self.sensor_id = int(self.get_parameter('sensor_id').value)
        self.sensor_mode = int(self.get_parameter('sensor_mode').value)

        self.capture_width = int(self.get_parameter('capture_width').value)
        self.capture_height = int(self.get_parameter('capture_height').value)
        self.capture_fps = float(self.get_parameter('capture_fps').value)

        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.publish_camera_info = bool(self.get_parameter('publish_camera_info').value)

        self.flip_method = int(self.get_parameter('flip_method').value)
        self.show_debug_window = bool(self.get_parameter('show_debug_window').value)

        self.warn_period_sec = float(self.get_parameter('warn_period_sec').value)
        self.reopen_on_fail = bool(self.get_parameter('reopen_on_fail').value)
        self.max_consecutive_failures = int(self.get_parameter('max_consecutive_failures').value)
        self.reopen_cooldown_sec = float(self.get_parameter('reopen_cooldown_sec').value)

        # ============================================================
        # State
        # ============================================================
        self.cap = None
        self.consecutive_failures = 0
        self.last_fail_warn_time = 0.0
        self.last_reopen_time = 0.0
        self.frame_seq = 0
        self.first_frame_logged = False

        # ============================================================
        # QoS
        # ============================================================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.image_pub = self.create_publisher(Image, self.image_topic, sensor_qos)
        self.info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, sensor_qos)

        # ============================================================
        # Open camera
        # ============================================================
        self.open_camera_or_raise()

        timer_period = 1.0 / max(self.fps, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_cb)

        self.get_logger().info(
            f'myagv_camera_node started | backend={self.backend} '
            f'publish={self.width}x{self.height}@{self.fps:.1f}Hz '
            f'topic={self.image_topic}'
        )

    # ============================================================
    # Camera open / close
    # ============================================================
    def open_camera_or_raise(self):
        self.close_camera()

        if self.backend == 'argus':
            self.cap = self.open_argus_camera()
        elif self.backend == 'v4l2':
            self.cap = self.open_v4l2_camera()
        else:
            raise RuntimeError(
                f'Unknown backend: {self.backend}. Use backend:=argus or backend:=v4l2'
            )

        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error('Failed to open camera.')
            raise RuntimeError('Failed to open camera.')

        self.consecutive_failures = 0
        self.first_frame_logged = False

        self.get_logger().info('Camera opened successfully.')
        self.get_logger().info(f'backend={self.backend}')
        self.get_logger().info(f'sensor_id={self.sensor_id}')
        self.get_logger().info(f'sensor_mode={self.sensor_mode}')
        self.get_logger().info(
            f'capture={self.capture_width}x{self.capture_height}@{self.capture_fps:.1f}'
        )
        self.get_logger().info(f'publish={self.width}x{self.height}@{self.fps:.1f}')
        self.get_logger().info(f'flip_method={self.flip_method}')
        self.get_logger().info(f'Publishing image: {self.image_topic}')
        self.get_logger().info(f'Publishing camera_info: {self.camera_info_topic}')
        self.get_logger().info('QoS reliability: BEST_EFFORT')

    def close_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.show_debug_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def open_argus_camera(self):
        capture_fps_int = int(round(self.capture_fps))

        # 핵심:
        #   nvarguscamerasrc는 안정적인 capture mode로 열고,
        #   nvvidconv에서 ROS publish 크기인 width/height로 줄인다.
        #
        # sensor-mode=2를 넣어 1280x720 120fps mode로 잡히는 것을 피한다.
        pipeline = (
            f'nvarguscamerasrc sensor-id={self.sensor_id} '
            f'sensor-mode={self.sensor_mode} ! '
            f'video/x-raw(memory:NVMM), '
            f'width=(int){self.capture_width}, '
            f'height=(int){self.capture_height}, '
            f'framerate=(fraction){capture_fps_int}/1 ! '
            f'nvvidconv flip-method={self.flip_method} ! '
            f'video/x-raw, '
            f'width=(int){self.width}, '
            f'height=(int){self.height}, '
            f'format=(string)BGRx ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )

        self.get_logger().info('Opening camera with GStreamer nvarguscamerasrc:')
        self.get_logger().info(pipeline)

        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        return cap

    def open_v4l2_camera(self):
        self.get_logger().info(f'Opening camera with V4L2: /dev/video{self.device_id}')

        cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        return cap

    def reopen_camera_if_needed(self):
        if not self.reopen_on_fail:
            return

        if self.consecutive_failures < self.max_consecutive_failures:
            return

        now = time.time()
        if now - self.last_reopen_time < self.reopen_cooldown_sec:
            return

        self.last_reopen_time = now

        self.get_logger().warn(
            f'Too many camera read failures '
            f'({self.consecutive_failures}), reopening camera...'
        )

        try:
            self.open_camera_or_raise()
        except Exception as e:
            self.get_logger().error(f'Camera reopen failed: {e}')

    # ============================================================
    # Message helpers
    # ============================================================
    def make_image_msg(self, frame):
        msg = Image()

        now_msg = self.get_clock().now().to_msg()
        msg.header.stamp = now_msg
        msg.header.frame_id = self.frame_id

        h, w = frame.shape[:2]

        msg.height = int(h)
        msg.width = int(w)
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = int(w * 3)
        msg.data = frame.tobytes()

        return msg

    def make_camera_info_msg(self):
        msg = CameraInfo()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.height = int(self.height)
        msg.width = int(self.width)

        # Calibration을 아직 안 했으므로 정확한 intrinsics는 아니다.
        # CameraInfo subscriber가 죽지 않도록 기본 pinhole 형태만 채운다.
        cx = float(self.width) / 2.0
        cy = float(self.height) / 2.0
        fx = float(self.width)
        fy = float(self.width)

        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]

        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        return msg

    # ============================================================
    # Timer
    # ============================================================
    def timer_cb(self):
        if self.cap is None or not self.cap.isOpened():
            self.consecutive_failures += 1
            self.warn_read_failure('Camera is not opened')
            self.reopen_camera_if_needed()
            return

        ok, frame = self.cap.read()


        if not ok or frame is None:
            self.consecutive_failures += 1
            self.warn_read_failure('Failed to read frame from camera')
            self.reopen_camera_if_needed()
            return

        self.consecutive_failures = 0
        
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3].copy()

        if not self.first_frame_logged:
            self.first_frame_logged = True
            h, w = frame.shape[:2]
            self.get_logger().info(f'First frame received: {w}x{h}, shape={frame.shape}')

        # 혹시 pipeline에서 resize가 안 먹은 경우를 대비한 fallback.
        # 정상이라면 여기로 거의 안 들어와야 한다.
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if self.show_debug_window:
            try:
                cv2.imshow('myagv_camera_node', frame)
                cv2.waitKey(1)
            except Exception:
                pass

        image_msg = self.make_image_msg(frame)
        self.image_pub.publish(image_msg)

        if self.publish_camera_info:
            info_msg = self.make_camera_info_msg()
            info_msg.header.stamp = image_msg.header.stamp
            self.info_pub.publish(info_msg)

        self.frame_seq += 1

    def warn_read_failure(self, text):
        now = time.time()
        if now - self.last_fail_warn_time >= self.warn_period_sec:
            self.last_fail_warn_time = now
            self.get_logger().warn(
                f'{text} | consecutive_failures={self.consecutive_failures}'
            )


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = MyAGVCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.close_camera()
            except Exception:
                pass

            try:
                node.destroy_node()
            except Exception:
                pass

        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
