from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    inspect_only = LaunchConfiguration('inspect_only')
    freeze_transition = LaunchConfiguration('freeze_transition')
    enable_drive = LaunchConfiguration('enable_drive')
    publish_debug = LaunchConfiguration('publish_debug')
    use_compressed_input = LaunchConfiguration('use_compressed_input')
    image_topic = LaunchConfiguration('image_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    odom_topic = LaunchConfiguration('odom_topic')
    telemetry_csv = LaunchConfiguration('telemetry_csv')
    debug_perception_rate = LaunchConfiguration('debug_perception_rate')
    debug_snapshot_dir = LaunchConfiguration('debug_snapshot_dir')

    return LaunchDescription([
        DeclareLaunchArgument(
            'inspect_only',
            default_value='true',
            description='true면 주행/상태전이 없이 현재 카메라 인식 결과만 publish'
        ),
        DeclareLaunchArgument(
            'freeze_transition',
            default_value='false',
            description='true면 자동 state/phase 전이를 막고 현재 장면에서 멈춰 디버깅'
        ),
        DeclareLaunchArgument(
            'enable_drive',
            default_value='false',
            description='true면 /cmd_vel_raw 발행 허용. 디버그 기본값은 false'
        ),
        DeclareLaunchArgument(
            'publish_debug',
            default_value='true',
            description='true면 /line_tracer/debug/compressed publish'
        ),
        DeclareLaunchArgument(
            'use_compressed_input',
            default_value='true',
            description='true면 CompressedImage 입력 사용'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/camera/color/image_raw/compressed',
            description='카메라 color 입력 토픽'
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/camera/camera/aligned_depth_to_color/image_raw',
            description='aligned depth 입력 토픽'
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odometry/filtered',
            description='odometry 입력 토픽'
        ),
        DeclareLaunchArgument(
            'debug_perception_rate',
            default_value='5.0',
            description='/line_tracer/perception publish Hz'
        ),
        DeclareLaunchArgument(
            'telemetry_csv',
            default_value='/home/er/myagv_ros2/src/tracer/log/debug_perception.csv',
            description='텔레메트리 CSV 저장 경로. 빈 문자열이면 비활성'
        ),
        DeclareLaunchArgument(
            'debug_snapshot_dir',
            default_value='/home/er/myagv_ros2/src/tracer/log/snapshots',
            description='SNAPSHOT 명령으로 저장되는 이미지/meta 경로'
        ),

        Node(
            package='tracer',
            executable='tracer_node',
            name='tracer',
            output='screen',
            parameters=[{
                'inspect_only': inspect_only,
                'freeze_transition': freeze_transition,
                'enable_drive': enable_drive,
                'publish_debug': publish_debug,
                'use_compressed_input': use_compressed_input,
                'image_topic': image_topic,
                'depth_topic': depth_topic,
                'odom_topic': odom_topic,
                'debug_perception_rate': debug_perception_rate,
                'telemetry_csv': telemetry_csv,
                'debug_snapshot_dir': debug_snapshot_dir,
                'start_idle': True,
            }]
        )
    ])
