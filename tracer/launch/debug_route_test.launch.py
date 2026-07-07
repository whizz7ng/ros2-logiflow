from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    goals_yaml = LaunchConfiguration('goals_yaml')
    route_name = LaunchConfiguration('route_name')
    autostart = LaunchConfiguration('autostart')

    goal_timeout_sec = LaunchConfiguration('goal_timeout_sec')
    hard_timeout_sec = LaunchConfiguration('hard_timeout_sec')
    near_goal_dist = LaunchConfiguration('near_goal_dist')
    fine_timeout_sec = LaunchConfiguration('fine_timeout_sec')

    debug_dir = LaunchConfiguration('debug_dir')
    log_rate_hz = LaunchConfiguration('log_rate_hz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'goals_yaml',
            default_value='/home/er/myagv_ros2/src/tracer/config/goals.yaml'
        ),
        DeclareLaunchArgument(
            'route_name',
            default_value='debug_to_obj_only'
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='false'
        ),

        # legacy compatibility용.
        # 이전 코드가 goal_timeout_sec를 쓰고 있어도 15초로 잘리지 않게 기본값 90으로 둠.
        DeclareLaunchArgument(
            'goal_timeout_sec',
            default_value='90.0'
        ),

        # 새 timeout 구조
        DeclareLaunchArgument(
            'hard_timeout_sec',
            default_value='90.0'
        ),
        DeclareLaunchArgument(
            'near_goal_dist',
            default_value='0.18'
        ),
        DeclareLaunchArgument(
            'fine_timeout_sec',
            default_value='15.0'
        ),

        DeclareLaunchArgument(
            'debug_dir',
            default_value='/home/er/nav_debug'
        ),
        DeclareLaunchArgument(
            'log_rate_hz',
            default_value='10.0'
        ),

        Node(
            package='tracer',
            executable='nav_debug_logger',
            name='nav_debug_logger',
            output='screen',
            parameters=[{
                'debug_dir': debug_dir,
                'log_rate_hz': log_rate_hz,
                'nav_status_topic': '/debug/nav_status',
                'goals_yaml': goals_yaml,
                'odom_topic': '/odometry/filtered',
            }]
        ),

        Node(
            package='tracer',
            executable='debug_route_runner',
            name='debug_route_runner',
            output='screen',
            parameters=[{
                'goals_yaml': goals_yaml,
                'route_name': route_name,
                'autostart': autostart,

                # legacy + new timeout params
                'goal_timeout_sec': goal_timeout_sec,
                'hard_timeout_sec': hard_timeout_sec,
                'near_goal_dist': near_goal_dist,
                'fine_timeout_sec': fine_timeout_sec,

                # /nav_status 충돌 방지
                'status_topic': '/debug/nav_status',

                # 코드에서 안 쓰면 무시돼도 됨
                'continue_on_failure': True,
            }]
        ),
    ])
