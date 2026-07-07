from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    goals_yaml = LaunchConfiguration('goals_yaml')
    route_name = LaunchConfiguration('route_name')
    autostart = LaunchConfiguration('autostart')

    debug_dir = LaunchConfiguration('debug_dir')
    log_rate_hz = LaunchConfiguration('log_rate_hz')

    drive_vx = LaunchConfiguration('drive_vx')
    min_drive_vx = LaunchConfiguration('min_drive_vx')
    slow_down_dist = LaunchConfiguration('slow_down_dist')
    turn_wz = LaunchConfiguration('turn_wz')
    xy_tolerance = LaunchConfiguration('xy_tolerance')
    heading_tolerance = LaunchConfiguration('heading_tolerance')
    drive_heading_tolerance = LaunchConfiguration('drive_heading_tolerance')
    final_yaw_tolerance = LaunchConfiguration('final_yaw_tolerance')
    goal_timeout_sec = LaunchConfiguration('goal_timeout_sec')
    amcl_timeout_sec = LaunchConfiguration('amcl_timeout_sec')
    odom_topic = LaunchConfiguration('odom_topic')
    odom_timeout_sec = LaunchConfiguration('odom_timeout_sec')
    use_odom_between_amcl = LaunchConfiguration('use_odom_between_amcl')

    enable_aruco_after_goal = LaunchConfiguration('enable_aruco_after_goal')
    aruco_goal_names = LaunchConfiguration('aruco_goal_names')
    aruco_cmd_topic = LaunchConfiguration('aruco_cmd_topic')
    aruco_done_topic = LaunchConfiguration('aruco_done_topic')
    aruco_start_delay_sec = LaunchConfiguration('aruco_start_delay_sec')
    aruco_timeout_sec = LaunchConfiguration('aruco_timeout_sec')

    return LaunchDescription([
        DeclareLaunchArgument(
            'goals_yaml',
            default_value='/home/er/myagv_ros2/src/tracer/config/goals.yaml'
        ),
        DeclareLaunchArgument(
            'route_name',
            default_value='full_mission_b'
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='false'
        ),
        DeclareLaunchArgument(
            'debug_dir',
            default_value='/home/er/nav_debug'
        ),
        DeclareLaunchArgument(
            'log_rate_hz',
            default_value='10.0'
        ),
        DeclareLaunchArgument(
            'drive_vx',
            default_value='0.18'
        ),
        DeclareLaunchArgument(
            'min_drive_vx',
            default_value='0.07'
        ),
        DeclareLaunchArgument(
            'slow_down_dist',
            default_value='0.0'
        ),
        DeclareLaunchArgument(
            'turn_wz',
            default_value='0.45'
        ),
        DeclareLaunchArgument(
            'xy_tolerance',
            default_value='0.20'
        ),
        DeclareLaunchArgument(
            'heading_tolerance',
            default_value='0.18'
        ),
        DeclareLaunchArgument(
            'drive_heading_tolerance',
            default_value='0.24'
        ),
        DeclareLaunchArgument(
            'final_yaw_tolerance',
            default_value='0.30'
        ),
        DeclareLaunchArgument(
            'goal_timeout_sec',
            default_value='120.0'
        ),
        DeclareLaunchArgument(
            'amcl_timeout_sec',
            default_value='5.0'
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odometry/filtered'
        ),
        DeclareLaunchArgument(
            'odom_timeout_sec',
            default_value='0.80'
        ),
        DeclareLaunchArgument(
            'use_odom_between_amcl',
            default_value='true'
        ),
        DeclareLaunchArgument(
            'enable_aruco_after_goal',
            default_value='true'
        ),
        DeclareLaunchArgument(
            'aruco_goal_names',
            default_value='to_obj,to_qr_a,to_qr_b,to_qr_c'
        ),
        DeclareLaunchArgument(
            'aruco_cmd_topic',
            default_value='/aruco_align_cmd'
        ),
        DeclareLaunchArgument(
            'aruco_done_topic',
            default_value='/aruco_align_done'
        ),
        DeclareLaunchArgument(
            'aruco_start_delay_sec',
            default_value='0.80'
        ),
        DeclareLaunchArgument(
            'aruco_timeout_sec',
            default_value='60.0'
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
            executable='primitive_route_runner',
            name='primitive_route_runner',
            output='screen',
            parameters=[{
                'goals_yaml': goals_yaml,
                'route_name': route_name,
                'autostart': autostart,

                'cmd_topic': '/primitive_route_cmd',
                'cmd_vel_topic': '/cmd_vel_nav',
                'status_topic': '/debug/nav_status',
                'amcl_topic': '/amcl_pose',
                'odom_topic': odom_topic,

                'drive_vx': drive_vx,
                'min_drive_vx': min_drive_vx,
                'slow_down_dist': slow_down_dist,
                'turn_wz': turn_wz,
                'xy_tolerance': xy_tolerance,
                'heading_tolerance': heading_tolerance,
                'drive_heading_tolerance': drive_heading_tolerance,
                'final_yaw_tolerance': final_yaw_tolerance,
                'goal_timeout_sec': goal_timeout_sec,
                'amcl_timeout_sec': amcl_timeout_sec,
                'use_odom_between_amcl': use_odom_between_amcl,
                'odom_timeout_sec': odom_timeout_sec,

                'enable_aruco_after_goal': enable_aruco_after_goal,
                'aruco_goal_names': aruco_goal_names,
                'aruco_cmd_topic': aruco_cmd_topic,
                'aruco_done_topic': aruco_done_topic,
                'aruco_start_delay_sec': aruco_start_delay_sec,
                'aruco_timeout_sec': aruco_timeout_sec,

                'continue_on_failure': False,
            }]
        ),
    ])
