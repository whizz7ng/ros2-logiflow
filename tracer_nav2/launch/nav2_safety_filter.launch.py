import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def generate_launch_description():
    map_yaml = LaunchConfiguration('map')
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Filter params
    publish_hz = LaunchConfiguration('filter_publish_hz')
    max_vx = LaunchConfiguration('filter_max_vx')
    max_vy = LaunchConfiguration('filter_max_vy')
    max_wz = LaunchConfiguration('filter_max_wz')
    max_acc_vx = LaunchConfiguration('filter_max_acc_vx')
    max_acc_vy = LaunchConfiguration('filter_max_acc_vy')
    max_acc_wz = LaunchConfiguration('filter_max_acc_wz')

    nav2_launch_path = os.path.expanduser(
        '~/myagv_ros2/src/myagv_navigation2/launch/navigation2_active.launch.py'
    )

    # Nav2 controller_server publishes /cmd_vel. Remap it to /cmd_vel_nav so
    # the v2 filter becomes the single /cmd_vel publisher to the myAGV driver.
    nav2_with_cmd_vel_remap = GroupAction([
        SetRemap(src='/cmd_vel', dst='/cmd_vel_nav'),
        SetRemap(src='cmd_vel', dst='/cmd_vel_nav'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch_path),
            launch_arguments={
                'map': map_yaml,
                'use_rviz': use_rviz,
                'use_sim_time': use_sim_time,
            }.items()
        ),
    ])

    cmd_vel_safety_filter = Node(
        package='tracer_nav2',
        executable='vel_filter_node',
        name='cmd_vel_safety_filter',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,

            # topics
            'input_topic': '/cmd_vel_nav',
            'output_topic': '/cmd_vel',
            'state_topic': '/cmd_vel_safety_filter/state',

            # frequency / timeout
            # Keep 20Hz to reduce driver timeout risk without old 40Hz load.
            'publish_hz': publish_hz,
            'cmd_timeout_sec': 0.35,
            'publish_zero_when_idle': True,

            # hard clamps
            'max_vx': max_vx,
            'max_vy': max_vy,
            'max_wz': max_wz,
            'block_reverse': True,

            # acceleration limiting
            'enable_acc_limit': True,
            'max_acc_vx': max_acc_vx,
            'max_acc_vy': max_acc_vy,
            'max_acc_wz': max_acc_wz,

            # preserve Nav2 curved driving unless angular.z is actually large
            'enable_high_wz_vx_limit': True,
            'high_wz_start': 0.35,
            'high_wz_full': 0.55,
            'vx_limit_at_high_wz_start': 0.12,
            'vx_limit_at_high_wz_full': 0.04,

            # keep these for compatibility with existing aruco launch patch
            'straight_vx_on': 0.03,
            'kill_vy_in_straight': False,
            'kill_vy_in_turn': False,

            # transition protections found useful in surge-probe experiments
            'enable_zero_cross_guard': True,
            'zero_cross_threshold': 0.30,
            'zero_cross_hold_sec': 0.25,
            'zero_cross_vx_limit': 0.00,
            'zero_cross_vy_limit': 0.00,

            'enable_turn_exit_hold': True,
            'turn_exit_hold_sec': 0.45,
            'turn_exit_min_prev_wz': 0.35,
            'turn_exit_target_wz': 0.08,
            'turn_exit_max_vx': 0.03,
            'turn_exit_max_vy': 0.02,
            'allow_turn_during_exit_hold': True,

            # old pulse/deadzone adapter is disabled by default in v2
            'enable_small_wz_deadband': True,
            'small_wz_deadband': 0.025,
            'enable_wz_deadzone_adapter': False,
            'use_pulse_adapter': False,
            'hw_min_wz': 0.40,
            'min_pulse_duty': 0.35,
            'max_pulse_duty': 0.60,
            'pulse_period': 0.45,

            'print_mode_change': True,
            'debug_period_sec': 0.50,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=os.path.expanduser('~/myagv_ros2/maps/mymap.yaml')
        ),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        # v2 safety filter parameters
        DeclareLaunchArgument('filter_publish_hz', default_value='20.0'),
        DeclareLaunchArgument('filter_max_vx', default_value='0.20'),
        DeclareLaunchArgument('filter_max_vy', default_value='0.06'),
        DeclareLaunchArgument('filter_max_wz', default_value='0.60'),
        DeclareLaunchArgument('filter_max_acc_vx', default_value='0.20'),
        DeclareLaunchArgument('filter_max_acc_vy', default_value='0.15'),
        DeclareLaunchArgument('filter_max_acc_wz', default_value='0.80'),

        nav2_with_cmd_vel_remap,
        cmd_vel_safety_filter,
    ])
