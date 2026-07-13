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

    nav2_launch_path = os.path.expanduser(
        '~/myagv_ros2/src/myagv_navigation2/launch/navigation2_active.launch.py'
    )

    nav2_with_cmd_vel_remap = GroupAction([
        # Nav2 controller_server가 내는 cmd_vel을 /cmd_vel_nav로 변경
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
        package='tracer',
        executable='vel_filter_node',
        name='cmd_vel_safety_filter',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            # 기존 filter가 /cmd_vel_raw를 subscribe한다면,
            # 그 입력을 Nav2 출력 /cmd_vel_nav로 연결
            ('/cmd_vel_raw', '/cmd_vel_nav'),
            ('cmd_vel_raw', '/cmd_vel_nav'),

            # filter 출력은 최종 /cmd_vel 유지
            ('/cmd_vel', '/cmd_vel'),
            ('cmd_vel', '/cmd_vel'),
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=os.path.expanduser('~/myagv_ros2/maps/mymap.yaml')
        ),

        DeclareLaunchArgument(
            'use_rviz',
            default_value='false'
        ),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false'
        ),

        nav2_with_cmd_vel_remap,
        cmd_vel_safety_filter,
    ])
