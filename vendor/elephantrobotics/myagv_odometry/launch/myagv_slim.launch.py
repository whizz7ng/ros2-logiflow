import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ekf_config = os.path.join(
        get_package_share_directory('myagv_odometry'),
        'config',
        'ekf.yaml'
    )

    return LaunchDescription([
        Node(
            package='myagv_odometry',
            executable='myagv_odometry_node',
            name='myagv_odometry_node',
            output='screen'
        ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_node',
            output='screen',
            parameters=[ekf_config]
        ),
    ])
