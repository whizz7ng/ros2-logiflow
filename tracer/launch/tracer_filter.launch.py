#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tracer',
            executable='tracer_node',
            name='line_tracer',
            output='screen',
        ),

        Node(
            package='tracer',
            executable='vel_filter_node',
            name='cmd_vel_safety_filter',
            output='screen',
        ),
    ])
