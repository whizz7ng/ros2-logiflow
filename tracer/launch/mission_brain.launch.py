from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('primitive_cmd_topic', default_value='/primitive_route_cmd'),
        DeclareLaunchArgument('nav_status_topic', default_value='/debug/nav_status'),
        DeclareLaunchArgument('brain_status_topic', default_value='/brain_status'),
        DeclareLaunchArgument('agv_align_enable_topic', default_value='/agv_align_enable'),

        DeclareLaunchArgument('place_target_topic', default_value='/place_target'),
        DeclareLaunchArgument('order_request_topic', default_value='/order_request'),
        DeclareLaunchArgument('arm_status_topic', default_value='/arm_status'),
        DeclareLaunchArgument('go_parking_topic', default_value='/go_parking'),

        DeclareLaunchArgument('stop_obj_topic', default_value='/stop_obj'),
        DeclareLaunchArgument('stop_qr_topic', default_value='/stop_qr'),

        DeclareLaunchArgument('aruco_done_topic', default_value='/aruco_align_done'),
        DeclareLaunchArgument('aruco_status_topic', default_value='/aruco_align_status'),
        DeclareLaunchArgument('aruco_target_name_topic', default_value='/marker_align_target'),

        DeclareLaunchArgument('align_cmd_topic', default_value='/align_cmd'),
        DeclareLaunchArgument('align_goal_topic', default_value='/align_goal'),

        DeclareLaunchArgument('pickup_goal_sequence', default_value='way12,to_obj'),
        DeclareLaunchArgument('pickup_route_name', default_value=''),
        DeclareLaunchArgument('to_obj_goal_name', default_value='to_obj'),

        DeclareLaunchArgument('obj_to_qr_route_template', default_value='obj_to_qr_{target_lower}'),
        DeclareLaunchArgument('qr_to_parking_route_template',default_value='qr_{from_qr_lower}_to_parking'),

        DeclareLaunchArgument('valid_targets', default_value='A,B,C'),
        DeclareLaunchArgument('default_qr_target', default_value='B'),

        DeclareLaunchArgument('auto_switch_aruco_target', default_value='true'),
        DeclareLaunchArgument('rack_aruco_target_name', default_value='RACK'),
        DeclareLaunchArgument('qr_aruco_target_map', default_value='A:A,B:B,C:C'),
        DeclareLaunchArgument('set_rack_target_on_pickup_start', default_value='true'),
        DeclareLaunchArgument('set_qr_target_before_qr_route', default_value='true'),
        DeclareLaunchArgument('set_target_again_on_aruco_wait_event', default_value='true'),

        DeclareLaunchArgument('stop_obj_delay_sec', default_value='1.0'),
        DeclareLaunchArgument('stop_qr_delay_sec', default_value='1.0'),
        DeclareLaunchArgument('command_timeout_sec', default_value='240.0'),

        DeclareLaunchArgument('publish_stop_obj_on_aruco_timeout', default_value='true'),
        DeclareLaunchArgument('publish_stop_obj_on_manual_aruco_stop', default_value='true'),
        DeclareLaunchArgument('publish_stop_qr_on_aruco_timeout', default_value='true'),
        DeclareLaunchArgument('publish_stop_qr_on_manual_aruco_stop', default_value='true'),
        DeclareLaunchArgument('allow_new_target_when_busy', default_value='false'),
        
        DeclareLaunchArgument('force_stop_on_long_aruco_align', default_value='true'),
        DeclareLaunchArgument('aruco_force_stop_sec', default_value='15.0'),
        DeclareLaunchArgument('stop_aruco_on_force_stop', default_value='true'),
        DeclareLaunchArgument('stop_primitive_on_force_stop', default_value='true'),
        DeclareLaunchArgument('qr_to_obj_route_template', default_value='qr_{from_qr_lower}_to_obj'),
        DeclareLaunchArgument('return_to_obj_on_new_target_from_qr', default_value='true'),
        DeclareLaunchArgument('aruco_cmd_topic', default_value='/aruco_align_cmd'),

        Node(
            package='tracer',
            executable='mission_brain_node',
            name='mission_brain_node',
            output='screen',
            parameters=[{
                'primitive_cmd_topic': LaunchConfiguration('primitive_cmd_topic'),
                'nav_status_topic': LaunchConfiguration('nav_status_topic'),
                'brain_status_topic': LaunchConfiguration('brain_status_topic'),
                'agv_align_enable_topic': LaunchConfiguration('agv_align_enable_topic'),

                'place_target_topic': LaunchConfiguration('place_target_topic'),
                'order_request_topic': LaunchConfiguration('order_request_topic'),
                'arm_status_topic': LaunchConfiguration('arm_status_topic'),
                'go_parking_topic': LaunchConfiguration('go_parking_topic'),

                'stop_obj_topic': LaunchConfiguration('stop_obj_topic'),
                'stop_qr_topic': LaunchConfiguration('stop_qr_topic'),

                'aruco_done_topic': LaunchConfiguration('aruco_done_topic'),
                'aruco_status_topic': LaunchConfiguration('aruco_status_topic'),
                'aruco_target_name_topic': LaunchConfiguration('aruco_target_name_topic'),

                'align_cmd_topic': LaunchConfiguration('align_cmd_topic'),
                'align_goal_topic': LaunchConfiguration('align_goal_topic'),

                'pickup_goal_sequence': LaunchConfiguration('pickup_goal_sequence'),
                'pickup_route_name': LaunchConfiguration('pickup_route_name'),
                'to_obj_goal_name': LaunchConfiguration('to_obj_goal_name'),

                'obj_to_qr_route_template': LaunchConfiguration('obj_to_qr_route_template'),
                'qr_to_parking_route_template': LaunchConfiguration('qr_to_parking_route_template'),

                'valid_targets': LaunchConfiguration('valid_targets'),
                'default_qr_target': LaunchConfiguration('default_qr_target'),

                'auto_switch_aruco_target': LaunchConfiguration('auto_switch_aruco_target'),
                'rack_aruco_target_name': LaunchConfiguration('rack_aruco_target_name'),
                'qr_aruco_target_map': LaunchConfiguration('qr_aruco_target_map'),
                'set_rack_target_on_pickup_start': LaunchConfiguration('set_rack_target_on_pickup_start'),
                'set_qr_target_before_qr_route': LaunchConfiguration('set_qr_target_before_qr_route'),
                'set_target_again_on_aruco_wait_event': LaunchConfiguration('set_target_again_on_aruco_wait_event'),

                'stop_obj_delay_sec': LaunchConfiguration('stop_obj_delay_sec'),
                'stop_qr_delay_sec': LaunchConfiguration('stop_qr_delay_sec'),
                'command_timeout_sec': LaunchConfiguration('command_timeout_sec'),

                'publish_stop_obj_on_aruco_timeout': LaunchConfiguration('publish_stop_obj_on_aruco_timeout'),
                'publish_stop_obj_on_manual_aruco_stop': LaunchConfiguration('publish_stop_obj_on_manual_aruco_stop'),
                'publish_stop_qr_on_aruco_timeout': LaunchConfiguration('publish_stop_qr_on_aruco_timeout'),
                'publish_stop_qr_on_manual_aruco_stop': LaunchConfiguration('publish_stop_qr_on_manual_aruco_stop'),
                'allow_new_target_when_busy': LaunchConfiguration('allow_new_target_when_busy'),

                'force_stop_on_long_aruco_align': LaunchConfiguration('force_stop_on_long_aruco_align'),
                'aruco_force_stop_sec': LaunchConfiguration('aruco_force_stop_sec'),
                'stop_aruco_on_force_stop': LaunchConfiguration('stop_aruco_on_force_stop'),
                'stop_primitive_on_force_stop': LaunchConfiguration('stop_primitive_on_force_stop'),
                'qr_to_obj_route_template': LaunchConfiguration('qr_to_obj_route_template'),
                'return_to_obj_on_new_target_from_qr': LaunchConfiguration('return_to_obj_on_new_target_from_qr'),
                'aruco_cmd_topic': LaunchConfiguration('aruco_cmd_topic'),
            }]
        )
    ])
