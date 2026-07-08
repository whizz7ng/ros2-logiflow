from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Camera args
    backend = LaunchConfiguration('backend')
    device_id = LaunchConfiguration('device_id')
    sensor_id = LaunchConfiguration('sensor_id')
    width = LaunchConfiguration('width')
    height = LaunchConfiguration('height')
    fps = LaunchConfiguration('fps')
    flip_method = LaunchConfiguration('flip_method')

    image_topic = LaunchConfiguration('image_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    frame_id = LaunchConfiguration('frame_id')
    show_debug_window = LaunchConfiguration('show_debug_window')
    
    sensor_mode = LaunchConfiguration('sensor_mode')
    capture_width = LaunchConfiguration('capture_width')
    capture_height = LaunchConfiguration('capture_height')
    capture_fps = LaunchConfiguration('capture_fps')

    # ArUco align args
    dict_name = LaunchConfiguration('dict_name')
    target_id = LaunchConfiguration('target_id')
    target_name = LaunchConfiguration('target_name')
    marker_id_map = LaunchConfiguration('marker_id_map')
    process_every_n_frames = LaunchConfiguration('process_every_n_frames')

    cmd_topic = LaunchConfiguration('cmd_topic')
    status_topic = LaunchConfiguration('status_topic')
    done_topic = LaunchConfiguration('done_topic')
    debug_image_topic = LaunchConfiguration('debug_image_topic')
    debug_json_topic = LaunchConfiguration('debug_json_topic')
    command_topic = LaunchConfiguration('command_topic')
    target_name_topic = LaunchConfiguration('target_name_topic')

    target_size_px = LaunchConfiguration('target_size_px')
    size_tolerance_px = LaunchConfiguration('size_tolerance_px')
    target_cx_px = LaunchConfiguration('target_cx_px')
    center_tolerance_px = LaunchConfiguration('center_tolerance_px')
    center_first = LaunchConfiguration('center_first')
    done_required_count = LaunchConfiguration('done_required_count')
    marker_smoothing_alpha = LaunchConfiguration('marker_smoothing_alpha')

    kp_vx = LaunchConfiguration('kp_vx')
    kp_vy = LaunchConfiguration('kp_vy')
    max_vx = LaunchConfiguration('max_vx')
    min_vx = LaunchConfiguration('min_vx')
    max_vy = LaunchConfiguration('max_vy')
    min_vy = LaunchConfiguration('min_vy')
    invert_y = LaunchConfiguration('invert_y')

    lost_timeout_sec = LaunchConfiguration('lost_timeout_sec')
    enable_lost_recovery = LaunchConfiguration('enable_lost_recovery')
    lost_recovery_start_sec = LaunchConfiguration('lost_recovery_start_sec')
    lost_recovery_sec = LaunchConfiguration('lost_recovery_sec')
    lost_recovery_vy = LaunchConfiguration('lost_recovery_vy')
    lost_recovery_min_err_px = LaunchConfiguration('lost_recovery_min_err_px')
    lost_recovery_max_age_sec = LaunchConfiguration('lost_recovery_max_age_sec')
    lost_recovery_debug_period_sec = LaunchConfiguration('lost_recovery_debug_period_sec')
    allow_reverse = LaunchConfiguration('allow_reverse')
    stop_publish_sec = LaunchConfiguration('stop_publish_sec')
    publish_debug_image = LaunchConfiguration('publish_debug_image')
    print_debug = LaunchConfiguration('print_debug')
    publish_debug_json = LaunchConfiguration('publish_debug_json')
    debug_json_period_sec = LaunchConfiguration('debug_json_period_sec')
    min_control_side_px = LaunchConfiguration('min_control_side_px')
    max_control_aspect_ratio = LaunchConfiguration('max_control_aspect_ratio')

    # Safety filter patch
    set_filter_params = LaunchConfiguration('set_filter_params')
    filter_node_name = LaunchConfiguration('filter_node_name')
    filter_max_vy = LaunchConfiguration('filter_max_vy')
    filter_straight_vx_on = LaunchConfiguration('filter_straight_vx_on')
    filter_kill_vy_in_straight = LaunchConfiguration('filter_kill_vy_in_straight')
    filter_max_acc_vy = LaunchConfiguration('filter_max_acc_vy')

    camera_node = Node(
        package='tracer_nav2',
        executable='camera_node',
        name='myagv_camera_node',
        output='screen',
        parameters=[{
            'backend': backend,
            'device_id': device_id,
            'sensor_id': sensor_id,
            'width': width,
            'height': height,
            'fps': fps,
            'flip_method': flip_method,
            'frame_id': frame_id,
            'image_topic': image_topic,
            'camera_info_topic': camera_info_topic,
            'publish_camera_info': False,
            'show_debug_window': show_debug_window,
            'sensor_mode': sensor_mode,
            'capture_width': capture_width,
            'capture_height': capture_height,
            'capture_fps': capture_fps,
        }]
    )

    align_node = Node(
        package='tracer_nav2',
        executable='aruco_align_node',
        name='aruco_align_node',
        output='screen',
        parameters=[{
            'image_topic': image_topic,
            'cmd_topic': cmd_topic,
            'status_topic': status_topic,
            'done_topic': done_topic,
            'debug_image_topic': debug_image_topic,
            'debug_json_topic': debug_json_topic,

            'command_topic': command_topic,
            'target_name_topic': target_name_topic,

            'dict_name': dict_name,
            'target_id': target_id,
            'target_name': target_name,
            'marker_id_map': marker_id_map,
            'process_every_n_frames': process_every_n_frames,

            'min_control_side_px': min_control_side_px,
            'max_control_aspect_ratio': max_control_aspect_ratio,

            'target_size_px': target_size_px,
            'size_tolerance_px': size_tolerance_px,
            'target_cx_px': target_cx_px,
            'center_tolerance_px': center_tolerance_px,
            'center_first': center_first,
            'done_required_count': done_required_count,
            'marker_smoothing_alpha': marker_smoothing_alpha,

            'kp_vx': kp_vx,
            'kp_vy': kp_vy,
            'max_vx': max_vx,
            'min_vx': min_vx,
            'max_vy': max_vy,
            'min_vy': min_vy,
            'invert_y': invert_y,

            'lost_timeout_sec': lost_timeout_sec,
            'enable_lost_recovery': enable_lost_recovery,
            'lost_recovery_start_sec': lost_recovery_start_sec,
            'lost_recovery_sec': lost_recovery_sec,
            'lost_recovery_vy': lost_recovery_vy,
            'lost_recovery_min_err_px': lost_recovery_min_err_px,
            'lost_recovery_max_age_sec': lost_recovery_max_age_sec,
            'lost_recovery_debug_period_sec': lost_recovery_debug_period_sec,
            'allow_reverse': allow_reverse,
            'stop_publish_sec': stop_publish_sec,
            'publish_debug_image': publish_debug_image,
            'publish_debug_json': publish_debug_json,
            'debug_json_period_sec': debug_json_period_sec,
            'print_debug': print_debug,
        }]
    )

    patch_filter_params = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'param', 'set', filter_node_name, 'max_vy', filter_max_vy],
                output='screen',
                condition=IfCondition(set_filter_params),
            ),
            ExecuteProcess(
                cmd=['ros2', 'param', 'set', filter_node_name, 'straight_vx_on', filter_straight_vx_on],
                output='screen',
                condition=IfCondition(set_filter_params),
            ),
            ExecuteProcess(
                cmd=['ros2', 'param', 'set', filter_node_name, 'kill_vy_in_straight', filter_kill_vy_in_straight],
                output='screen',
                condition=IfCondition(set_filter_params),
            ),
            ExecuteProcess(
                cmd=['ros2', 'param', 'set', filter_node_name, 'max_acc_vy', filter_max_acc_vy],
                output='screen',
                condition=IfCondition(set_filter_params),
            ),
        ]
    )

    return LaunchDescription([
        # Camera
        DeclareLaunchArgument('backend', default_value='argus'),
        DeclareLaunchArgument('device_id', default_value='0'),
        DeclareLaunchArgument('sensor_id', default_value='0'),
        DeclareLaunchArgument('width', default_value='640'),
        DeclareLaunchArgument('height', default_value='360'),
        DeclareLaunchArgument('fps', default_value='15.0'),
        DeclareLaunchArgument('flip_method', default_value='2'),

        DeclareLaunchArgument('image_topic', default_value='/myagv_camera/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/myagv_camera/camera_info'),
        DeclareLaunchArgument('frame_id', default_value='myagv_camera_frame'),
        DeclareLaunchArgument('show_debug_window', default_value='false'),
        
        DeclareLaunchArgument('sensor_mode', default_value='2'),
        DeclareLaunchArgument('capture_width', default_value='1920'),
        DeclareLaunchArgument('capture_height', default_value='1080'),
        DeclareLaunchArgument('capture_fps', default_value='30.0'),

        # Keep old topic names for compatibility with primitive_route_runner.
        DeclareLaunchArgument('cmd_topic', default_value='/cmd_vel_nav'),
        DeclareLaunchArgument('status_topic', default_value='/aruco_align_status'),
        DeclareLaunchArgument('done_topic', default_value='/aruco_align_done'),
        DeclareLaunchArgument('debug_image_topic', default_value='/aruco_align/debug_image'),
        DeclareLaunchArgument('debug_json_topic', default_value='/aruco_align/debug_json'),
        DeclareLaunchArgument('command_topic', default_value='/aruco_align_cmd'),
        DeclareLaunchArgument('target_name_topic', default_value='/marker_align_target'),

        # New marker setup: all markers are ArUco.
        DeclareLaunchArgument('dict_name', default_value='DICT_4X4_100'),
        DeclareLaunchArgument('target_id', default_value='-1'),
        DeclareLaunchArgument('target_name', default_value='RACK'),
        DeclareLaunchArgument(
            'marker_id_map',
            default_value='C:73,B:87,A:88,RACK:89,OBJ:89,TO_OBJ:89'
        ),
        DeclareLaunchArgument('process_every_n_frames', default_value='1'),
        DeclareLaunchArgument('min_control_side_px', default_value='15.0'),
        DeclareLaunchArgument('max_control_aspect_ratio', default_value='1.50'),

        # User requirement:
        # center tolerance 60px, min side >= 210px
        DeclareLaunchArgument('target_size_px', default_value='50.0'),
        DeclareLaunchArgument('size_tolerance_px', default_value='0.0'),
        DeclareLaunchArgument('target_cx_px', default_value='-1.0'),
        DeclareLaunchArgument('center_tolerance_px', default_value='60.0'),

        DeclareLaunchArgument('center_first', default_value='true'),
        DeclareLaunchArgument('done_required_count', default_value='3'),
        DeclareLaunchArgument('marker_smoothing_alpha', default_value='0.35'),

        DeclareLaunchArgument('kp_vx', default_value='0.00100'),
        DeclareLaunchArgument('kp_vy', default_value='0.00015'),
        DeclareLaunchArgument('max_vx', default_value='0.045'),
        DeclareLaunchArgument('min_vx', default_value='0.018'),
        DeclareLaunchArgument('max_vy', default_value='0.015'),
        DeclareLaunchArgument('min_vy', default_value='0.012'),
        DeclareLaunchArgument('invert_y', default_value='false'),

        DeclareLaunchArgument('lost_timeout_sec', default_value='1.5'),
        DeclareLaunchArgument('enable_lost_recovery', default_value='true'),
        DeclareLaunchArgument('lost_recovery_start_sec', default_value='0.25'),
        DeclareLaunchArgument('lost_recovery_sec', default_value='1.40'),
        DeclareLaunchArgument('lost_recovery_vy', default_value='0.012'),
        DeclareLaunchArgument('lost_recovery_min_err_px', default_value='25.0'),
        DeclareLaunchArgument('lost_recovery_max_age_sec', default_value='3.0'),
        DeclareLaunchArgument('lost_recovery_debug_period_sec', default_value='0.30'),
        DeclareLaunchArgument('allow_reverse', default_value='false'),
        DeclareLaunchArgument('stop_publish_sec', default_value='1.0'),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        DeclareLaunchArgument('publish_debug_json', default_value='true'),
        DeclareLaunchArgument('debug_json_period_sec', default_value='0.20'),
        DeclareLaunchArgument('print_debug', default_value='true'),

        # Safety filter patch for pure vy lateral alignment
        DeclareLaunchArgument('set_filter_params', default_value='false'),
        DeclareLaunchArgument('filter_node_name', default_value='/cmd_vel_safety_filter'),
        DeclareLaunchArgument('filter_max_vy', default_value='0.06'),
        DeclareLaunchArgument('filter_straight_vx_on', default_value='0.0'),
        DeclareLaunchArgument('filter_kill_vy_in_straight', default_value='false'),
        DeclareLaunchArgument('filter_max_acc_vy', default_value='0.15'),

        camera_node,
        align_node,
        patch_filter_params,
    ])
