from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    """
    tracer_nav2 mission stack.

    전제:
      - myAGV driver/odom/ekf가 실행 중이거나 별도 launch에서 실행됨
      - Nav2 + vel_filter_v2는 nav2_safety_filter.launch.py로 실행 중이거나 별도 실행
      - 이 launch는 mission layer만 실행함:
          camera + aruco_align + nav2_route_runner + mission_brain + agv_align_bridge

    중요한 구조:
      mission_brain_node -> /nav2_route_cmd
      nav2_route_runner_node -> Nav2 NavigateToPose action
      nav2_route_runner_node -> /debug/nav_status
      mission_brain_node <- /debug/nav_status
    """

    goals_yaml = LaunchConfiguration('goals_yaml')
    target_size_px = LaunchConfiguration('target_size_px')

    tracer_nav2_share = FindPackageShare('tracer_nav2')

    # ============================================================
    # Optional auto initial pose
    # ============================================================
    auto_initial_pose_node = Node(
        package='tracer_nav2',
        executable='auto_initial_pose_node',
        name='auto_initial_pose_node',
        output='screen',
        parameters=[{
            'auto_on_start': True,

            # parking_region pose
            'start_x': 0.28207093477249146,
            'start_y': 0.02868373692035675,
            'start_yaw': 0.005267,

            'initialpose_topic': '/initialpose',
            'amcl_topic': '/amcl_pose',
            'odom_topic': '/odometry/filtered',

            # filter를 태우기 위해 /cmd_vel_nav 사용
            'cmd_vel_topic': '/cmd_vel_nav',

            'enable_spin_scan': True,
            'spin_wz': 0.40,
            'spin_angle_rad': 6.28318530718,
            'spin_min_sec': 3.0,
            'spin_max_sec': 25.0,

            'max_cov_xx': 1.0,
            'max_cov_yy': 1.0,
            'max_cov_yaw': 1.0,
        }]
    )

    # ============================================================
    # Camera + ArUco align
    # ============================================================
    aruco_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                tracer_nav2_share,
                'launch',
                'aruco_align_test.launch.py'
            ])
        ),
        launch_arguments={
            'target_name': 'RACK',
            'target_size_px': target_size_px,
            'set_filter_params': 'false',
            'cmd_topic': '/cmd_vel_nav',

            # Nano low-load camera setting
            'width': '320',
            'height': '180',
            'fps': '5.0',

            'sensor_mode': '4',
            'capture_width': '1280',
            'capture_height': '720',
            'capture_fps': '5.0',

            'process_every_n_frames': '1',

            # 320x180 scale
            'done_required_count': '2',
            'center_tolerance_px': '35.0',
            'min_control_side_px': '15.0',

            'enable_lost_recovery': 'true',
            'lost_recovery_start_sec': '0.15',
            'lost_recovery_sec': '2.50',
            'lost_recovery_vy': '0.015',
            'lost_recovery_min_err_px': '10.0',
            'lost_recovery_max_age_sec': '5.0',
            'lost_timeout_sec': '2.5',

            'publish_debug_image': 'false',
            'publish_debug_json': 'true',
        }.items()
    )

    # ============================================================
    # Nav2 route runner
    # primitive_route_runner 대체.
    # mission_brain이 보내는 "goal xxx" / "route xxx" 명령을 받아
    # Nav2 NavigateToPose action으로 실행한다.
    # ============================================================
    nav2_route_runner_node = Node(
        package='tracer_nav2',
        executable='nav2_route_runner_node',
        name='nav2_route_runner_node',
        output='screen',
        parameters=[{
            'goals_yaml': goals_yaml,
            'command_topic': '/nav2_route_cmd',
            'status_topic': '/debug/nav_status',
            'action_name': '/navigate_to_pose',

            'goal_timeout_sec': 180.0,
            'server_wait_sec': 10.0,

            'enable_aruco_after_goal': True,
            'aruco_goal_names': 'to_obj,to_qr_a,to_qr_b,to_qr_c',
            'aruco_cmd_topic': '/aruco_align_cmd',
            'aruco_done_topic': '/aruco_align_done',
            'aruco_timeout_sec': 20.0,
            'continue_on_aruco_timeout': True,

            'print_debug': True,
        }]
    )

    # ============================================================
    # Mission brain
    # 기존 mission_brain_node 내부 parameter 이름은 primitive_cmd_topic이지만,
    # 실제 topic은 Nav2용 /nav2_route_cmd로 넘긴다.
    # ============================================================
    mission_brain_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                tracer_nav2_share,
                'launch',
                'mission_brain.launch.py'
            ])
        ),
        launch_arguments={
            'primitive_cmd_topic': '/nav2_route_cmd',
            'nav_status_topic': '/debug/nav_status',

            'order_request_topic': '/order_request',
            'place_target_topic': '/place_target',
            'arm_status_topic': '/arm_status',
            'go_parking_topic': '/go_parking',

            'auto_switch_aruco_target': 'true',
            'agv_align_enable_topic': '/agv_align_enable',
            'external_nav_status_topic': '/nav_status',

            # legacy /stop_obj, /stop_qr는 기본 OFF
            'publish_legacy_stop_topics': 'false',
        }.items()
    )

    # ============================================================
    # AGV align bridge
    # external brain_node -> /agv_align -> /cmd_vel_nav
    # ============================================================
    agv_align_bridge_node = Node(
        package='tracer_nav2',
        executable='agv_align_bridge_node',
        name='agv_align_bridge_node',
        output='screen',
        parameters=[{
            'input_topic': '/agv_align',
            'output_topic': '/cmd_vel_nav',
            'enable_topic': '/agv_align_enable',
            'brain_status_topic': '/brain_status',
            'status_topic': '/agv_align_bridge/status',

            'cmd_timeout_sec': 0.35,
            'publish_hz': 20.0,

            'max_vx': 0.100,
            'max_vy': 0.100,
            'max_wz': 0.400,

            'require_enable': True,
            'allow_brain_status_fallback': True,
            'block_z_axes': True,
            'print_debug': True,
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'goals_yaml',
            default_value='/home/er/myagv_ros2/src/tracer_nav2/config/goals.yaml'
        ),

        # 320x180 기준 ArUco marker 목표 크기
        DeclareLaunchArgument(
            'target_size_px',
            default_value='50.0'
        ),

        auto_initial_pose_node,
        aruco_launch,
        nav2_route_runner_node,
        mission_brain_launch,
        agv_align_bridge_node,
    ])
