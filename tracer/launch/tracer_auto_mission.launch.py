from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    """
    Tracer mission automation stack.

    Assumption:
      myagv_navigation2/navigation2_filter_active.launch.py is already running
      or is launched separately, so /cmd_vel_safety_filter, /amcl_pose,
      /odometry/filtered are available.

    This launch starts:
      - auto_initial_pose_node
      - myAGV built-in camera
      - aruco_align_node
      - primitive_route_runner + nav_debug_logger
      - mission_brain_node (emergency_stop / go_home / idle)
      - agv_align_bridge_node

    After this stack is up, Orin brain_node can start a mission by publishing:
      /place_target std_msgs/String "A"|"B"|"C"
    or:
      /order_request std_msgs/String "A"|"B"|"C" / JSON containing target.

    Camera / ArUco note:
      Nano 내장 카메라 + Python OpenCV node에서 640x360@10Hz는 CPU 병목이 있었으므로,
      최종 안정값은 320x180@5Hz로 낮춘다.
      이에 맞춰 ArUco target_size, center_tolerance, min_control_side도 절반 수준으로 낮춘다.
    """

    goals_yaml = LaunchConfiguration('goals_yaml')
    target_size_px = LaunchConfiguration('target_size_px')

    tracer_share = FindPackageShare('tracer')

    # ============================================================
    # Auto initial pose node
    # RViz 2D Pose Estimate를 자동으로 발행하고,
    # 필요하면 제자리 360도 회전으로 AMCL 수렴을 돕는다.
    # ============================================================
    auto_initial_pose_node = Node(
        package='tracer',
        executable='auto_initial_pose_node',
        name='auto_initial_pose_node',
        output='screen',
        parameters=[{
            'auto_on_start': True,

            # parking_region pose
            'start_x': 0.28207093477249146,
            'start_y': 0.02868373692035675,
            'start_yaw': 0.005267,

            # RViz 2D Pose Estimate와 같은 역할
            'initialpose_topic': '/initialpose',
            'amcl_topic': '/amcl_pose',
            'odom_topic': '/odometry/filtered',

            # filter를 태우기 위해 /cmd_vel_nav 사용
            'cmd_vel_topic': '/cmd_vel_nav',

            # 360도 scan
            'enable_spin_scan': True,
            'spin_wz': 0.40,
            'spin_angle_rad': 6.28318530718,
            'spin_min_sec': 3.0,
            'spin_max_sec': 25.0,

            # covariance 기준은 처음엔 넉넉하게
            'max_cov_xx': 1.0,
            'max_cov_yy': 1.0,
            'max_cov_yaw': 1.0,
        }]
    )

    # ============================================================
    # ArUco camera + align node
    # ============================================================
    aruco_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                tracer_share,
                'launch',
                'aruco_align_test.launch.py'
            ])
        ),
        launch_arguments={
            # target
            'target_name': 'RACK',
            'target_size_px': target_size_px,
            'set_filter_params': 'true',
            'cmd_topic': '/cmd_vel_nav',

            # ====================================================
            # Camera low-load setting
            # ====================================================
            # Nano에서 640x360@10Hz 이상은 CPU가 과하게 올라갔으므로
            # 320x180@5Hz로 안정화한다.
            'width': '320',
            'height': '180',
            'fps': '5.0',

            # Argus capture setting
            # mode 4 = 1280x720 sensor mode.
            # capture_fps도 5로 낮춰 pipeline 자체 부하를 줄인다.
            'sensor_mode': '4',
            'capture_width': '1280',
            'capture_height': '720',
            'capture_fps': '5.0',

            'process_every_n_frames': '1',

            # ====================================================
            # Align success condition for 320x180
            # ====================================================
            # 기존 640x360 기준:
            #   target_size_px ~= 200
            #   center_tolerance_px ~= 70
            #   min_control_side_px ~= 60
            #
            # 320x180에서는 거의 절반으로 스케일링.
            'done_required_count': '3',
            'center_tolerance_px': '30.0',
            'min_control_side_px': '17.0',
            # lateral align
            'kp_vy': '0.00022',
            'max_vy': '0.025',
            'min_vy': '0.014',

            # forward align
            'kp_vx': '0.00100',
            'max_vx': '0.035',
            'min_vx': '0.016',


            # ====================================================
            # Rough front/yaw alignment using marker skew
            # ====================================================
            # 정밀 solvePnP 없이 ArUco 사각형의 좌/우 변 길이 차이로
            # 대략 정면 여부를 판단하고 angular.z를 살짝 낸다.
            'enable_front_align': 'true',
            'front_align_only_when_center_ok': 'true',

            # front_ok 판단 기준
            'front_skew_tol': '0.06',
            'front_lr_ratio_max': '1.15',
            'front_tb_ratio_max': '1.20',
            'front_angle_tol_deg': '10.0',

            # wz control
            'kp_wz': '0.45',
            'max_wz': '0.10',
            'min_wz': '0.035',
            'invert_wz': 'false',

            # frame-cut / lost-marker recovery
            'enable_lost_recovery': 'true',
            'lost_recovery_start_sec': '0.45',
            'lost_recovery_sec': '1.50',
            'lost_recovery_vy': '0.010',
            'lost_recovery_min_err_px': '10.0',
            'lost_recovery_max_age_sec': '3.0',
            'lost_timeout_sec': '2.5',

            'marker_smoothing_alpha': '0.55',

            # debug load reduction
            'publish_debug_image': 'true',
            'publish_debug_json': 'true',
        }.items()
    )

    # ============================================================
    # Primitive route runner + nav debug logger
    # ============================================================
    primitive_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                tracer_share,
                'launch',
                'primitive_route_test.launch.py'
            ])
        ),
        launch_arguments={
            'goals_yaml': goals_yaml,
            'autostart': 'false',
            'enable_aruco_after_goal': 'true',
            'aruco_goal_names': 'to_obj,to_qr_a,to_qr_b,to_qr_c',
            'aruco_timeout_sec': '60.0',
        }.items()
    )

    # ============================================================
    # Mission brain
    # ============================================================
    mission_brain_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                tracer_share,
                'launch',
                'mission_brain.launch.py'
            ])
        ),
        launch_arguments={
            'order_request_topic': '/order_request_unused',
            'place_target_topic': '/place_target',
            'arm_status_topic': '/arm_status',
            'go_parking_topic': '/go_parking',

            # External safety commands. Any String payload triggers.
            'emergency_stop_topic': '/emergency_stop',
            'emergency_reset_topic': '/retry_pick',
            'go_home_topic': '/go_home',
            'go_home_release_topic': '/go_home_release',
            'idle_topic': '/idle',
            'odom_topic': '/odometry/filtered',
            'parking_goal_name': 'parking_region',

            # Stop confirmation before IDLE / direct parking.
            'stop_confirm_vx': '0.02',
            'stop_confirm_vy': '0.02',
            'stop_confirm_wz': '0.05',
            'stop_confirm_count': '3',
            'stop_confirm_max_wait_sec': '1.0',
            'stop_odom_timeout_sec': '0.30',

            'auto_switch_aruco_target': 'true',

            # /stop_obj, /stop_qr 이후 외부 brain_node의 /agv_align을 허용하기 위한 enable topic
            'agv_align_enable_topic': '/agv_align_enable',
            'external_nav_status_topic': '/nav_status',
        }.items()
    )

    # ============================================================
    # AGV align bridge
    # External brain_node can publish low-speed correction to /agv_align.
    # This bridge forwards it to /cmd_vel_nav only while mission_brain allows it.
    # ============================================================
    agv_align_bridge_node = Node(
        package='tracer',
        executable='agv_align_bridge_node',
        name='agv_align_bridge_node',
        output='screen',
        parameters=[{
            'input_topic': '/agv_align',
            'output_topic': '/cmd_vel_nav',
            'enable_topic': '/agv_align_enable',
            'status_topic': '/agv_align_bridge/status',

            # 외부 보정 명령은 WAIT_PICKED / WAIT_NEXT 등 mission_brain이 허용할 때만 전달
            'cmd_timeout_sec': 0.35,
            'publish_hz': 20.0,

            # low-speed micro alignment limits
            'max_vx': 0.100,
            'max_vy': 0.100,
            'max_wz': 0.400,

            'publish_stop_when_disabled': True,
            'print_debug': True,
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'goals_yaml',
            default_value='/home/er/myagv_ros2/src/tracer/config/goals.yaml'
        ),

        # 320x180 기준 ArUco marker 목표 크기.
        # 기존 640x360에서 200px 쓰던 것을 절반으로 낮춤.
        DeclareLaunchArgument(
            'target_size_px',
            default_value='50.0'
        ),

        auto_initial_pose_node,
        aruco_launch,
        primitive_launch,
        mission_brain_launch,
        agv_align_bridge_node,
    ])
