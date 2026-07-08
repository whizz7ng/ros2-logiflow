# ROS2 LogiFlow

ROS2 기반 물류 패키지 운반 및 분류 자율주행 로봇 프로젝트

## 프로젝트 개요
myAGV 2023 기반 로봇이 패키지를 인식하고 목적지로 자율주행하여 운반 및 분류하는 시스템

## 개발 환경
- Robot: myAGV 2023
- Framework: ROS2
- AI: YOLO (객체 인식)
- Language: Python

## 주요 기능
- 자율주행 (Navigation)
- 패키지 인식 (YOLO)
- 물류 분류
- 시스템 모니터링

## 팀 브랜치
- main (최종)
- feat/zzz
- feat/aaa
- feat/bbb


----------------------
# Jetson Orin Nano 개발환경 셋업 가이드

> 물품 자동 분류·배송 로봇 프로젝트 / 메인 허브(Orin Nano) 기준
> 베이스: **JetPack 6.2 (L4T 36.4.3) / Ubuntu 22.04 / Python 3.10**

---

## ⚠️ 가장 먼저 읽을 주의사항

1. **시스템/펌웨어 업그레이드 금지**
   - `sudo apt upgrade`, `apt full-upgrade`, 배포판 업그레이드(22.04→24.04) **하지 말 것**
   - 이 보드는 EEPROM 보드ID가 비어있는 개체 문제가 있어, L4T 패키지가 올라가면 부팅이 깨짐
   - L4T 관련 패키지는 `hold` 걸려 있음 (`apt-mark showhold`로 확인 가능)
   - 일반 앱 업데이트(LibreOffice 등)는 무방하나, 목록에 `nvidia-l4t-*` / 커널 / 부트로더가 보이면 중단

2. **네트워크 (공용망 asia-edu, 수동 고정 IP)**
   - Orin Nano: `192.168.0.35`
   - 라즈베리파이(myCobot): `192.168.0.36`
   - 증상 "갑자기 SSH 안 됨(Connection refused)" → 공용망 IP 혼선. 해당 기기에서 WiFi 재시작:
     ```
     sudo nmcli con down "asia-edu" && sudo nmcli con up "asia-edu"
     ```

---

## 1. CUDA / cuDNN / TensorRT

JetPack에 이미 포함되어 있음 (CUDA 12.6 / cuDNN 9.3 / TensorRT 10.3).
`nvcc`가 안 잡히면 PATH만 추가:

```bash
echo '' >> ~/.bashrc
echo '# CUDA' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version   # release 12.6 확인
```

---

## 2. ROS2 Humble

> 교육은 Jazzy로 받았으나 Jetson(22.04)에서는 **Humble**이 표준.
> Jazzy는 Ubuntu 24.04 전용이라 apt 설치 불가.

```bash
# 저장소 + 키
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

# 설치
sudo apt install -y ros-humble-desktop
sudo apt install -y ros-dev-tools python3-colcon-common-extensions

# 환경변수
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc

echo $ROS_DISTRO   # humble 확인
```

---

## 3. RealSense D435i

```bash
sudo apt install -y ros-humble-realsense2-camera ros-humble-realsense2-description
```

동작 확인 (카메라는 **USB 3.0 / 파란 포트**에 연결):

```bash
# 터미널 A
ros2 launch realsense2_camera rs_launch.py
# 터미널 B
ros2 topic list | grep camera
ros2 topic hz /camera/camera/color/image_raw   # ~30Hz면 정상
```

---

## 4. PyTorch + YOLOv8 (venv)

> **numpy 충돌**을 피하려고 venv를 쓴다. torch(Jetson 빌드)는 numpy 1.x 필요.
> venv는 `--system-site-packages`로 만들어 ROS2(rclpy)도 함께 보이게 한다.

### venv 생성 및 활성화
```bash
sudo apt install -y python3-venv
python3 -m venv ~/yolo_env --system-site-packages
source ~/yolo_env/bin/activate     # 프롬프트에 (yolo_env) 표시
pip install --upgrade pip
```

### numpy 고정 → torch 설치 (순서 중요)
```bash
pip install "numpy==1.26.4"
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

GPU 확인:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.8.0 True Orin  → 정상
```

### ultralytics 설치 (numpy 재충돌 주의)
```bash
pip install ultralytics --no-deps
pip install opencv-python pillow pyyaml requests scipy matplotlib tqdm psutil py-cpuinfo pandas seaborn polars ultralytics-thop

# 위 과정에서 numpy가 2.x로 튀므로 반드시 다시 고정
pip install "numpy==1.26.4"

python3 -c "import torch, numpy, cv2; print(torch.cuda.is_available(), numpy.__version__, cv2.__version__)"
# True 1.26.4 ... → 정상
```

### 추론 테스트
```bash
yolo predict model=yolov8s.pt source='https://ultralytics.com/images/bus.jpg' device=0
```

---

## 5. venv 사용법 (팀원용)

```bash
source ~/yolo_env/bin/activate   # YOLO/torch 작업 시작 전 항상 실행
# ... 작업 ...
deactivate                        # 종료
```

- ROS2 노드에서 YOLO를 쓰려면 venv 활성화 상태로 노드를 실행하면 됨
  (torch + rclpy 둘 다 인식됨)
- **주의**: venv 안에서 `pip install`로 새 패키지를 깔 때 numpy가 2.x로 올라가면
  `pip install "numpy==1.26.4"`로 다시 내릴 것

---

## 시스템 요약

| 항목 | 버전/값 |
|------|---------|
| JetPack / L4T | 6.2 / 36.4.3 |
| Ubuntu / Python | 22.04 / 3.10 |
| CUDA / cuDNN / TensorRT | 12.6 / 9.3 / 10.3 |
| ROS2 | Humble |
| PyTorch / torchvision | 2.8.0 / 0.23.0 |
| numpy (고정) | 1.26.4 |
| venv 경로 | `~/yolo_env` |
| Orin Nano IP | 192.168.0.35 |
| 라즈베리파이(myCobot) IP | 192.168.0.36 |

---

## CycloneDDS로 Orin, Pi, AGV 통신
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
echo 'export ROS_DOMAIN_ID=35' >> ~/.bashrc
source ~/.bashrc





## ROS2 Topic Interface

### 메인 흐름 (주문 → 픽업 → 배송)

| 토픽명                | 발신                   | 수신                   | 타입                           | 내용 / 비고                                                                                 |
| ------------------ | -------------------- | -------------------- | ---------------------------- | --------------------------------------------------------------------------------------- |
| `/order_request`   | `wms_dashboard_node` | `brain_node`         | `std_msgs/String`            | 주문 정보. `"물품라벨:구역:층"` 형식. 예: `"green_clover:A:2"`. 층 생략 시 기본 1층                          |
| `/place_target`    | `brain_node`         | `nav_node`           | `std_msgs/String`            | 포장 목적지. 값: `"A"`, `"B"`, `"C"`. 주문 시작 시 AGV 목적지 설정                                      |
| `/nav_status`      | `nav_node`           | `brain_node`         | `std_msgs/String`            | AGV 이동 상태. 값: `"arrived_objects"`, `"arrived"`, `"parked"`                              |
| `/observe_move`    | `brain_node`         | `pick_node`          | `std_msgs/String`            | eye-in-hand 관측 자세 이동 명령. 데이터는 층 번호 `"1"` 또는 `"2"`                                       |
| `/observe_ready`   | `pick_node`          | `brain_node`         | `std_msgs/String`            | 로봇팔이 관측 자세에 도착했음을 알림. 값: `"ready"`                                                      |
| `/observe_pose`    | `pick_node`          | `vision_node`        | `std_msgs/Float32MultiArray` | 실제 관측 자세의 로봇팔 좌표. `[x, y, z, rx, ry, rz]`. vision_node가 동적 `T_cam2base` 계산에 사용          |
| `/vision_activate` | `brain_node`         | `vision_node`        | `std_msgs/String`            | 비전 인식 활성화/중지. 블록 검출은 `"물품라벨:층"` 형식. 예: `"green_clover:2"`. `"stop"` 시 중지                |
| `/box_pose`        | `vision_node`        | `brain_node`         | `std_msgs/Float32MultiArray` | 인식된 블록의 로봇팔 기준 3D 목표 좌표. `[x, y, z, rx, ry, rz]`                                        |
| `/pick_command`    | `brain_node`         | `pick_node`          | `std_msgs/Float32MultiArray` | 피킹 명령 및 목표 좌표. `/box_pose`를 받아 그대로 전달. `[x, y, z, rx, ry, rz]`                          |
| `/pick_status`     | `pick_node`          | `brain_node`         | `std_msgs/String`            | 피킹/플레이싱 결과. 값: `"done"`, `"pick_failed"`, `"placing_done"`, `"realign_fail"`, `"error"` |
| `/arm_status`      | `brain_node`         | `nav_node`           | `std_msgs/String`            | 로봇팔 작업 상태. 값: `"picked"`, `"placed"`. AGV 이동 트리거로 사용                                    |
| `/place_command`   | `brain_node`         | `pick_node`          | `std_msgs/Float32MultiArray` | 플레이싱 명령 및 내려놓기 좌표. 현재는 `ZONE_TO_PLACE` 고정 좌표 사용. `[x, y, z, rx, ry, rz]`                |
| `/go_parking`      | `brain_node`         | `nav_node`           | `std_msgs/Empty`             | 모든 주문 완료 후 주차 복귀 명령                                                                     |
| `/wms_update`      | `brain_node`         | `wms_dashboard_node` | `std_msgs/String`            | 주문 완료/실패 알림. `"물품라벨:구역:상태"` 예: `"green_clover:A:done"`                             

Vision 거리 판정 및 AGV 차체보정 흐름

| 토픽명                 | 발신                                   | 수신                      | 타입                           | 내용 / 비고                                                          |
| ------------------- | ------------------------------------ | ----------------------- | ---------------------------- | ---------------------------------------------------------------- |
| `/distance_status`  | `vision_node`                        | `brain_node`            | `std_msgs/String`            | 블록까지의 거리 상태. 값 예: `"ok:311"`, `"too_close:239"`, `"too_far:370"` |
| `/marker_agv_pose`  | `vision_node`                        | `agv_align_node`        | `std_msgs/Float32MultiArray` | ArUco 마커의 AGV 기준 좌표. `[level, Lx, Ly, Rx, Ry]`. 안 보이는 마커는 `NaN`  |
| `/agv_align`        | `agv_align_node`                     | `agv_align_bridge_node` | `geometry_msgs/Twist`        | AGV 차체보정 속도 명령. `linear.x`, `linear.y`, `angular.z` 사용           |
| `/align_status`     | `agv_align_node`                     | `brain_node`            | `std_msgs/String`            | 차체보정 상태. `"step_done"`은 한 번 보정 이동 완료, `"aligned"`는 정렬 완료         |
| `/agv_align_enable` | `brain_node` 또는 `mission_brain_node` | `agv_align_bridge_node` | `std_msgs/Bool`              | AGV align 명령 허용 여부. `True`일 때만 `/agv_align`을 실제 주행 명령으로 전달       |
| `/cmd_vel_nav`      | `agv_align_bridge_node`              | `cmd_vel_safety_filter` | `geometry_msgs/Twist`        | bridge를 통과한 중간 속도 명령                                             |
| `/cmd_vel`          | `cmd_vel_safety_filter`              | `myAGV_driver`          | `geometry_msgs/Twist`        | 최종 AGV 구동 명령                                                     |


차체보정 판단 기준

| 상황              | vision_node 동작                                                 | brain_node 동작                                             | align_node 동작         |
| --------------- | -------------------------------------------------------------- | --------------------------------------------------------- | --------------------- |
| 블록 거리 정상        | `/distance_status = "ok:mm"` 발행 후 `/box_pose` 발행               | `waiting_align_step=False`, `/box_pose` 수신 시 `PICKING` 전환 | 동작 없음                 |
| 블록이 너무 가까움      | `/distance_status = "too_close:mm"` 발행 후 `/marker_agv_pose` 발행 | `waiting_align_step=True`, `/align_status step_done` 대기   | 후진 방향 `/agv_align` 발행 |
| 블록이 너무 멂        | `/distance_status = "too_far:mm"` 발행 후 `/marker_agv_pose` 발행   | `waiting_align_step=True`, `/align_status step_done` 대기   | 전진 방향 `/agv_align` 발행 |
| depth 실패 반복     | `/marker_agv_pose` 발행 또는 `realign_fail`                        | 재관측 또는 ERROR 처리                                           | 마커 기준 보정              |
| 늦은 step_done 수신 | 해당 없음                                                          | `waiting_align_step=False`이면 무시                           | 해당 없음                 |

J1 보정 및 마커 보정 보조 토픽

| 토픽명               | 발신            | 수신                         | 타입                            | 내용 / 비고                                                                     |
| ----------------- | ------------- | -------------------------- | ----------------------------- | --------------------------------------------------------------------------- |
| `/j1_correction`  | `vision_node` | `pick_node`                | `std_msgs/String`             | 로봇팔 1번축 보정 명령. 예: `"2:8.5"`. `"realign_fail"`이면 팔 보정으로 해결 불가                |
| `/detected_image` | `vision_node` | `dashboard_node` 또는 디버그 뷰어 | `sensor_msgs/CompressedImage` | YOLO 검출 결과 이미지                                                              |
| `/depth_qr`       | `vision_node` | `nav_node` 또는 `brain_node` | `std_msgs/String`             | QR 검증 결과. 예: `"A:0.80"`. 현재 배송 검증용으로 사용 가능                                  |
| `/place_pose`     | `vision_node` | `brain_node` 예정            | `std_msgs/Float32MultiArray`  | QR 기반 플레이싱 좌표. `[x, y, z, rx, ry, rz]`. 현재 vision에는 기능이 있으나 brain 연동은 추후 작업 |


현재 FSM 상태값

| 상태               | 의미                         |
| ---------------- | -------------------------- |
| `IDLE`           | 대기 상태                      |
| `NAV_TO_RACK`    | AGV가 물체 위치로 이동 중           |
| `OBSERVING`      | 로봇팔이 관측 자세로 이동 중           |
| `VISION`         | vision_node가 블록 또는 마커 인식 중 |
| `PICKING`        | pick_node가 블록 파지 중         |
| `NAV_TO_DEST`    | AGV가 포장 목적지로 이동 중          |
| `PLACING`        | 로봇팔이 물체를 내려놓는 중            |
| `GO_PARKING`     | 모든 주문 완료 후 주차 위치로 복귀 중     |
| `ERROR`          | 오류 상태                      |
| `EMERGENCY_STOP` | 비상정지 상태                    |


/pick_status 값 정의
| 값                | 의미                                |
| ---------------- | --------------------------------- |
| `"done"`         | 피킹 성공                             |
| `"pick_failed"`  | 파지 실패. brain_node가 재관측 후 한 번 더 시도 |
| `"placing_done"` | 플레이싱 완료                           |
| `"realign_fail"` | 로봇팔 J1 보정으로 해결 불가. AGV 차체 보정 필요   |
| `"error"`        | pick_node 내부 오류                   |


/vision_activate 값 정의

| 값                  | 의미                                                           |
| ------------------ | ------------------------------------------------------------ |
| `"green_clover:2"` | 2층에서 `green_clover` 블록 검출 시작                                 |
| `"red_cross:1"`    | 1층에서 `red_cross` 블록 검출 시작                                    |
| `"stop"`           | 비전 인식 중지                                                     |
| `"qr_place"`       | QR 기반 place 좌표 계산 모드. 현재 vision_node 기능은 있으나 brain 연동은 추후 작업 |
| `"align:2"`        | ArUco 정렬 전용 모드. align-first 구조 전환 시 사용 예정                    |

현재 기준 전체 흐름

/order_request "green_clover:A:2"
↓
brain_node: NAV_TO_RACK
↓
/place_target "A"
↓
nav_node: 물체 위치 이동
↓
/nav_status "arrived_objects"
↓
brain_node: OBSERVING
↓
/observe_move "2"
↓
pick_node: 2층 관측 자세 이동
↓
/observe_ready "ready"
/observe_pose [x,y,z,rx,ry,rz]
↓
brain_node: VISION
↓
/vision_activate "green_clover:2"
↓
vision_node: YOLO + depth
↓
거리 정상:
  /distance_status "ok:311"
  /box_pose [x,y,z,rx,ry,rz]
  ↓
  brain_node: PICKING
  ↓
  /pick_command
  ↓
  /pick_status "done"

거리 비정상:
  /distance_status "too_close:239" 또는 "too_far:370"
  /marker_agv_pose [level,Lx,Ly,Rx,Ry]
  ↓
  agv_align_node: /agv_align 발행
  ↓
  /align_status "step_done"
  ↓
  brain_node: OBSERVING으로 돌아가 재관측
  

### 보조 / 검증

| 토픽명 | 발신 | 수신 | 타입 | 내용 / 비고 |
| --- | --- | --- | --- | --- |
| `/qr_result` | `qr_node` | `nav_node` | `std_msgs/String` | AGV 내부 QR 인식 및 정밀 정차 신호. AGV가 구역 판단/재시도 자체 처리 |
| `/depth_qr` | `vision_node` | 측정·로그용 | `std_msgs/String` | D435i 뎁스 기반 구역 QR 검증. `"A:0.90"` = `구역:성공률`. FSM 미관여, `NAV_TO_DEST`에서만 동작 |
| `/emergency_stop` | `keyboard_estop_node` / `wms_dashboard_node` | `brain_node` / `pick_node` / `nav_node` | `std_msgs/String` | 비상정지/해제. 값: `"stop"`, `"reset"` |

### 카메라 (realsense2_camera 드라이버 공유)

D435i 한 대를 드라이버가 열고, vision_node·대시보드·라인트레이싱이 토픽을 구독해 공유.

| 토픽명 | 발신 | 수신 | 타입 | 내용 / 비고 |
| --- | --- | --- | --- | --- |
| `/camera/camera/color/image_raw` | `realsense2_camera` | `vision_node`, `line_tracer` | `sensor_msgs/Image` | 컬러 원본 (YOLO 추론, 라인트레이싱) |
| `/camera/camera/color/image_raw/compressed` | `realsense2_camera` | `wms_dashboard_node` | `sensor_msgs/CompressedImage` | 대시보드용 압축 영상. format=`"rgb8; jpeg compressed bgr8"` |
| `/camera/camera/aligned_depth_to_color/image_raw` | `realsense2_camera` | `vision_node` | `sensor_msgs/Image` | 정렬 depth (3D 좌표 계산용) |
| `/camera/camera/color/camera_info` | `realsense2_camera` | `vision_node` | `sensor_msgs/CameraInfo` | intrinsic (deproject용) |
| `/detected_image` | `vision_node` | `wms_dashboard_node` | `sensor_msgs/CompressedImage` | YOLO 검출 결과 영상(박스 표시). 검출 시점 발행 |

**드라이버 실행:**
```
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true align_depth.enable:=true rgb_camera.color_profile:=640x480x30
ros2 launch realsense2_camera rs_launch.py   qos_overrides./camera/camera.color_qos:=1   qos_overrides./camera/camera.depth_qos:=1   enable_color:=true   enable_depth:=true   align_depth.enable:=true

#short range mode
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  json_file_path:=/home/zzz/short_range.json

#short range mode
ros2 param set /camera/camera depth_module.visual_preset 5

D435i 픽셀 노이즈 제거
ros2 param set /camera/camera spatial_filter.enable true
ros2 param set /camera/camera temporal_filter.enable true
ros2 param set /camera/camera hole_filling_filter.enable true


```

---

### YOLO 클래스 (물품 라벨)

`/order_request`, `/vision_activate`의 물품라벨은 반드시 아래 5개 중 하나:

- `blue_pentagon`
- `green_clover`
- `green_dome`
- `red_cross`
- `red_square`

> ⚠️ 대시보드 DB의 `yolo_label`도 위 5개로 통일 필요 (기존 `red_triangle`, `blue_square` 등은 모델에 없음)

---

### 변경 이력

1. `/order_request` 예시 클래스명: `red_triangle` → 실제 클래스(`red_cross` 등)
2. `/wms_update` 형식 확정: `"물품라벨:구역:상태"` 3개 필드 (brain `_finish_current_order`에서 `f"{item}:{zone}:done"` 발행)
3. 카메라 구조 변경: vision_node 직접 발행 → realsense2_camera 드라이버 공유 (옵션 A)
4. 노드명 정정: `wms_node` → `wms_dashboard_node`

----------------------
# 대시 보드 키는법

(백앤드)
cd ~/proj/robot-dashboard/backend

uvicorn main:app --host 0.0.0.0 --port 8000

(프론트)
cd ~/proj/robot-dashboard/frontend

npm run dev -- --host 0.0.0.0

