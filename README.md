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

| 토픽명 (Topic Name)   | 발신 (Publisher)        | 수신 (Subscriber)                         | 메시지 타입                       | 내용 / 비고                                                                                                         |
| ------------------ | --------------------- | --------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `/order_request`   | `wms_node`            | `brain_node`                            | `std_msgs/String`            | 주문 정보 전달. `"물품라벨:구역"` 형식으로 사용. 예: `"red_triangle:A"`<br>물품 라벨과 배송구역은 콜론(`:`)으로 구분되며, 서로 독립적으로 처리된다.             |
| `/vision_activate` | `brain_node`          | `vision_node`                           | `std_msgs/String`            | 비전 인식 활성화/중지 명령.<br>물품 라벨 수신 시 해당 물품 검출 시작, `"stop"` 수신 시 검출 중지.                                                |
| `/box_pose`        | `vision_node`         | `brain_node`                            | `std_msgs/Float32MultiArray` | 인식된 블록의 3D 목표 좌표 전달.<br>형식: `[x, y, z, rx, ry, rz]`                                                             |
| `/pick_command`    | `brain_node`          | `pick_node` (Pi)                        | `std_msgs/Float32MultiArray` | 피킹 명령 및 물체 목표 좌표 전달.<br>형식: `[x, y, z, rx, ry, rz]`                                                             |
| `/place_command`   | `brain_node`          | `pick_node` (Pi)                        | `std_msgs/Float32MultiArray` | 플레이싱 명령 및 포장구역 내려놓기 좌표 전달.<br>형식: `[x, y, z, rx, ry, rz]`                                                       |
| `/pick_status`     | `pick_node` (Pi)      | `brain_node`                            | `std_msgs/String`            | 피킹/플레이싱 결과 신호.<br>사용 값: `"done"`, `"placing_done"`, `"error"`                                                   |
| `/place_target`    | `brain_node`          | `nav_node`                              | `std_msgs/String`            | 포장 목적지 지정.<br>사용 값: `"A"`, `"B"`, `"C"`<br>`nav_node`가 물체 위치 및 포장구역 이동 흐름을 관리한다.                                |
| `/arm_status`      | `brain_node`          | `nav_node`                              | `std_msgs/String`            | 로봇팔 작업 상태 알림.<br>사용 값: `"picked"`, `"placed"`<br>AGV 회전 또는 다음 이동 트리거에 사용된다.                                     |
| `/go_parking`      | `brain_node`          | `nav_node`                              | `std_msgs/Empty`             | 모든 주문 완료 후 주차 위치로 복귀 명령.                                                                                        |
| `/nav_status`      | `nav_node`            | `brain_node`                            | `std_msgs/String`            | AGV 이동 상태 피드백.<br>사용 값: `"arrived_objects"`, `"arrived"`, `"parked"`                                            |
| `/qr_result`       | `qr_node`             | `nav_node`                              | `std_msgs/String`            | AGV 내부 QR 인식 결과 및 정밀 정차 신호.<br>AGV가 구역 판단과 재시도를 자체 처리한다.                                                        |
| `/depth_qr`        | `vision_node`         | 측정·로그용                                  | `std_msgs/String`            | D435i 뎁스 기반 구역 QR 검증 결과.<br>형식: `"A:0.90"` = `구역:검출성공률`<br>FSM 흐름에는 관여하지 않는 백업/측정용이며, `NAV_TO_DEST` 상태에서만 동작한다. |
| `/emergency_stop`  | `keyboard_estop_node` | `brain_node` / `pick_node` / `nav_node` | `std_msgs/String`            | 비상정지 및 해제 명령.<br>사용 값: `"stop"`, `"reset"`                                                                      |
| `/wms_update`      | `brain_node`          | `wms_node`                              | `std_msgs/String`            | 재고 차감 및 주문 완료 처리 요청.                                                                                            |
| `/brain_state`     | `brain_node`          | `dashboard_node`                        | `std_msgs/String`            | FSM 현재 상태를 대시보드로 전달.                                                                                            |


