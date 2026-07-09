#!/bin/bash

# Jetson에서 Brain / Align / Vision 노드만 실행하는 스크립트
# 위치 기준:
#   ~/pj3_ws/src/brain_pkg/brain_pkg/start_all_jetson.sh
#
# 실행:
#   cd ~/pj3_ws/src/brain_pkg/brain_pkg
#   ./start_all_jetson.sh
#
# 종료:
#   Ctrl + C
# 또는
#   pkill -f brain_node.py
#   pkill -f align_node.py
#   pkill -f vision_eyeinhand_node.py

echo "[ALL] Jetson nodes start"

echo "[ALL] start brain_node..."
python3 brain_node.py &

echo "[ALL] start align_node..."
python3 align_node.py &

echo "[ALL] start vision_eyeinhand_node with yolo_env..."
(
    source ~/yolo_env/bin/activate
    echo "[ALL] vision venv python: $(which python)"
    python3 vision_eyeinhand_node.py
) &

echo "[ALL] Jetson nodes started."
echo "[ALL] stop command:"
echo "pkill -f brain_node.py; pkill -f align_node.py; pkill -f vision_eyeinhand_node.py"

wait
