#!/bin/bash

TEST_NO=${1:-01}
BASE=~/pj3_ws/deburg/kpi_before_align/test_${TEST_NO}
NODE_DIR="$BASE/node_logs"
TOPIC_DIR="$BASE/topic_logs"

mkdir -p "$NODE_DIR"
mkdir -p "$TOPIC_DIR"

echo "[ALL] Jetson test_${TEST_NO} start"
echo "[ALL] node log dir : $NODE_DIR"
echo "[ALL] topic log dir: $TOPIC_DIR"

echo "[ALL] start Jetson topic logs..."

ros2 topic echo /brain_state      > "$TOPIC_DIR/topic_brain_state.log" 2>&1 &
ros2 topic echo /nav_status       > "$TOPIC_DIR/topic_nav_status.log" 2>&1 &
ros2 topic echo /vision_activate  > "$TOPIC_DIR/topic_vision_activate.log" 2>&1 &
ros2 topic echo /distance_status  > "$TOPIC_DIR/topic_distance_status.log" 2>&1 &
ros2 topic echo /align_status     > "$TOPIC_DIR/topic_align_status.log" 2>&1 &
ros2 topic echo /marker_agv_pose  > "$TOPIC_DIR/topic_marker_agv_pose.log" 2>&1 &

ros2 topic echo /observe_move     > "$TOPIC_DIR/topic_observe_move.log" 2>&1 &
ros2 topic echo /observe_ready    > "$TOPIC_DIR/topic_observe_ready.log" 2>&1 &
ros2 topic echo /observe_pose     > "$TOPIC_DIR/topic_observe_pose.log" 2>&1 &

ros2 topic echo /box_pose         > "$TOPIC_DIR/topic_box_pose.log" 2>&1 &
ros2 topic echo /pick_command     > "$TOPIC_DIR/topic_pick_command.log" 2>&1 &
ros2 topic echo /pick_status      > "$TOPIC_DIR/topic_pick_status.log" 2>&1 &
ros2 topic echo /j1_correction    > "$TOPIC_DIR/topic_j1_correction.log" 2>&1 &

ros2 topic echo /place_pose       > "$TOPIC_DIR/topic_place_pose.log" 2>&1 &
ros2 topic echo /place_command    > "$TOPIC_DIR/topic_place_command.log" 2>&1 &
ros2 topic echo /place_status     > "$TOPIC_DIR/topic_place_status.log" 2>&1 &

ros2 topic echo /emergency_stop   > "$TOPIC_DIR/topic_emergency_stop.log" 2>&1 &
ros2 topic echo /go_parking       > "$TOPIC_DIR/topic_go_parking.log" 2>&1 &
ros2 topic echo /wms_update       > "$TOPIC_DIR/topic_wms_update.log" 2>&1 &

echo "[ALL] start Jetson nodes..."

python3 brain_node.py 2>&1 | tee "$NODE_DIR/brain.log" &
python3 align_node.py 2>&1 | tee "$NODE_DIR/agv_align.log" &

(
    source ~/yolo_env/bin/activate
    echo "[KPI] vision venv python: $(which python)"
    python3 vision_eyeinhand_node.py 2>&1 | tee "$NODE_DIR/vision.log"
) &

echo "[ALL] Jetson all started."
echo "[ALL] watch brain:"
echo "tail -f $NODE_DIR/brain.log"
echo "[ALL] stop:"
echo "pkill -f brain_node.py; pkill -f vision_eyeinhand_node.py; pkill -f align_node.py; pkill -f 'ros2 topic echo'"

wait
