import asyncio
import base64

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

from database import SessionLocal
from models import Product, Zone, MissionItem, HistoryRecord, AppState
from ws_manager import manager


node_instance = None


def _get_state(db):
    row = db.query(AppState).filter(
        AppState.key == "mission_state"
    ).first()

    return row.value if row else "idle"


def _set_state(db, state):
    row = db.query(AppState).filter(
        AppState.key == "mission_state"
    ).first()

    if row:
        row.value = state
    else:
        db.add(
            AppState(
                key="mission_state",
                value=state
            )
        )

    db.commit()


class WmsNode(Node):

    def __init__(self, main_loop):
        super().__init__("wms_dashboard_node")

        self.main_loop = main_loop
        self.current_item_id = None

        # 카메라 프레임 제한
        self._frame_skip = 0

        self.order_pub = self.create_publisher(
            String,
            "/order_request",
            10
        )

        self.estop_pub = self.create_publisher(
            String,
            "/emergency_stop",
            10
        )

        self.retry_pick_pub = self.create_publisher(
            String,
            "/retry_pick",
            10
        )

        self.go_home_pub = self.create_publisher(
            String,
            "/go_home",
            10
        )


        self.create_subscription(
            String,
            "/brain_state",
            self._on_brain_state,
            10
        )

        self.create_subscription(
            String,
            "/nav_status",
            self._on_nav_status,
            10
        )

        self.create_subscription(
            String,
            "/arm_status",
            self._on_pick_status,
            10
        )

        self.create_subscription(
            String,
            "/wms_update",
            self._on_wms_update,
            10
        )

        # 변경: BEST_EFFORT 구독
        self.create_subscription(
            CompressedImage,
            "/camera/camera/color/image_raw/compressed",
            self._on_camera,
            qos_profile_sensor_data
        )

    def _broadcast(self, type_, payload):
        asyncio.run_coroutine_threadsafe(
            manager.broadcast(
                {
                    "type": type_,
                    "payload": payload
                }
            ),
            self.main_loop
        )

    def _log(self, topic, text):
        self._broadcast(
            "topic_log",
            {
                "topic": topic,
                "text": text
            }
        )

    def _publish_next_order(self):

        if self.current_item_id is not None:
            return

        db = SessionLocal()

        try:
            if _get_state(db) != "running":
                return

            item = (
                db.query(MissionItem)
                .order_by(MissionItem.position)
                .first()
            )

            if not item:
                return

            zone = (
                db.query(Zone)
                .filter(Zone.id == item.zone_id)
                .first()
            )

            zone_code = zone.code if zone else ""

            msg = String()
            msg.data = f"{item.yolo_label}:{zone_code}:{item.rack_level}"

            self.order_pub.publish(msg)

            self.current_item_id = item.id

            self._log(
                "/order_request",
                msg.data
            )

        finally:
            db.close()

    def _on_brain_state(self, msg):
        self._log(
            "/brain_state",
            msg.data
        )

    def _on_nav_status(self, msg):

        self._broadcast(
            "robot_status",
            {
                "agv": {
                    "state": msg.data
                }
            }
        )

        self._log(
            "/nav_status",
            msg.data
        )

    def _on_pick_status(self, msg):

        self._broadcast(
            "robot_status",
            {
                "cobot": {
                    "state": msg.data
                }
            }
        )

        self._log(
            "/arm_status",
            msg.data
        )

        if msg.data == "error":

            db = SessionLocal()

            try:
                _set_state(db, "paused")
            finally:
                db.close()

            self.current_item_id = None

            self._broadcast("mission_state", {"state": "paused"})
            self._broadcast(
                "intervention",
                {"source": "pick_status", "message": "피킹 실패"}
            )

        elif msg.data == "placed":

            db = SessionLocal()

            try:
                item = (
                    db.query(MissionItem)
                    .filter(MissionItem.id == self.current_item_id)
                    .first()
                )

                if not item:
                    return

                # 재고 차감
                product = (
                    db.query(Product)
                    .filter(Product.yolo_label == item.yolo_label)
                    .first()
                )

                remaining_stock = 0

                if product:
                    product.stock = max(0, product.stock - 1)
                    db.commit()
                    remaining_stock = product.stock

                    self._broadcast(
                        "stock_update",
                        {"id": product.id, "stock": product.stock}
                    )

                # 배송 기록 추가
                record = HistoryRecord(
                    product_name=item.name,
                    yolo_label=item.yolo_label,
                    zone_id=item.zone_id,
                    zone_name=item.zone_name,
                    remaining_stock=remaining_stock,
                    status="완료"
                )
                db.add(record)
                db.commit()
                db.refresh(record)

                self._broadcast(
                    "history_add",
                    {
                        "id": record.id,
                        "productName": record.product_name,
                        "yoloLabel": record.yolo_label,
                        "zoneId": record.zone_id,
                        "zoneName": record.zone_name,
                        "duration": record.duration,
                        "remainingStock": record.remaining_stock,
                        "confidence": record.confidence,
                        "status": record.status,
                    }
                )

                # 큐에서 완료 항목 삭제 + position 재정렬
                item_id = item.id
                db.delete(item)
                db.commit()

                remaining_items = (
                    db.query(MissionItem)
                    .order_by(MissionItem.position)
                    .all()
                )
                for i, m in enumerate(remaining_items):
                    m.position = i
                db.commit()

                self._broadcast("queue_remove", {"id": item_id})

                self.current_item_id = None

                if remaining_items:
                    self._publish_next_order()
                else:
                    _set_state(db, "idle")
                    self._broadcast("mission_state", {"state": "idle"})

                # cobot 상태 리셋 — "placed" 배지가 다음 arm_status 올 때까지 남는 문제 방지
                self._broadcast(
                    "robot_status",
                    {"cobot": {"state": "idle"}}
                )

            finally:
                db.close()



    def _on_wms_update(self, msg):
        self._log(
            "/wms_update",
            msg.data
        )

    def _on_camera(self, msg):

        # 3프레임 중 1프레임만 전송
        self._frame_skip += 1

        if self._frame_skip % 3 != 0:
            return

        try:
            b64 = base64.b64encode(
                msg.data
            ).decode()

            self._broadcast(
                "camera_frame",
                {
                    "format": "jpeg",
                    "data": b64
                }
            )

        except Exception as e:

            self.get_logger().warning(
                f"camera send failed: {e}"
            )


def start_ros(main_loop):
    global node_instance

    rclpy.init()

    node_instance = WmsNode(
        main_loop
    )

    rclpy.spin(
        node_instance
    )


def stop_ros():

    if node_instance:

        node_instance.destroy_node()

        rclpy.shutdown()


def publish_next_order():

    if node_instance is None:
        print("⚠️ ROS 노드 미시작")
        return

    node_instance._publish_next_order()


def reset_current_item():

    if node_instance:
        node_instance.current_item_id = None


def publish_estop(command):

    if node_instance is None:
        print("⚠️ ROS 노드 미시작")
        return

    msg = String()

    msg.data = command

    node_instance.estop_pub.publish(msg)
    
def broadcast_mission_state(state):
    if node_instance:
        node_instance._broadcast("mission_state", {"state": state})


def reset_cobot_status():
    if node_instance:
        node_instance._broadcast("robot_status", {"cobot": {"state": "idle"}})
        
def retry_pick():

    if node_instance is None:
        print("⚠️ ROS 노드 미시작")
        return

    msg = String()
    msg.data = "reset"

    node_instance.retry_pick_pub.publish(msg)

    node_instance._log(
        "/retry_pick",
        msg.data
    )
    

def go_home():

    if node_instance is None:
        print("⚠️ ROS 노드 미시작")
        return

    msg = String()
    msg.data = "home"

    node_instance.go_home_pub.publish(msg)

    node_instance._log(
        "/go_home",
        msg.data
    )