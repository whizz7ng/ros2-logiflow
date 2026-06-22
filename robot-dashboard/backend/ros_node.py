import asyncio
import base64
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

from database import SessionLocal
from models import Product, Zone, MissionItem, HistoryRecord, AppState
from ws_manager import manager

node_instance = None


def _get_state(db):
    row = db.query(AppState).filter(AppState.key == "mission_state").first()
    return row.value if row else "idle"


def _set_state(db, state):
    row = db.query(AppState).filter(AppState.key == "mission_state").first()
    if row:
        row.value = state
    else:
        db.add(AppState(key="mission_state", value=state))
    db.commit()


class WmsNode(Node):
    def __init__(self, main_loop):
        super().__init__("wms_dashboard_node")
        self.main_loop = main_loop
        self.current_item_id = None

        self.order_pub = self.create_publisher(String, "/order_request", 10)
        self.estop_pub = self.create_publisher(String, "/emergency_stop", 10)

        self.create_subscription(String, "/brain_state", self._on_brain_state, 10)
        self.create_subscription(String, "/nav_status", self._on_nav_status, 10)
        self.create_subscription(String, "/pick_status", self._on_pick_status, 10)
        self.create_subscription(String, "/wms_update", self._on_wms_update, 10)
        self.create_subscription(CompressedImage, "/camera/image_compressed", self._on_camera, 10)

    def _broadcast(self, type_, payload):
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"type": type_, "payload": payload}), self.main_loop
        )

    def _log(self, topic, text):
        self._broadcast("topic_log", {"topic": topic, "text": text})

    def _publish_next_order(self):
        if self.current_item_id is not None:
            return  # 이미 진행 중인 주문 있음
        db = SessionLocal()
        try:
            if _get_state(db) != "running":
                return
            item = db.query(MissionItem).order_by(MissionItem.position).first()
            if not item:
                return
            zone = db.query(Zone).filter(Zone.id == item.zone_id).first()
            zone_code = zone.code if zone else ""
            msg = String()
            msg.data = f"{item.yolo_label}:{zone_code}"
            self.order_pub.publish(msg)
            self.current_item_id = item.id
            self._log("/order_request", msg.data)
        finally:
            db.close()

    def _on_brain_state(self, msg):
        self._log("/brain_state", msg.data)

    def _on_nav_status(self, msg):
        self._broadcast("robot_status", {"agv": {"state": msg.data}})
        self._log("/nav_status", msg.data)

    def _on_pick_status(self, msg):
        self._broadcast("robot_status", {"cobot": {"state": msg.data}})
        self._log("/pick_status", msg.data)

        if msg.data == "error":
                    db = SessionLocal()
                    try:
                        _set_state(db, "paused")
                    finally:
                        db.close()
                    self.current_item_id = None          # ← 추가 (재시도 위해 비움)
                    self._broadcast("mission_state", {"state": "paused"})
                    self._broadcast("intervention", {"source": "pick_status", "message": "피킹 실패"})

    def _on_wms_update(self, msg):
        self._log("/wms_update", msg.data)
        parts = msg.data.split(":")
        if len(parts) != 3:
            return
        label, zone_code, status = parts

        db = SessionLocal()
        try:
            if status == "done":
                item_id = self.current_item_id
                item = db.query(MissionItem).filter(MissionItem.id == item_id).first() if item_id else None

                product = db.query(Product).filter(Product.yolo_label == label).first()
                if product:
                    product.stock = max(0, product.stock - 1)
                    db.commit()
                    self._broadcast("stock_update", {"id": product.id, "stock": product.stock})

                zone = db.query(Zone).filter(Zone.code == zone_code).first()
                history = HistoryRecord(
                    product_name=product.name if product else label,
                    yolo_label=label,
                    zone_id=zone.id if zone else 0,
                    zone_name=zone.name if zone else "",
                    duration="-",
                    remaining_stock=product.stock if product else 0,
                    confidence="-",
                    status="완료",
                )
                db.add(history)
                db.commit()
                db.refresh(history)
                self._broadcast("history_add", {
                    "id": history.id,
                    "timestamp": history.timestamp.isoformat() if history.timestamp else None,
                    "productName": history.product_name,
                    "yoloLabel": history.yolo_label,
                    "zoneId": history.zone_id,
                    "zoneName": history.zone_name,
                    "duration": history.duration,
                    "remainingStock": history.remaining_stock,
                    "confidence": history.confidence,
                    "status": history.status,
                })

                if item:
                    db.delete(item)
                    db.commit()
                    self._broadcast("queue_remove", {"id": item_id})
                    remaining = db.query(MissionItem).order_by(MissionItem.position).all()
                    for i, m in enumerate(remaining):
                        m.position = i
                    db.commit()

                self.current_item_id = None
                if _get_state(db) == "running":
                    self._publish_next_order()

            elif status == "error":
                _set_state(db, "paused")
                self.current_item_id = None      # ← 추가 (재시도 위해 비움)
                self._broadcast("mission_state", {"state": "paused"})
                self._broadcast("intervention", {"source": "wms_update", "message": f"{label} 처리 실패"})
        finally:
            db.close()

    def _on_camera(self, msg):
        b64 = base64.b64encode(bytes(msg.data)).decode()
        self._broadcast("camera_frame", {"format": msg.format, "data": b64})


def start_ros(main_loop):
    global node_instance
    rclpy.init()
    node_instance = WmsNode(main_loop)
    rclpy.spin(node_instance)


def stop_ros():
    if node_instance is not None:
        node_instance.destroy_node()
    rclpy.shutdown()


def publish_next_order():
    if node_instance is None:
        print("⚠️ ROS 노드 미시작")
        return
    node_instance._publish_next_order()

def reset_current_item():
    if node_instance is not None:
        node_instance.current_item_id = None

def publish_estop(command: str):
    if node_instance is None:
        print("⚠️ ROS 노드 미시작")
        return
    msg = String()
    msg.data = command
    node_instance.estop_pub.publish(msg)