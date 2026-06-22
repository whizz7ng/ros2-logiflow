import threading
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from database import init_db, SessionLocal
from models import Product, Zone, AppState

from routers import products, zones, missions, history
from ws_manager import manager
import ros_node


def seed_data():
    db = SessionLocal()
    try:
        if db.query(Product).count() == 0:
            db.add_all([
                Product(color="빨강", shape="세모", name="스킨 (150ml)", yolo_label="red_triangle", zone_id=1, stock=12, note="", status="활성"),
                Product(color="파랑", shape="네모", name="로션 (120ml)", yolo_label="blue_square", zone_id=2, stock=9, note="", status="활성"),
                Product(color="노랑", shape="오각형", name="립글로즈", yolo_label="yellow_pentagon", zone_id=3, stock=4, note="재고 부족 주의", status="활성"),
                Product(color="초록", shape="동그라미", name="핸드크림", yolo_label="green_circle", zone_id=3, stock=15, note="", status="활성"),
                Product(color="주황", shape="십자가", name="선크림 (50ml)", yolo_label="orange_cross", zone_id=1, stock=7, note="자외선 차단", status="활성"),
            ])
            db.commit()

        if db.query(Zone).count() == 0:
            db.add_all([
                Zone(name="구역 A", code="A", desc="기초 케어", color="#FDE68A", qr="QR-A01", status="운영 중"),
                Zone(name="구역 B", code="B", desc="메이크업", color="#BFDBFE", qr="QR-B01", status="운영 중"),
                Zone(name="구역 C", code="C", desc="바디 케어", color="#DDD6FE", qr="QR-C01", status="운영 중"),
            ])
            db.commit()

        row = db.query(AppState).filter(AppState.key == "mission_state").first()
        if row:
            row.value = "idle"
        else:
            db.add(AppState(key="mission_state", value="idle"))
        db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_data()
    print("✅ DB 초기화 + 시드 완료")

    main_loop = asyncio.get_running_loop()
    thread = threading.Thread(target=ros_node.start_ros, args=(main_loop,), daemon=True)
    thread.start()
    print("✅ ROS2 노드 스레드 시작")

    yield

    ros_node.stop_ros()
    print("🛑 ROS2 노드 종료")


app = FastAPI(title="분류로봇 대시보드 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|192\.168\.0\.\d+):5173",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(products.router)
app.include_router(zones.router)
app.include_router(missions.router)
app.include_router(history.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.post("/api/robot/estop")
def emergency_stop():
    ros_node.publish_estop("stop")
    return {"ok": True}


@app.post("/api/robot/reset")
def reset_estop():
    ros_node.publish_estop("reset")
    return {"ok": True}