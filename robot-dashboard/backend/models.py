from sqlalchemy import Column, Integer, String, Float, DateTime, func
from database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    color = Column(String, nullable=False)        # 빨강/노랑/파랑/초록/주황
    shape = Column(String, nullable=False)        # 세모/네모/동그라미/...
    name = Column(String, nullable=False)
    yolo_label = Column(String, default="")
    zone_id = Column(Integer, default=0)
    stock = Column(Integer, default=0)
    note = Column(String, default="")
    status = Column(String, default="활성")        # 활성/비활성


class Zone(Base):
    __tablename__ = "zones"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    code = Column(String, default="") 
    desc = Column(String, default="")
    color = Column(String, default="#BBF7D0")     # 카드 상단 border 색
    qr = Column(String, default="")
    status = Column(String, default="운영 중")     # 운영 중/점검 중


class MissionItem(Base):
    __tablename__ = "mission_queue"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, default=0)
    name = Column(String, default="")
    yolo_label = Column(String, default="")
    zone_id = Column(Integer, default=0)
    zone_name = Column(String, default="")
    position = Column(Integer, default=0)         # 큐 내 순서
    created_at = Column(DateTime, server_default=func.now())


class HistoryRecord(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    product_name = Column(String, default="")
    yolo_label = Column(String, default="")
    zone_id = Column(Integer, default=0)
    zone_name = Column(String, default="")
    duration = Column(String, default="")         # "17초" 등
    remaining_stock = Column(Integer, default=0)
    confidence = Column(String, default="")       # "0.87" 등
    status = Column(String, default="완료")        # 완료/이동 중/실패


class AppState(Base):
    """미션 상태 등 싱글톤 설정값"""
    __tablename__ = "app_state"

    key = Column(String, primary_key=True)
    value = Column(String, default="")
