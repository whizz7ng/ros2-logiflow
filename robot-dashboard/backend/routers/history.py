from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from database import get_db
from models import HistoryRecord

router = APIRouter(prefix="/api/history", tags=["history"])


class HistoryCreate(BaseModel):
    product_name: str = ""
    yolo_label: str = ""
    zone_id: int = 0
    zone_name: str = ""
    duration: str = ""
    remaining_stock: int = 0
    confidence: str = ""
    status: str = "완료"


@router.get("")
def list_history(limit: int = 100, db: Session = Depends(get_db)):
    return db.query(HistoryRecord).order_by(HistoryRecord.id.desc()).limit(limit).all()


@router.post("", status_code=201)
def create_history(body: HistoryCreate, db: Session = Depends(get_db)):
    r = HistoryRecord(**body.model_dump())
    db.add(r)
    db.commit()
    db.refresh(r)
    return r
