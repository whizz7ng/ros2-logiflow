from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from database import get_db
from models import Zone

router = APIRouter(prefix="/api/zones", tags=["zones"])


class ZoneCreate(BaseModel):
    name: str
    code: str = ""
    desc: str = ""
    color: str = "#BBF7D0"
    qr: str = ""
    status: str = "운영 중"

class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
    desc: Optional[str] = None
    color: Optional[str] = None
    qr: Optional[str] = None
    status: Optional[str] = None


@router.get("")
def list_zones(db: Session = Depends(get_db)):
    return db.query(Zone).all()


@router.post("", status_code=201)
def create_zone(body: ZoneCreate, db: Session = Depends(get_db)):
    z = Zone(**body.model_dump())
    db.add(z)
    db.commit()
    db.refresh(z)
    return z


@router.put("/{zone_id}")
def update_zone(zone_id: int, body: ZoneUpdate, db: Session = Depends(get_db)):
    z = db.query(Zone).filter(Zone.id == zone_id).first()
    if not z:
        raise HTTPException(404, "구역을 찾을 수 없습니다")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(z, k, v)
    db.commit()
    db.refresh(z)
    return z


@router.delete("/{zone_id}")
def delete_zone(zone_id: int, db: Session = Depends(get_db)):
    z = db.query(Zone).filter(Zone.id == zone_id).first()
    if not z:
        raise HTTPException(404, "구역을 찾을 수 없습니다")
    db.delete(z)
    db.commit()
    return {"ok": True}
