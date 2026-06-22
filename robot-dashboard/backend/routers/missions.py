from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import get_db
from models import MissionItem, AppState
from ros_node import publish_next_order, reset_current_item

router = APIRouter(prefix="/api/missions", tags=["missions"])


class MissionAdd(BaseModel):
    product_id: int = 0
    name: str = ""
    yolo_label: str = ""
    zone_id: int = 0
    zone_name: str = ""

class MissionCommand(BaseModel):
    action: str  # start / pause / resume / cancel
    


def _get_mission_state(db: Session) -> str:
    row = db.query(AppState).filter(AppState.key == "mission_state").first()
    return row.value if row else "idle"


def _set_mission_state(db: Session, state: str):
    row = db.query(AppState).filter(AppState.key == "mission_state").first()
    if row:
        row.value = state
    else:
        db.add(AppState(key="mission_state", value=state))
    db.commit()


@router.get("/queue")
def list_queue(db: Session = Depends(get_db)):
    items = db.query(MissionItem).order_by(MissionItem.position).all()
    return items


@router.post("/queue", status_code=201)
def add_to_queue(body: MissionAdd, db: Session = Depends(get_db)):
    max_pos = db.query(MissionItem).count()
    item = MissionItem(**body.model_dump(), position=max_pos)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.delete("/queue/{item_id}")
def remove_from_queue(item_id: int, db: Session = Depends(get_db)):
    item = db.query(MissionItem).filter(MissionItem.id == item_id).first()
    if not item:
        raise HTTPException(404, "미션 아이템을 찾을 수 없습니다")
    db.delete(item)
    db.commit()
    # 순서 재정렬
    for i, m in enumerate(db.query(MissionItem).order_by(MissionItem.position).all()):
        m.position = i
    db.commit()
    return {"ok": True}


@router.delete("/queue")
def clear_queue(db: Session = Depends(get_db)):
    db.query(MissionItem).delete()
    db.commit()
    return {"ok": True}


@router.get("/state")
def get_state(db: Session = Depends(get_db)):
    return {"state": _get_mission_state(db)}


@router.post("/command")
def mission_command(body: MissionCommand, db: Session = Depends(get_db)):
    current = _get_mission_state(db)
    action = body.action

    transitions = {
        ("idle", "start"): "running",
        ("cancelled", "start"): "running",
        ("running", "pause"): "paused",
        ("paused", "resume"): "running",
        ("running", "cancel"): "cancelled",
        ("paused", "cancel"): "cancelled",
    }

    new_state = transitions.get((current, action))
    if new_state is None:
        raise HTTPException(400, f"'{current}' 상태에서 '{action}' 불가")

    _set_mission_state(db, new_state)

    if new_state == "cancelled":
        db.query(MissionItem).delete()
        db.commit()
        reset_current_item()

    if new_state == "paused":
        reset_current_item() 

    if new_state == "running":
        publish_next_order()

    return {"state": new_state}