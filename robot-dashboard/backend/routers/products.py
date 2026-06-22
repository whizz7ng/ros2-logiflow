from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from database import get_db
from models import Product

router = APIRouter(prefix="/api/products", tags=["products"])


# ── Pydantic 스키마 ──
class ProductCreate(BaseModel):
    color: str
    shape: str
    name: str
    yolo_label: str = ""
    zone_id: int = 0
    stock: int = 0
    note: str = ""
    status: str = "활성"

class ProductUpdate(BaseModel):
    color: Optional[str] = None
    shape: Optional[str] = None
    name: Optional[str] = None
    yolo_label: Optional[str] = None
    zone_id: Optional[int] = None
    stock: Optional[int] = None
    note: Optional[str] = None
    status: Optional[str] = None

class StockAdjust(BaseModel):
    delta: int


# ── 엔드포인트 ──
@router.get("")
def list_products(db: Session = Depends(get_db)):
    return db.query(Product).all()


@router.post("", status_code=201)
def create_product(body: ProductCreate, db: Session = Depends(get_db)):
    p = Product(**body.model_dump())
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.put("/{product_id}")
def update_product(product_id: int, body: ProductUpdate, db: Session = Depends(get_db)):
    p = db.query(Product).filter(Product.id == product_id).first()
    if not p:
        raise HTTPException(404, "상품을 찾을 수 없습니다")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(p, k, v)
    db.commit()
    db.refresh(p)
    return p


@router.delete("/{product_id}")
def delete_product(product_id: int, db: Session = Depends(get_db)):
    p = db.query(Product).filter(Product.id == product_id).first()
    if not p:
        raise HTTPException(404, "상품을 찾을 수 없습니다")
    db.delete(p)
    db.commit()
    return {"ok": True}


@router.patch("/{product_id}/stock")
def adjust_stock(product_id: int, body: StockAdjust, db: Session = Depends(get_db)):
    p = db.query(Product).filter(Product.id == product_id).first()
    if not p:
        raise HTTPException(404, "상품을 찾을 수 없습니다")
    p.stock = max(0, p.stock + body.delta)
    db.commit()
    db.refresh(p)
    return p
