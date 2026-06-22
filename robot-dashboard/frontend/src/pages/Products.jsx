import { useState } from "react";
import useStore from "../store";
import {
  createProduct,
  updateProduct,
  deleteProduct as apiDeleteProduct,
  adjustStock as apiAdjustStock,
  uploadLabelModel,
} from "../api";
import "./Products.css";

const COLORS = {
  빨강: { bg: "#FEE2E2", text: "#991B1B", dot: "#EF4444", en: "Red" },
  파랑: { bg: "#DBEAFE", text: "#1E40AF", dot: "#3B82F6", en: "Blue" },
  노랑: { bg: "#FEF3C7", text: "#92400E", dot: "#F59E0B", en: "Yellow" },
  초록: { bg: "#DCFCE7", text: "#166534", dot: "#22C55E", en: "Green" },
  주황: { bg: "#FFEDD5", text: "#9A3412", dot: "#F97316", en: "Orange" },
};

const SHAPES = ["세모", "네모", "동그라미", "십자가", "오각형", "ㄷ", "클로바", "평행사변형", "ㄱ", "U"];
const SHAPE_SYMBOL = {
  세모: "▲", 네모: "■", 동그라미: "●", 십자가: "✚", 오각형: "⬟",
  ㄷ: "ㄷ", 클로바: "☘", 평행사변형: "▱", ㄱ: "ㄱ", U: "U",
};

export default function Products() {
  const products = useStore((s) => s.products);
  const zones = useStore((s) => s.zones);
  const labelModel = useStore((s) => s.labelModel);

  const [showModal, setShowModal] = useState(false);
  const [editTarget, setEditTarget] = useState(null);

  const handleAdjustStock = (id, delta) => apiAdjustStock(id, delta);

  const handleDelete = (id) => {
    if (window.confirm("정말 삭제하시겠습니까?")) apiDeleteProduct(id);
  };

  const openAdd = () => { setEditTarget(null); setShowModal(true); };
  const openEdit = (product) => { setEditTarget(product); setShowModal(true); };

  const saveProduct = async (data) => {
    if (editTarget) {
      await updateProduct(editTarget.id, data);
    } else {
      await createProduct(data);
    }
    setShowModal(false);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (file) await uploadLabelModel(file);
  };

  // 라벨 매핑 현황: 상품에 연결된 라벨 vs labelModel의 전체 클래스
  const mappedLabels = products.map((p) => ({
    label: p.yoloLabel,
    color: COLORS[p.color]?.dot || "#9CA3AF",
  }));
  const unmappedLabels = labelModel
    ? labelModel.classes.filter((c) => !products.some((p) => p.yoloLabel === c.label))
    : [];

  return (
    <div className="products">
      <div className="page-header">
        <span className="page-title">상품 관리</span>
        <div className="header-actions">
          <label className="btn-outline">
            📤 모델 파일 업로드
            <input type="file" accept=".yaml,.json" style={{ display: "none" }} onChange={handleFileUpload} />
          </label>
          <button className="btn-accent" onClick={openAdd}>＋ 상품 추가</button>
        </div>
      </div>

      <div className="label-banner">
        <div className="label-top">
          <span className="label-title">🏷 YOLO 클래스 라벨 매핑</span>
          <span className="label-status">
            모델: <b>{labelModel ? labelModel.fileName : "없음"}</b> · <b>{mappedLabels.length}</b>개 매핑됨
          </span>
        </div>
        <div className="label-chips">
          {mappedLabels.map((l) => (
            <span className="chip-mapped" key={l.label}>
              <span className="chip-dot" style={{ background: l.color }} />{l.label}
            </span>
          ))}
          {unmappedLabels.map((l) => (
            <span className="chip-unmapped" key={l.label}>
              <span className="chip-dot" style={{ background: "#9CA3AF" }} />{l.label} ⚠
            </span>
          ))}
        </div>
        <div className="upload-drop">
          📁 <span>.yaml / .json 파일 드래그 또는 버튼으로 업로드 — YOLO data.yaml · classes.json 형식 지원</span>
        </div>
      </div>

      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: "9%" }}>색상</th>
              <th style={{ width: "10%" }}>모양</th>
              <th style={{ width: "16%" }}>상품명</th>
              <th style={{ width: "16%" }}>YOLO 라벨</th>
              <th style={{ width: "10%" }}>배송 구역</th>
              <th style={{ width: "14%" }}>재고</th>
              <th style={{ width: "11%" }}>비고</th>
              <th style={{ width: "7%" }}>상태</th>
              <th style={{ width: "7%" }}>관리</th>
            </tr>
          </thead>
          <tbody>
            {products.map((p) => {
              const c = COLORS[p.color] || {};
              const zone = zones.find((z) => z.id === p.zoneId);
              return (
                <tr key={p.id}>
                  <td>
                    <span className="color-chip" style={{ background: c.bg, color: c.text }}>
                      <span className="color-dot" style={{ background: c.dot }} />{c.en || p.color}
                    </span>
                  </td>
                  <td><span className="shape-chip">{SHAPE_SYMBOL[p.shape] || p.shape} {p.shape}</span></td>
                  <td>{p.name}</td>
                  <td><span className="mono">{p.yoloLabel}</span></td>
                  <td><span className="badge b-zone">{zone ? zone.name : "—"}</span></td>
                  <td>
                    <div className="qty-ctrl">
                      <button className="qty-btn" onClick={() => handleAdjustStock(p.id, -1)}>−</button>
                      <span className={`qty-num ${p.stock <= 5 ? "qty-low" : ""}`}>{p.stock}</span>
                      <button className="qty-btn" onClick={() => handleAdjustStock(p.id, 1)}>+</button>
                      {p.stock <= 5 && <span className="qty-warn">⚠ 부족</span>}
                    </div>
                  </td>
                  <td className="note-cell">{p.note || "—"}</td>
                  <td>
                    <span className={`badge ${p.status === "활성" ? "b-teal" : "b-gray"}`}>
                      {p.status}
                    </span>
                  </td>
                  <td>
                    <div className="action-btns">
                      <button className="act-btn" onClick={() => openEdit(p)}>✏</button>
                      <button className="act-btn" onClick={() => handleDelete(p.id)}>🗑</button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {showModal && (
        <ProductModal
          product={editTarget}
          zones={zones}
          onSave={saveProduct}
          onClose={() => setShowModal(false)}
        />
      )}
    </div>
  );
}

function ProductModal({ product, zones, onSave, onClose }) {
  const [form, setForm] = useState(
    product
      ? { ...product }
      : { color: "빨강", shape: "세모", name: "", yoloLabel: "", zoneId: zones[0]?.id || 1, stock: 0, note: "", status: "활성" }
  );

  const update = (key, val) => setForm({ ...form, [key]: val });

  const handleSubmit = () => {
    if (!form.name.trim()) { alert("상품명을 입력하세요"); return; }
    onSave(form);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span>{product ? "상품 수정" : "상품 추가"}</span>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <div className="form-row">
            <label>색상</label>
            <select value={form.color} onChange={(e) => update("color", e.target.value)}>
              {Object.keys(COLORS).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="form-row">
            <label>모양</label>
            <select value={form.shape} onChange={(e) => update("shape", e.target.value)}>
              {SHAPES.map((s) => <option key={s} value={s}>{SHAPE_SYMBOL[s]} {s}</option>)}
            </select>
          </div>
          <div className="form-row">
            <label>상품명</label>
            <input value={form.name} onChange={(e) => update("name", e.target.value)} placeholder="예: 스킨 (150ml)" />
          </div>
          <div className="form-row">
            <label>YOLO 라벨</label>
            <input value={form.yoloLabel} onChange={(e) => update("yoloLabel", e.target.value)} placeholder="예: red_triangle" />
          </div>
          <div className="form-row">
            <label>배송 구역</label>
            <select value={form.zoneId} onChange={(e) => update("zoneId", Number(e.target.value))}>
              {zones.map((z) => <option key={z.id} value={z.id}>{z.name}</option>)}
            </select>
          </div>
          <div className="form-row">
            <label>재고</label>
            <input type="number" value={form.stock} onChange={(e) => update("stock", parseInt(e.target.value) || 0)} />
          </div>
          <div className="form-row">
            <label>비고</label>
            <input value={form.note} onChange={(e) => update("note", e.target.value)} />
          </div>
          <div className="form-row">
            <label>상태</label>
            <select value={form.status} onChange={(e) => update("status", e.target.value)}>
              <option value="활성">활성</option>
              <option value="비활성">비활성</option>
            </select>
          </div>
        </div>
        <div className="modal-footer">
          <button className="btn-outline" onClick={onClose}>취소</button>
          <button className="btn-accent" onClick={handleSubmit}>{product ? "수정" : "추가"}</button>
        </div>
      </div>
    </div>
  );
}