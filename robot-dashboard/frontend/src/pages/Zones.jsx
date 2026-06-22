import { useState } from "react";
import useStore from "../store";
import {
  createZone,
  updateZone,
  deleteZone as apiDeleteZone,
} from "../api";
import "./Zones.css";

const ZONE_COLORS = {
  "#FDE68A": { badge: { bg: "#FEF3C7", text: "#92400E" } },
  "#BFDBFE": { badge: { bg: "#DBEAFE", text: "#1E40AF" } },
  "#DDD6FE": { badge: { bg: "#EDE9FE", text: "#5B21B6" } },
  "#BBF7D0": { badge: { bg: "#ECFDF5", text: "#065F46" } },
};
const DEFAULT_BADGE = { bg: "#F1F5F9", text: "#475569" };

const SHAPE_SYMBOL = {
  세모: "▲", 네모: "■", 동그라미: "●", 십자가: "✚", 오각형: "⬟",
  ㄷ: "ㄷ", 클로바: "☘", 평행사변형: "▱", ㄱ: "ㄱ", U: "U",
};

const COLORS_EN = { 빨강: "Red", 파랑: "Blue", 노랑: "Yellow", 초록: "Green", 주황: "Orange" };

export default function Zones() {
  const zones = useStore((s) => s.zones);
  const products = useStore((s) => s.products);

  const [showModal, setShowModal] = useState(false);
  const [editTarget, setEditTarget] = useState(null);

  const openAdd = () => { setEditTarget(null); setShowModal(true); };
  const openEdit = (zone) => { setEditTarget(zone); setShowModal(true); };

  const handleDelete = (id) => {
    if (window.confirm("정말 삭제하시겠습니까?")) apiDeleteZone(id);
  };

  const saveZone = async (data) => {
    if (editTarget) {
      await updateZone(editTarget.id, data);
    } else {
      await createZone(data);
    }
    setShowModal(false);
  };

  // 구역별 담당 상품 문자열 동적 생성
  const getZoneProducts = (zoneId) => {
    const list = products.filter((p) => p.zoneId === zoneId);
    if (list.length === 0) return "—";
    return list.map((p) => `${p.name} (${COLORS_EN[p.color] || p.color} ${SHAPE_SYMBOL[p.shape] || p.shape})`).join(", ");
  };

  const getZoneProductCount = (zoneId) => products.filter((p) => p.zoneId === zoneId).length;

  return (
    <div className="zones">
      <div className="page-header">
        <span className="page-title">배송 구역 관리</span>
        <button className="btn-accent" onClick={openAdd}>＋ 구역 추가</button>
      </div>

      <div className="zone-grid">
        {zones.map((z) => {
          const c = ZONE_COLORS[z.color] || { badge: DEFAULT_BADGE };
          const count = getZoneProductCount(z.id);
          return (
            <div className="zone-card" key={z.id} style={{ borderTopColor: z.color || "#E2E8F0" }}>
              <div className="zc-name">{z.name}</div>
              <div className="zc-desc">{z.desc}</div>
              <div className="zc-info">상품 {count}건 · {z.qr}</div>
              <div className="zc-status">
                <span className={`badge ${z.status === "운영 중" ? "b-teal" : "b-amber"}`}>{z.status}</span>
              </div>
              <div className="zc-actions">
                <button className="act-btn" onClick={() => openEdit(z)}>✏</button>
                <button className="act-btn" onClick={() => handleDelete(z.id)}>🗑</button>
              </div>
            </div>
          );
        })}
      </div>

      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: "12%" }}>구역명</th>
              <th style={{ width: "18%" }}>설명</th>
              <th style={{ width: "34%" }}>담당 상품</th>
              <th style={{ width: "12%" }}>상태</th>
              <th style={{ width: "12%" }}>QR ID</th>
              <th style={{ width: "12%" }}>관리</th>
            </tr>
          </thead>
          <tbody>
            {zones.map((z) => {
              const c = ZONE_COLORS[z.color] || { badge: DEFAULT_BADGE };
              return (
                <tr key={z.id}>
                  <td>
                    <span className="badge" style={{ background: c.badge.bg, color: c.badge.text }}>{z.name}</span>
                  </td>
                  <td className="desc-cell">{z.desc}</td>
                  <td className="desc-cell">{getZoneProducts(z.id)}</td>
                  <td>
                    <span className={`badge ${z.status === "운영 중" ? "b-teal" : "b-amber"}`}>{z.status}</span>
                  </td>
                  <td className="mono-cell">{z.qr}</td>
                  <td>
                    <div className="action-btns">
                      <button className="act-btn" onClick={() => openEdit(z)}>✏</button>
                      <button className="act-btn" onClick={() => handleDelete(z.id)}>🗑</button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {showModal && (
        <ZoneModal
          zone={editTarget}
          onSave={saveZone}
          onClose={() => setShowModal(false)}
        />
      )}
    </div>
  );
}

function ZoneModal({ zone, onSave, onClose }) {
  const [form, setForm] = useState(
    zone || { name: "", desc: "", color: "#BBF7D0", qr: "", status: "운영 중" }
  );

  const update = (key, val) => setForm({ ...form, [key]: val });

  const handleSubmit = () => {
    if (!form.name.trim()) { alert("구역명을 입력하세요"); return; }
    onSave(form);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span>{zone ? "구역 수정" : "구역 추가"}</span>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <div className="form-row">
            <label>구역명</label>
            <input value={form.name} onChange={(e) => update("name", e.target.value)} placeholder="예: 구역 E" />
          </div>
          <div className="form-row">
            <label>설명</label>
            <input value={form.desc} onChange={(e) => update("desc", e.target.value)} placeholder="예: 헤어 케어" />
          </div>
          <div className="form-row">
            <label>QR ID</label>
            <input value={form.qr} onChange={(e) => update("qr", e.target.value)} placeholder="예: QR-E01" />
          </div>
          <div className="form-row">
            <label>상태</label>
            <select value={form.status} onChange={(e) => update("status", e.target.value)}>
              <option>운영 중</option>
              <option>점검 중</option>
            </select>
          </div>
        </div>
        <div className="modal-footer">
          <button className="btn-outline" onClick={onClose}>취소</button>
          <button className="btn-accent" onClick={handleSubmit}>{zone ? "수정" : "추가"}</button>
        </div>
      </div>
    </div>
  );
}