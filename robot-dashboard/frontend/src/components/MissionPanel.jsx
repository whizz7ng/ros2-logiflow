import { useState } from "react";
import useStore from "../store";
import {
  addMissionItem,
  removeMissionItem,
  clearMissionQueue,
  startMission,
  pauseMission,
  resumeMission,
  cancelMission,
} from "../api";
import "./MissionPanel.css";

const SHAPE_SYMBOL = {
  세모: "▲", 네모: "■", 동그라미: "●", 십자가: "✚", 오각형: "⬟",
  ㄷ: "ㄷ", 클로바: "☘", 평행사변형: "▱", ㄱ: "ㄱ", U: "U",
};

const STATE_INFO = {
  idle: { label: "대기", cls: "b-gray" },
  running: { label: "실행 중", cls: "b-teal" },
  paused: { label: "일시정지", cls: "b-amber" },
  cancelled: { label: "취소됨", cls: "b-red" },
};

export default function MissionPanel() {
  const products = useStore((s) => s.products);
  const zones = useStore((s) => s.zones);
  const queue = useStore((s) => s.missionQueue);
  const missionState = useStore((s) => s.missionState);

  const [selected, setSelected] = useState("");

  // 상품 DB(store) 기준으로 선택 목록 구성 — 더 이상 하드코딩 배열 없음
  const productOptions = products.map((p) => {
    const zone = zones.find((z) => z.id === p.zoneId);
    return {
      id: p.id,
      name: p.name,
      label: p.yoloLabel,
      zoneId: p.zoneId,
      zoneName: zone ? zone.name : "—",
      display: `${p.name} — ${p.color} ${SHAPE_SYMBOL[p.shape] || p.shape}`,
    };
  });

  const picked = productOptions.find((p) => p.id === Number(selected));

  const addQueue = async () => {
    if (!picked) return;
    await addMissionItem({
      productId: picked.id,
      name: picked.name,
      yoloLabel: picked.label,
      zoneId: picked.zoneId,
      zoneName: picked.zoneName,
    });
    setSelected("");
  };

  const si = STATE_INFO[missionState];

  return (
    <div className="mission-card">
      <div className="card-hd">📨 미션 명령</div>

      <div className="sel-row">
        <div className="sel-wrap">
          <div className="sel-label">상품 선택 <b>→ 구역 자동 매핑</b></div>
          <select className="ps" value={selected} onChange={(e) => setSelected(e.target.value)}>
            <option value="">-- 상품을 선택하세요 --</option>
            {productOptions.map((p) => (
              <option key={p.id} value={p.id}>{p.display}</option>
            ))}
          </select>
        </div>
        <div className="zone-auto">
          <div className="za-lbl">배송 구역</div>
          <div className="za-val">{picked ? picked.zoneName : "—"}</div>
          <div className="za-cls">{picked ? picked.label : "label"}</div>
        </div>
      </div>

      <button className="add-q-btn" onClick={addQueue}>＋ 큐에 추가</button>

      <div className="q-section">
        <div className="q-hd">
          <span>대기 미션 큐</span>
          <div className="q-hd-right">
            <b>{queue.length}건</b>
            {queue.length > 0 && (
              <button className="q-clear" onClick={() => clearMissionQueue()}>전체 삭제</button>
            )}
          </div>
        </div>
        <div className="q-list">
          {queue.length === 0 && (
            <div className="q-empty">큐가 비어있습니다</div>
          )}
          {queue.map((q, i) => (
            <div className={`q-item ${i === 0 && missionState === "running" ? "q-active" : ""}`} key={q.id}>
              <span className="q-n">{i + 1}</span>
              <span className="q-name">{q.name}</span>
              <span className="q-cls">{q.yoloLabel}</span>
              <span className="q-arr">→</span>
              <span className="q-dest">{q.zoneName}</span>
              {i === 0 && missionState === "running" && (
                <span className="badge b-amber" style={{ fontSize: "9px" }}>실행 중</span>
              )}
              <button className="q-del" onClick={() => removeMissionItem(q.id)}>✕</button>
            </div>
          ))}
        </div>
      </div>

      <div className="mission-ctrl">
        <div className="state-row">
          <span className="state-label">미션 상태</span>
          <span className={`badge ${si.cls}`}>{si.label}</span>
        </div>
        <div className="ctrl-3">
          <button className="cb cb-primary" disabled={!(missionState === "idle" || missionState === "cancelled") || queue.length === 0} onClick={() => startMission()}>▶ 시작</button>
          {missionState === "paused"
            ? <button className="cb" onClick={() => resumeMission()}>▶ 재개</button>
            : <button className="cb" disabled={missionState !== "running"} onClick={() => pauseMission()}>⏸ 일시정지</button>}
          <button className="cb cb-danger" disabled={missionState === "idle" || missionState === "cancelled"} onClick={() => cancelMission()}>✕ 취소</button>
        </div>

      </div>
    </div>
  );
}
