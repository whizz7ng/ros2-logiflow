import useStore from "../store";
import "./StatusMap.css";

const NAV_LABEL = {
  parked:          { label: "정차 / 대기",   cls: "b-gray" },
  arrived:         { label: "목적지 도착",   cls: "b-teal" },
  arrived_objects: { label: "적재 위치 도착", cls: "b-blue" },
};
const PICK_LABEL = {
  done:         { label: "피킹 완료", cls: "b-teal" },
  placing_done: { label: "적재 완료", cls: "b-teal" },
  error:        { label: "오류",      cls: "b-red" },
};
const MISSION_LABEL = {
  idle:      { label: "대기",     cls: "b-gray" },
  running:   { label: "실행 중",  cls: "b-teal" },
  paused:    { label: "일시정지", cls: "b-amber" },
  cancelled: { label: "취소됨",   cls: "b-red" },
};
const pick = (map, v) => map[v] || { label: v || "—", cls: "b-gray" };

export default function StatusMap() {
  const robotStatus = useStore((s) => s.robotStatus);
  const missionState = useStore((s) => s.missionState);

  const agv = pick(NAV_LABEL, robotStatus.agv?.state);
  const cobot = pick(PICK_LABEL, robotStatus.cobot?.state);
  const mission = pick(MISSION_LABEL, missionState);

  return (
    <div className="status-map">
      <div className="sm-card">
        <div className="card-hd">📈 실시간 상태</div>
        <div className="status-row">
          <span className="status-k">📍 AGV</span>
          <span className={`badge ${agv.cls}`}>{agv.label}</span>
        </div>
        <div className="status-row">
          <span className="status-k">🦾 myCobot</span>
          <span className={`badge ${cobot.cls}`}>{cobot.label}</span>
        </div>
        <div className="status-row">
          <span className="status-k">🚩 미션</span>
          <span className={`badge ${mission.cls}`}>{mission.label}</span>
        </div>
      </div>

      <div className="sm-card">
        <div className="card-hd">🗺 AGV 이동 상태</div>
        <div className="status-row">
          <span className="status-k">📍 현재 상태</span>
          <span className={`badge ${agv.cls}`}>{agv.label}</span>
        </div>
      </div>
    </div>
  );
}