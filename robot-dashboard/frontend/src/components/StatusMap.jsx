import useStore from "../store";
import "./StatusMap.css";

const PICK_LABEL = {
  done:         { label: "피킹 완료",   cls: "b-teal" },
  placing_done: { label: "플레이싱 완료", cls: "b-teal" },
  error:        { label: "오류 / 비상정지", cls: "b-red" },
};
const MISSION_LABEL = {
  idle:      { label: "대기",     cls: "b-gray" },
  running:   { label: "실행 중",  cls: "b-teal" },
  paused:    { label: "일시정지", cls: "b-amber" },
  cancelled: { label: "취소됨",   cls: "b-red" },
};
const pick = (map, v) => map[v] || { label: v ? v : "대기", cls: "b-gray" };

export default function StatusMap() {
  const robotStatus = useStore((s) => s.robotStatus);
  const missionState = useStore((s) => s.missionState);

  const cobot = pick(PICK_LABEL, robotStatus.cobot?.state);
  const mission = pick(MISSION_LABEL, missionState);

  return (
    <div className="status-map">
      <div className="sm-card">
        <div className="card-hd">📈 실시간 상태</div>
        <div className="status-row">
          <span className="status-k">🚩 미션</span>
          <span className={`badge ${mission.cls}`}>{mission.label}</span>
        </div>
      </div>

      <div className="sm-card">
        <div className="card-hd">🦾 myCobot 상태</div>
        <div className="status-row">
          <span className="status-k">🦾 현재 상태</span>
          <span className={`badge ${cobot.cls}`}>{cobot.label}</span>
        </div>
      </div>
    </div>
  );
}