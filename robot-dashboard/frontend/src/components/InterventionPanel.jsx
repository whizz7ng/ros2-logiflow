import useStore from "../store";
import { emergencyStop } from "../api";
import "./InterventionPanel.css";

export default function InterventionPanel() {
  const alert = useStore((s) => s.interventionAlert);
  const clearAlert = useStore((s) => s.clearInterventionAlert);

  const handleEstop = async () => {
    try {
      await emergencyStop();
    } catch (e) {
      console.error("E-STOP 전송 실패", e);
    }
  };

  return (
    <div className="int-card">
      <div className="card-hd">✋ 사용자 개입</div>

      {alert && (
        <div
          style={{
            background: "#FEE2E2",
            color: "#991B1B",
            padding: "8px 12px",
            borderRadius: 6,
            marginBottom: 10,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontSize: 13,
          }}
        >
          <span>⚠ {alert.message}</span>
          <button
            onClick={clearAlert}
            style={{ background: "none", border: "none", cursor: "pointer", color: "#991B1B", fontWeight: "bold" }}
          >
            ✕
          </button>
        </div>
      )}

      <div className="int-sect">현재 미션</div>
      <div className="btn-row">
        <button className="ib" disabled title="미구현">🔄 피킹 재시도</button>
        <button className="ib" disabled title="미구현">⬇ 낙하물 회수</button>
      </div>
      <div className="divider" />
      <div className="int-sect">하드웨어</div>
      <div className="btn-row">
        <button className="ib" disabled title="미구현">🏠 홈 복귀</button>
        <button className="estop" onClick={handleEstop}>⚠ 긴급 정지 (E-STOP)</button>
      </div>
    </div>
  );
}