import useStore from "../store";
import "./History.css";

const ZONE_BADGE = {
  "구역 A": { bg: "#FEF3C7", text: "#92400E" },
  "구역 B": { bg: "#DBEAFE", text: "#1E40AF" },
  "구역 C": { bg: "#EDE9FE", text: "#5B21B6" },
};

export default function History() {
  const history = useStore((s) => s.history);
  const topicLog = useStore((s) => s.topicLog);
  const zones = useStore((s) => s.zones);

  const getZoneName = (zoneId) => {
    const z = zones.find((z) => z.id === zoneId);
    return z ? z.name : "—";
  };

  return (
    <div className="history">
      <div className="page-header">
        <span className="page-title">배송 기록</span>
        <button className="btn-outline">📥 CSV 내보내기</button>
      </div>

      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: "6%" }}>#</th>
              <th style={{ width: "10%" }}>시각</th>
              <th style={{ width: "18%" }}>YOLO 라벨</th>
              <th style={{ width: "14%" }}>상품명</th>
              <th style={{ width: "10%" }}>구역</th>
              <th style={{ width: "8%" }}>소요</th>
              <th style={{ width: "10%" }}>잔여재고</th>
              <th style={{ width: "8%" }}>신뢰도</th>
              <th style={{ width: "8%" }}>상태</th>
            </tr>
          </thead>
          <tbody>
            {history.length === 0 && (
              <tr><td colSpan={9} style={{ textAlign: "center", color: "#9CA3AF", padding: "24px" }}>배송 기록이 없습니다</td></tr>
            )}
            {history.map((r, i) => {
              const zoneName = r.zoneName || getZoneName(r.zoneId);
              const zc = ZONE_BADGE[zoneName] || { bg: "#F1F5F9", text: "#475569" };
              const timeStr = r.timestamp ? new Date(r.timestamp).toLocaleTimeString("ko-KR", { hour12: false }) : "—";
              return (
                <tr key={r.id || i}>
                  <td className="id-cell">#{String(r.id).padStart(3, "0")}</td>
                  <td className="time-cell">{timeStr}</td>
                  <td><span className="mono">{r.yoloLabel || "—"}</span></td>
                  <td>{r.productName || "—"}</td>
                  <td><span className="badge" style={{ background: zc.bg, color: zc.text }}>{zoneName}</span></td>
                  <td className="dim-cell">{r.duration || "—"}</td>
                  <td className={r.remainingStock <= 5 ? "stock-low" : ""}>{r.remainingStock ?? "—"}</td>
                  <td className="dim-cell">{r.confidence || "—"}</td>
                  <td>
                    <span className={`badge ${r.status === "완료" ? "b-teal" : "b-amber"}`}>
                      {r.status || "—"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="log-card">
        <div className="log-hd">📋 ROS2 토픽 로그</div>
        <div className="log-box">
          {topicLog.length === 0 && (
            <div className="log-line log-dim">로그가 없습니다</div>
          )}
          {topicLog.map((l, i) => (
            <div className={`log-line ${l.cls || "log-green"}`} key={i}>
              [{l.timestamp}] {l.text}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}