import useStore from "../store";
import "./Monitor.css";
import CameraFeed from "../components/CameraFeed";
import StatusMap from "../components/StatusMap";
import MissionPanel from "../components/MissionPanel";
import InterventionPanel from "../components/InterventionPanel";

export default function Monitor() {
  const history = useStore((s) => s.history);
  const queue = useStore((s) => s.missionQueue);
  const missionState = useStore((s) => s.missionState);

  const todayCount = history.filter((h) => {
    if (!h.timestamp) return false;
    return new Date(h.timestamp).toDateString() === new Date().toDateString();
  }).length;

  const activeItem = missionState === "running" && queue.length > 0 ? queue[0] : null;

  const avgDuration = (() => {
    const completed = history.filter((h) => h.duration && h.status === "완료");
    if (completed.length === 0) return "—";
    const nums = completed.map((h) => parseInt(h.duration) || 0).filter(Boolean);
    return nums.length > 0 ? Math.round(nums.reduce((a, b) => a + b, 0) / nums.length) : "—";
  })();

  const METRICS = [
    { label: "오늘 총 배송", value: String(todayCount), unit: "건", sub: "실시간 집계", icon: "📦" },
    { label: "처리 중", value: activeItem ? "1" : "0", unit: "건", sub: activeItem ? `${activeItem.yoloLabel} → ${activeItem.zoneName}` : "없음", icon: "⏳" },
    { label: "대기 미션", value: String(queue.length), unit: "건", sub: "큐 잔여", icon: "📋" },
    { label: "평균 처리", value: String(avgDuration), unit: avgDuration === "—" ? "" : "초", sub: avgDuration === "—" ? "데이터 없음" : "정상 범위", icon: "⏱️" },
  ];

  return (
    <div className="monitor">
      <div className="metric-grid">
        {METRICS.map((m, i) => (
          <div className="metric" key={i}>
            <div className="metric-top">
              <span className="metric-label">{m.label}</span>
              <span className="metric-icon">{m.icon}</span>
            </div>
            <div className="metric-val">
              {m.value}<span className="metric-unit">{m.unit}</span>
            </div>
            <div className="metric-sub">{m.sub}</div>
          </div>
        ))}
      </div>

      <div className="monitor-body">
        <div className="left-col">
          <CameraFeed />
          <StatusMap />
        </div>
        <div className="right-col">
          <MissionPanel />
          <InterventionPanel />
        </div>
      </div>
    </div>
  );
}