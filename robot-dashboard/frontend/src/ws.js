import useStore from "./store";

let socket = null;

export function connectWS() {
  if (socket) return socket;
  socket = new WebSocket(`ws://${location.hostname}:8000/ws`);
  
  socket.onopen = () => console.log("[WS] 연결됨");
  socket.onerror = (e) => console.error("[WS] 에러:", e);
  socket.onclose = () => {
    console.log("[WS] 연결 종료, 3초 후 재연결");
    socket = null;
    setTimeout(connectWS, 3000);
  };

  socket.onmessage = (e) => {
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch {
      return;
    }
    const { type, payload } = msg;
    const s = useStore.getState();

    switch (type) {
      case "robot_status":
        s.setRobotStatus({
          agv: { ...s.robotStatus.agv, ...(payload.agv || {}) },
          cobot: { ...s.robotStatus.cobot, ...(payload.cobot || {}) },
        });
        break;
      case "topic_log":
        s.addTopicLog({ timestamp: new Date().toISOString(), ...payload });
        break;
      case "stock_update":
        s.updateProduct(payload.id, { stock: payload.stock });
        break;
      case "queue_remove":
        s.removeFromQueue(payload.id);
        break;
      case "history_add":
        s.addHistoryRecord(payload);
        break;
      case "mission_state":
        s.setMissionState(payload.state);
        break;
      case "camera_frame":
        s.setCameraFrame(payload);
        break;
      case "intervention":
        s.setInterventionAlert(payload);
        break;
      default:
        console.warn("[WS] 알 수 없는 타입:", type);
    }
  };

  return socket;
}