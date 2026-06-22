import { useRef, useEffect, useState } from "react";
import useStore from "../store";
import "./CameraFeed.css";

export default function CameraFeed() {
  const frame = useStore((s) => s.cameraFrame);

  const canvasRef = useRef(document.createElement("canvas"));
  const recorderRef = useRef(null);
  const blobsRef = useRef([]);
  const [recording, setRecording] = useState(false);

  // 들어오는 프레임을 (숨은) 캔버스에 그려서 녹화 소스로 사용
  useEffect(() => {
    if (!frame) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      if (canvas.width !== img.width || canvas.height !== img.height) {
        canvas.width = img.width;
        canvas.height = img.height;
      }
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/${frame.format};base64,${frame.data}`;
  }, [frame]);

  const toggleRecord = () => {
    if (!recording) {
      // 시작
      const canvas = canvasRef.current;
      if (!canvas.width) return; // 아직 프레임 없음
      blobsRef.current = [];
      const stream = canvas.captureStream(30);
      const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : "video/webm";
      const rec = new MediaRecorder(stream, { mimeType: mime });
      rec.ondataavailable = (e) => { if (e.data.size > 0) blobsRef.current.push(e.data); };
      rec.onstop = () => {
        const blob = new Blob(blobsRef.current, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `camera_${Date.now()}.webm`;
        a.click();
        URL.revokeObjectURL(url);
      };
      rec.start();
      recorderRef.current = rec;
      setRecording(true);
    } else {
      // 중지 → onstop에서 저장
      recorderRef.current?.stop();
      setRecording(false);
    }
  };

  return (
    <div className="cam-card">
      <div className="card-hd">📷 카메라 피드 (D435i · YOLO)</div>
      <div className="cam-box">
        {frame ? (
          <img
            src={`data:image/${frame.format};base64,${frame.data}`}
            alt="camera"
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#94A3B8" }}>
            📡 카메라 스트림 대기 중...
          </div>
        )}
        <div className="cam-hud"><span className="rec" />{frame ? "LIVE" : "대기"}</div>
      </div>
      <div className="cam-info">
        <span className="cam-tag">⚙ YOLO v8</span>
        <span className="cam-tag">📐 D435i</span>
        <button
          onClick={toggleRecord}
          disabled={!frame}
          style={{
            marginLeft: "auto", padding: "4px 12px", borderRadius: "4px", cursor: frame ? "pointer" : "not-allowed",
            border: "1px solid", borderColor: recording ? "#EF4444" : "#00C2FF",
            background: recording ? "#EF4444" : "transparent",
            color: recording ? "#fff" : "#00C2FF", fontSize: "12px",
          }}
        >
          {recording ? "⏹ 녹화 중지" : "⏺ 녹화 시작"}
        </button>
      </div>
    </div>
  );
}