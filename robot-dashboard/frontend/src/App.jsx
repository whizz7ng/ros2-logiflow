import { useState } from "react";
import "./index.css";
import Monitor from "./pages/Monitor";
import Products from "./pages/Products";
import Zones from "./pages/Zones";
import History from "./pages/History";
import { useEffect } from "react";
import { loadAll } from "./api";
import { connectWS } from "./ws";

const TABS = [
  { id: "monitor", label: "모니터링 & 제어", icon: "📊" },
  { id: "products", label: "상품 관리", icon: "📦" },
  { id: "zones", label: "배송 구역", icon: "📍" },
  { id: "history", label: "배송 기록", icon: "🕐" },
];

const PAGES = {
  monitor: Monitor,
  products: Products,
  zones: Zones,
  history: History,
};

export default function App() {
  const [tab, setTab] = useState("monitor");
  const Page = PAGES[tab];

  useEffect(() => {
    loadAll();
    connectWS(); 
  }, []);
  
  return (
    <div className="app">
      <header className="topbar">
        <div className="logo-wrap">
          <div className="logo-icon">🤖</div>
          <div>
            <div className="logo-text">분류로봇 대시보드</div>
            <div className="logo-sub">AI Scanning Package Classifier</div>
          </div>
        </div>
        <div className="topbar-right">
          <span className="pill"><span className="conn-dot dot-on" />Orin Nano</span>
          <span className="pill"><span className="conn-dot dot-on" />AGV</span>
          <span className="pill"><span className="conn-dot dot-warn" />ROS2 DDS</span>
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            <span>{t.icon}</span>{t.label}
          </button>
        ))}
      </nav>

      <main className="content">
        <Page />
      </main>
    </div>
  );
}