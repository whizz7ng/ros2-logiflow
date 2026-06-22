import { create } from "zustand";

// ── 초기 시드 데이터 ────────────────────────────────
const initialZones = [
  { id: 1, name: "구역 A", desc: "기초 케어", color: "#FDE68A", qr: "QR-A01", status: "운영 중" },
  { id: 2, name: "구역 B", desc: "메이크업", color: "#BFDBFE", qr: "QR-B01", status: "운영 중" },
  { id: 3, name: "구역 C", desc: "바디 케어", color: "#DDD6FE", qr: "QR-C01", status: "운영 중" },
];

const initialProducts = [
  { id: 1, color: "빨강", shape: "세모", name: "스킨 (150ml)", yoloLabel: "red_triangle", zoneId: 1, stock: 12, note: "", status: "활성" },
  { id: 2, color: "파랑", shape: "네모", name: "로션 (120ml)", yoloLabel: "blue_square", zoneId: 2, stock: 9, note: "", status: "활성" },
  { id: 3, color: "노랑", shape: "오각형", name: "립글로즈", yoloLabel: "yellow_pentagon", zoneId: 3, stock: 4, note: "재고 부족 주의", status: "활성" },
  { id: 4, color: "초록", shape: "동그라미", name: "핸드크림", yoloLabel: "green_circle", zoneId: 3, stock: 15, note: "", status: "활성" },
  { id: 5, color: "주황", shape: "십자가", name: "선크림 (50ml)", yoloLabel: "orange_cross", zoneId: 1, stock: 7, note: "", status: "활성" },
];

const initialMissionQueue = [
  { id: 9001, productId: 3, name: "립글로즈", yoloLabel: "yellow_pentagon", zoneId: 3, zoneName: "구역 C" },
  { id: 9002, productId: 4, name: "핸드크림", yoloLabel: "green_circle", zoneId: 3, zoneName: "구역 C" },
];

const initialRobotStatus = {
  agv: { state: "idle", battery: 87, position: "구역 A" },
  cobot: { state: "idle" },
};

// ── 스토어 ────────────────────────────────────────
const useStore = create((set, get) => ({
  // state
  products: initialProducts,
  zones: initialZones,
  missionQueue: initialMissionQueue,
  missionState: "idle", // idle | running | paused | cancelled
  robotStatus: initialRobotStatus,
  history: [],
  topicLog: [],
  labelModel: null, // { fileName, classes: [{ label, mapped }] }
  cameraFrame: null,        // { format, data(base64) }
  interventionAlert: null,  // { source, message }

  // setter 추가
  setCameraFrame: (cameraFrame) => set({ cameraFrame }),
  setInterventionAlert: (interventionAlert) => set({ interventionAlert }),
  clearInterventionAlert: () => set({ interventionAlert: null }),

  // ── products ──
  setProducts: (products) => set({ products }),
  addProduct: (product) => set((s) => ({ products: [...s.products, product] })),
  updateProduct: (id, patch) =>
    set((s) => ({ products: s.products.map((p) => (p.id === id ? { ...p, ...patch } : p)) })),
  deleteProduct: (id) => set((s) => ({ products: s.products.filter((p) => p.id !== id) })),
  adjustStock: (id, delta) =>
    set((s) => ({
      products: s.products.map((p) =>
        p.id === id ? { ...p, stock: Math.max(0, p.stock + delta) } : p
      ),
    })),

  // ── zones ──
  setZones: (zones) => set({ zones }),
  addZone: (zone) => set((s) => ({ zones: [...s.zones, zone] })),
  updateZone: (id, patch) =>
    set((s) => ({ zones: s.zones.map((z) => (z.id === id ? { ...z, ...patch } : z)) })),
  deleteZone: (id) => set((s) => ({ zones: s.zones.filter((z) => z.id !== id) })),

  // ── mission queue ──
  addToQueue: (item) => set((s) => ({ missionQueue: [...s.missionQueue, item] })),
  removeFromQueue: (id) => set((s) => ({ missionQueue: s.missionQueue.filter((q) => q.id !== id) })),
  clearQueue: () => set({ missionQueue: [] }),
  setMissionState: (missionState) => set({ missionState }),
  setMissionQueue: (missionQueue) => set({ missionQueue }),

  // ── robot status ──
  setRobotStatus: (patch) => set((s) => ({
  robotStatus: {
    agv:   { ...s.robotStatus.agv,   ...(patch.agv   || {}) },
    cobot: { ...s.robotStatus.cobot, ...(patch.cobot || {}) },
    },
  })),

  // ── history ──
  addHistoryRecord: (record) => set((s) => ({ history: [record, ...s.history] })),
  setHistory: (history) => set({ history }),

  // ── topic log (최근 100개만 유지) ──
  addTopicLog: (log) => set((s) => ({ topicLog: [log, ...s.topicLog].slice(0, 100) })),

  // ── label model ──
  setLabelModel: (labelModel) => set({ labelModel }),
}));

export default useStore;
