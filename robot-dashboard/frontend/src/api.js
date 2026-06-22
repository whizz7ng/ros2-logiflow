import useStore from "./store";

const BASE = `http://${location.hostname}:8000/api`;
const delay = (ms = 250) => new Promise((res) => setTimeout(res, ms));

// ── case 변환 헬퍼 ──────────────────────────────────
function toCamel(obj) {
  if (Array.isArray(obj)) return obj.map(toCamel);
  if (obj !== null && typeof obj === "object") {
    return Object.fromEntries(
      Object.entries(obj).map(([k, v]) => [
        k.replace(/_([a-z])/g, (_, c) => c.toUpperCase()),
        toCamel(v),
      ])
    );
  }
  return obj;
}

function toSnake(obj) {
  if (Array.isArray(obj)) return obj.map(toSnake);
  if (obj !== null && typeof obj === "object") {
    return Object.fromEntries(
      Object.entries(obj).map(([k, v]) => [
        k.replace(/[A-Z]/g, (c) => "_" + c.toLowerCase()),
        toSnake(v),
      ])
    );
  }
  return obj;
}

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    console.error(`API 오류: ${options.method || "GET"} ${path} → ${res.status}`);
    throw new Error(`API request failed: ${path} (${res.status})`);
  }
  if (res.status === 204) return null;
  return toCamel(await res.json());
}

// ── 초기 로딩 (App 마운트 시 1회) ───────────────────────
export async function loadAll() {
  const [products, zones, queue, state, history] = await Promise.all([
    request("/products"),
    request("/zones"),
    request("/missions/queue"),
    request("/missions/state"),
    request("/history"),
  ]);
  useStore.getState().setProducts(products);
  useStore.getState().setZones(zones);
  useStore.getState().setMissionQueue(queue);
  useStore.getState().setMissionState(state.state);
  useStore.getState().setHistory(history);
}

// ── products ──────────────────────────────────────
export async function getProducts() {
  const products = await request("/products");
  useStore.getState().setProducts(products);
  return products;
}

export async function createProduct(data) {
  const product = await request("/products", {
    method: "POST",
    body: JSON.stringify(toSnake(data)),
  });
  useStore.getState().addProduct(product);
  return product;
}

export async function updateProduct(id, patch) {
  const product = await request(`/products/${id}`, {
    method: "PUT",
    body: JSON.stringify(toSnake(patch)),
  });
  useStore.getState().updateProduct(id, product);
  return product;
}

export async function deleteProduct(id) {
  await request(`/products/${id}`, { method: "DELETE" });
  useStore.getState().deleteProduct(id);
}

export async function adjustStock(id, delta) {
  const product = await request(`/products/${id}/stock`, {
    method: "PATCH",
    body: JSON.stringify({ delta }),
  });
  useStore.getState().updateProduct(id, product);
}

// ── zones ─────────────────────────────────────────
export async function getZones() {
  const zones = await request("/zones");
  useStore.getState().setZones(zones);
  return zones;
}

export async function createZone(data) {
  const zone = await request("/zones", {
    method: "POST",
    body: JSON.stringify(toSnake(data)),
  });
  useStore.getState().addZone(zone);
  return zone;
}

export async function updateZone(id, patch) {
  const zone = await request(`/zones/${id}`, {
    method: "PUT",
    body: JSON.stringify(toSnake(patch)),
  });
  useStore.getState().updateZone(id, zone);
  return zone;
}

export async function deleteZone(id) {
  await request(`/zones/${id}`, { method: "DELETE" });
  useStore.getState().deleteZone(id);
}

// ── mission queue / 상태머신 ─────────────────────────
export async function getMissionQueue() {
  const queue = await request("/missions/queue");
  useStore.getState().setMissionQueue(queue);
  return queue;
}

export async function addMissionItem(item) {
  const queueItem = await request("/missions/queue", {
    method: "POST",
    body: JSON.stringify(toSnake(item)),
  });
  useStore.getState().addToQueue(queueItem);
  return queueItem;
}

export async function removeMissionItem(id) {
  await request(`/missions/queue/${id}`, { method: "DELETE" });
  useStore.getState().removeFromQueue(id);
}

export async function clearMissionQueue() {
  await request("/missions/queue", { method: "DELETE" });
  useStore.getState().clearQueue();
}

async function sendMissionCommand(command) {
  const result = await request("/missions/command", {
    method: "POST",
    body: JSON.stringify({ action: command }),   // command → action
  });
  useStore.getState().setMissionState(result.state);
  if (command === "cancel") useStore.getState().clearQueue();
}

export async function startMission() {
  await sendMissionCommand("start");
}

export async function pauseMission() {
  await sendMissionCommand("pause");
}

export async function resumeMission() {
  await sendMissionCommand("resume");
}

export async function cancelMission() {
  await sendMissionCommand("cancel");
}

// ── history ───────────────────────────────────────
export async function getHistory() {
  const history = await request("/history");
  useStore.getState().setHistory(history);
  return history;
}

export async function addHistoryRecord(record) {
  const entry = await request("/history", {
    method: "POST",
    body: JSON.stringify(toSnake(record)),
  });
  useStore.getState().addHistoryRecord(entry);
  return entry;
}

// ── robot status / topic log (Phase 5에서 WebSocket으로 대체) ──
export async function getRobotStatus() {
  await delay(100);
  return useStore.getState().robotStatus;
}

export async function getTopicLog() {
  await delay(100);
  return useStore.getState().topicLog;
}

export async function emergencyStop() {
  return request("/robot/estop", { method: "POST" });
}

// ── 라벨(.yaml/.json) 업로드 (Phase 5에서 실제 파싱) ─────
export async function uploadLabelModel(file) {
  await delay(500);
  const model = { fileName: file.name, classes: [] };
  useStore.getState().setLabelModel(model);
  return model;
}