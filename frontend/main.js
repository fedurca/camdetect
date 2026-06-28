import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { colorFor, cssColor, hexColor, CLASS_COLORS } from "/colors.js";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let appConfig = null;
let labels = {};                       // class -> Czech label
let center = { x: 0, y: 0 };
const objectMeshes = new Map();        // id -> { group, box, label, class }
let latestObjects = [];
let latestAudio = {};                  // cam -> audio result
let selectedAudioCam = null;
const prevPos = new Map();             // id -> {x,y} for heading arrows

const BEHAVIOR_CS = {
  standing: "stoji", walking: "jde", running: "bezi", loitering: "postava",
  moving: "jede", stopped: "stoji",
};
const EVENT_CS = {
  engine: "motor", drone: "dron", bark: "stekot", speech: "rec", loud: "hluk",
};
const EVENT_COLORS = {
  engine: "#38bdf8", drone: "#fafafa", bark: "#fb923c", speech: "#a78bfa",
  loud: "#facc15",
};

function labelFor(cls) { return labels[cls] || cls; }

function objectLabel(o) {
  let s = `${labelFor(o.class)} #${o.id} ${(o.prob ?? 0).toFixed(2)}`;
  if (o.behavior) s += ` · ${BEHAVIOR_CS[o.behavior] || o.behavior}`;
  if (o.age) s += ` · ${o.age}`;
  if (o.engine_type) s += ` · ${o.engine_type}`;
  if (o.plate) s += ` · ${o.plate}`;
  if (o.make) s += ` · ${o.make} ${o.model || ""}`.trimEnd();
  if (o.speech) s += ` · "${o.speech}"`;
  return s;
}

// Footprint (meters) per class for 3D boxes.
const FOOTPRINT = {
  person: [0.6, 0.6], bicycle: [1.6, 0.6], car: [4.4, 1.9], motorcycle: [2.0, 0.8],
  bus: [10, 2.5], truck: [7, 2.5], "trash bin": [0.6, 0.6], scooter: [1.2, 0.5],
  skates: [0.6, 0.4], drone: [0.5, 0.5], dog: [0.8, 0.4],
};
function footprint(name) { return FOOTPRINT[name] || [0.7, 0.7]; }

function worldToScene(X, Y, up = 0) {
  return new THREE.Vector3(X - center.x, up, -(Y - center.y));
}

// ---------------------------------------------------------------------------
// Three.js 3D scene
// ---------------------------------------------------------------------------
const sceneEl = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
sceneEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f17);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
camera.position.set(0, 18, 22);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(10, 30, 10);
scene.add(dir);

function resize() {
  const w = sceneEl.clientWidth || 1;
  const h = sceneEl.clientHeight || 1;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  resizeTopdown();
}
window.addEventListener("resize", resize);

function buildGround(sizeMeters) {
  scene.add(new THREE.GridHelper(sizeMeters, Math.max(4, Math.round(sizeMeters)),
    0x334155, 0x1e293b));
  const geo = new THREE.PlaneGeometry(sizeMeters, sizeMeters);
  geo.rotateX(-Math.PI / 2);
  const mat = new THREE.MeshStandardMaterial({ color: 0x0f172a, transparent: true, opacity: 0.6 });
  const plane = new THREE.Mesh(geo, mat);
  plane.position.y = -0.01;
  scene.add(plane);
  new THREE.TextureLoader().load("/assets/satellite.png", (tex) => {
    tex.colorSpace = THREE.SRGBColorSpace;
    mat.map = tex; mat.color.set(0xffffff); mat.opacity = 1; mat.needsUpdate = true;
  }, undefined, () => {});
}

function labelSprite(text, rgb) {
  const canvas = document.createElement("canvas");
  canvas.width = 320; canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(10,14,22,0.85)"; ctx.fillRect(0, 0, 320, 64);
  ctx.strokeStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  ctx.lineWidth = 4; ctx.strokeRect(2, 2, 316, 60);
  ctx.fillStyle = "#fff"; ctx.font = "bold 24px sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, 160, 34);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
  spr.scale.set(3.0, 0.6, 1);
  return spr;
}

function buildCameras(cameras) {
  const cov = (appConfig && appConfig.coverage) || {};
  for (const cam of cameras) {
    const [X, Y] = cam.world_xy;
    const pos = worldToScene(X, Y, cam.height_m);
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.4, 0.9, 4),
      new THREE.MeshStandardMaterial({ color: 0x38bdf8 }));
    cone.position.copy(pos); cone.rotation.x = Math.PI;
    scene.add(cone);
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
      [pos, worldToScene(X, Y, 0)]),
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.5 })));

    // coverage wedge on the ground, oriented along the camera azimuth
    const cv = cov[cam.id];
    if (cv) {
      const az = (cv.azimuth_deg * Math.PI) / 180;
      const half = (cv.fov_deg * Math.PI) / 180 / 2;
      const R = cv.range_m;
      const shape = new THREE.Shape();
      shape.moveTo(0, 0);
      const steps = 24;
      for (let i = 0; i <= steps; i++) {
        const a = az - half + (2 * half * i) / steps;
        shape.lineTo(R * Math.cos(a), R * Math.sin(a));
      }
      const geo = new THREE.ShapeGeometry(shape);
      geo.rotateX(-Math.PI / 2);  // lay flat on ground (XZ), shape XY -> XZ
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color: 0x38bdf8, transparent: true, opacity: 0.08, side: THREE.DoubleSide,
        depthWrite: false,
      }));
      const g = worldToScene(X, Y, 0.02);
      mesh.position.set(g.x, g.y, g.z);
      scene.add(mesh);
    } else {
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
        [pos, worldToScene(center.x, center.y, 0.2)]),
        new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.25 })));
    }

    const tag = labelSprite(cam.id, [56, 189, 248]);
    tag.position.copy(pos.clone().add(new THREE.Vector3(0, 0.9, 0)));
    tag.scale.set(1.6, 0.35, 1);
    scene.add(tag);
  }
}

function makeObject(o) {
  const [fw, fd] = footprint(o.class);
  const h = o.height || 1.7;
  const group = new THREE.Group();
  const box = new THREE.Mesh(new THREE.BoxGeometry(fw, h, fd),
    new THREE.MeshStandardMaterial({ color: hexColor(o.class), transparent: true, opacity: 0.55 }));
  box.position.y = h / 2; group.add(box);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry),
    new THREE.LineBasicMaterial({ color: hexColor(o.class) }));
  edges.position.y = h / 2; group.add(edges);
  const label = labelSprite(objectLabel(o), colorFor(o.class));
  label.position.y = h + 0.6; group.add(label);
  scene.add(group);

  // drones get a trajectory polyline in world coordinates (absolute)
  let trail = null;
  if (o.class === "drone") {
    trail = new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: hexColor(o.class) }));
    scene.add(trail);
  }
  return { group, box, label, trail, class: o.class, text: objectLabel(o) };
}

function updateObjects(objects) {
  const seen = new Set();
  for (const o of objects) {
    seen.add(o.id);
    let e = objectMeshes.get(o.id);
    const text = objectLabel(o);
    if (!e || e.class !== o.class) {
      if (e) scene.remove(e.group);
      e = makeObject(o);
      objectMeshes.set(o.id, e);
    } else if (e.text !== text) {
      e.group.remove(e.label);
      e.label = labelSprite(text, colorFor(o.class));
      e.label.position.y = (o.height || 1.7) + 0.6;
      e.group.add(e.label);
      e.text = text;
    }
    const p = worldToScene(o.x, o.y, 0);
    e.group.position.set(p.x, 0, p.z);
    // update drone trajectory (drawn at a small altitude)
    if (e.trail && o.trail && o.trail.length > 1) {
      e.trail.geometry.setFromPoints(
        o.trail.map(([x, y]) => worldToScene(x, y, 1.2)));
    }
  }
  for (const [id, e] of objectMeshes) {
    if (!seen.has(id)) {
      scene.remove(e.group);
      if (e.trail) scene.remove(e.trail);
      objectMeshes.delete(id);
    }
  }
}

// ---------------------------------------------------------------------------
// Top-down 2D view
// ---------------------------------------------------------------------------
const td = document.getElementById("topdown");
const tdx = td.getContext("2d");
let bounds = { minX: -10, maxX: 10, minY: -10, maxY: 10 };

function resizeTopdown() {
  td.width = td.clientWidth || 300;
  td.height = td.clientHeight || 200;
}

function computeBounds(cameras) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const c of cameras) {
    minX = Math.min(minX, c.world_xy[0]); maxX = Math.max(maxX, c.world_xy[0]);
    minY = Math.min(minY, c.world_xy[1]); maxY = Math.max(maxY, c.world_xy[1]);
  }
  const px = (maxX - minX) * 0.35 + 3, py = (maxY - minY) * 0.35 + 3;
  bounds = { minX: minX - px, maxX: maxX + px, minY: minY - py, maxY: maxY + py };
}

function tdPt(X, Y) {
  const { minX, maxX, minY, maxY } = bounds;
  const sx = td.width / (maxX - minX), sy = td.height / (maxY - minY);
  const s = Math.min(sx, sy);
  const ox = (td.width - (maxX - minX) * s) / 2;
  const oy = (td.height - (maxY - minY) * s) / 2;
  return [ox + (X - minX) * s, td.height - oy - (Y - minY) * s];
}

function drawTopdown() {
  if (!appConfig) return;
  tdx.clearRect(0, 0, td.width, td.height);
  tdx.fillStyle = "#0a0e16"; tdx.fillRect(0, 0, td.width, td.height);
  // grid lines every 2 m
  tdx.strokeStyle = "#1e293b"; tdx.lineWidth = 1;
  for (let g = Math.ceil(bounds.minX); g <= bounds.maxX; g += 2) {
    const [x] = tdPt(g, 0); tdx.beginPath(); tdx.moveTo(x, 0); tdx.lineTo(x, td.height); tdx.stroke();
  }
  for (let g = Math.ceil(bounds.minY); g <= bounds.maxY; g += 2) {
    const [, y] = tdPt(0, g); tdx.beginPath(); tdx.moveTo(0, y); tdx.lineTo(td.width, y); tdx.stroke();
  }
  // camera coverage wedges (from the UniFi coverage map)
  const cov = appConfig.coverage || {};
  for (const c of appConfig.cameras) {
    const cv = cov[c.id];
    if (!cv) continue;
    const az = (cv.azimuth_deg * Math.PI) / 180;
    const half = (cv.fov_deg * Math.PI) / 180 / 2;
    const R = cv.range_m;
    const [px, py] = tdPt(cv.x, cv.y);
    tdx.beginPath();
    tdx.moveTo(px, py);
    const steps = 16;
    for (let i = 0; i <= steps; i++) {
      const a = az - half + (2 * half * i) / steps;
      const [wx, wy] = [cv.x + R * Math.cos(a), cv.y + R * Math.sin(a)];
      const [ex, ey] = tdPt(wx, wy);
      tdx.lineTo(ex, ey);
    }
    tdx.closePath();
    tdx.fillStyle = "rgba(56,189,248,0.10)";
    tdx.strokeStyle = "rgba(56,189,248,0.35)";
    tdx.fill(); tdx.stroke();
  }
  // camera markers
  for (const c of appConfig.cameras) {
    const [px, py] = tdPt(c.world_xy[0], c.world_xy[1]);
    tdx.fillStyle = "#38bdf8"; tdx.beginPath();
    tdx.moveTo(px, py - 6); tdx.lineTo(px - 5, py + 5); tdx.lineTo(px + 5, py + 5);
    tdx.closePath(); tdx.fill();
    tdx.fillStyle = "#94a3b8"; tdx.font = "10px sans-serif"; tdx.fillText(c.id, px + 7, py + 4);
  }
  // drone trajectories
  for (const o of latestObjects) {
    if (o.class !== "drone" || !o.trail || o.trail.length < 2) continue;
    tdx.strokeStyle = cssColor(o.class, 0.7); tdx.lineWidth = 1.5;
    tdx.beginPath();
    o.trail.forEach(([x, y], i) => {
      const [tx, ty] = tdPt(x, y);
      if (i === 0) tdx.moveTo(tx, ty); else tdx.lineTo(tx, ty);
    });
    tdx.stroke();
  }
  // objects
  for (const o of latestObjects) {
    const [px, py] = tdPt(o.x, o.y);
    // heading arrow from previous position
    const prev = prevPos.get(o.id);
    if (prev) {
      const [ppx, ppy] = tdPt(prev.x, prev.y);
      const dx = px - ppx, dy = py - ppy, len = Math.hypot(dx, dy);
      if (len > 1.5) {
        tdx.strokeStyle = cssColor(o.class); tdx.lineWidth = 2;
        tdx.beginPath(); tdx.moveTo(px, py);
        tdx.lineTo(px + dx / len * 12, py + dy / len * 12); tdx.stroke();
      }
    }
    tdx.fillStyle = cssColor(o.class);
    tdx.beginPath(); tdx.arc(px, py, 5, 0, Math.PI * 2); tdx.fill();
    tdx.fillStyle = "#e2e8f0"; tdx.font = "10px sans-serif";
    tdx.fillText(`${labelFor(o.class)} #${o.id}`, px + 7, py + 3);
  }
}

// ---------------------------------------------------------------------------
// Audio panel
// ---------------------------------------------------------------------------
const spectro = document.getElementById("spectrogram");
const freqC = document.getElementById("freq");
const freqX = freqC.getContext("2d");
const eventsEl = document.getElementById("audio-events");

function buildAudioTabs(cameras) {
  const el = document.getElementById("audio-cams");
  el.innerHTML = "";
  cameras.forEach((c, i) => {
    const b = document.createElement("span");
    b.className = "tab" + (i === 0 ? " active" : "");
    b.textContent = c.id;
    b.onclick = () => selectAudioCam(c.id);
    el.appendChild(b);
  });
  if (cameras.length) selectAudioCam(cameras[0].id);
}

function selectAudioCam(cam) {
  selectedAudioCam = cam;
  spectro.src = `/audio/${cam}/spectrogram`;
  document.querySelectorAll("#audio-cams .tab").forEach((t) =>
    t.classList.toggle("active", t.textContent === cam));
}

const CAM_COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#a78bfa", "#f472b6"];

// Combined frequency analysis across all 3 cameras in a single plot:
// grouped bars per band (low/mid/high), one colored bar per camera.
function drawFreq() {
  freqX.clearRect(0, 0, freqC.width, freqC.height);
  const cams = (appConfig?.cameras || []).map((c) => c.id);
  if (!cams.length) return;
  const bands = ["low", "mid", "high"];
  const groupW = freqC.width / 3;
  const barW = Math.max(5, (groupW - 24) / cams.length);
  bands.forEach((band, bi) => {
    cams.forEach((cam, ci) => {
      const a = latestAudio[cam];
      const v = a ? Math.min(1, a.bands[band]) : 0;
      const h = v * (freqC.height - 34);
      const x = bi * groupW + 12 + ci * barW;
      freqX.fillStyle = CAM_COLORS[ci % CAM_COLORS.length];
      freqX.fillRect(x, freqC.height - 20 - h, barW - 2, h);
    });
    freqX.fillStyle = "#94a3b8"; freqX.font = "10px sans-serif";
    freqX.fillText(band, bi * groupW + groupW / 2 - 8, freqC.height - 5);
  });
  // header: combined level + per-camera legend
  const levels = cams.map((c) => (latestAudio[c] ? latestAudio[c].level : 0));
  const avg = levels.reduce((s, x) => s + x, 0) / cams.length;
  freqX.fillStyle = "#e2e8f0"; freqX.font = "11px sans-serif";
  freqX.fillText(`vsechny kamery · level avg ${(avg * 100).toFixed(0)}%`, 8, 13);
  cams.forEach((cam, ci) => {
    freqX.fillStyle = CAM_COLORS[ci % CAM_COLORS.length];
    freqX.fillText(cam, freqC.width - 36 * cams.length + ci * 36, 13);
  });
  const eng = cams.map((c) => latestAudio[c] && latestAudio[c].engine_type).find(Boolean);
  if (eng) { freqX.fillStyle = "#38bdf8"; freqX.fillText(`motor ${eng}`, 8, 27); }
}

function drawAudioEvents() {
  const a = latestAudio[selectedAudioCam];
  eventsEl.innerHTML = "";
  if (!a || !a.events || !a.events.length) {
    const li = document.createElement("li");
    li.style.color = "#64748b"; li.textContent = "(zadne udalosti)";
    eventsEl.appendChild(li); return;
  }
  for (const ev of a.events) {
    const li = document.createElement("li");
    const sw = document.createElement("span");
    sw.className = "swatch"; sw.style.background = EVENT_COLORS[ev.type] || "#94a3b8";
    li.appendChild(sw);
    li.appendChild(document.createTextNode(
      `${EVENT_CS[ev.type] || ev.type} ${(ev.conf * 100).toFixed(0)}%`));
    eventsEl.appendChild(li);
  }
}

let latestTranscripts = {};
function drawTranscript() {
  const el = document.getElementById("transcript");
  const segs = latestTranscripts[selectedAudioCam] || [];
  if (!segs.length) { el.innerHTML = '<div class="muted">(zadny prepis)</div>'; return; }
  el.innerHTML = segs.map((s) => {
    const who = s.person ? `${s.person} (${s.speaker || "S1"})` : (s.speaker || "S1");
    return `<div class="seg"><span class="spk">${who}</span>${s.text}</div>`;
  }).join("");
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
let activeTab = "live";
let debugTimer = null;

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll(".tab-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === tab));
  for (const v of ["live", "history", "report", "benchmark", "debug"]) {
    document.getElementById(`view-${v}`).classList.toggle("hidden", v !== tab);
  }
  if (tab === "live") setTimeout(resize, 50);
  if (tab === "history") { loadHistory(); loadRecordings(); }
  if (tab === "report") loadReport();
  if (debugTimer) { clearInterval(debugTimer); debugTimer = null; }
  if (tab === "debug") {
    pollLogs(); pollEvents();
    debugTimer = setInterval(() => { pollLogs(); pollEvents(); }, 1000);
  }
}

// ---------------------------------------------------------------------------
// Daily report
// ---------------------------------------------------------------------------
async function loadReport() {
  const dateEl = document.getElementById("report-date");
  if (!dateEl.value) dateEl.value = new Date().toISOString().slice(0, 10);
  const date = dateEl.value;
  try {
    const r = await fetch(`/api/report?date=${date}`).then((x) => x.json());
    document.getElementById("report-text").textContent = r.text || "(prazdne)";
    const grid = document.getElementById("report-images");
    grid.innerHTML = (r.images || []).map((name) =>
      `<figure style="margin:0"><img loading="lazy" src="/report-image/${date}/${name}" />` +
      `<figcaption class="cap">${name}</figcaption></figure>`).join("") ||
      '<div class="muted">Zadne snimky pro tento den.</div>';
  } catch (e) {
    document.getElementById("report-text").textContent = "Chyba nacitani reportu.";
  }
}

function wireTabs() {
  document.querySelectorAll(".tab-btn").forEach((b) =>
    b.addEventListener("click", () => switchTab(b.dataset.tab)));
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------
async function loadHistory() {
  try {
    const [objs, evts, stats] = await Promise.all([
      fetch("/api/history/objects?limit=200").then((r) => r.json()),
      fetch("/api/history/events?limit=200").then((r) => r.json()),
      fetch("/api/history/stats").then((r) => r.json()),
    ]);
    renderHistory(objs.objects || [], evts.events || [], stats);
  } catch (e) { /* ignore */ }
}

function attrPills(attrs) {
  const keep = ["behavior", "age", "engine_type", "plate", "make", "model",
    "vehicle_age", "drivetrain", "speaker", "speech"];
  return keep.filter((k) => attrs[k]).map((k) =>
    `<span class="pill">${attrs[k]}</span>`).join("");
}

function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleString("cs-CZ");
}

function renderHistory(objects, events, stats) {
  document.getElementById("hist-stats").textContent =
    `${stats.objects || 0} objektu, ${stats.events || 0} udalosti`;
  const ob = document.querySelector("#hist-objects tbody");
  ob.innerHTML = objects.map((o) =>
    `<tr><td>${o.id}</td><td>${labelFor(o.class)}</td><td>${attrPills(o.attrs || {})}</td>` +
    `<td>${o.observations}</td><td>${fmtTime(o.last_seen)}</td></tr>`).join("");
  const eb = document.querySelector("#hist-events tbody");
  eb.innerHTML = events.map((e) =>
    `<tr><td>${fmtTime(e.ts)}</td><td>${e.kind}</td><td>${e.cam || ""}</td>` +
    `<td>${e.label || ""} ${e.data ? JSON.stringify(e.data) : ""}</td></tr>`).join("");
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------
async function runBenchmark() {
  const el = document.getElementById("bench-result");
  el.textContent = "Mereni…";
  try {
    const r = await fetch("/api/benchmark").then((x) => x.json());
    const dev = r.cuda ? `GPU (${(r.gpus || []).join(", ")})` : "CPU";
    el.textContent =
      `Zarizeni: ${dev}\nDevice: ${r.device}\nModel: ${r.model || "?"} @ imgsz ${r.imgsz || "?"}\n` +
      `Latence: ${r.latency_ms ?? "?"} ms/snimek\nFPS (1 stream): ${r.fps_single ?? "?"}\n` +
      `Doporucene FPS/kamera: ${r.suggested_fps ?? "?"}` + (r.error ? `\nChyba: ${r.error}` : "");
  } catch (e) { el.textContent = "Chyba benchmarku."; }
}

async function saveStartup() {
  await pushSettings();
  try {
    await fetch("/api/settings/save-startup", { method: "POST" });
    document.getElementById("bench-saved").textContent = "Ulozeno (pouzije se pri pristim startu).";
  } catch (e) { document.getElementById("bench-saved").textContent = "Ulozeni selhalo."; }
}

// ---------------------------------------------------------------------------
// Debug log
// ---------------------------------------------------------------------------
let lastLogSeq = 0;
async function pollLogs() {
  try {
    const r = await fetch(`/api/logs?after=${lastLogSeq}&limit=500`).then((x) => x.json());
    const pre = document.getElementById("debug-log");
    for (const l of r.logs || []) {
      lastLogSeq = l.seq;
      const div = document.createElement("div");
      div.className = `lvl-${l.level}`;
      div.textContent = l.msg;
      pre.appendChild(div);
    }
    while (pre.childNodes.length > 1000) pre.removeChild(pre.firstChild);
    if (document.getElementById("debug-autoscroll").checked)
      pre.scrollTop = pre.scrollHeight;
  } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Detection events log (second debug window)
// ---------------------------------------------------------------------------
async function pollEvents() {
  try {
    const r = await fetch("/api/history/events?limit=100").then((x) => x.json());
    const pre = document.getElementById("events-log");
    const evs = (r.events || []).slice().reverse();
    pre.innerHTML = "";
    for (const e of evs) {
      const div = document.createElement("div");
      const t = new Date(e.ts * 1000).toLocaleTimeString("cs-CZ");
      const extra = e.data ? " " + JSON.stringify(e.data) : "";
      div.textContent = `${t} [${e.kind}] ${e.cam || ""} ${e.label || ""}${extra}`;
      pre.appendChild(div);
    }
    if (document.getElementById("debug-autoscroll").checked)
      pre.scrollTop = pre.scrollHeight;
  } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Recordings
// ---------------------------------------------------------------------------
function buildRecCams() {
  const sel = document.getElementById("rec-cam");
  sel.innerHTML = (appConfig.cameras || []).map((c) => `<option>${c.id}</option>`).join("");
}

async function startRecording() {
  const cam = document.getElementById("rec-cam").value;
  const dur = parseFloat(document.getElementById("rec-dur").value);
  const st = document.getElementById("rec-status");
  st.textContent = "Spoustim…";
  try {
    const r = await fetch("/api/record/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam, duration: dur }),
    }).then((x) => x.json());
    st.textContent = r.error ? `Chyba: ${r.error}` : `Nahravam ${cam} (${dur}s)…`;
    loadRecordings();
    setTimeout(loadRecordings, (dur + 3) * 1000);
  } catch (e) { st.textContent = "Chyba nahravani."; }
}

async function loadRecordings() {
  try {
    const r = await fetch("/api/recordings").then((x) => x.json());
    const tb = document.querySelector("#rec-list tbody");
    tb.innerHTML = (r.recordings || []).map((x) =>
      `<tr><td>${x.file}</td><td>${(x.size / 1e6).toFixed(1)} MB</td>` +
      `<td>${x.recording ? "nahrava se" : "hotovo"}</td>` +
      `<td>${x.recording ? "" : `<a href="/recordings/${x.file}" download>stahnout</a>`}</td></tr>`
    ).join("");
  } catch (e) { /* ignore */ }
}

function wireExtraControls() {
  document.getElementById("hist-refresh").addEventListener("click", loadHistory);
  document.getElementById("bench-run").addEventListener("click", runBenchmark);
  document.getElementById("bench-save").addEventListener("click", saveStartup);
  document.getElementById("rec-start").addEventListener("click", startRecording);
  document.getElementById("report-load").addEventListener("click", loadReport);
  document.getElementById("debug-level").addEventListener("change", async (e) => {
    await fetch("/api/log-level", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ level: e.target.value }),
    });
  });
  document.getElementById("debug-clear").addEventListener("click", () => {
    document.getElementById("debug-log").innerHTML = "";
  });
}

// ---------------------------------------------------------------------------
// Legend + status
// ---------------------------------------------------------------------------
function buildLegend(colors) {
  const el = document.getElementById("legend");
  const map = colors || CLASS_COLORS;
  el.innerHTML = "";
  for (const [name, rgb] of Object.entries(map)) {
    const item = document.createElement("span");
    item.className = "item";
    item.innerHTML = `<span class="swatch" style="background: rgb(${rgb[0]},${rgb[1]},${rgb[2]})"></span>${labelFor(name)}`;
    el.appendChild(item);
  }
}

function setCameraStatus(status) {
  for (const [cid, ok] of Object.entries(status || {})) {
    document.querySelectorAll(`.dot[data-cam="${cid}"]`).forEach((d) =>
      d.classList.toggle("on", !!ok));
  }
}

// ---------------------------------------------------------------------------
// Settings panel
// ---------------------------------------------------------------------------
const SETTINGS_KEY = "camdetect.settings";
let settings = null;

const els = {
  videoEnabled: "set-video-enabled", videoFps: "set-video-fps", videoImgsz: "set-video-imgsz",
  videoConf: "set-video-conf", ovEnabled: "set-ov-enabled", ovPrompts: "set-ov-prompts",
  attrBehavior: "set-attr-behavior", attrAge: "set-attr-age", minCams: "set-min-cams",
  vehEnabled: "set-veh-enabled", vehPlates: "set-veh-plates", vehMakeModel: "set-veh-makemodel",
  audioEnabled: "set-audio-enabled", audioEvents: "set-audio-events",
  audioEngine: "set-audio-engine", audioWindow: "set-audio-window", audioHop: "set-audio-hop",
  trEnabled: "set-tr-enabled", trDiar: "set-tr-diar", trRecord: "set-tr-record",
  droneEnabled: "set-drone-enabled", droneVisual: "set-drone-visual",
  droneAudio: "set-drone-audio", droneFuse: "set-drone-fuse", droneSens: "set-drone-sens",
};
const $ = (id) => document.getElementById(id);

function applySettingsToControls(s) {
  $(els.videoEnabled).checked = s.video.enabled;
  $(els.videoFps).value = s.video.fps; $("val-video-fps").textContent = s.video.fps;
  $(els.videoImgsz).value = String(s.video.imgsz);
  $(els.videoConf).value = s.video.confidence; $("val-video-conf").textContent = s.video.confidence;
  $(els.ovEnabled).checked = s.video.open_vocabulary.enabled;
  $(els.ovPrompts).value = (s.video.open_vocabulary.prompts || []).join(", ");
  if (s.video.min_cameras) $(els.minCams).value = String(s.video.min_cameras);
  $(els.attrBehavior).checked = s.attributes.behavior;
  $(els.attrAge).checked = s.attributes.age;
  $(els.vehEnabled).checked = s.vehicles.enabled;
  $(els.vehPlates).checked = s.vehicles.plates;
  $(els.vehMakeModel).checked = s.vehicles.make_model;
  $(els.audioEnabled).checked = s.audio.enabled;
  $(els.audioEvents).checked = s.audio.events;
  $(els.audioEngine).checked = s.audio.engine_2t4t;
  $(els.audioWindow).value = s.audio.window_s; $("val-audio-window").textContent = s.audio.window_s;
  $(els.audioHop).value = s.audio.hop_s; $("val-audio-hop").textContent = s.audio.hop_s;
  $(els.trEnabled).checked = s.transcription.enabled;
  $(els.trDiar).checked = s.transcription.diarization;
  $(els.trRecord).checked = s.transcription.record;
  if (s.drone) {
    $(els.droneEnabled).checked = s.drone.enabled;
    $(els.droneVisual).checked = s.drone.visual;
    $(els.droneAudio).checked = s.drone.audio;
    $(els.droneFuse).checked = s.drone.fuse;
    $(els.droneSens).value = s.drone.sensitivity;
    $("val-drone-sens").textContent = s.drone.sensitivity;
  }
}

function collectSettings() {
  return {
    video: {
      enabled: $(els.videoEnabled).checked,
      fps: parseFloat($(els.videoFps).value),
      imgsz: parseInt($(els.videoImgsz).value, 10),
      confidence: parseFloat($(els.videoConf).value),
      open_vocabulary: {
        enabled: $(els.ovEnabled).checked,
        prompts: $(els.ovPrompts).value.split(",").map((s) => s.trim()).filter(Boolean),
      },
      min_cameras: parseInt($(els.minCams).value, 10),
    },
    attributes: { behavior: $(els.attrBehavior).checked, age: $(els.attrAge).checked },
    vehicles: {
      enabled: $(els.vehEnabled).checked,
      plates: $(els.vehPlates).checked,
      make_model: $(els.vehMakeModel).checked,
    },
    audio: {
      enabled: $(els.audioEnabled).checked,
      events: $(els.audioEvents).checked,
      engine_2t4t: $(els.audioEngine).checked,
      window_s: parseFloat($(els.audioWindow).value),
      hop_s: parseFloat($(els.audioHop).value),
    },
    transcription: {
      enabled: $(els.trEnabled).checked,
      diarization: $(els.trDiar).checked,
      record: $(els.trRecord).checked,
    },
    drone: {
      enabled: $(els.droneEnabled).checked,
      visual: $(els.droneVisual).checked,
      audio: $(els.droneAudio).checked,
      fuse: $(els.droneFuse).checked,
      sensitivity: parseFloat($(els.droneSens).value),
    },
  };
}

async function pushSettings() {
  const patch = collectSettings();
  $("val-video-fps").textContent = patch.video.fps;
  $("val-video-conf").textContent = patch.video.confidence;
  $("val-audio-window").textContent = patch.audio.window_s;
  $("val-audio-hop").textContent = patch.audio.hop_s;
  $("val-drone-sens").textContent = patch.drone.sensitivity;
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(patch));
  try {
    await fetch("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
  } catch (e) { /* ignore */ }
}

function wireSettings() {
  for (const id of Object.values(els)) {
    const el = $(id);
    el.addEventListener("change", pushSettings);
    if (el.type === "range") el.addEventListener("input", pushSettings);
  }
  $("btn-settings").addEventListener("click", () => {
    $("drawer").classList.toggle("hidden");
    $("btn-settings").classList.toggle("active");
  });
  $("btn-cameras").addEventListener("click", toggleCameras);
  $("btn-audio").addEventListener("click", toggleAudio);
}

function toggleCameras() {
  const hidden = document.body.classList.toggle("cameras-hidden");
  $("btn-cameras").classList.toggle("active", !hidden);
  localStorage.setItem("camdetect.cameras", hidden ? "0" : "1");
  document.querySelectorAll("img.stream").forEach((img) => {
    if (hidden) { img.removeAttribute("src"); }
    else { img.src = `/stream/${img.dataset.cam}`; }
  });
  setTimeout(resize, 50);
}

function toggleAudio() {
  const hidden = document.body.classList.toggle("audio-hidden");
  $("btn-audio").classList.toggle("active", !hidden);
  localStorage.setItem("camdetect.audio", hidden ? "0" : "1");
  // stop/refresh the spectrogram stream to save bandwidth when hidden
  if (hidden) spectro.removeAttribute("src");
  else if (selectedAudioCam) spectro.src = `/audio/${selectedAudioCam}/spectrogram`;
  setTimeout(resize, 50);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function boot() {
  appConfig = await (await fetch("/api/config")).json();
  labels = appConfig.labels || {};

  const cs = appConfig.cameras.map((c) => c.world_xy);
  center.x = cs.reduce((s, p) => s + p[0], 0) / cs.length;
  center.y = cs.reduce((s, p) => s + p[1], 0) / cs.length;
  let span = 8;
  for (const c of appConfig.cameras)
    span = Math.max(span, Math.abs(c.world_xy[0] - center.x), Math.abs(c.world_xy[1] - center.y));

  buildGround(Math.ceil(span * 2 + 8));
  buildCameras(appConfig.cameras);
  computeBounds(appConfig.cameras);
  buildLegend(appConfig.colors);
  buildAudioTabs(appConfig.cameras);
  buildRecCams();
  document.getElementById("mode").textContent = appConfig.mode + " mode";
  if (appConfig.build) {
    document.getElementById("ver").textContent =
      `v${appConfig.build.version} (${appConfig.build.commit})`;
  }

  // settings: server values, overlaid by any saved local preferences
  settings = await (await fetch("/api/settings")).json();
  const saved = localStorage.getItem(SETTINGS_KEY);
  if (saved) {
    try { settings = deepMerge(settings, JSON.parse(saved)); } catch (e) {}
  }
  applySettingsToControls(settings);
  wireSettings();
  wireTabs();
  wireExtraControls();
  await pushSettings();

  // camera visibility preference (default hidden)
  if (localStorage.getItem("camdetect.cameras") === "1") toggleCameras();
  // audio panel preference (default shown)
  if (localStorage.getItem("camdetect.audio") === "0") toggleAudio();

  resize();
  connectWs();
  animate();
}

function deepMerge(base, patch) {
  const out = Array.isArray(base) ? base.slice() : { ...base };
  for (const k in patch) {
    if (patch[k] && typeof patch[k] === "object" && !Array.isArray(patch[k]) &&
        base[k] && typeof base[k] === "object")
      out[k] = deepMerge(base[k], patch[k]);
    else out[k] = patch[k];
  }
  return out;
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  const conn = document.getElementById("conn");
  ws.onopen = () => { conn.textContent = "zive"; conn.className = "badge badge-on"; };
  ws.onclose = () => {
    conn.textContent = "odpojeno"; conn.className = "badge badge-off";
    setTimeout(connectWs, 1500);
  };
  ws.onmessage = (ev) => {
    const state = JSON.parse(ev.data);
    // remember previous positions for heading arrows
    for (const o of latestObjects) prevPos.set(o.id, { x: o.x, y: o.y });
    latestObjects = state.objects || [];
    latestAudio = state.audio || {};
    latestTranscripts = state.transcripts || {};
    updateObjects(latestObjects);
    setCameraStatus(state.cameras);
    document.getElementById("objcount").textContent = `${latestObjects.length} objektu`;
  };
}

function animate() {
  requestAnimationFrame(animate);
  if (activeTab === "live") {
    controls.update();
    drawTopdown();
    drawFreq();
    drawAudioEvents();
    drawTranscript();
    renderer.render(scene, camera);
  }
}

boot();
