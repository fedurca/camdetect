import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { colorFor, cssColor, hexColor, CLASS_COLORS } from "/colors.js";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let appConfig = null;
let center = { x: 0, y: 0 };          // world-frame center used to center the scene
const objectMeshes = new Map();        // id -> { group, mesh, label, class }
let latestObjects = [];

// Footprint (meters) per class for the 3D boxes.
const FOOTPRINT = {
  person: [0.6, 0.6], bicycle: [1.6, 0.6], car: [4.4, 1.9],
  motorcycle: [2.0, 0.8], bus: [10, 2.5], truck: [7, 2.5],
};
function footprint(name) { return FOOTPRINT[name] || [0.7, 0.7]; }

// World (X,Y[,up]) -> scene coords. X east, Y north, Z up in world; in three.js
// we use X right, Y up, Z toward viewer, so north maps to -Z.
function worldToScene(X, Y, up = 0) {
  return new THREE.Vector3(X - center.x, up, -(Y - center.y));
}

// ---------------------------------------------------------------------------
// Three.js setup
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
controls.target.set(0, 0, 0);

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
}
window.addEventListener("resize", resize);

// ---------------------------------------------------------------------------
// Ground + cameras
// ---------------------------------------------------------------------------
function buildGround(sizeMeters) {
  const grid = new THREE.GridHelper(sizeMeters, Math.max(4, Math.round(sizeMeters)),
    0x334155, 0x1e293b);
  scene.add(grid);

  const geo = new THREE.PlaneGeometry(sizeMeters, sizeMeters);
  geo.rotateX(-Math.PI / 2);
  const mat = new THREE.MeshStandardMaterial({
    color: 0x0f172a, transparent: true, opacity: 0.6,
  });
  const plane = new THREE.Mesh(geo, mat);
  plane.position.y = -0.01;
  scene.add(plane);

  // Try to drape a satellite image; ignore if it isn't present.
  new THREE.TextureLoader().load(
    "/assets/satellite.png",
    (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      mat.map = tex;
      mat.color.set(0xffffff);
      mat.opacity = 1;
      mat.needsUpdate = true;
    },
    undefined,
    () => {/* no satellite image - keep grid */}
  );
}

function labelSprite(text, rgb) {
  const canvas = document.createElement("canvas");
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = `rgba(10,14,22,0.85)`;
  ctx.fillRect(0, 0, 256, 64);
  ctx.strokeStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  ctx.lineWidth = 4; ctx.strokeRect(2, 2, 252, 60);
  ctx.fillStyle = "#fff";
  ctx.font = "bold 28px sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 34);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
  spr.scale.set(2.4, 0.6, 1);
  return spr;
}

function buildCameras(cameras) {
  for (const cam of cameras) {
    const [X, Y] = cam.world_xy;
    const pos = worldToScene(X, Y, cam.height_m);
    const geo = new THREE.ConeGeometry(0.4, 0.9, 4);
    const mat = new THREE.MeshStandardMaterial({ color: 0x38bdf8 });
    const cone = new THREE.Mesh(geo, mat);
    cone.position.copy(pos);
    cone.rotation.x = Math.PI; // point down
    scene.add(cone);

    // pole to the ground
    const pole = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([pos, worldToScene(X, Y, 0)]),
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.5 })
    );
    scene.add(pole);

    // sight line toward courtyard center
    const sight = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([pos, worldToScene(center.x, center.y, 0.2)]),
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.25 })
    );
    scene.add(sight);

    const tag = labelSprite(cam.id, [56, 189, 248]);
    tag.position.copy(pos.clone().add(new THREE.Vector3(0, 0.9, 0)));
    tag.scale.set(1.6, 0.4, 1);
    scene.add(tag);
  }
}

// ---------------------------------------------------------------------------
// Object boxes
// ---------------------------------------------------------------------------
function makeObject(o) {
  const [fw, fd] = footprint(o.class);
  const h = o.height || 1.7;
  const rgb = colorFor(o.class);
  const group = new THREE.Group();

  const box = new THREE.Mesh(
    new THREE.BoxGeometry(fw, h, fd),
    new THREE.MeshStandardMaterial({
      color: hexColor(o.class), transparent: true, opacity: 0.55,
    })
  );
  box.position.y = h / 2;
  group.add(box);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(box.geometry),
    new THREE.LineBasicMaterial({ color: hexColor(o.class) })
  );
  edges.position.y = h / 2;
  group.add(edges);

  const label = labelSprite(`${o.class} #${o.id} ${o.prob.toFixed(2)}`, rgb);
  label.position.y = h + 0.6;
  group.add(label);

  scene.add(group);
  return { group, box, label, class: o.class };
}

function updateObjects(objects) {
  const seen = new Set();
  for (const o of objects) {
    seen.add(o.id);
    let entry = objectMeshes.get(o.id);
    if (!entry || entry.class !== o.class) {
      if (entry) scene.remove(entry.group);
      entry = makeObject(o);
      objectMeshes.set(o.id, entry);
    } else {
      // refresh label (probability changes)
      entry.group.remove(entry.label);
      entry.label = labelSprite(`${o.class} #${o.id} ${o.prob.toFixed(2)}`, colorFor(o.class));
      entry.label.position.y = (o.height || 1.7) + 0.6;
      entry.group.add(entry.label);
    }
    const p = worldToScene(o.x, o.y, 0);
    entry.group.position.set(p.x, 0, p.z);
  }
  for (const [id, entry] of objectMeshes) {
    if (!seen.has(id)) {
      scene.remove(entry.group);
      objectMeshes.delete(id);
    }
  }
}

// ---------------------------------------------------------------------------
// Minimap (top-down)
// ---------------------------------------------------------------------------
const mini = document.getElementById("minimap");
const mctx = mini.getContext("2d");
let miniBounds = { minX: -10, maxX: 10, minY: -10, maxY: 10 };

function computeBounds(cameras) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const c of cameras) {
    minX = Math.min(minX, c.world_xy[0]); maxX = Math.max(maxX, c.world_xy[0]);
    minY = Math.min(minY, c.world_xy[1]); maxY = Math.max(maxY, c.world_xy[1]);
  }
  const padX = (maxX - minX) * 0.3 + 3;
  const padY = (maxY - minY) * 0.3 + 3;
  miniBounds = { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
}

function miniPt(X, Y) {
  const { minX, maxX, minY, maxY } = miniBounds;
  const px = ((X - minX) / (maxX - minX)) * mini.width;
  // invert Y so north is up
  const py = mini.height - ((Y - minY) / (maxY - minY)) * mini.height;
  return [px, py];
}

function drawMinimap() {
  mctx.clearRect(0, 0, mini.width, mini.height);
  mctx.fillStyle = "rgba(5,8,14,0.6)";
  mctx.fillRect(0, 0, mini.width, mini.height);

  if (appConfig) {
    mctx.fillStyle = "#38bdf8";
    for (const c of appConfig.cameras) {
      const [px, py] = miniPt(c.world_xy[0], c.world_xy[1]);
      mctx.beginPath();
      mctx.moveTo(px, py - 5); mctx.lineTo(px - 4, py + 4); mctx.lineTo(px + 4, py + 4);
      mctx.closePath(); mctx.fill();
    }
  }
  for (const o of latestObjects) {
    const [px, py] = miniPt(o.x, o.y);
    mctx.fillStyle = cssColor(o.class);
    mctx.beginPath();
    mctx.arc(px, py, 4, 0, Math.PI * 2);
    mctx.fill();
    mctx.fillStyle = "#e2e8f0";
    mctx.font = "9px sans-serif";
    mctx.fillText(`${o.class[0]}${o.id}`, px + 5, py + 3);
  }
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
    item.innerHTML =
      `<span class="swatch" style="background: rgb(${rgb[0]},${rgb[1]},${rgb[2]})"></span>${name}`;
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
// Boot
// ---------------------------------------------------------------------------
async function boot() {
  const res = await fetch("/api/config");
  appConfig = await res.json();

  const cs = appConfig.cameras.map((c) => c.world_xy);
  center.x = cs.reduce((s, p) => s + p[0], 0) / cs.length;
  center.y = cs.reduce((s, p) => s + p[1], 0) / cs.length;

  let span = 8;
  for (const c of appConfig.cameras) {
    span = Math.max(span, Math.abs(c.world_xy[0] - center.x), Math.abs(c.world_xy[1] - center.y));
  }
  buildGround(Math.ceil(span * 2 + 8));
  buildCameras(appConfig.cameras);
  computeBounds(appConfig.cameras);
  buildLegend(appConfig.colors);

  document.getElementById("mode").textContent = appConfig.mode + " mode";

  // Point each stream <img> at its MJPEG endpoint.
  document.querySelectorAll("img.stream").forEach((img) => {
    img.src = `/stream/${img.dataset.cam}`;
  });

  resize();
  connectWs();
  animate();
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  const conn = document.getElementById("conn");
  ws.onopen = () => { conn.textContent = "live"; conn.className = "badge badge-on"; };
  ws.onclose = () => {
    conn.textContent = "disconnected"; conn.className = "badge badge-off";
    setTimeout(connectWs, 1500);
  };
  ws.onmessage = (ev) => {
    const state = JSON.parse(ev.data);
    latestObjects = state.objects || [];
    updateObjects(latestObjects);
    setCameraStatus(state.cameras);
    document.getElementById("objcount").textContent = `${latestObjects.length} objects`;
  };
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  drawMinimap();
  renderer.render(scene, camera);
}

boot();
