/* Xiao Hei perception viewer.
 *
 * Draws what `scripts/export_viz.py` dumped: the accumulated lidar cloud, the
 * scan and lifted points of one frame, our fused object boxes and the ground
 * truth boxes, in the map frame. Geometry arrives as one float32 blob and the
 * manifest indexes into it, so nothing here parses coordinates out of JSON.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const OURS = 0x35d0d8, GT = 0xff9f43, DET = 0xa78bfa, PATH = 0xf472b6;
const STRUCTURE = /^(wall|walls|exterior wall|interior wall|partition wall|floor|ceiling|carpet|rug)$/;

let renderer, scene, camera, controls, raycaster;
let data = null, blob = null;      // manifest + Float32Array of xyz
let frameIdx = 0, playing = false, lastStep = 0;
let gWorld, gGtCloud, gScan, gDet, gOurs, gGt, gDetBox, gPath, gRobot;
let visibleBoxes = [];             // what picking and labels iterate
let picked = null;

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------

function init() {
  const stage = $('stage');
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  stage.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1116);
  camera = new THREE.PerspectiveCamera(55, 1, 0.05, 500);
  camera.up.set(0, 0, 1);                       // map frame is Z-up
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  raycaster = new THREE.Raycaster();

  gWorld = points(0.55); scene.add(gWorld);
  gGtCloud = points(0.75); scene.add(gGtCloud);
  gScan = points(0.9); scene.add(gScan);
  gDet = points(1.0); scene.add(gDet);
  gOurs = lines(); scene.add(gOurs);
  gGt = lines(); scene.add(gGt);
  gDetBox = lines(); scene.add(gDetBox);
  gPath = lines(); scene.add(gPath);
  gRobot = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 12),
                          new THREE.MeshBasicMaterial({ color: 0xffffff }));
  scene.add(gRobot);

  addEventListener('resize', resize);
  resize();
  wireUi();
  renderer.domElement.addEventListener('click', onPick);
  loadIndex();
  tick();
}

function points(opacity) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(3), 3));
  const m = new THREE.PointsMaterial({
    size: 1.8, sizeAttenuation: false, vertexColors: true,
    transparent: true, opacity, depthWrite: false });
  const p = new THREE.Points(g, m);
  p.frustumCulled = false;
  return p;
}

function lines() {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
  const l = new THREE.LineSegments(
    g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true }));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(3), 3));
  l.frustumCulled = false;
  return l;
}

function resize() {
  const s = $('stage').getBoundingClientRect();
  // updateStyle must stay on: with it off the canvas keeps its default
  // 300x150 CSS box and the whole scene renders into the top-left corner.
  renderer.setSize(s.width, s.height, true);
  camera.aspect = s.width / Math.max(s.height, 1);
  camera.updateProjectionMatrix();
}

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------

async function loadIndex() {
  const idx = await (await fetch('data/index.json')).json();
  const sel = $('scene');
  sel.innerHTML = '';
  for (const s of idx.scenes) {
    const o = document.createElement('option');
    o.value = o.textContent = s;
    sel.appendChild(o);
  }
  sel.onchange = () => loadScene(sel.value);
  if (idx.scenes.length) loadScene(idx.scenes[0]);
  else $('loading').textContent = 'no scenes exported — run scripts/export_viz.py';
}

async function loadScene(name) {
  $('loading').style.display = 'grid';
  $('loading').textContent = `loading ${name}…`;
  data = await (await fetch(`data/${name}.json`)).json();
  blob = new Float32Array(
    await (await fetch(`data/${data.bin}`)).arrayBuffer());
  data.structSet = data.struct_labels ? new Set(data.struct_labels) : null;

  frameIdx = 0;
  const fr = $('frame');
  fr.max = Math.max(data.frames.length - 1, 0);
  fr.value = 0;

  buildWorld();
  buildGtCloud();
  buildPath();
  rebuildBoxes();
  showFrame(0);
  fitView();
  $('loading').style.display = 'none';
  $('ver').textContent = `· ${name}`;
}

/** Copy `[offset, count]` point pairs out of the blob into a flat array. */
function gather(refs) {
  let n = 0;
  for (const r of refs) n += r[1];
  const out = new Float32Array(n * 3);
  let w = 0;
  for (const r of refs) {
    out.set(blob.subarray(r[0] * 3, (r[0] + r[1]) * 3), w);
    w += r[1] * 3;
  }
  return out;
}

function setPoints(obj, pos, colorFn) {
  const n = pos.length / 3;
  const col = new Float32Array(pos.length);
  for (let i = 0; i < n; i++) colorFn(col, i * 3, pos);
  obj.geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  obj.geometry.setAttribute('color', new THREE.BufferAttribute(col, 3));
  obj.geometry.attributes.position.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// layers
// ---------------------------------------------------------------------------

/** Per-axis percentile box. A few stray returns down a corridor otherwise set
 *  the extent, and the room ends up a dot in the middle of empty space. */
function robustBounds(pos, lowPct = 0.015, highPct = 0.985) {
  const n = pos.length / 3;
  if (!n) return { min: [-1, -1, -1], max: [1, 1, 1] };
  const step = Math.max(1, Math.floor(n / 60000));
  const min = [], max = [];
  for (let a = 0; a < 3; a++) {
    const v = [];
    for (let i = 0; i < n; i += step) v.push(pos[i * 3 + a]);
    v.sort((x, y) => x - y);
    min.push(v[Math.floor(v.length * lowPct)]);
    max.push(v[Math.min(v.length - 1, Math.floor(v.length * highPct))]);
  }
  return { min, max };
}

function buildWorld() {
  const pos = gather([data.world]);
  data.fit = robustBounds(pos);
  const lo = data.fit.min[2], hi = data.fit.max[2];
  const span = Math.max(hi - lo, 1e-3);
  setPoints(gWorld, pos, (c, i, p) => {
    const t = Math.min(Math.max((p[i + 2] - lo) / span, 0), 1);
    c[i] = 0.20 + 0.30 * t; c[i + 1] = 0.26 + 0.34 * t; c[i + 2] = 0.34 + 0.40 * t;
  });
}

/** The scene's own `map.ply`, thinned. This is the real geometry -- our lidar
 *  cloud next to it shows coverage, and a ground-truth box next to it shows
 *  what that box was drawn around. */
function buildGtCloud() {
  const ref = data.gt_world;
  if (!ref || !ref[1]) { gGtCloud.visible = false; return; }
  const pos = gather([ref]);
  const lo = data.fit.min[2], hi = data.fit.max[2];
  const span = Math.max(hi - lo, 1e-3);
  setPoints(gGtCloud, pos, (c, i, p) => {
    const t = Math.min(Math.max((p[i + 2] - lo) / span, 0), 1);
    c[i] = 0.55 + 0.33 * t; c[i + 1] = 0.40 + 0.28 * t; c[i + 2] = 0.28 + 0.24 * t;
  });
  gGtCloud.visible = $('lGtCloud').checked;
}

function buildPath() {
  const f = data.frames, v = [], c = [];
  for (let i = 1; i < f.length; i++) {
    v.push(...f[i - 1].p, ...f[i].p);
    c.push(0.96, 0.45, 0.71, 0.96, 0.45, 0.71);
  }
  setLines(gPath, v, c);
}

function setLines(obj, verts, cols) {
  obj.geometry.setAttribute('position',
    new THREE.BufferAttribute(new Float32Array(verts), 3));
  obj.geometry.setAttribute('color',
    new THREE.BufferAttribute(new Float32Array(cols), 3));
}

/** 12 edges of an axis-aligned box, as 24 vertices. */
function boxEdges(lo, hi, out) {
  const x = [lo[0], hi[0]], y = [lo[1], hi[1]], z = [lo[2], hi[2]];
  const c = [];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++)
    c.push([x[i], y[j], z[k]]);
  // corner index = i*4 + j*2 + k
  const E = [[0,1],[2,3],[4,5],[6,7], [0,2],[1,3],[4,6],[5,7], [0,4],[1,5],[2,6],[3,7]];
  for (const [a, b] of E) out.push(...c[a], ...c[b]);
}

/** Same 12 edges, rotated about the up axis by the annotation's heading. */
function boxEdgesOriented(c, s, h, out) {
  const ca = Math.cos(h), sa = Math.sin(h), corner = [];
  for (let i = -1; i <= 1; i += 2)
    for (let j = -1; j <= 1; j += 2)
      for (let k = -1; k <= 1; k += 2) {
        const x = i * s[0] / 2, y = j * s[1] / 2;
        corner.push([c[0] + x * ca - y * sa, c[1] + x * sa + y * ca,
                     c[2] + k * s[2] / 2]);
      }
  const E = [[0,1],[2,3],[4,5],[6,7], [0,2],[1,3],[4,6],[5,7],
             [0,4],[1,5],[2,6],[3,7]];
  for (const [a, b] of E) out.push(...corner[a], ...corner[b]);
}

function pushColor(cols, hex, n) {
  const r = ((hex >> 16) & 255) / 255, g = ((hex >> 8) & 255) / 255, b = (hex & 255) / 255;
  for (let i = 0; i < n; i++) cols.push(r, g, b);
}

function passes(o, isGt) {
  const name = data.labels[o.l] || '';
  // `structSet` comes from vocab.is_structure, so a detection and the object
  // it fused into get the same verdict. The regex is only a fallback for data
  // exported before the manifest carried the flags.
  const struct = o.struct || (data.structSet ? data.structSet.has(o.l)
                                             : STRUCTURE.test(name));
  if ($('hideStruct').checked && struct) return false;
  const q = $('q').value.trim().toLowerCase();
  if (q && !name.toLowerCase().includes(q)) return false;
  if (isGt) return true;
  if (o.s !== undefined && o.s < parseFloat($('score').value)) return false;
  if (o.n_obs !== undefined && o.n_obs < parseInt($('minObs').value, 10)) return false;
  // The fused map is the whole run's answer; these two replay how it was
  // built, which is the only way to see a node appear twice for one object.
  const mode = $('objMode').value;
  if (mode !== 'all' && o.fs) {
    if (mode === 'cum') return o.f0 <= frameIdx;
    if (mode === 'cur') return o.fs.indexOf(frameIdx) >= 0;
  }
  return true;
}

function rebuildBoxes() {
  visibleBoxes = [];
  const ov = [], oc = [];
  for (const o of data.objects) {
    if (!passes(o, false)) continue;
    boxEdges(o.lo, o.hi, ov);
    pushColor(oc, picked === o ? 0xffffff : OURS, 24);
    visibleBoxes.push({ o, kind: 'ours' });
  }
  setLines(gOurs, ov, oc);

  const gv = [], gc = [];
  const obb = $('gtObb').checked;
  for (const o of data.gt) {
    if (!passes(o, true)) continue;
    // perception/eval.py scores against `center +- size/2` with the heading
    // dropped, so the axis-aligned form is the one the numbers use; the
    // oriented form is what the scene actually annotates.
    if (obb && o.sz && o.h !== undefined) boxEdgesOriented(o.c, o.sz, o.h, gv);
    else boxEdges(o.lo, o.hi, gv);
    pushColor(gc, picked === o ? 0xffffff : GT, 24);
    visibleBoxes.push({ o, kind: 'gt' });
  }
  setLines(gGt, gv, gc);
  updateHud();
}

function showFrame(i) {
  frameIdx = Math.max(0, Math.min(i, data.frames.length - 1));
  const f = data.frames[frameIdx];
  $('frame').value = frameIdx;
  $('frameLbl').textContent = `${frameIdx + 1} / ${data.frames.length}`;
  $('tickLbl').textContent = `tick ${f.t}`;

  setPoints(gScan, gather([f.scan]), (c, i) => {
    c[i] = 0.56; c[i + 1] = 0.83; c[i + 2] = 1.0;
  });

  const minS = parseFloat($('score').value);
  const dets = f.dets.filter((d) => d.s >= minS && passes(
    { l: d.l, struct: false, s: d.s }, false));
  const refs = dets.map((d) => d.pts);
  const pos = gather(refs);
  // Colour each lifted cloud by its label so two objects overlapping in space
  // are still separable on screen.
  const hues = [];
  for (const d of dets) {
    const h = hue(data.labels[d.l] || '');
    for (let k = 0; k < d.pts[1]; k++) hues.push(h);
  }
  let hi = 0;
  setPoints(gDet, pos, (c, i) => {
    const [r, g, b] = hues[hi++] || [1, 1, 1];
    c[i] = r; c[i + 1] = g; c[i + 2] = b;
  });

  const dv = [], dc = [];
  for (const d of dets) { boxEdges(d.lo, d.hi, dv); pushColor(dc, DET, 24); }
  setLines(gDetBox, dv, dc);

  gRobot.position.set(f.p[0], f.p[1], f.p[2]);
  if ($('follow').checked) {
    controls.target.set(f.p[0], f.p[1], f.p[2]);
  }
  if ($('objMode').value !== 'all') rebuildBoxes();
  else updateHud();
}

/** Stable pastel colour per label name. */
function hue(name) {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xffff;
  const a = (h % 360) / 60, s = 0.55, v = 1.0;
  const c = v * s, x = c * (1 - Math.abs((a % 2) - 1)), m = v - c;
  const t = [[c,x,0],[x,c,0],[0,c,x],[0,x,c],[x,0,c],[c,0,x]][Math.floor(a) % 6];
  return [t[0] + m, t[1] + m, t[2] + m];
}

// ---------------------------------------------------------------------------
// labels, hud, picking
// ---------------------------------------------------------------------------

function drawLabels() {
  const ov = $('overlay');
  if (!$('lLabels').checked || !data) { ov.innerHTML = ''; return; }
  const rect = renderer.domElement.getBoundingClientRect();
  const v = new THREE.Vector3();
  const items = [];
  for (const b of visibleBoxes) {
    if (b.kind === 'ours' && !$('lOurs').checked) continue;
    if (b.kind === 'gt' && !$('lGt').checked) continue;
    const o = b.o;
    v.set((o.lo[0] + o.hi[0]) / 2, (o.lo[1] + o.hi[1]) / 2, o.hi[2]);
    const d = v.distanceTo(camera.position);
    v.project(camera);
    if (v.z > 1 || v.x < -1 || v.x > 1 || v.y < -1 || v.y > 1) continue;
    items.push({ d, x: (v.x * 0.5 + 0.5) * rect.width,
                 y: (-v.y * 0.5 + 0.5) * rect.height,
                 name: data.labels[o.l], kind: b.kind });
  }
  items.sort((a, b) => a.d - b.d);
  ov.innerHTML = items.slice(0, 50).map((t) =>
    `<div class="tag" style="left:${t.x.toFixed(0)}px;top:${t.y.toFixed(0)}px;` +
    `color:${t.kind === 'ours' ? '#35d0d8' : '#ff9f43'}">${esc(t.name)}</div>`
  ).join('');
}

const esc = (s) => String(s).replace(/[&<>]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function updateHud() {
  if (!data) return;
  const ours = visibleBoxes.filter((b) => b.kind === 'ours').length;
  const gt = visibleBoxes.filter((b) => b.kind === 'gt').length;
  const f = data.frames[frameIdx];
  const mode = { all: 'whole run', cum: 'up to this frame',
                 cur: 'seen this frame' }[$('objMode').value];
  $('hud').innerHTML =
    `<b>${ours}</b> ours <span style="opacity:.7">(${mode})</span> &nbsp;·&nbsp; ` +
    `<b>${gt}</b> gt &nbsp;·&nbsp; ` +
    `<b>${f ? f.dets.length : 0}</b> dets this frame &nbsp;·&nbsp; ` +
    `stride ${data.stride}`;
}

/** Ray/AABB against the filtered boxes — exact, and cheaper than mesh picking. */
function onPick(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(
    ((ev.clientX - r.left) / r.width) * 2 - 1,
    -((ev.clientY - r.top) / r.height) * 2 + 1), camera);
  const ro = raycaster.ray.origin, rd = raycaster.ray.direction;
  let best = null, bestT = Infinity;
  for (const b of visibleBoxes) {
    if (b.kind === 'ours' && !$('lOurs').checked) continue;
    if (b.kind === 'gt' && !$('lGt').checked) continue;
    let t0 = -Infinity, t1 = Infinity, ok = true;
    for (let a = 0; a < 3; a++) {
      const d = rd.getComponent(a), o = ro.getComponent(a);
      if (Math.abs(d) < 1e-9) { if (o < b.o.lo[a] || o > b.o.hi[a]) { ok = false; break; } continue; }
      let ta = (b.o.lo[a] - o) / d, tb = (b.o.hi[a] - o) / d;
      if (ta > tb) [ta, tb] = [tb, ta];
      t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
      if (t0 > t1) { ok = false; break; }
    }
    if (ok && t1 > 0 && t0 < bestT) { bestT = t0; best = b; }
  }
  picked = best ? best.o : null;
  const box = $('pick');
  if (!best) { box.style.display = 'none'; rebuildBoxes(); return; }
  const o = best.o, sz = [0, 1, 2].map((i) => (o.hi[i] - o.lo[i]).toFixed(2));
  const rows = [['source', best.kind === 'ours' ? 'ours (fused)' : 'ground truth'],
                ['label', data.labels[o.l]],
                ['size m', sz.join(' × ')],
                ['centre', o.c ? o.c.map((v) => v.toFixed(2)).join(', ') : '—']];
  if (o.s !== undefined) rows.push(['score', o.s]);
  if (o.n_obs !== undefined) rows.push(['obs / pts', `${o.n_obs} / ${o.n_pts}`]);
  box.innerHTML = rows.map(([k, v]) =>
    `<div><span>${k}</span><span>${esc(v)}</span></div>`).join('');
  box.style.display = 'block';
  rebuildBoxes();
}

// ---------------------------------------------------------------------------
// camera + ui
// ---------------------------------------------------------------------------

/** Frame the box so it fills the viewport, honouring both fov and aspect. */
function frame(box, dir) {
  const lo = box.min, hi = box.max;
  const c = [0, 1, 2].map((i) => (lo[i] + hi[i]) / 2);
  const size = [0, 1, 2].map((i) => Math.max(hi[i] - lo[i], 0.1));
  const radius = Math.hypot(size[0], size[1], size[2]) / 2;
  const vFov = THREE.MathUtils.degToRad(camera.fov);
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
  // The bounding *sphere* of a wide flat room is a lot bigger than its
  // silhouette from an oblique angle, so back off the textbook distance.
  const dist = 0.85 * radius / Math.sin(Math.min(vFov, hFov) / 2);

  controls.target.set(c[0], c[1], c[2]);
  camera.position.set(c[0] + dir[0] * dist, c[1] + dir[1] * dist,
                      c[2] + dir[2] * dist);
  camera.near = Math.max(dist / 1000, 0.02);
  camera.far = dist * 10 + radius * 4;
  camera.updateProjectionMatrix();
  controls.update();
}

const norm3 = (v) => {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
};

function fitView() {
  camera.up.set(0, $('yUp').checked ? 1 : 0, $('yUp').checked ? 0 : 1);
  frame(data.fit || data.bounds, norm3([0.55, -0.72, 0.42]));
}

function topView() {
  // Looking straight down the up-axis is degenerate for OrbitControls, so the
  // top view borrows a different up rather than nudging the eye off-axis.
  camera.up.set(0, 1, 0);
  frame(data.fit || data.bounds, [0, 0.001, 1]);
}

function wireUi() {
  const redrawBoxes = () => rebuildBoxes();
  const redrawFrame = () => showFrame(frameIdx);

  $('frame').oninput = (e) => showFrame(parseInt(e.target.value, 10));
  $('prev').onclick = () => showFrame(frameIdx - 1);
  $('next').onclick = () => showFrame(frameIdx + 1);
  $('play').onclick = (e) => { playing = !playing; e.target.classList.toggle('on', playing);
                               e.target.textContent = playing ? 'Pause' : 'Play'; };
  $('fit').onclick = () => fitView();
  $('top').onclick = () => topView();

  for (const [id, obj] of [['lWorld', gWorld], ['lGtCloud', gGtCloud],
                           ['lScan', gScan], ['lDetPts', gDet],
                           ['lDetBox', gDetBox], ['lOurs', gOurs], ['lGt', gGt],
                           ['lPath', gPath]]) {
    $(id).onchange = () => { obj.visible = $(id).checked; updateHud(); };
    obj.visible = $(id).checked;
  }
  $('objMode').onchange = () => redrawBoxes();
  $('gtObb').onchange = () => redrawBoxes();
  $('q').oninput = () => { redrawBoxes(); redrawFrame(); };
  $('hideStruct').onchange = () => { redrawBoxes(); redrawFrame(); };
  $('score').oninput = (e) => { $('scoreLbl').textContent = (+e.target.value).toFixed(2);
                                redrawBoxes(); redrawFrame(); };
  $('minObs').oninput = (e) => { $('obsLbl').textContent = e.target.value; redrawBoxes(); };
  $('ps').oninput = (e) => {
    $('psLbl').textContent = (+e.target.value).toFixed(1);
    for (const p of [gWorld, gGtCloud, gScan, gDet]) p.material.size = +e.target.value;
  };
  $('op').oninput = (e) => {
    $('opLbl').textContent = (+e.target.value).toFixed(2);
    gWorld.material.opacity = +e.target.value;
    gGtCloud.material.opacity = Math.min(1, +e.target.value + 0.2);
  };
  $('yUp').onchange = (e) => {
    camera.up.set(0, e.target.checked ? 1 : 0, e.target.checked ? 0 : 1);
    controls.update();
  };

  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
    const layer = { '1': 'lWorld', '2': 'lScan', '3': 'lDetPts', '4': 'lDetBox',
                    '5': 'lOurs', '6': 'lGt', '7': 'lPath', '8': 'lGtCloud',
                    'l': 'lLabels', 'L': 'lLabels' }[e.key];
    if (layer) { const c = $(layer); c.checked = !c.checked;
                 c.onchange && c.onchange(); e.preventDefault(); return; }
    if (e.key === 'ArrowLeft') showFrame(frameIdx - 1);
    else if (e.key === 'ArrowRight') showFrame(frameIdx + 1);
    else if (e.key === 'f' || e.key === 'F') fitView();
    else if (e.key === ' ') { $('play').onclick({ target: $('play') }); e.preventDefault(); }
  });
}

function tick(now) {
  requestAnimationFrame(tick);
  if (playing && data && now - lastStep > 90) {
    lastStep = now;
    showFrame(frameIdx + 1 >= data.frames.length ? 0 : frameIdx + 1);
  }
  controls.update();
  renderer.render(scene, camera);
  drawLabels();
}

init();
