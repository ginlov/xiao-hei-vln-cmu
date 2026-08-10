"""Streamlit + plotly viewer for the perception benchmark debug data.

Loads the per-scene dumps from dump_debug.py and shows, interactively:
  - the mask-overlay image (phases 1+2: input + detect),
  - two 3D views: the CUMULATIVE scene graph with the accumulated LiDAR cloud,
    and what THIS viewpoint contributed on its own — with a viewpoint slider
    so you can watch the graph build up,
  - the per-viewpoint detection/lift table,
  - a COMPARE tab putting several dump directories side by side at the same
    viewpoint, for eyeballing a parameter sweep (sweep_keyframes.sh).

No sidecar needed (reads the dumped data).

    uv run --with streamlit --with plotly streamlit run perception_benchmark/viz_app.py

PERCEPTION_DEBUG_DIR selects the dump directory; its *siblings* that also
contain dumps are offered in the sidebar, so a sweep laid out as
debug_k0/ … debug_k10/ is browsable without restarting the app.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# The sidecar's geometry (perception/geometry.py at the repo root) builds the
# face-unwrap LUTs; make it importable when streamlit runs from perception_benchmark/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Optional: clicking INSIDE a 3D scene. st.plotly_chart's on_select cannot do
# this — plotly has no selection layer for scatter3d — but plotly's raw
# `plotly_click` does fire in 3D, and this component forwards it. Its frontend
# bundle predates Streamlit 1.60, so treat it as experimental and keep the 2D
# map working as the reliable path.
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:                                  # noqa: BLE001
    HAS_PLOTLY_EVENTS = False

MAX_CLOUD_PTS = 40000          # plotly gets sluggish much beyond this
DEBUG_DIR = Path(os.environ.get("PERCEPTION_DEBUG_DIR",
                                "perception_benchmark/debug"))
# Only consulted for headings, and only when the dump predates viz.json
# carrying a "yaw" field (see _yaws below).
CAP_DIR = Path(os.environ.get("PERCEPTION_CAP_DIR",
                              "perception_benchmark/captures_nav"))
HEADING_LEN_M = 1.0            # length of the drawn heading arrow
VIDEO_DIR = Path(os.environ.get("PERCEPTION_VIDEO_DIR",
                                "perception_benchmark/videos"))

st.set_page_config(page_title="Perception benchmark debug", layout="wide")


def _natkey(p: str):
    """Sort debug_k2 before debug_k10 — a sweep is read in numeric order."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]


def available_runs() -> list[str]:
    """Dump directories sitting alongside the selected one.

    A sweep writes debug_k0/ … debug_k10/ next to each other, so the siblings
    of PERCEPTION_DEBUG_DIR are exactly the runs worth comparing. The selected
    directory is always included even if it is the only one.
    """
    runs = {str(DEBUG_DIR)}
    for p in glob.glob(str(DEBUG_DIR.parent / "*" / "*" / "viz.json")):
        runs.add(str(Path(p).parent.parent))
    return sorted(runs, key=_natkey)


def _box_edges(bmin, bmax):
    x0, y0, z0 = bmin; x1, y1, z1 = bmax
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in e:
        xs += [c[a][0], c[b][0], None]; ys += [c[a][1], c[b][1], None]; zs += [c[a][2], c[b][2], None]
    return xs, ys, zs


def _boxes_trace(boxes, color, name, width=3):
    xs, ys, zs, texts = [], [], [], []
    for b in boxes:
        x, y, z = _box_edges(b["bmin"], b["bmax"])
        xs += x; ys += y; zs += z
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=color, width=width), name=name, hoverinfo="name")


def _highlight_traces(box, color="gold"):
    """A selected box, drawn to be unmissable in 3D.

    scatter3d ignores `line.width` (WebGL renders 3D lines ~1px regardless), so
    a thicker outline is not an option — the highlight has to be a solid
    translucent volume plus a large centre marker.
    """
    x0, y0, z0 = box["bmin"]
    x1, y1, z1 = box["bmax"]
    pad = 0.02                                   # lift off the crimson wireframe
    x0, y0, z0 = x0 - pad, y0 - pad, z0 - pad
    x1, y1, z1 = x1 + pad, y1 + pad, z1 + pad
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return [
        go.Mesh3d(
            x=[v[0] for v in verts], y=[v[1] for v in verts], z=[v[2] for v in verts],
            i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
            j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
            k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
            color=color, opacity=0.35, flatshading=True,
            name="selected", showlegend=True, hoverinfo="name"),
        go.Scatter3d(
            x=[box["center"][0]], y=[box["center"][1]], z=[box["center"][2]],
            mode="markers+text", marker=dict(size=12, color=color, symbol="diamond"),
            text=[box["label"]], textposition="top center",
            textfont=dict(color=color, size=13),
            name="selected centre", showlegend=False, hoverinfo="text"),
    ]


def _labels_trace(boxes, color, name, size=3, show_ids=False):
    """Centre markers labelled with the node id.

    ``#<id>`` is the same number ``dump_debug`` draws beside the mask on the 2D
    overlay, so a detection in the image can be matched to its box here. Ids
    are drawn as text (not just hover) because matching is the point — you need
    to read both panels at once.
    """
    def _tag(b):
        # Per-viewpoint boxes carry cum_ids: their own node_id belongs to a
        # different map and would read as a false match against the crimson
        # ids. Several ids means this frame's blob spans several cumulative
        # nodes — shown as #a+#b rather than hidden.
        if "cum_ids" in b:
            ids = b["cum_ids"]
            return ("+".join(f"#{i}" for i in ids) + " ") if ids else ""
        nid = b.get("node_id")
        return f"#{nid} " if nid is not None else ""

    return go.Scatter3d(
        x=[b["center"][0] for b in boxes], y=[b["center"][1] for b in boxes],
        z=[b["center"][2] for b in boxes],
        mode="markers+text" if show_ids else "markers",
        marker=dict(size=size, color=color),
        text=[f"{_tag(b)}{b['label']}"
              + (f" (n={b.get('n_obs')})" if "n_obs" in b else "") for b in boxes],
        textposition="top center", textfont=dict(size=9, color=color),
        hoverinfo="text", name=name)


def _match_in(target, node):
    """Nearest counterpart to `node` among `target`, preferring the same label.

    Per-viewpoint node ids are local and do NOT correspond to cumulative ids,
    so the link has to be geometric.
    """
    if not target or node is None:
        return None
    c = np.array(node["center"])
    same = [b for b in target if b["label"] == node["label"]] or target
    return min(same, key=lambda b: float(np.linalg.norm(np.array(b["center"]) - c)))


@st.cache_data
def _yaws(run: str, scene: str, n: int):
    """Robot heading per viewpoint, in radians.

    Prefers the ``yaw`` written into viz.json. Dumps made before that field
    existed fall back to the capture manifest, which records a yaw per frame.

    Heading is deliberately *not* derived from consecutive path points: a
    navigation capture is stationary for most of its frames, and a zero-length
    step has no direction — the arrow would spin randomly exactly where it
    matters most.
    """
    have = [v.get("yaw") for v in load_scene(run, scene)["viewpoints"]]
    if all(y is not None for y in have) and have:
        return have
    man = CAP_DIR / scene / "manifest.json"
    if man.is_file():
        try:
            frames = json.loads(man.read_text()).get("viewpoints", [])
            if len(frames) >= n and all("yaw" in f for f in frames[:n]):
                return [float(f["yaw"]) for f in frames[:n]]
        except (ValueError, KeyError, TypeError):
            pass
    return None


def _heading_traces(pose, yaw, length=HEADING_LEN_M):
    """Shaft + cone showing which way the robot faces.

    Yaw is a rotation about +z, so the heading is (cos, sin, 0) in the map
    frame — the vehicle's forward axis, which is what "where is it looking"
    means for a robot carrying a 360° camera.
    """
    x, y, z = pose
    dx, dy = float(np.cos(yaw)), float(np.sin(yaw))
    tipx, tipy = x + dx * length, y + dy * length
    return [
        go.Scatter3d(
            x=[x, tipx], y=[y, tipy], z=[z, z], mode="lines",
            line=dict(color="red", width=6), name="heading", hoverinfo="skip"),
        go.Cone(
            x=[tipx], y=[tipy], z=[z], u=[dx], v=[dy], w=[0.0],
            sizemode="absolute", sizeref=length * 0.45, anchor="tail",
            showscale=False, colorscale=[[0, "red"], [1, "red"]],
            name="heading", hoverinfo="skip"),
    ]


@st.cache_data
def load_scene(run, scene):
    data = json.load(open(Path(run) / scene / "viz.json"))
    return data


@st.cache_data(persist="disk")
def run_summary(run: str, scene: str) -> dict | None:
    """Scalars only, so a whole sweep can be tabulated without holding every
    viz.json in the cache — they run 8–15 MB each.

    Persisted to disk: the table covers *every* run, so a cold render parses
    the whole sweep (~10 s for 11 dumps). Paying that once per machine rather
    than once per app start is worth the staleness risk — re-dumping a run
    means clearing the cache from the Streamlit menu.

    Returns None when this run has no dump for the scene, which is normal:
    the sweep may have been driven with a subset of scenes.
    """
    f = Path(run) / scene / "viz.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    vps = d["viewpoints"]
    return {
        "run": Path(run).name,
        "frames": len(vps),
        "dets": sum(v["n_det"] for v in vps),
        "lifts": sum(v["n_lift"] for v in vps),
        "final nodes": len(vps[-1]["nodes"]) if vps else 0,
        "gt": len(d.get("gt", [])),
        "params": d.get("params", {}),
    }


def available_scenes(run=None):
    root = Path(run) if run else DEBUG_DIR
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(str(root / "*" / "viz.json")))


# matplotlib tab20, inlined so the viewer stays streamlit+plotly+numpy only.
# The pre-baked overlay colours detections by tab20(i % 20); we match it so the
# live re-composite reads the same as the PNG.
_TAB20 = [
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
]


@st.cache_data(max_entries=12)
def _frame_masks(cap_dir: str, scene: str, vpid: str):
    """Raw RGB image + per-detection (label, yolo, sam, mask, orig_idx).

    Read from the captures the viewer already points at — image.npy is the raw
    equirect, detections.npz the frozen masks — so the 2D overlay can be
    re-composited for a chosen class set without re-dumping. ``sam`` is SAM's
    predicted mask IoU (1.0 for pre-B5 dumps that lack the field). Cached to a
    dozen frames so scrubbing stays responsive without holding the whole scene
    (each mask stack is tens of MB). Returns (None, []) when the frame lacks a
    frozen detection file.
    """
    base = Path(cap_dir) / scene / vpid
    img_f, det_f = base / "image.npy", base / "detections.npz"
    if not (img_f.is_file() and det_f.is_file()):
        return None, []
    bgr = np.load(img_f)
    rgb = bgr[:, :, ::-1] if bgr.ndim == 3 else bgr
    z = np.load(det_f, allow_pickle=True)
    n = len(z["labels"])
    sam = z["sam_scores"] if "sam_scores" in z.files else np.ones(n, np.float32)
    dets = [(str(lbl), float(sc), float(ss), m, i)
            for i, (m, lbl, sc, ss) in enumerate(
                zip(z["masks"], z["labels"], z["scores"], sam))]
    return rgb, dets


def _composite_overlay(rgb, dets):
    """Blend each detection's mask onto ``rgb`` (0.5 alpha, tab20 by orig
    index) and return the uint8 image plus label anchors, mirroring
    debug_viewpoint._overlay_masks so the live view matches the baked PNG.

    The label carries both scores — ``Y`` YOLO box confidence, ``S`` SAM mask
    quality — so mask fit is visible alongside detection confidence."""
    out = rgb.astype(np.float32) / 255.0
    anchors = []
    for label, score, sam, mask, orig in dets:
        color = np.array(_TAB20[orig % 20], dtype=np.float32) / 255.0
        m = np.asarray(mask, dtype=bool)
        out[m] = 0.5 * out[m] + 0.5 * color
        ys, xs = np.nonzero(m)
        if len(xs):
            anchors.append((int(xs.mean()), int(ys.mean()),
                            f"{label} Y{score:.2f} S{sam:.2f}",
                            _TAB20[orig % 20]))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), anchors


# Faces are built in perception/geometry.FACE_YAWS order [0, 90, 180, 270] =
# front, right, back, left. Labelled by that index.
_FACE_LABELS = {0: "Front 0°", 1: "Right 90°", 2: "Back 180°", 3: "Left 270°"}
# Display left-to-right as they appear across the concatenated equirect —
# verified by each face's centre column (back@0, left@480, front@960, right@1440),
# so the strip reads back | left | front | right, with front centred.
_FACE_ORDER_LR = [2, 3, 0, 1]


@st.cache_resource
def _forward_luts():
    """Per-face integer sample indices into the equirect, built once.

    ``build_forward_luts`` returns the same (map_x, map_y) the sidecar feeds
    ``cv2.remap`` to unwrap each 640² face from the panorama. Rounded to int
    here for a pure-numpy nearest-neighbour gather — no cv2 in the viewer. The
    faces this produces are pixel-faithful to what YOLO/SAM actually ran on.
    """
    from perception.geometry import EQUIRECT_H, EQUIRECT_W, build_forward_luts
    luts = []
    for map_x, map_y in build_forward_luts():
        xi = np.clip(np.rint(map_x).astype(np.int32), 0, EQUIRECT_W - 1)
        yi = np.clip(np.rint(map_y).astype(np.int32), 0, EQUIRECT_H - 1)
        luts.append((yi, xi))
    return luts


def _unwrap_faces(equirect_rgb):
    """Unwrap an equirect image (raw or already mask-composited) into the four
    perspective faces, so masks drawn on the panorama carry onto the faces."""
    return [equirect_rgb[yi, xi] for yi, xi in _forward_luts()]


@st.cache_data
def class_options(run: str, scene: str) -> list[str]:
    """Every label that appears anywhere in the run — predicted or GT.

    Taken over all viewpoints rather than the last one: same-label NMS can
    remove a node late, and a class you can no longer see at the end is
    exactly the one worth filtering to.
    """
    d = load_scene(run, scene)
    return sorted({n["label"] for v in d["viewpoints"] for n in v["nodes"]}
                  | {g["label"] for g in d["gt"]})


def _keep(items, classes):
    """Filter boxes/detections by label; an empty selection means all."""
    return items if not classes else [x for x in items if x["label"] in classes]


def _common_traces(data, vps, idx, yaws, *, path_to_here, show_gt, show_path,
                   show_heading, classes=()):
    """GT boxes, robot path and marker — shared by every 3D panel."""
    vp = vps[idx]
    t = []
    gt = _keep(data["gt"], classes)
    if show_gt and gt:
        t.append(_boxes_trace(gt, "rgba(80,80,80,0.5)", "GT boxes", 2))
        t.append(_labels_trace(gt, "gray", "GT labels"))
    if show_path and path_to_here:
        poses = np.array([v["pose"] for v in vps[:idx + 1]])
        if len(poses):
            t.append(go.Scatter3d(x=poses[:, 0], y=poses[:, 1], z=poses[:, 2],
                     mode="lines+markers", line=dict(color="orange", width=3),
                     marker=dict(size=3, color="orange"), name="robot path"))
    t.append(go.Scatter3d(x=[vp["pose"][0]], y=[vp["pose"][1]], z=[vp["pose"][2]],
             mode="markers", marker=dict(size=7, color="red"),
             name="robot (this vp)"))
    if show_heading and yaws is not None:
        t += _heading_traces(vp["pose"], yaws[idx])
    return t


@st.cache_data
def _new_node_frames(run: str, scene: str, classes: tuple) -> list[int]:
    """Viewpoints where the cumulative graph gained a node of these classes.

    On a 316-frame capture almost nothing happens at most viewpoints — the
    robot is stationary for two thirds of `arabic_room`. These are the frames
    worth landing on, and hunting for them by dragging a slider is hopeless.
    """
    d = load_scene(run, scene)
    counts = [len(_keep(v["nodes"], classes)) for v in d["viewpoints"]]
    return [i for i in range(1, len(counts)) if counts[i] > counts[i - 1]]


def _viewpoint_nav(run: str, scene: str, vps: list, classes: tuple) -> int:
    """Slider + stepper + event jumps, returning the chosen viewpoint index.

    The index lives in session_state so the buttons can move it. Every button
    writes it from an `on_click` callback, which is the only place Streamlit
    allows a widget-backed key to be assigned.
    """
    n = len(vps)
    key = "vp_idx"
    # Switching run or scene can shorten the capture; a stale index would make
    # the slider raise before anything renders.
    if st.session_state.get(key) is None or st.session_state[key] > n - 1:
        st.session_state[key] = 0

    def _go(target: int) -> None:
        st.session_state[key] = int(min(max(target, 0), n - 1))

    def _step(delta: int) -> None:
        _go(st.session_state[key] + delta)

    def _event(delta: int) -> None:
        here = st.session_state[key]
        marks = _new_node_frames(run, scene, classes)
        nxt = ([m for m in marks if m > here] if delta > 0 else
               [m for m in reversed(marks) if m < here])
        if nxt:
            _go(nxt[0])

    st.sidebar.markdown("**Viewpoint**")
    st.sidebar.slider("Viewpoint", 0, n - 1, key=key, format="vp %d",
                      label_visibility="collapsed")

    b = st.sidebar.columns(4)
    b[0].button("⏮", on_click=_go, args=(0,), help="first",
                use_container_width=True)
    b[1].button("◀", on_click=_step, args=(-1,), help="previous viewpoint",
                use_container_width=True)
    b[2].button("▶", on_click=_step, args=(1,), help="next viewpoint",
                use_container_width=True)
    b[3].button("⏭", on_click=_go, args=(n - 1,), help="last",
                use_container_width=True)

    # A separate key: two widgets cannot share one, so this box does not track
    # the slider. It is a "go to" action, not a second display of the index —
    # hence the label.
    st.sidebar.number_input(
        "jump to vp #", 0, n - 1, key="vp_goto", step=1,
        on_change=lambda: _go(int(st.session_state["vp_goto"])),
        help="Type an index and press Enter. Faster than dragging when you "
             "already know the viewpoint you want.")

    marks = _new_node_frames(run, scene, classes)
    what = ", ".join(classes) if classes else "any class"
    e = st.sidebar.columns(2)
    e[0].button("◀ new node", on_click=_event, args=(-1,),
                use_container_width=True,
                help=f"previous viewpoint that added a node ({what})")
    e[1].button("new node ▶", on_click=_event, args=(1,),
                use_container_width=True,
                help=f"next viewpoint that added a node ({what})")

    i = st.session_state[key]
    gained = len(_keep(vps[i]["nodes"], classes)) - (
        len(_keep(vps[i - 1]["nodes"], classes)) if i else 0)
    st.sidebar.caption(
        f"`{vps[i]['id']}` · {i + 1} of {n} · "
        f"{len(_keep(vps[i]['nodes'], classes))} nodes"
        + (f" (**+{gained}** here)" if gained > 0 else "")
        + f"  ·  {len(marks)} growth frames")
    return i


def _object_points(run, scene, vps, idx):
    """Union of the lifted inliers from every viewpoint up to `idx`."""
    chunks = []
    for v in vps[:idx + 1]:
        f = Path(run) / scene / f"{v['id']}_pts.npy"
        if f.exists():
            chunks.append(np.load(f))
    pts = np.vstack(chunks) if chunks else np.empty((0, 3))
    if len(pts) > MAX_CLOUD_PTS:                     # keep plotly responsive
        pts = pts[np.random.default_rng(0).choice(len(pts), MAX_CLOUD_PTS, False)]
    return pts


# ---- sidebar ----
st.sidebar.title("Perception debug")
runs = available_runs()
run = st.sidebar.selectbox(
    "Dump directory", runs, index=runs.index(str(DEBUG_DIR)),
    format_func=lambda r: Path(r).name,
    help="Siblings of PERCEPTION_DEBUG_DIR that contain dumps. A parameter "
         "sweep writes one per setting — compare them in the Compare tab.")
RUN_DIR = Path(run)

scenes = available_scenes(run)
if not scenes:
    st.error("No debug dumps found. Run: uv run --extra perception python "
             "perception_benchmark/dump_debug.py --all"); st.stop()

scene = st.sidebar.selectbox("Scene", scenes)
data = load_scene(run, scene)
vps = data["viewpoints"]
n = len(vps)
yaws = _yaws(run, scene, n)

# The class filter is read by the viewpoint navigator below ("next new node"),
# so it has to be chosen first. It also reads better this way: pick what you
# are looking at, then move through it.
classes = st.sidebar.multiselect(
    "Classes (phases 3 & 4)", class_options(run, scene), default=[],
    help="Restrict the two 3D panels, the picker, the lift-input list and the "
         "detection table to these labels. Empty = every class. The "
         "accumulated point cloud is NOT filtered — lifted points carry no "
         "label once they are merged into the cloud.")

idx = _viewpoint_nav(run, scene, vps, tuple(classes))

cloud_mode = st.sidebar.selectbox(
    "Cumulative panel cloud", ["object points", "full scene", "none"],
    help="object points = the lifted inliers behind the predicted boxes; "
         "full scene = every LiDAR sweep, including surfaces nothing was "
         "detected on (needs dump_cloud.py).")
show_ids = st.sidebar.checkbox(
    "Node ids on boxes", value=True,
    help="Print #N beside each cumulative box. dump_debug draws the same #N "
         "next to the mask on the 2D overlay, so the two panels can be "
         "matched by eye. Needs a dump made after this feature landed.")
show_heading = st.sidebar.checkbox(
    "Heading arrow", value=True,
    help="Red arrow from the robot showing which way it faces at this "
         "viewpoint.")
if show_heading and yaws is None:
    st.sidebar.caption(
        "_No heading available: this dump predates the `yaw` field and no "
        f"manifest was found under `{CAP_DIR}`. Re-run dump_debug.py, or point "
        "PERCEPTION_CAP_DIR at the captures._")
show_gt = st.sidebar.checkbox("GT boxes", value=True)
show_pred = st.sidebar.checkbox("Predicted boxes", value=True)
show_pts = st.sidebar.checkbox("Lifted points (this vp)", value=True,
    help="Blue points in the per-viewpoint panel.")
show_path = st.sidebar.checkbox("Robot path", value=True)
click3d = HAS_PLOTLY_EVENTS and st.sidebar.checkbox(
    "Click inside the 3D view (experimental)", value=False,
    help="Uses streamlit-plotly-events. If the plot renders blank, turn this "
         "off — the 2D map above the panels always works.")
if not HAS_PLOTLY_EVENTS:
    st.sidebar.caption("_3D clicking needs `--with streamlit-plotly-events`_")
    click3d = False
st.sidebar.caption(f"params: {data['params']}")

tab_graph, tab_2d, tab_cmp = st.tabs(
    ["Scene graph", "2D detector", f"Compare runs ({len(runs)})"])

with tab_graph:
    vp = vps[idx]
    # Phases 3 and 4 work off these; phases 1-2 are the raw frame and stay
    # whole, since the overlay image is a PNG the dump already baked.
    cum_nodes = _keep(vp["nodes"], classes)
    gt_boxes = _keep(data["gt"], classes)
    st.title(f"{scene} — {vp['id']}"
             + (f"  ·  {', '.join(classes)}" if classes else ""))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("detections", vp["n_det"])
    c2.metric("lifted", vp["n_lift"])
    c3.metric("nodes (cumulative)", len(cum_nodes),
              delta=(f'of {len(vp["nodes"])} all classes' if classes else
                     (f'{len(vp["nodes_vp"])} this vp'
                      if "nodes_vp" in vp else None)),
              delta_color="off")
    c4.metric("GT (scoreable)", len(gt_boxes),
              delta=(f"of {len(data['gt'])}" if classes else None),
              delta_color="off")

    # ---- phase 1+2: overlay image ----
    st.subheader("Phase 1–2 · input image + model detections vs ground truth")
    st.caption("**Coloured masks + white labels = MODEL DETECTIONS** (what perception found). "
               "**Lime ◇ + green labels = GROUND TRUTH** objects (from object_list) projected "
               "into this view. `[OK]`/`[x]` on a detection = whether it produced a 3D point "
               "(cleared min_inliers). Compare the two: GT with no overlapping mask = a miss.")
    img_path = RUN_DIR / scene / f"{vp['id']}.png"

    # The baked PNG carries every class. When a class filter is set we can
    # re-composite the raw frame with only those masks — same blend as the dump,
    # done live from the captures. A toggle keeps the (faster) PNG available and
    # lets you filter the image even with no sidebar class selected.
    rgb2d, dets2d = _frame_masks(str(CAP_DIR), scene, vp["id"])
    can_recomp = rgb2d is not None
    recomp = st.checkbox(
        "Re-composite overlay (filter masks by class)", value=bool(classes),
        disabled=not can_recomp,
        help="Redraw the overlay from the raw image with only the sidebar's "
             "classes. Off = the pre-baked PNG (all classes). Needs the frozen "
             "detections.npz next to the captures.")
    if not can_recomp and classes:
        st.caption(f"_No frozen detections under `{CAP_DIR}/{scene}/{vp['id']}` "
                   "— showing the all-class PNG. Run dump_detections.py to enable "
                   "class filtering here._")

    # The equirect the faces get unwrapped from: the filtered composite when
    # re-compositing, else the raw frame. So masks/filter carry onto the faces.
    equirect_src = None
    if recomp and can_recomp:
        shown = [d for d in dets2d if not classes or d[0] in classes]
        overlay, anchors = _composite_overlay(rgb2d, shown)
        equirect_src = overlay
        fig2d = go.Figure(go.Image(z=overlay))
        if anchors:
            fig2d.add_trace(go.Scatter(
                x=[a[0] for a in anchors], y=[a[1] for a in anchors],
                mode="markers+text",
                marker=dict(size=6, color=[f"rgb{a[3]}" for a in anchors]),
                text=[a[2] for a in anchors], textposition="top center",
                textfont=dict(size=9, color="white"), hoverinfo="text",
                name="detections"))
        fig2d.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False,
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False))
        fig2d.update_xaxes(range=[0, rgb2d.shape[1]])
        fig2d.update_yaxes(range=[rgb2d.shape[0], 0])
        st.plotly_chart(fig2d, use_container_width=True, key="overlay2d")
        st.caption(f"_Re-composited live · {len(shown)} of {len(dets2d)} "
                   "detections shown · zoom/pan enabled. GT diamonds are only on "
                   "the baked PNG (toggle off to see them)._")
    else:
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        if can_recomp:
            equirect_src = rgb2d      # faces off the raw frame (no masks)

    # ---- the 4 perspective faces the detector actually ran on ----
    # Reconstructed from the equirect via the sidecar's forward LUTs (faces are
    # not stored anywhere) — pixel-faithful to YOLO/SAM's input. Unwrapping the
    # *composited* equirect above means the class-filtered masks appear on them.
    show_faces = st.checkbox(
        "Show 4 face views (front / right / back / left)", value=False,
        disabled=equirect_src is None,
        help="The 100° perspective crops the panorama is split into before "
             "detection. Adjacent faces overlap ~10°, so a seam object shows "
             "in two. Inherits the class filter / re-composite above.")
    if show_faces and equirect_src is not None:
        faces = _unwrap_faces(equirect_src)
        for col, fi in zip(st.columns(4), _FACE_ORDER_LR, strict=True):
            col.image(faces[fi], caption=_FACE_LABELS[fi],
                      use_container_width=True)

    # ---- animations built by make_video.py ----
    # The slider shows one viewpoint; these show the graph assembling itself,
    # which is where a class that fragments gives itself away.
    vids = sorted(VIDEO_DIR.glob(f"{scene}_*.mp4")) + \
        sorted(VIDEO_DIR.glob(f"{scene}_*.gif"))
    with st.expander(f"Animations ({len(vids)})", expanded=False):
        if not vids:
            st.caption(
                "_None yet. Build one with:_\n\n"
                "`uv run --with imageio --with imageio-ffmpeg python "
                "perception_benchmark/make_video.py --scene "
                f"{scene} --labels \"potted plant\" --run {run}`")
        else:
            pick = st.selectbox("Clip", vids, format_func=lambda p: p.name,
                                key=f"vid_{scene}")
            if pick.suffix == ".mp4":
                st.video(str(pick))          # scrubber, pause, speed
            else:
                st.image(str(pick))          # GIFs just loop
            # The host is headless, so the only ways to see a clip are this
            # browser tab or your own machine. If the embedded player stays
            # blank — a browser without an H.264 decoder will do that, and
            # Firefox on Linux needs a system ffmpeg for one — download it
            # here and play it locally, or build the GIF variant, which is
            # just an image and cannot fail to render.
            st.download_button(f"Download {pick.name}", pick.read_bytes(),
                               file_name=pick.name,
                               mime=("video/mp4" if pick.suffix == ".mp4"
                                     else "image/gif"))
            st.caption(f"`{pick}` · {pick.stat().st_size / 1e6:.1f} MB. Player "
                       "blank? Download it, or re-render with `--format gif` "
                       "(image, always renders) or `--compat` (H.264 baseline "
                       "profile).")

    # ---- phase 3+4: two 3D panels, cumulative vs this viewpoint ----
    def _panel_traces(*, path_to_here: bool):
        return _common_traces(data, vps, idx, yaws, path_to_here=path_to_here,
                              show_gt=show_gt, show_path=show_path,
                              show_heading=show_heading, classes=classes)

    def _render(traces, key):
        fig = go.Figure(traces)
        fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0),
                          scene=dict(aspectmode="data", xaxis_title="x",
                                     yaxis_title="y", zaxis_title="z"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.0))
        st.plotly_chart(fig, use_container_width=True, key=key)

    # ── A. cumulative ───────────────────────────────────────────────────────
    st.subheader("Phase 4 · Cumulative scene graph")
    st.caption("Every observation through this viewpoint, fused. Blue points are the "
               "**lifted inliers** accumulated so far — the evidence behind the boxes, "
               "so a box floating clear of them is unsupported. Switch the sidebar to "
               "*full scene* to see surfaces nothing was detected on.")
    if classes:
        st.caption(f"_Filtered to **{', '.join(classes)}** — "
                   f"{len(cum_nodes)} of {len(vp['nodes'])} nodes shown. The "
                   "point cloud is unfiltered (points carry no label once "
                   "merged)._")
    traces = _panel_traces(path_to_here=True)
    if cloud_mode == "object points":
        # Union of the lifted inliers from every viewpoint so far: the evidence
        # the boxes were actually built from, not the whole room.
        pts = _object_points(run, scene, vps, idx)
        if len(pts):
            traces.insert(0, go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                marker=dict(size=1.6, color="royalblue", opacity=0.45),
                name=f"object points ({len(pts)})", hoverinfo="skip"))
    elif cloud_mode == "full scene":
        cloud_path = RUN_DIR / scene / "cloud.npz"
        if cloud_path.exists():
            C = np.load(cloud_path)
            pts = C["points"][C["first_vp"] <= idx]
            if len(pts) > MAX_CLOUD_PTS:
                pts = pts[np.random.default_rng(0).choice(len(pts), MAX_CLOUD_PTS, False)]
            if len(pts):
                traces.insert(0, go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                    marker=dict(size=1, color="lightgray", opacity=0.35),
                    name=f"scene cloud ({len(pts)})", hoverinfo="skip"))
        else:
            st.caption("_No scene cloud — run "
                       "`uv run python perception_benchmark/dump_cloud.py --all`_")
    # plotly's scatter3d has no selection layer (3D drag modes are orbit/zoom/pan
    # only), so cross-panel linking is driven by an explicit picker rather than a
    # click or hover on the plot itself.
    vp_nodes = _keep(vps[idx].get("nodes_vp", []), classes)
    st.markdown("**Inspect an object** — click a dot on the map, or use the list. "
                "The choice is highlighted in gold in *both* panels below.")
    pick_l, pick_r = st.columns([2, 1])

    # Clicking inside a 3D scene is impossible: plotly has no selection layer for
    # scatter3d (its 3D drag modes are orbit/turntable/zoom/pan only), so
    # st.plotly_chart's on_select never fires there. 2D charts DO support it, so
    # the click target is this top-down map of the same nodes.
    with pick_l:
        if vp_nodes:
            cx = [n["center"][0] for n in vp_nodes]
            cy = [n["center"][1] for n in vp_nodes]
            fmap = go.Figure([
                go.Scatter(x=[v["pose"][0] for v in vps[:idx + 1]],
                           y=[v["pose"][1] for v in vps[:idx + 1]],
                           mode="lines+markers", line=dict(color="orange", width=2),
                           marker=dict(size=5, color="orange"), name="robot path",
                           hoverinfo="skip"),
                go.Scatter(x=cx, y=cy, mode="markers",
                           marker=dict(size=15, color="mediumseagreen",
                                       line=dict(color="white", width=1)),
                           customdata=list(range(len(vp_nodes))),
                           text=[f"{i}: {n['label']}" for i, n in enumerate(vp_nodes)],
                           hoverinfo="text", name="nodes (this vp)"),
            ])
            fmap.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0),
                               showlegend=False, dragmode="select",
                               xaxis_title="x", yaxis_title="y",
                               yaxis=dict(scaleanchor="x", scaleratio=1))
            ev = st.plotly_chart(fmap, use_container_width=True,
                                 key=f"map_{scene}_{vp['id']}",
                                 on_select="rerun", selection_mode=("points", "box", "lasso"))
            hits = [q.get("customdata") for q in (ev or {}).get("selection", {}).get("points", [])]
            hits = [h[0] if isinstance(h, list) else h for h in hits]
            clicked = next((h for h in hits if isinstance(h, int) and h < len(vp_nodes)), None)
        else:
            clicked = None
            st.caption("_no nodes at this viewpoint_")

    with pick_r:
        opts = ["(none)"] + [f"{i}: {n['label']}  ·  score {n['score']:.2f}"
                             for i, n in enumerate(vp_nodes)]
        default = (clicked + 1) if clicked is not None else 0
        choice = st.selectbox("or pick from the list", opts, index=default,
                              key=f"pick_{scene}_{vp['id']}_{clicked}")
    picked = vp_nodes[int(choice.split(":")[0])] if choice != "(none)" else None
    click_key = f"click3d_{scene}_{vp['id']}"
    if picked is None and st.session_state.get(click_key) is not None:
        i3 = st.session_state[click_key]
        if i3 < len(vp_nodes):
            picked = vp_nodes[i3]
    match = _match_in(cum_nodes, picked)
    if show_pred and cum_nodes:
        traces.append(_boxes_trace(cum_nodes, "crimson", "pred boxes", 4))
        traces.append(_labels_trace(cum_nodes, "crimson", "pred labels",
                                    show_ids=show_ids))
    if match is not None:
        traces += _highlight_traces(match)
        gap = float(np.linalg.norm(np.array(match["center"]) - np.array(picked["center"])))
        note = (f"matched to cumulative node {match['node_id']} "
                f"(**{match['label']}**, n_obs {match.get('n_obs')}, {gap:.2f} m away)")
        (st.success if match["label"] == picked["label"] else st.warning)(
            f"Showing **{picked['label']}** from this viewpoint — {note}."
            + ("" if match["label"] == picked["label"]
               else "  Labels differ, so this may be the wrong object."))
    _render(traces, key="fig_cumulative")

    # ── B. this viewpoint only ──────────────────────────────────────────────
    st.subheader("Phase 3 · This viewpoint alone")
    has_vp_only = "nodes_vp" in vps[idx]
    if not has_vp_only:
        st.warning("This dump predates per-viewpoint nodes — re-run dump_debug.py.")
    st.caption("Fused from this frame's lifts only, with the points it lifted. "
               "What the viewpoint contributed, separate from what it inherited. "
               "Node ids here are local to the viewpoint.")
    if classes:
        st.caption(f"_Filtered to **{', '.join(classes)}** — {len(vp_nodes)} of "
                   f"{len(vps[idx].get('nodes_vp', []))} nodes this viewpoint._")
    traces = _panel_traces(path_to_here=False)
    if show_pts:
        p = np.load(RUN_DIR / scene / f"{vp['id']}_pts.npy")
        if len(p):
            traces.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                          marker=dict(size=2, color="royalblue", opacity=0.7),
                          name="lifted pts (this vp)", hoverinfo="skip"))
    if show_pred and has_vp_only and vp_nodes:
        traces.append(_boxes_trace(vp_nodes, "mediumseagreen", "pred boxes", 4))
        # Labelled with cum_ids (see _tag), NOT nodes_vp's own numbering.
        traces.append(_labels_trace(vp_nodes, "mediumseagreen",
                                    "pred labels", size=6, show_ids=show_ids))
        if picked is not None:
            traces += _highlight_traces(picked)
    if click3d and has_vp_only and vp_nodes:
        # the node-centre markers are the only clickable trace with one point
        # per node, so its position in the list is what maps a click to a node
        marker_curve = next(i for i, t in enumerate(traces) if t.name == "pred labels")
        st.caption("Click a **green node marker** in the plot. Rotate first, then click — "
                   "a drag that ends on a point still registers as a click.")
        fig = go.Figure(traces)
        fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0),
                          scene=dict(aspectmode="data", xaxis_title="x",
                                     yaxis_title="y", zaxis_title="z"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.0))
        ev = plotly_events(fig, click_event=True, override_height=620,
                           key=f"ev_{scene}_{vp['id']}")
        if ev:
            e = ev[0]
            if e.get("curveNumber") == marker_curve:
                i3 = e.get("pointNumber", e.get("pointIndex"))
                if isinstance(i3, int) and st.session_state.get(click_key) != i3:
                    st.session_state[click_key] = i3
                    st.rerun()
            else:
                st.caption(f"_that was the '{traces[e['curveNumber']].name}' trace — "
                           "click the green centre markers_")
        if st.session_state.get(click_key) is not None and st.button("clear 3D selection"):
            st.session_state[click_key] = None
            st.rerun()
    else:
        _render(traces, key="fig_viewpoint")

    # ---- phase 3 · LIFT INPUT (per detection) ----
    st.subheader("Phase 3 · Lift INPUT — registered scan & what each mask selects")
    st.caption("The lifter's inputs are **(mask, registered scan, pose)**. Pick a detection to "
               "see each stage strip points away: **gray = registered scan** (LiDAR input) · "
               "**orange = in-mask** (what the mask selects) · **yellow = after the z-buffer** "
               "occlusion gate · **green = after voxel clustering** — the points actually "
               "medianed · **red ◇ = lifted position**. A mask needs ≥ `min_inliers` green "
               "points to produce a 3D point. Orange→yellow is occlusion; yellow→green is "
               "mask spill onto a neighbouring surface.")
    lift_path = RUN_DIR / scene / f"{vp['id']}_lift.npz"
    if not lift_path.exists():
        st.info("No lift-input dump for this scene. Run: uv run python "
                "perception_benchmark/dump_lift_input.py --all")
    else:
        L = np.load(lift_path, allow_pickle=True)
        dets = list(L["dets"]); mi = int(L["min_inliers"])
        if not dets:
            st.write("No detections at this viewpoint.")
        else:
            def _lab(i):
                r = dets[i]
                n_kept = r.get("n_kept", r["n_front"])
                return (f"{i}: {r['label']} ({r['score']:.2f}) — mask {r['n_in_mask']} → "
                        f"zbuf {r['n_front']} → kept {n_kept}"
                        f"  {'✓ LIFT' if r['lifted'] else '✗ drop'}")
            # The options are ORIGINAL indices, not positions in a filtered
            # list — `sel` indexes the npz point arrays (`inmask_det == sel`),
            # so renumbering here would select the wrong detection's points.
            opts_det = [i for i in range(len(dets))
                        if not classes or dets[i]["label"] in classes]
            if not opts_det:
                # Not st.stop() — that would abandon the rest of the page,
                # including the other tabs.
                st.info(f"No **{', '.join(classes)}** detection at this "
                        "viewpoint; listing all classes instead.")
                opts_det = list(range(len(dets)))
            sel = st.selectbox("Detection (mask → scan points)", opts_det,
                               format_func=_lab, key="liftdet")
            r = dets[sel]
            n_kept = r.get("n_kept", r["n_front"])
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("in-mask", r["n_in_mask"])
            m2.metric("after z-buffer", r["n_front"], delta=r["n_front"] - r["n_in_mask"],
                      delta_color="off")
            m3.metric("after clustering", n_kept, delta=n_kept - r["n_front"],
                      delta_color="off")
            m4.metric("min_inliers gate", mi)
            m5.metric("lifted?", "yes" if r["lifted"] else "no")
            if "cluster_voxel_m" in L:
                st.caption(f"cluster_voxel_m = {float(L['cluster_voxel_m']):.2f} m"
                           + ("  (clustering disabled)"
                           if float(L["cluster_voxel_m"]) == 0 else ""))
            scan = L["scan"]
            im = L["inmask_pts"][L["inmask_det"] == sel]
            fr = L["front_pts"][L["front_det"] == sel]
            t = [go.Scatter3d(x=scan[:, 0], y=scan[:, 1], z=scan[:, 2], mode="markers",
                              marker=dict(size=1, color="lightgray"),
                           name="registered scan", hoverinfo="skip")]
            if len(im):
                t.append(go.Scatter3d(x=im[:, 0], y=im[:, 1], z=im[:, 2], mode="markers",
                         marker=dict(size=2.5, color="orange"), name="in-mask (pre z-buf)"))
            if len(fr):
                t.append(go.Scatter3d(x=fr[:, 0], y=fr[:, 1], z=fr[:, 2], mode="markers",
                         marker=dict(size=2.5, color="gold"), name="after z-buffer"))
            if "kept_pts" in L:
                kp = L["kept_pts"][L["kept_det"] == sel]
                if len(kp):
                    t.append(go.Scatter3d(x=kp[:, 0], y=kp[:, 1], z=kp[:, 2], mode="markers",
                             marker=dict(size=3.5, color="limegreen"), name="after clustering"))
            if r["position"]:
                p = r["position"]
                t.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                         marker=dict(size=9, color="red"),
                      name="lifted position"))
            rob = L["robot"]
            t.append(go.Scatter3d(x=[rob[0]], y=[rob[1]], z=[rob[2]], mode="markers",
                     marker=dict(size=6, color="black", symbol="x"), name="robot"))
            f2 = go.Figure(t)
            f2.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
                             scene=dict(aspectmode="data", xaxis_title="x",
                                 yaxis_title="y", zaxis_title="z"),
                             legend=dict(orientation="h", yanchor="bottom", y=1.0))
            st.plotly_chart(f2, use_container_width=True)

    # ---- phase 3 table ----
    st.subheader("Phase 3 · detection / lift table")
    det_rows = _keep(vp["detections"], classes)
    if classes:
        st.caption(f"_{len(det_rows)} of {len(vp['detections'])} detections._")
    rows = sorted(det_rows, key=lambda r: (-r["lifted"], -r["score"]))
    st.dataframe(rows, use_container_width=True, height=360)

# ===========================================================================
# 2D DETECTOR TAB — scores the masks alone: no lifting, no fusion.
# ===========================================================================
with tab_2d:
    d2_path = RUN_DIR / scene / "detect2d.json"
    if not d2_path.exists():
        st.info("No 2D evaluation for this scene. Run:\n\n"
                "`uv run python perception_benchmark/detect_2d_eval.py --all "
                "--dump perception_benchmark/debug`")
    else:
        with open(d2_path) as fh:
            D2 = json.load(fh)
        by_id = {v["id"]: v for v in D2["viewpoints"]}
        cur = by_id.get(vp["id"])

        st.subheader("Detector quality, isolated from the 3D pipeline")
        st.caption("GT object centres are projected into this frame; a detection **hits** when "
                   "its mask contains a visible GT centre, and is **correct** when its label "
                   "also matches. Everything here is 2D — lift and fusion errors are excluded, "
                   "so this is the ceiling the rest of the pipeline works under.")

        t = D2["totals"]
        det, loc, lab = t["detections"], t["localised"], t["labelled"]
        vis, cov, covl = t["gt_visible"], t["gt_covered"], t["gt_covered_lbl"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("detections (scene)", det)
        c2.metric("hit a GT object", f"{100 * loc / max(det, 1):.1f}%")
        c3.metric("...and right label", f"{100 * lab / max(det, 1):.1f}%",
                  delta=f"{100 * lab / max(loc, 1):.0f}% of hits", delta_color="off")
        c4.metric("GT recall (correct label)", f"{100 * covl / max(vis, 1):.1f}%",
                  delta=f"{100 * cov / max(vis, 1):.0f}% any label", delta_color="off")

        if cur is None:
            st.warning(f"No 2D record for {vp['id']}.")
        else:
            st.markdown(f"#### {vp['id']}")
            m1, m2, m3, m4 = st.columns(4)
            n = len(cur["detections"])
            hits = sum(x["hit"] for x in cur["detections"])
            good = sum(x["correct"] for x in cur["detections"])
            m1.metric("detections", n)
            m2.metric("hit", hits)
            m3.metric("correct label", good)
            m4.metric("GT visible here", cur["gt_visible"],
                      delta=f"{cur['gt_covered_lbl']} found correctly", delta_color="off")

            img_path = RUN_DIR / scene / f"{vp['id']}.png"
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
                st.caption("Coloured masks + white labels = detections · lime ◇ = ground truth "
                           "observable from here.")

            st.markdown("**Per detection** — `inside` lists the GT objects whose centre "
                        "falls in the mask; empty means the mask hit nothing real.")
            st.dataframe(
                [{"label": r["label"], "score": r["score"], "hit": r["hit"],
                  "correct": r["correct"], "inside": ", ".join(r["inside"])}
                 for r in cur["detections"]],
                use_container_width=True, height=300)

            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Missed entirely** — visible GT with no mask on it")
                st.write(", ".join(cur["missed"]) or "_none_")
            with colB:
                st.markdown("**Found but mislabelled** — a mask covered it, label was wrong")
                st.write(", ".join(cur["missed_label_only"]) or "_none_")

        st.markdown("#### Per-label accuracy across this scene")
        agg = {}
        for v in D2["viewpoints"]:
            for r in v["detections"]:
                a = agg.setdefault(r["label"], {"n": 0, "hit": 0, "correct": 0})
                a["n"] += 1
                a["hit"] += r["hit"]
                a["correct"] += r["correct"]
        rows = [{"label": k, "detections": a["n"],
                 "hit %": round(100 * a["hit"] / a["n"], 1),
                 "correct %": round(100 * a["correct"] / a["n"], 1)}
                for k, a in agg.items()]
        rows.sort(key=lambda r: (r["correct %"], -r["detections"]))
        st.caption("Sorted worst-first. A label with many detections and a low correct% is one "
                   "the detector cannot use — a candidate to drop from the class list.")
        st.dataframe(rows, use_container_width=True, height=320)

# ===========================================================================
# COMPARE TAB — the same scene, the same viewpoint, several dump directories.
#
# For a sweep this is the only honest way to judge the trade: the score table
# says which run has the better number, this says whether the extra nodes are
# real objects or fragments of one.
# ===========================================================================
with tab_cmp:
    st.title(f"Compare runs — {scene}")

    if len(runs) < 2:
        st.info(f"Only one dump directory found next to `{DEBUG_DIR.parent}`. "
                "Produce more with:\n\n"
                "`DUMP=1 K_VALUES=\"0 2 5 10\" perception_benchmark/"
                "sweep_keyframes.sh " + scene + "`")
        st.stop()

    # ---- 1. the whole sweep at a glance (scalars only — no viz.json held) ----
    st.subheader("All runs")
    st.caption("Totals over the scene. Detections are identical across a "
               "keyframe sweep by construction (the masks are frozen and "
               "replayed), so every difference below comes from the "
               "accumulator. **final nodes** against **gt** is the "
               "fragmentation you are trying to see.")
    summ = [s for s in (run_summary(r_, scene) for r_ in runs) if s]
    st.dataframe(
        [{"run": s["run"], "frames": s["frames"], "dets": s["dets"],
          "lifts": s["lifts"],
          "lift %": round(100 * s["lifts"] / max(s["dets"], 1), 1),
          "final nodes": s["final nodes"], "gt": s["gt"],
          "nodes/gt": round(s["final nodes"] / max(s["gt"], 1), 2)}
         for s in summ],
        use_container_width=True, hide_index=True)

    # ---- 2. which runs to put side by side ----
    # Default to the selected run and its neighbour in the sweep — adjacent k
    # is the comparison that shows what one step of the parameter did.
    have = [r_ for r_ in runs if scene in available_scenes(r_)]
    i0 = have.index(str(RUN_DIR)) if str(RUN_DIR) in have else 0
    nbr = have[i0 - 1] if i0 > 0 else (have[1] if len(have) > 1 else None)
    default = [have[i0]] + ([nbr] if nbr else [])
    sel = st.multiselect("Runs to show", have, default=default,
                         format_func=lambda r_: Path(r_).name,
                         help="Each selected run loads its viz.json (8–15 MB), "
                              "so pick the two or three worth looking at.")
    if not sel:
        st.stop()

    c1, c2, c3 = st.columns([1, 1, 2])
    per_row = c1.number_input("panels per row", 1, 4, min(len(sel), 2))
    what = c2.selectbox("Panel", ["Cumulative 3D", "This viewpoint 3D",
                                  "Overlay image"])
    view = c3.radio("Camera", ["isometric", "top-down", "free"],
                    horizontal=True,
                    help="Panels are separate plotly figures, so orbiting one "
                         "does not move the others — pick a preset to put them "
                         "all in the same pose. *free* leaves each where you "
                         "left it.")
    st.caption(f"Showing **{vps[idx]['id']}** — move the sidebar *Viewpoint* "
               "slider to step through together.")

    CAMERAS = {
        "isometric": dict(eye=dict(x=1.4, y=1.4, z=1.1)),
        "top-down": dict(eye=dict(x=0, y=0, z=2.6), up=dict(x=0, y=1, z=0)),
    }

    # A shared axis range, so a box is the same size in every panel. Without
    # this, aspectmode="data" scales each panel to its own extent and a run
    # with one stray far-away node silently shrinks the room.
    def _scene_range(datasets):
        pts = []
        for d_ in datasets:
            pts += [b["bmin"] for b in d_["gt"]] + [b["bmax"] for b in d_["gt"]]
            pts += [v_["pose"] for v_ in d_["viewpoints"]]
        if not pts:
            return None
        a = np.array(pts, dtype=float)
        lo, hi = a.min(0) - 0.5, a.max(0) + 0.5
        return [[float(lo[i]), float(hi[i])] for i in range(3)]

    loaded = [(r_, load_scene(r_, scene)) for r_ in sel]
    rng = _scene_range([d_ for _, d_ in loaded])

    for start in range(0, len(loaded), int(per_row)):
        chunk = loaded[start:start + int(per_row)]
        for col, (r_, d_) in zip(st.columns(len(chunk)), chunk, strict=True):
            with col:
                vps_ = d_["viewpoints"]
                j = min(idx, len(vps_) - 1)
                vp_ = vps_[j]
                st.markdown(f"**{Path(r_).name}** · `{vp_['id']}`")
                a, b = st.columns(2)
                a.metric("nodes (cum)", len(_keep(vp_["nodes"], classes)),
                         delta=(f'of {len(vp_["nodes"])}' if classes else None),
                         delta_color="off")
                b.metric("lifted here", vp_["n_lift"],
                         delta=f'{vp_["n_det"]} det', delta_color="off")

                if what == "Overlay image":
                    p_ = Path(r_) / scene / f"{vp_['id']}.png"
                    if p_.exists():
                        st.image(str(p_), use_container_width=True)
                    else:
                        st.caption("_no overlay_")
                    continue

                cum = what == "Cumulative 3D"
                t_ = _common_traces(
                    d_, vps_, j, _yaws(r_, scene, len(vps_)),
                    path_to_here=cum, show_gt=show_gt, show_path=show_path,
                    show_heading=show_heading, classes=classes)
                if cum and cloud_mode == "object points":
                    pts_ = _object_points(r_, scene, vps_, j)
                    if len(pts_):
                        t_.insert(0, go.Scatter3d(
                            x=pts_[:, 0], y=pts_[:, 1], z=pts_[:, 2],
                            mode="markers",
                            marker=dict(size=1.4, color="royalblue", opacity=0.4),
                            name=f"object points ({len(pts_)})", hoverinfo="skip"))
                elif not cum:
                    p_ = Path(r_) / scene / f"{vp_['id']}_pts.npy"
                    if show_pts and p_.exists():
                        q_ = np.load(p_)
                        if len(q_):
                            t_.append(go.Scatter3d(
                                x=q_[:, 0], y=q_[:, 1], z=q_[:, 2],
                                mode="markers",
                                marker=dict(size=2, color="royalblue", opacity=0.7),
                                name="lifted pts", hoverinfo="skip"))
                key_ = "nodes" if cum else "nodes_vp"
                boxes = _keep(vp_.get(key_, []), classes)
                if show_pred and boxes:
                    colour = "crimson" if cum else "mediumseagreen"
                    t_.append(_boxes_trace(boxes, colour, "pred boxes", 4))
                    # Ids are per-run counters; they do NOT correspond between
                    # panels. Labels and positions are what compare.
                    t_.append(_labels_trace(boxes, colour, "pred labels",
                                            size=4 if cum else 6,
                                            show_ids=show_ids))

                fig_ = go.Figure(t_)
                sc = dict(aspectmode="data", xaxis_title="x", yaxis_title="y",
                          zaxis_title="z")
                if rng:
                    sc["xaxis"] = dict(range=rng[0], title="x")
                    sc["yaxis"] = dict(range=rng[1], title="y")
                    sc["zaxis"] = dict(range=rng[2], title="z")
                if view in CAMERAS:
                    sc["camera"] = CAMERAS[view]
                fig_.update_layout(
                    height=520, margin=dict(l=0, r=0, t=0, b=0), scene=sc,
                    showlegend=False,
                    # keyed on the camera preset alone: changing viewpoint must
                    # NOT reset an orbit the user set up, changing the preset
                    # must apply it.
                    uirevision=f"{r_}|{view}")
                st.plotly_chart(fig_, use_container_width=True,
                                key=f"cmp_{r_}_{scene}_{j}")

    # ---- 3. how the graph grows, per run ----
    st.subheader("Cumulative node count")
    st.caption("Where the runs diverge. A curve that keeps climbing after the "
               "robot has stopped covering new ground is fragmentation, not "
               "discovery — the vertical line is the current viewpoint.")
    growth = go.Figure()
    for r_, d_ in loaded:
        growth.add_trace(go.Scatter(
            x=list(range(len(d_["viewpoints"]))),
            y=[len(_keep(v_["nodes"], classes)) for v_ in d_["viewpoints"]],
            mode="lines", name=Path(r_).name))
    growth.add_trace(go.Scatter(
        x=[0, max(len(d_["viewpoints"]) for _, d_ in loaded)],
        y=[len(_keep(data["gt"], classes))] * 2, mode="lines", name="GT count",
        line=dict(color="gray", dash="dash")))
    growth.add_vline(x=idx, line=dict(color="red", width=1, dash="dot"))
    growth.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                         xaxis_title="viewpoint", yaxis_title="nodes",
                         legend=dict(orientation="h", yanchor="bottom", y=1.0))
    st.plotly_chart(growth, use_container_width=True, key="cmp_growth")
