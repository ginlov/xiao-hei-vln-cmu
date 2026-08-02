"""Streamlit + plotly viewer for the perception benchmark debug data.

Loads the per-scene dumps from dump_debug.py and shows, interactively:
  - the mask-overlay image (phases 1+2: input + detect),
  - two 3D views: the CUMULATIVE scene graph with the accumulated LiDAR cloud,
    and what THIS viewpoint contributed on its own — with a viewpoint slider
    so you can watch the graph build up,
  - the per-viewpoint detection/lift table.

No sidecar needed (reads the dumped data).

    uv run --with streamlit --with plotly streamlit run perception_benchmark/viz_app.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

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
SCORES_DIR = Path("perception_benchmark/scores")

st.set_page_config(page_title="Perception benchmark debug", layout="wide")


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


def _labels_trace(boxes, color, name, size=3):
    """Centre markers, hover shows the label."""
    return go.Scatter3d(
        x=[b["center"][0] for b in boxes], y=[b["center"][1] for b in boxes],
        z=[b["center"][2] for b in boxes], mode="markers",
        marker=dict(size=size, color=color),
        text=[f"{b['label']}" + (f" (n={b.get('n_obs')})" if 'n_obs' in b else "") for b in boxes],
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
def load_scene(scene):
    data = json.load(open(DEBUG_DIR / scene / "viz.json"))
    return data


def available_scenes():
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(str(DEBUG_DIR / "*" / "viz.json")))


# ---- sidebar ----
scenes = available_scenes()
if not scenes:
    st.error("No debug dumps found. Run: uv run --extra perception python "
             "perception_benchmark/dump_debug.py --all"); st.stop()

st.sidebar.title("Perception debug")
scene = st.sidebar.selectbox("Scene", scenes)
data = load_scene(scene)
vps = data["viewpoints"]
n = len(vps)
idx = st.sidebar.slider("Viewpoint", 0, n - 1, 0, format="vp %d")
cloud_mode = st.sidebar.selectbox(
    "Cumulative panel cloud", ["object points", "full scene", "none"],
    help="object points = the lifted inliers behind the predicted boxes; "
         "full scene = every LiDAR sweep, including surfaces nothing was "
         "detected on (needs dump_cloud.py).")
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

tab_graph, tab_2d = st.tabs(["Scene graph", "2D detector"])

with tab_graph:
    vp = vps[idx]
    st.title(f"{scene} — {vp['id']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("detections", vp["n_det"])
    c2.metric("lifted", vp["n_lift"])
    c3.metric("nodes (cumulative)", len(vp["nodes"]),
              delta=(f'{len(vp["nodes_vp"])} this vp' if "nodes_vp" in vp else None),
              delta_color="off")
    c4.metric("GT (scoreable)", len(data["gt"]))

    # ---- phase 1+2: overlay image ----
    st.subheader("Phase 1–2 · input image + model detections vs ground truth")
    st.caption("**Coloured masks + white labels = MODEL DETECTIONS** (what perception found). "
               "**Lime ◇ + green labels = GROUND TRUTH** objects (from object_list) projected "
               "into this view. `[OK]`/`[x]` on a detection = whether it produced a 3D point "
               "(cleared min_inliers). Compare the two: GT with no overlapping mask = a miss.")
    img_path = DEBUG_DIR / scene / f"{vp['id']}.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    # ---- phase 3+4: two 3D panels, cumulative vs this viewpoint ----
    def _common_traces(*, path_to_here: bool):
        """GT boxes, robot path and marker — shared by both panels."""
        t = []
        if show_gt:
            t.append(_boxes_trace(data["gt"], "rgba(80,80,80,0.5)", "GT boxes", 2))
            t.append(_labels_trace(data["gt"], "gray", "GT labels"))
        if show_path and path_to_here:
            poses = np.array([v["pose"] for v in vps[:idx + 1]])
            if len(poses):
                t.append(go.Scatter3d(x=poses[:, 0], y=poses[:, 1], z=poses[:, 2],
                         mode="lines+markers", line=dict(color="orange", width=3),
                         marker=dict(size=3, color="orange"), name="robot path"))
        t.append(go.Scatter3d(x=[vp["pose"][0]], y=[vp["pose"][1]], z=[vp["pose"][2]],
                 mode="markers", marker=dict(size=7, color="red"),
                 name="robot (this vp)"))
        return t

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
    traces = _common_traces(path_to_here=True)
    if cloud_mode == "object points":
        # Union of the lifted inliers from every viewpoint so far: the evidence
        # the boxes were actually built from, not the whole room.
        chunks = []
        for v in vps[:idx + 1]:
            f = DEBUG_DIR / scene / f"{v['id']}_pts.npy"
            if f.exists():
                chunks.append(np.load(f))
        pts = np.vstack(chunks) if chunks else np.empty((0, 3))
        if len(pts) > MAX_CLOUD_PTS:                 # keep plotly responsive
            pts = pts[np.random.default_rng(0).choice(len(pts), MAX_CLOUD_PTS, False)]
        if len(pts):
            traces.insert(0, go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                marker=dict(size=1.6, color="royalblue", opacity=0.45),
                name=f"object points ({len(pts)})", hoverinfo="skip"))
    elif cloud_mode == "full scene":
        cloud_path = DEBUG_DIR / scene / "cloud.npz"
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
    vp_nodes = vps[idx].get("nodes_vp", [])
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
    match = _match_in(vps[idx]["nodes"], picked)
    if show_pred:
        traces.append(_boxes_trace(vps[idx]["nodes"], "crimson", "pred boxes", 4))
        traces.append(_labels_trace(vps[idx]["nodes"], "crimson", "pred labels"))
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
    traces = _common_traces(path_to_here=False)
    if show_pts:
        p = np.load(DEBUG_DIR / scene / f"{vp['id']}_pts.npy")
        if len(p):
            traces.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                          marker=dict(size=2, color="royalblue", opacity=0.7),
                          name="lifted pts (this vp)", hoverinfo="skip"))
    if show_pred and has_vp_only:
        traces.append(_boxes_trace(vps[idx]["nodes_vp"], "mediumseagreen", "pred boxes", 4))
        traces.append(_labels_trace(vps[idx]["nodes_vp"], "mediumseagreen",
                                    "pred labels", size=6))
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
    lift_path = DEBUG_DIR / scene / f"{vp['id']}_lift.npz"
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
            sel = st.selectbox("Detection (mask → scan points)", range(len(dets)),
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
    rows = sorted(vp["detections"], key=lambda r: (-r["lifted"], -r["score"]))
    st.dataframe(rows, use_container_width=True, height=360)

# ===========================================================================
# 2D DETECTOR TAB — scores the masks alone: no lifting, no fusion.
# ===========================================================================
with tab_2d:
    d2_path = DEBUG_DIR / scene / "detect2d.json"
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

            img_path = DEBUG_DIR / scene / f"{vp['id']}.png"
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
