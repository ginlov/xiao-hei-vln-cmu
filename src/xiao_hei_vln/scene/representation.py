"""Scene representation built incrementally from VLM tick snapshots.

Adapts SysNav's three-level hierarchical graph to the challenge sensor suite:

    Room      — one node for the entire scene; holds the best-coverage camera
                frame seen so far and the registered-scan bounding box.
    Viewpoint — one node per distinct robot pose; added when the robot moves
                more than `viewpoint_radius` m from every existing node.
    Object    — one node per detected instance; populated by the caller after
                VLM inference via `add_object()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from xiao_hei_vln.messages.common import Quaternion, Vector3
from xiao_hei_vln.messages.inputs import VLMInput

if TYPE_CHECKING:
    from xiao_hei_vln.messages.sensors import ImageFrame, LidarScan


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialRelation:
    """A directed spatial relation from one object to another.

    Stored on the source `ObjectObservation`. ``relation`` is a free
    string set by the caller (e.g. ``"left_of"``, ``"near"``,
    ``"on_top_of"``). Three identifiers point at the target:

    - ``target_index`` is the positional index into
      ``SceneRepresentation._objects`` at the time the edge was created.
      Currently authoritative; suitable for direct list lookup.
    - ``target_object_id`` is the stable id assigned by
      :meth:`SceneRepresentation.add_object` (monotonic, never reused).
      Designed to remain valid even if a future sweeper removes or
      reorders objects (which would silently corrupt ``target_index``).
    - ``target_label`` is the target's label at edge-creation time,
      kept for human-readable display and JSON-replay debugging only —
      it is not authoritative.

    For now ``target_object_id`` is informational; dedup and renderer
    lookup still go through ``target_index``. Migrating to id-based
    lookup is the next step once an object-removal path lands.
    """

    target_label: str
    target_index: int
    relation: str
    target_object_id: int = 0


@dataclass
class RoomNode:
    """Global scene node — exactly one per `SceneRepresentation`.

    `best_image` is the camera frame captured at the most recently added
    viewpoint — updated each time the robot reaches a novel position.
    `scene_bounds` is the axis-aligned bounding box of the registered scan
    in the map frame, refreshed every tick a scan is available.
    """

    label: str = "scene"
    best_image: ImageFrame | None = None
    best_image_tick_id: int | None = None
    best_image_position: Vector3 | None = None
    scene_bounds: tuple[Vector3, Vector3] | None = None  # (min_xyz, max_xyz)
    viewpoint_tick_ids: list[int] = field(default_factory=list)  # edge: Room → Viewpoint


@dataclass(frozen=True)
class ViewpointNode:
    """One distinct pose from which the robot observed the scene."""

    tick_id: int
    position: Vector3
    yaw: float  # heading in radians (0 = +x axis)


@dataclass
class ObjectObservation:
    """A single object instance reported by VLM output.

    `position` is in the map frame.  `bbox_min`/`bbox_max` form an AABB;
    both may be `None` when LiDAR-based localisation is unavailable.
    Confidence and position are updated in-place when a higher-confidence
    observation of the same label arrives within `merge_radius`.

    ``object_id`` is a stable identifier assigned by
    :meth:`SceneRepresentation.add_object` — monotonic across the scene
    lifetime and never reused. Callers should treat the default ``0``
    as "unassigned" and not rely on it for identity; the scene
    representation stamps every accepted observation with a real id.
    """

    label: str
    position: Vector3
    confidence: float = 1.0
    # Appearance colour, sampled from the pixels inside the detection mask.
    # ``color_rgb`` is the median (R, G, B) in 0-255; ``color_name`` is the
    # nearest basic-colour label (e.g. "red", "brown"). Both are ``None``
    # when no producer supplied a colour (e.g. LiDAR-only observations).
    # Refreshed in lock-step with ``position`` when a higher-confidence
    # observation of the same object arrives.
    color_rgb: tuple[int, int, int] | None = None
    color_name: str | None = None
    bbox_min: Vector3 | None = None
    bbox_max: Vector3 | None = None
    object_id: int = 0
    first_tick_id: int = 0
    last_tick_id: int = 0
    # Stable viewpoint ids (i.e. ``ViewpointNode.tick_id`` values) from
    # which this object has been observed. Each entry corresponds to a
    # real Viewpoint node; ``add_object`` resolves the *current*
    # viewpoint (the latest viewpoint added at-or-before the observing
    # tick) so this list always doubles as the reverse Viewpoint→Object
    # edge.
    observing_viewpoint_ids: list[int] = field(default_factory=list)  # edge: Viewpoint → Object (reverse)
    spatial_relations: list[SpatialRelation] = field(default_factory=list)  # edge: Object → Object


# ---------------------------------------------------------------------------
# SceneRepresentation
# ---------------------------------------------------------------------------


class SceneRepresentation:
    """Incrementally maintained three-level scene graph.

    Instantiate once per question and call `update()` on every tick before
    VLM inference. After inference, call `add_object()` whenever the VLM
    output identifies an object instance.
    """

    def __init__(
        self,
        *,
        viewpoint_radius: float = 2.0,
        merge_radius: float = 1.5,
    ) -> None:
        """
        Args:
            viewpoint_radius: Minimum distance (m) the robot must travel
                before a new `ViewpointNode` is added.
            merge_radius: Maximum distance (m) between two detections of the
                same label for them to be merged into one node.
        """
        self._viewpoint_radius = viewpoint_radius
        self._merge_radius = merge_radius

        self._room = RoomNode()
        self._viewpoints: list[ViewpointNode] = []
        self._objects: list[ObjectObservation] = []
        self._tick_id: int = 0
        # Monotonic id counter for new objects. Never reused — even if a
        # future sweeper removes objects, ids stay unique across the
        # scene's lifetime.
        self._next_object_id: int = 1
        # Stable ObjectMap node_id -> scene object_id, so the fused-map sync
        # path (sync_from_object_map) keeps object identity across ticks even
        # as the export is rebuilt each tick.
        self._node_object_ids: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def room(self) -> RoomNode:
        return self._room

    @property
    def viewpoints(self) -> list[ViewpointNode]:
        return list(self._viewpoints)

    @property
    def objects(self) -> list[ObjectObservation]:
        return list(self._objects)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, snapshot: VLMInput) -> None:
        """Ingest a tick snapshot and refresh the internal graph."""
        self._tick_id = snapshot.tick_id

        if snapshot.pose is not None:
            self._maybe_add_viewpoint(snapshot)

        if snapshot.registered_scan is not None:
            self._update_scene_bounds(snapshot.registered_scan)

    def add_object(self, obs: ObjectObservation) -> None:
        """Register an object detection from VLM output.

        If an existing node has the same `label` within `merge_radius`, the
        higher-confidence observation wins; otherwise a new node is appended.

        ``observing_viewpoint_ids`` is appended with the *current*
        viewpoint id — i.e. the latest viewpoint added at-or-before
        ``self._tick_id`` — so the field is always a real edge into the
        viewpoint nodes (no synthetic / non-viewpoint ticks land in it).
        When no viewpoint exists yet (object reported before the robot
        moved past the first viewpoint radius), nothing is appended; the
        renderer falls back to anchoring such objects manually.
        """
        obs.last_tick_id = self._tick_id
        vp_id = self._current_viewpoint_id()
        for existing in self._objects:
            if (
                existing.label == obs.label
                and _dist3(existing.position, obs.position) < self._merge_radius
            ):
                if obs.confidence >= existing.confidence:
                    existing.position = obs.position
                    existing.confidence = obs.confidence
                    existing.bbox_min = obs.bbox_min
                    existing.bbox_max = obs.bbox_max
                    existing.last_tick_id = self._tick_id
                    # Keep colour in step with position — only overwrite when
                    # the new observation actually carries one, so a colourless
                    # (e.g. LiDAR-only) update doesn't wipe a known colour.
                    if obs.color_rgb is not None:
                        existing.color_rgb = obs.color_rgb
                        existing.color_name = obs.color_name
                    if vp_id is not None and (
                        not existing.observing_viewpoint_ids
                        or existing.observing_viewpoint_ids[-1] != vp_id
                    ):
                        existing.observing_viewpoint_ids.append(vp_id)
                return
        obs.first_tick_id = self._tick_id
        if vp_id is not None:
            obs.observing_viewpoint_ids.append(vp_id)
        # Overwrite any caller-provided object_id — the scene rep is the
        # only authority on stable ids.
        obs.object_id = self._next_object_id
        self._next_object_id += 1
        self._objects.append(obs)

    def sync_from_object_map(self, nodes: list[dict]) -> None:
        """Replace the object layer with a fused ``ObjectMap`` export.

        This is the alternative to per-detection :meth:`add_object`: instead
        of merging one median point at a time, the responder accumulates each
        detection's LiDAR cloud in an :class:`ObjectMap`, then hands the fused
        snapshot here every tick. Each ``nodes`` entry is a dict from
        ``ObjectMap.export()`` (``node_id``, ``label``, ``score``,
        ``center_3d``, ``bbox_aabb``, ``color_rgb``, ``color_name``).

        Unlike ``add_object``, fused nodes carry a real 3D box, so
        ``bbox_min``/``bbox_max`` are populated. ``object_id`` is kept stable
        per ``node_id`` across ticks (via :attr:`_node_object_ids`), and the
        current viewpoint is recorded as an observing edge. Spatial relations
        already attached to a surviving object are preserved.
        """
        vp_id = self._current_viewpoint_id()
        prev = {o.object_id: o for o in self._objects}
        new_objects: list[ObjectObservation] = []
        for nd in nodes:
            nid = int(nd["node_id"])
            oid = self._node_object_ids.get(nid)
            if oid is None:
                oid = self._next_object_id
                self._next_object_id += 1
                self._node_object_ids[nid] = oid

            center = nd["center_3d"]
            bmin = nd["bbox_aabb"]["min"]
            bmax = nd["bbox_aabb"]["max"]
            color = nd.get("color_rgb")
            existing = prev.get(oid)

            vp_ids = list(existing.observing_viewpoint_ids) if existing else []
            if vp_id is not None and (not vp_ids or vp_ids[-1] != vp_id):
                vp_ids.append(vp_id)

            new_objects.append(ObjectObservation(
                label=nd["label"],
                position=Vector3(x=center[0], y=center[1], z=center[2]),
                confidence=float(nd["score"]),
                color_rgb=tuple(color) if color is not None else None,
                color_name=nd.get("color_name"),
                bbox_min=Vector3(x=bmin[0], y=bmin[1], z=bmin[2]),
                bbox_max=Vector3(x=bmax[0], y=bmax[1], z=bmax[2]),
                object_id=oid,
                first_tick_id=existing.first_tick_id if existing else self._tick_id,
                last_tick_id=self._tick_id,
                observing_viewpoint_ids=vp_ids,
                spatial_relations=list(existing.spatial_relations) if existing else [],
            ))
        self._objects = new_objects

    def add_spatial_relation_by_index(
        self, idx_a: int, idx_b: int, relation: str,
    ) -> bool:
        """Record a directed spatial relation from object ``idx_a`` to
        object ``idx_b``. Index-keyed (positional in :attr:`objects`).

        Dedup key is ``(target_index, relation)``. Idempotent —
        re-asserting the same edge is a no-op, so callers can replay
        every tick without inflating the graph.

        Returns ``True`` if a new edge was added, ``False`` if the edge
        already existed or either index is out of range.
        """
        n = len(self._objects)
        if not (0 <= idx_a < n and 0 <= idx_b < n):
            return False
        src = self._objects[idx_a]
        for existing in src.spatial_relations:
            if existing.target_index == idx_b and existing.relation == relation:
                return False
        tgt = self._objects[idx_b]
        src.spatial_relations.append(
            SpatialRelation(
                target_label=tgt.label,
                target_index=idx_b,
                relation=relation,
                target_object_id=tgt.object_id,
            ),
        )
        return True

    def add_spatial_relation(self, label_a: str, label_b: str, relation: str) -> None:
        """Label-keyed convenience wrapper around
        :meth:`add_spatial_relation_by_index`.

        Resolves ``label_a`` and ``label_b`` to their first matching
        object index, then forwards. No-op when either label is missing.

        When multiple nodes share the same label only the first match is
        used — for full multi-instance edges, call
        :meth:`add_spatial_relation_by_index` directly with the indices
        you want.
        """
        idx_a = next((i for i, o in enumerate(self._objects) if o.label == label_a), None)
        idx_b = next((i for i, o in enumerate(self._objects) if o.label == label_b), None)
        if idx_a is None or idx_b is None:
            return
        self.add_spatial_relation_by_index(idx_a, idx_b, relation)

    def derive_near_relations(self, threshold: float = 2.0) -> int:
        """Add bidirectional ``near`` edges for every object pair within
        ``threshold`` metres (XY distance).

        Pairs are matched by *index*, so multiple objects sharing the
        same label each get their own edges. Idempotent — replaying the
        same call every tick collapses to one edge per ordered pair.

        Returns the number of new edges added on this call.
        """
        added = 0
        n = len(self._objects)
        for i in range(n):
            a = self._objects[i]
            for j in range(i + 1, n):
                b = self._objects[j]
                dx = a.position.x - b.position.x
                dy = a.position.y - b.position.y
                if dx * dx + dy * dy > threshold * threshold:
                    continue
                added += int(self.add_spatial_relation_by_index(i, j, "near"))
                added += int(self.add_spatial_relation_by_index(j, i, "near"))
        return added

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable snapshot of the current graph.

        `best_image` is omitted (bytes); only its tick_id and position
        are recorded so a replayer can index back into raw frames.
        """
        bounds = None
        if self._room.scene_bounds is not None:
            mn, mx = self._room.scene_bounds
            bounds = [[mn.x, mn.y, mn.z], [mx.x, mx.y, mx.z]]
        return {
            "tick_id": self._tick_id,
            "room": {
                "label": self._room.label,
                "scene_bounds": bounds,
                "best_image_tick_id": self._room.best_image_tick_id,
                "best_image_position": _v3(self._room.best_image_position),
                "viewpoint_tick_ids": list(self._room.viewpoint_tick_ids),
            },
            "viewpoints": [
                {
                    "tick_id": vp.tick_id,
                    "position": _v3(vp.position),
                    "yaw": vp.yaw,
                }
                for vp in self._viewpoints
            ],
            "objects": [
                {
                    "object_id": o.object_id,
                    "label": o.label,
                    "position": _v3(o.position),
                    "confidence": o.confidence,
                    "color_rgb": list(o.color_rgb) if o.color_rgb is not None else None,
                    "color_name": o.color_name,
                    "bbox_min": _v3(o.bbox_min),
                    "bbox_max": _v3(o.bbox_max),
                    "first_tick_id": o.first_tick_id,
                    "last_tick_id": o.last_tick_id,
                    "observing_viewpoint_ids": list(o.observing_viewpoint_ids),
                    "spatial_relations": [
                        {
                            "target_label": r.target_label,
                            "target_index": r.target_index,
                            "target_object_id": r.target_object_id,
                            "relation": r.relation,
                        }
                        for r in o.spatial_relations
                    ],
                }
                for o in self._objects
            ],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _current_viewpoint_id(self) -> int | None:
        """Return the tick_id of the viewpoint the robot is currently
        observing from, or ``None`` if no viewpoint has been created yet.

        "Current" = the latest viewpoint whose ``tick_id`` is
        ``<= self._tick_id``. Viewpoints are appended in tick order by
        :meth:`_maybe_add_viewpoint`, so the last element is the answer
        unless the latest viewpoint somehow lives in the future
        (defensive — shouldn't happen in normal flow).
        """
        if not self._viewpoints:
            return None
        last = self._viewpoints[-1]
        if last.tick_id <= self._tick_id:
            return last.tick_id
        for vp in reversed(self._viewpoints):
            if vp.tick_id <= self._tick_id:
                return vp.tick_id
        return None

    def _maybe_add_viewpoint(self, snapshot: VLMInput) -> None:
        pos = snapshot.pose.position
        for vp in self._viewpoints:
            if _dist3(pos, vp.position) < self._viewpoint_radius:
                return
        yaw = _yaw_from_quaternion(snapshot.pose.orientation)
        self._viewpoints.append(
            ViewpointNode(tick_id=snapshot.tick_id, position=pos, yaw=yaw)
        )
        self._room.viewpoint_tick_ids.append(snapshot.tick_id)
        # New viewpoint = novel coverage; update the room's representative frame.
        if snapshot.image is not None:
            self._room.best_image = snapshot.image
            self._room.best_image_tick_id = snapshot.tick_id
            self._room.best_image_position = pos

    def _update_scene_bounds(self, scan: LidarScan) -> None:
        pts = scan.points  # (N, 4) float32: x, y, z, intensity
        if pts.shape[0] == 0:
            return
        mn = pts[:, :3].min(axis=0)
        mx = pts[:, :3].max(axis=0)
        self._room.scene_bounds = (
            Vector3(x=float(mn[0]), y=float(mn[1]), z=float(mn[2])),
            Vector3(x=float(mx[0]), y=float(mx[1]), z=float(mx[2])),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist3(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _yaw_from_quaternion(q: Quaternion) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _v3(v: Vector3 | None) -> list[float] | None:
    return [v.x, v.y, v.z] if v is not None else None
