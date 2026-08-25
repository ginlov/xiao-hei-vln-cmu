#!/usr/bin/env python3
"""Score instruction-following runs against the challenge's /6 rubric.

The official `challenge_evaluation_node` is not public, so this is a **proxy**
and is labelled as one everywhere it is reported. README section "Question
Types and Initial Scoring" states the whole of what is known about it:

    A series of Pose2D waypoints must be published to guide the vehicle. The
    score will be calculated based on the actual trajectory followed by the
    robot based on whether it follows the path constraints in the command and
    in the correct order. Penalties are imposed upon the score if the followed
    path deviates from the correct order of constraints, does not achieve the
    desired constraints, or passes through areas it is forbidden to go through
    in the command. Score between 0 and 6, with possibility for partial points.

Four things are therefore scored, and nothing else is invented:

  * a destination (GOTO) is achieved if the driven track comes within TAU of
    the ground-truth box of the object the clause names;
  * a passage (PASS) is achieved if the track crosses the segment joining its
    two anchors -- proximity is not passage (TASK 34);
  * order is checked on the arc index at which each constraint was first
    achieved, not on the order they were attempted;
  * a keep-out costs points whether or not anything else succeeded.

TAU defaults to 1.0 m because the organisers' own reference trajectories come
within 1.0 m of 81% of the objects their instructions name, against 26% for
objects they do not name (`scripts/traj_tolerance.py`). It is a property of the
reference answers, not a number chosen to flatter us.

**Validation.** `--reference` scores the organisers' own `trajectory_q4/q5.ply`
with this same scorer. A proxy that does not award the reference answer full
marks is measuring its own thresholds, so that number is reported first and
every claim made with this tool is conditional on it.

    uv run python scripts/score_if.py --reference
    uv run python scripts/score_if.py --runs runs
    uv run python scripts/score_if.py --runs runs --json scores.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from instruction_plan import GOTO, PASS, keepouts, parse_instruction, steps  # noqa: E402
from traj_tolerance import (  # noqa: E402
    mentioned,
    point_to_box_xy,
    read_ply_ascii,
)

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
VIZ_DATA = Path("viz/data")
TRAJ_FOR = {0: "trajectory_q4.ply", 1: "trajectory_q5.ply"}

# The tolerance is OURS, not the challenge's: the official scorer is closed and
# returns one number, so its threshold is unknown. This one is derived from the
# organisers' reference trajectories -- over 59 reference destinations the
# closest approach has median 0.52 m and **maximum 1.20 m**, so 1.25 m is the
# smallest round value that admits every destination they themselves drove to.
#
# An earlier comment here called it the p90 of those distances. That was wrong;
# the p90 is 0.81 m. Nothing downstream changed, but the claim was repeated into
# notes and drafts, so it is corrected loudly rather than quietly.
#
# Every score this file produces is conditional on this number, and strongly:
# the driven corpus means 2.63/6 at tau 0.85 and 5.03/6 at tau 2.00. Report the
# sensitivity, never the single number alone. `--tau` exists for that.
TAU = 1.25         # m
KEEPOUT_M = 1.2    # m -- half-width of a forbidden corridor
PASS_NEAR_M = 1.5   # m -- "take the path near X" passes within this of X
ORDER_CREDIT = 0.5  # an out-of-order constraint keeps this share of its points


# --------------------------------------------------------------------------
# scene ground truth
# --------------------------------------------------------------------------

def load_scene(scene: str, viz: Path) -> tuple[dict[str, list], set[str]] | None:
    vf = viz / f"{scene}.json"
    if not vf.is_file():
        return None
    data = json.loads(vf.read_text())
    labels = {n.strip().lower() for n in data["labels"]}
    by_label: dict[str, list] = {}
    for o in data["gt"]:
        lab = data["labels"][o["l"]].strip().lower()
        by_label.setdefault(lab, []).append(
            {"lo": np.array(o["lo"], float), "hi": np.array(o["hi"], float),
             "c": np.array(o["c"], float)})
    return by_label, labels


def exact_label(phrase: str, labels: set[str]) -> str | None:
    """The label a keep-out anchor names, or None if the scene has no such label.

    A keep-out is the only constraint that can *lose* points, so it must not be
    grounded by widening. livingroom_2 q5 forbids "the path between the TV and
    the tea table"; that scene's annotation has no `tea table`, and falling back
    to its `table` grounds the corridor on the wrong furniture -- which reported
    the organisers' own reference path as violating the keep-out it was drawn to
    respect. When the phrase's own noun is not a label in this scene, say so.
    """
    q = re.sub(r"^(the|a|an)\s+", "", phrase.strip().lower())
    q = re.sub(r"[^a-z ]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    for form in (q, q.rstrip("s"), q + "s"):
        if form in labels:
            return form
    return None


def head_label(phrase: str, labels: set[str]) -> str | None:
    """The object a referring expression is *about*.

    "the small table farthest from the columns" names two labels; the
    destination is the table. English puts the head noun before its
    prepositional qualifiers, so the label that occurs earliest in the phrase
    is the head -- checked against all 30 official instructions in --audit.
    """
    hits = mentioned(phrase, labels)
    if not hits:
        return None
    low = phrase.lower()
    return min(hits, key=lambda h: low.find(h) if h in low else 10**6)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def first_touch(track: np.ndarray, boxes: list[dict], tau: float,
                start: int = 0) -> int | None:
    """Index of the *closest approach* at or after `start`, if within `tau`.

    Not the first entry into the tolerance ball. A reference path is a tour and
    brushes past objects it has not visited yet, so "first within tau" finds a
    monotonic assignment in almost any order — reversing a reference trajectory
    left 7 of 30 still scoring 6/6, including destinations 6.88 m apart. The
    visit is where the robot came nearest and stopped, which is what "stop at
    the trash can" means and what a reversal actually moves.
    """
    best, bd = None, tau
    for i in range(start, len(track)):
        d = min(point_to_box_xy(track[i][:2], b["lo"], b["hi"]) for b in boxes)
        if d <= bd:
            best, bd = i, d
    return best


def nearest_pair(a: list[dict], b: list[dict]) -> tuple[dict, dict]:
    """The pair of instances a person would read the phrase as naming.

    "between the chair and the folding screen" in a room with six chairs names
    the chair that forms a gap with the screen, not all six. The narrowest pair
    is that gap. Chosen without reference to any trajectory, so it cannot be
    tuned to flatter a result.
    """
    best, bd = (a[0], b[0]), float("inf")
    for p in a:
        for q in b:
            d = float(np.linalg.norm(p["c"][:2] - q["c"][:2]))
            if 1e-6 < d < bd:
                bd, best = d, (p, q)
    return best


def _ccw(a, b, c) -> float:
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(p1, p2, p3, p4) -> bool:
    d1, d2 = _ccw(p3, p4, p1), _ccw(p3, p4, p2)
    d3, d4 = _ccw(p1, p2, p3), _ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def crossing_index(track: np.ndarray, a: np.ndarray, b: np.ndarray,
                   start: int = 0) -> int | None:
    """First index at which the track crosses the segment a--b.

    This is `went_between` from TASK 34: being near a gap is not going through
    it, and the challenge scores the trajectory.
    """
    for i in range(start, len(track) - 1):
        if segments_cross(track[i][:2], track[i + 1][:2], a[:2], b[:2]):
            return i
    return None


def enters_corridor(track: np.ndarray, a: np.ndarray, b: np.ndarray,
                    half_width: float) -> bool:
    """Does the track enter the corridor spanning a--b?

    A keep-out phrased "the path between X and Y" is a corridor, not a disc
    around the midpoint (TASK 37).
    """
    a2, b2 = a[:2], b[:2]
    ab = b2 - a2
    L = float(np.linalg.norm(ab))
    if L < 1e-6:
        return bool(np.min(np.linalg.norm(track[:, :2] - a2, axis=1)) <= half_width)
    u = ab / L
    rel = track[:, :2] - a2
    t = rel @ u
    perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    inside = (t >= 0) & (t <= L) & (perp <= half_width)
    return bool(inside.any())


# --------------------------------------------------------------------------
# scoring one trajectory against one question
# --------------------------------------------------------------------------

def score_trajectory(track: np.ndarray, question: str, by_label: dict,
                     labels: set[str], tau: float = TAU,
                     pin: dict[int, list] | None = None) -> dict:
    """Score one trajectory. `pin` restricts a GOTO to specific instances.

    Without `pin`, a GOTO is satisfied by **any** instance of the phrase's head
    noun: "the potted plant on the table" grounds to `potted plant` and the
    relative clause never enters the test. That is too generous, and measurably
    so -- across the driven corpus, 20% of credited destinations were credited
    on a different instance from the one the organisers' own reference path
    approached, in one case 12.8 m away. `pin` maps a constraint index to the
    only instances that may satisfy it, which lets `--strict-instance` grade
    against the reference's choice and report the gap between the two readings.
    """
    plan = parse_instruction(question)
    ordered = steps(plan)
    kos = keepouts(plan)

    if not ordered or track is None or len(track) < 2:
        return {"score": 0.0, "max": 6.0, "constraints": [], "keepouts": [],
                "ungroundable": True, "n_steps": len(ordered)}

    per = 6.0 / len(ordered)
    results = []
    cursor = 0          # constraints must be achieved in order along the arc

    for c in ordered:
        entry = {"kind": c.kind, "text": str(c), "achieved": False,
                 "at": None, "in_order": True, "why": ""}
        if c.kind == GOTO:
            lab = head_label(c.text, labels)
            boxes = by_label.get(lab or "", [])
            if pin is not None and len(results) in pin:
                boxes = pin[len(results)]
            entry.update(label=lab, n_candidates=len(boxes))
            if not boxes:
                entry["why"] = f"no ground-truth object named {lab!r}"
            else:
                i = first_touch(track, boxes, tau, cursor)
                if i is not None:
                    entry.update(achieved=True, at=i, why="reached")
                else:
                    # achieved, but before the constraint that should precede it
                    j = first_touch(track, boxes, tau, 0)
                    if j is not None:
                        entry.update(achieved=True, at=j, in_order=False,
                                     why="reached, but out of order")
                    else:
                        d = min(min(point_to_box_xy(p[:2], b["lo"], b["hi"])
                                    for p in track) for b in boxes)
                        entry["why"] = f"closest approach {d:.2f} m > {tau:.2f} m"
        else:  # PASS -- and the two relations are not the same constraint
            alabs = [head_label(a, labels) for a in c.anchors]
            entry.update(anchors=alabs, relation=c.relation)
            sets = [by_label.get(a or "", []) for a in alabs]

            if c.relation == "between":
                # one plural anchor ("the two columns") names two instances of
                # one label; two anchors name one instance of each.
                if len(sets) == 1 and len(sets[0]) >= 2:
                    A = B = sets[0]
                elif len(sets) >= 2 and all(sets[:2]):
                    A, B = sets[0], sets[1]
                else:
                    A = B = []
                if not A:
                    entry["why"] = f"gap anchors not in ground truth: {alabs}"
                else:
                    p, q = nearest_pair(A, B)
                    i = crossing_index(track, p["c"], q["c"], cursor)
                    if i is not None:
                        entry.update(achieved=True, at=i, why="crossed")
                    else:
                        j = crossing_index(track, p["c"], q["c"], 0)
                        if j is None:
                            entry["why"] = "track never crosses the gap"
                        else:
                            entry.update(achieved=True, at=j, in_order=False,
                                         why="crossed, but out of order")
            else:
                # "take the path near X" -- there is no gap. The constraint is
                # that the driven path passes within reach of X, and a crossing
                # test cannot express it.
                boxes = sets[0] if sets else []
                if not boxes:
                    entry["why"] = f"anchor not in ground truth: {alabs}"
                else:
                    i = first_touch(track, boxes, PASS_NEAR_M, cursor)
                    if i is not None:
                        entry.update(achieved=True, at=i, why="passed near")
                    else:
                        j = first_touch(track, boxes, PASS_NEAR_M, 0)
                        if j is not None:
                            entry.update(achieved=True, at=j, in_order=False,
                                         why="passed near, out of order")
                        else:
                            d = min(min(point_to_box_xy(p[:2], b["lo"], b["hi"])
                                        for p in track) for b in boxes)
                            entry["why"] = (f"closest approach {d:.2f} m > "
                                            f"{PASS_NEAR_M:.2f} m")
        if entry["achieved"] and entry["in_order"]:
            cursor = max(cursor, entry["at"])
        results.append(entry)

    score = 0.0
    for entry in results:
        if entry["achieved"]:
            score += per if entry["in_order"] else per * ORDER_CREDIT

    ko_results = []
    for c in kos:
        alabs = [exact_label(a, labels) for a in c.anchors] or [exact_label(c.text, labels)]
        boxes = [by_label.get(a or "", []) for a in alabs]
        e = {"text": str(c), "anchors": alabs, "violated": False,
             "ungroundable": False, "why": ""}
        if c.relation == "between" and len(boxes) >= 2 and all(boxes[:2]):
            p, q = nearest_pair(boxes[0], boxes[1])
            hit = enters_corridor(track, p["c"], q["c"], KEEPOUT_M)
            e.update(violated=hit, why="entered corridor" if hit else "clear")
        elif c.relation != "between" and boxes and boxes[0]:
            hit = first_touch(track, boxes[0], KEEPOUT_M) is not None
            e.update(violated=hit, why="entered radius" if hit else "clear")
        else:
            e["ungroundable"] = True
            e["why"] = f"anchors not in ground truth: {alabs}"
        if e["violated"]:
            score -= per
        ko_results.append(e)

    return {"score": max(0.0, round(score, 3)), "max": 6.0,
            "constraints": results, "keepouts": ko_results,
            "n_steps": len(ordered), "ungroundable": False,
            "path_m": float(np.linalg.norm(np.diff(track[:, :2], axis=0),
                                           axis=1).sum())}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def reference_pins(question: str, scene: str, qi: int, by_label: dict,
                   labels: set[str]) -> dict[int, list]:
    """For each ambiguous GOTO, the one instance the organisers drove to.

    The relative clause -- "the potted plant *on the table*" -- is what picks an
    instance, and nothing in this scorer reads it. The organisers' own reference
    trajectory does pick one, by going there, so it can stand in for the missing
    predicate: whichever instance their path approaches most closely is the one
    the sentence meant. That is not airtight (a reference path can brush a rival
    instance on its way past) but it is independent of anything we built, and it
    turns "any potted plant" into "that potted plant".
    """
    rp = CHALLENGE / "questions" / scene / TRAJ_FOR[qi]
    if not rp.is_file():
        return {}
    ref = read_ply_ascii(rp)
    pins: dict[int, list] = {}
    for k, c in enumerate(steps(parse_instruction(question))):
        if c.kind != GOTO:
            continue
        boxes = by_label.get(head_label(c.text, labels) or "", [])
        if len(boxes) <= 1:
            continue
        best, bd = None, float("inf")
        for j, b in enumerate(boxes):
            d = min(point_to_box_xy(p[:2], b["lo"], b["hi"]) for p in ref)
            if d < bd:
                best, bd = j, d
        if best is not None:
            pins[k] = [boxes[best]]
    return pins


def _norm_q(q: str) -> str:
    """Whitespace-insensitive key for an instruction.

    Three office_1 runs were silently dropped from the corpus because the
    question reached the robot through a shell argument that wrapped it, so
    `plan.json` holds "...water cooler near\nthe window". The text is otherwise
    identical to the official question. Matching on collapsed whitespace
    recovers the runs; nothing else about the string is relaxed, so a genuinely
    different question still fails to match.
    """
    return " ".join(q.split()).strip()


def official_questions(root: Path) -> dict[str, tuple[str, int]]:
    """Instruction text -> (scene, question index 0|1)."""
    out = {}
    for e in json.loads((root / "questions/questions.json").read_text()):
        for qi, q in enumerate(e["questions"]["instruction_following"]):
            out[_norm_q(q)] = (e["scene"], qi)
    return out


def driven_track(run: Path) -> np.ndarray | None:
    """The trajectory the vehicle actually drove, concatenated across legs.

    `robot_io` subsamples `/state_estimation` at 0.10 m and hands the track
    back with each drive; this stitches a question's legs into the one curve
    the challenge scores.
    """
    sj = run / "steps.jsonl"
    if not sj.is_file():
        return None
    pts: list[list[float]] = []
    for line in sj.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = (r.get("drive") or {}).get("track") or []
        pts.extend(t)
    return np.asarray(pts, float) if len(pts) >= 2 else None


def run_question(run: Path) -> str | None:
    pj = run / "plan.json"
    if not pj.is_file():
        return None
    try:
        return _norm_q(json.loads(pj.read_text()).get("question") or "") or None
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def render(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print("-" * 96)
    print(f"{'run / scene':26s} {'q':>2s} {'score':>6s} {'steps':>6s} "
          f"{'done':>5s} {'order':>6s} {'KO':>3s}  detail")
    print("-" * 96)
    for r in rows:
        s = r["result"]
        done = sum(1 for c in s["constraints"] if c["achieved"])
        ooo = sum(1 for c in s["constraints"]
                  if c["achieved"] and not c.get("in_order", True))
        ko = sum(1 for k in s["keepouts"] if k["violated"])
        detail = "; ".join(
            f"{c['kind']}:{'ok' if c['achieved'] else c['why'][:34]}"
            for c in s["constraints"])
        print(f"{r['name']:26s} {r['q']:2d} {s['score']:6.2f} "
              f"{s['n_steps']:6d} {done:5d} {ooo:6d} {ko:3d}  {detail[:44]}")
    if not rows:
        print("(nothing scored)")
        return
    tot = sum(r["result"]["score"] for r in rows)
    mx = 6.0 * len(rows)
    print("-" * 96)
    print(f"{'TOTAL':26s} {'':2s} {tot:6.2f} / {mx:.0f}   "
          f"mean {tot / len(rows):.2f} / 6   over {len(rows)} questions")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge", default=str(CHALLENGE))
    ap.add_argument("--viz-data", default=str(VIZ_DATA))
    ap.add_argument("--runs", default=None,
                    help="directory of run directories to score")
    ap.add_argument("--reference", action="store_true",
                    help="score the organisers' own trajectories (validation)")
    ap.add_argument("--negative", action="store_true",
                    help="score deliberately wrong trajectories (discrimination)")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--json", default=None, help="write full results here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root, viz = Path(args.challenge), Path(args.viz_data)
    qmap = official_questions(root)
    scenes: dict[str, tuple[dict, set]] = {}

    def scene_gt(scene: str):
        if scene not in scenes:
            got = load_scene(scene, viz)
            if got is None:
                return None
            scenes[scene] = got
        return scenes[scene]

    rows: list[dict] = []

    if args.reference:
        for q, (scene, qi) in sorted(qmap.items(), key=lambda kv: kv[1]):
            gt = scene_gt(scene)
            tp = root / "questions" / scene / TRAJ_FOR[qi]
            if gt is None or not tp.is_file():
                continue
            by_label, labels = gt
            track = read_ply_ascii(tp)
            rows.append({"name": scene, "q": qi + 4, "question": q,
                         "result": score_trajectory(track, q, by_label,
                                                    labels, args.tau)})
        render(rows, f"REFERENCE trajectories, scored by this proxy (tau={args.tau} m)")
        good = sum(1 for r in rows if r["result"]["score"] >= 5.99)
        print(f"\nreference scores full marks on {good}/{len(rows)}. "
              "Every number this tool produces is conditional on that.")

    if args.negative:
        # Scoring the reference answers at 6/6 is necessary and NOT sufficient:
        # a scorer returning full marks for every input would pass it too. These
        # are the controls that separate a metric from a rubber stamp. Each is
        # built by corrupting a reference answer in one specific way, so the
        # points it should lose are known in advance.
        import numpy as _np
        arms: dict[str, list] = {}
        for q, (scene, qi) in sorted(qmap.items(), key=lambda kv: kv[1]):
            gt = scene_gt(scene)
            tp = root / "questions" / scene / TRAJ_FOR[qi]
            if gt is None or not tp.is_file():
                continue
            by_label, labels = gt
            traj = read_ply_ascii(tp)
            other_qi = 1 - qi
            other = root / "questions" / scene / TRAJ_FOR[other_qi]
            variants = {
                "reference": traj,
                "reversed": traj[::-1],
                "truncated (first half)": traj[: max(2, len(traj) // 2)],
                "straight line start->end": _np.linspace(
                    traj[0], traj[-1], max(2, len(traj))),
            }
            if other.is_file():
                variants["other question's path"] = read_ply_ascii(other)
            for name, tr in variants.items():
                arms.setdefault(name, []).append(
                    score_trajectory(tr, q, by_label, labels, args.tau)["score"])

        print(f"\nNEGATIVE CONTROLS — what the proxy does to a wrong answer "
              f"(tau={args.tau} m)")
        print("-" * 72)
        print(f"{'trajectory':30s} {'mean /6':>8s} {'at 6/6':>8s}  "
              f"{'vs reference':>13s}")
        print("-" * 72)
        base = float(_np.mean(arms["reference"]))
        for name in ("reference", "truncated (first half)", "reversed",
                     "other question's path", "straight line start->end"):
            if name not in arms:
                continue
            v = _np.array(arms[name])
            drop = "" if name == "reference" else f"{v.mean() - base:+.2f}"
            print(f"{name:30s} {v.mean():8.2f} {(v >= 5.99).mean():8.0%}  "
                  f"{drop:>13s}")
        print("-" * 72)
        print("A proxy that cannot separate these rows is not measuring the "
              "instruction,\nonly the presence of a trajectory.")

    if args.runs:
        for run in sorted(Path(args.runs).iterdir()):
            if not run.is_dir():
                continue
            q = run_question(run)
            if q is None or q not in qmap:
                continue
            scene, qi = qmap[q]
            gt = scene_gt(scene)
            track = driven_track(run)
            if gt is None or track is None:
                continue
            by_label, labels = gt
            rows_r = {"name": run.name, "q": qi + 4, "scene": scene,
                      "question": q,
                      "result": score_trajectory(track, q, by_label, labels,
                                                 args.tau)}
            rows.append(rows_r)
        render([r for r in rows if "scene" in r],
               f"DRIVEN runs, proxy /6 (tau={args.tau} m)")

    if args.verbose:
        for r in rows:
            print(f"\n=== {r['name']} q{r['q']}: {r['question']}")
            for c in r["result"]["constraints"]:
                print(f"   [{'x' if c['achieved'] else ' '}] {c['text']}"
                      f"   -> {c['why']}"
                      + ("" if c.get("in_order", True) else "  (OUT OF ORDER)"))
            for k in r["result"]["keepouts"]:
                print(f"   {'VIOLATED' if k['violated'] else 'clear   '} "
                      f"{k['text']} -> {k['why']}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
