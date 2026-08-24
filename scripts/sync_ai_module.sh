#!/usr/bin/env bash
#
# Copy the instruction-following stack into the challenge repo's `ai_module/`.
#
# Only `ai_module/` may change in a submission, so the drive loop has to live
# there rather than be imported from here. Rather than refactor twelve tuned
# files' imports, the *shape* of this repo is mirrored under `ai_module/vlm/`:
# `scripts/`, `perception/` and `src/` sit in the same relative positions, so
# every `sys.path.insert(..., parent.parent / "perception")` in them resolves
# exactly as it does now and not one import line changes.
#
#   scripts/sync_ai_module.sh                    # copy, dev repo -> challenge
#   scripts/sync_ai_module.sh --check            # exit 1 if the two differ
#   CHALLENGE=/path/to/repo scripts/sync_ai_module.sh
#
# The dev repo is the source of truth. `--check` exists so that a fix made
# inside the container — which is where a failing eval run gets debugged — is
# never silently overwritten by the next sync.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHALLENGE="${CHALLENGE:-$(cd "$HERE/../CMU-VLN-Challenge-2026" && pwd)}"
DEST="$CHALLENGE/ai_module/vlm"

# The drive loop and its transitive imports. `vlm_gates.py` is deliberately not
# here: nothing imports it. `robot_io.py` is, even though the node no longer
# shells out to it, because it is still the way to poke one topic by hand from
# inside the container when a run misbehaves.
#
# `verify_binding.py` was missing from this list while `approach_loop.py`
# imported it at module scope, which would have taken the *whole* submission
# down at launch -- instruction following is 36 of the 51 available points and
# none of them survive a `ModuleNotFoundError` on import. It only stayed
# invisible because the copy under `ai_module/` had gone stale and predated the
# import. `--check` now catches that class of drift; keeping this list in step
# with the imports is the thing to watch when a new module lands.
SCRIPTS=(
  answer_numerical.py
  approach_loop.py
  count_view.py
  decompose.py
  execute_plan.py
  faces.py
  grab_faces.py
  instruction_plan.py
  numerical_plan.py
  robot_io.py
  verify_binding.py
  vlm_approach.py
  vlm_locate.py
  vlm_probe.py
  waypoint_converter_model.py
)

# `import geometry as G` resolves to this one, not dataset_generator's.
PERCEPTION=(geometry.py)

# Only the two leaves are taken. The real `xiao_hei_vln/perception/__init__.py`
# imports `PerceptionResponder`, which drags in httpx, pillow, pycocotools and
# the whole sidecar client — none of which the drive loop touches, and all of
# which would have to be installed into the ai_module image to satisfy an
# import of `sensor_to_camera_transform`. The vendored `__init__.py` is a stub;
# see the note it carries.
PKG=(geometry.py size_prior.py)

check=0
[[ "${1:-}" == "--check" ]] && check=1

fail=0
copy() {   # copy <src> <dst>
  local src="$1" dst="$2"
  if [[ $check -eq 1 ]]; then
    if ! diff -q "$src" "$dst" >/dev/null 2>&1; then
      echo "DRIFT  $dst"
      diff -u "$src" "$dst" | head -40 || true
      fail=1
    fi
  else
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "  $dst"
  fi
}

[[ $check -eq 1 ]] || echo "sync -> $DEST"

for f in "${SCRIPTS[@]}";    do copy "$HERE/scripts/$f"                        "$DEST/scripts/$f"; done
for f in "${PERCEPTION[@]}"; do copy "$HERE/perception/$f"                     "$DEST/perception/$f"; done
for f in "${PKG[@]}";        do copy "$HERE/src/xiao_hei_vln/perception/$f"    "$DEST/src/xiao_hei_vln/perception/$f"; done
copy "$HERE/src/xiao_hei_vln/__init__.py" "$DEST/src/xiao_hei_vln/__init__.py"

if [[ $check -eq 1 ]]; then
  [[ $fail -eq 0 ]] && echo "ai_module/vlm is in sync with $HERE"
  exit $fail
fi

# Not copied, authored in the challenge repo, listed here so the split is
# visible from one place:
#   vlm/challenge_node.py   the ROS node — what `dummyVLM` runs
#   vlm/robot_node.py       in-process replacement for `Robot`
#   vlm/classify.py         question-type router
#   vlm/src/xiao_hei_vln/perception/__init__.py   the stub described above
# Import the copied tree the way the container will. A module missing from
# SCRIPTS is invisible here until something imports it, and by then it is a
# launch failure in front of the graders.
if command -v python3 >/dev/null; then
  ( cd "$DEST" && python3 - <<'EOF' || echo "!! the synced tree does not import — see above"
import sys, pathlib
for d in ("scripts", "perception", "src"):
    sys.path.insert(0, str(pathlib.Path(d).resolve()))
import importlib.util, ast
missing, deferred = [], []
for f in sorted(pathlib.Path("scripts").glob("*.py")):
    tree = ast.parse(f.read_text())
    # A module-level import that is not in the image kills the process at
    # launch; one inside a function only kills the path that reaches it. Both
    # are worth knowing about and only the first is fatal.
    top = {id(n) for n in ast.walk(tree)
           if isinstance(n, (ast.Import, ast.ImportFrom))} & \
          {id(n) for n in tree.body}
    for node in ast.walk(tree):
        names = ([a.name.split(".")[0] for a in node.names]
                 if isinstance(node, ast.Import) else
                 [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom)
                 and node.module and node.level == 0 else [])
        for n in names:
            if (pathlib.Path("scripts") / f"{n}.py").exists():
                continue
            # What the ai_module image supplies and this laptop does not:
            # the Dockerfile's python3-opencv / python3-scipy / anthropic, and
            # everything ROS puts on the path. `google` is only reached by
            # `--backend gemini`, which the submission never selects.
            if n in {"cv2", "numpy", "scipy", "anthropic", "google",
                     "rclpy", "std_msgs", "sensor_msgs", "geometry_msgs",
                     "nav_msgs", "visualization_msgs", "builtin_interfaces"}:
                continue
            if importlib.util.find_spec(n) is None:
                where = missing if id(node) in top else deferred
                where.append(f"{f.name} imports {n}")
for line in sorted(set(deferred)):
    print(f"  note: {line} inside a function — only that path breaks")
if missing:
    print("\n".join(f"FATAL: {m} at module level" for m in sorted(set(missing))))
    raise SystemExit(1)
print("  every module-level import in scripts/ resolves inside the synced tree")
EOF
  )
fi

echo "done. node/robot/classify + the __init__ stub are authored in the"
echo "challenge repo and are not synced — see the tail of this script."
