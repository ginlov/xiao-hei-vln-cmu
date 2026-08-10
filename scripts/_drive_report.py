#!/usr/bin/env python3
"""Print what `scripts/drive.sh` got back. Reads one JSON object on stdin.

Split out of the shell script because the formatting has to survive being
nested inside ssh -> docker exec -> bash -lc, and every layer eats a level of
quoting.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "drive"
    raw = sys.stdin.read().strip()
    if not raw:
        print("no JSON came back — is the container up?", file=sys.stderr)
        return 1
    r = json.loads(raw.splitlines()[-1])

    if what == "where":
        p = (r.get("pose") or {}).get("position")
        print(f"pose  ({p[0]:+.2f}, {p[1]:+.2f})" if p else "no pose")
        return 0

    print(f"  why           {r.get('why')}")
    print(f"  moved         {(r.get('moved_m') or 0):.2f} m")
    if r.get("pose"):
        print(f"  final pose    ({r['pose'][0]:+.2f}, {r['pose'][1]:+.2f})")
    print(f"  gap to asked  {(r.get('dist_to_requested_m') or 0):.2f} m")
    track = r.get("track") or []
    print(f"  track         {len(track)} points")
    for p in track:
        print(f"      ({p[0]:+.2f}, {p[1]:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
