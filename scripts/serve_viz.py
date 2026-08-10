#!/usr/bin/env python3
"""Serve the perception viewer, fetching its two vendor scripts on first run.

The viewer is plain files, so any static server works; this one exists so that
`viz/vendor/` populates itself and the correct headers go out for the float32
blob. Runs on whichever machine holds `viz/data` -- on the box, reach it with

    ssh -N -L 8765:localhost:8765 "${XIAO_HEI_SIM_HOST:-xiaohei1}"

and open http://localhost:8765 in a local browser. No desktop session needed.

    uv run python scripts/serve_viz.py [--port 8765] [--bind 127.0.0.1]
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
import urllib.request
from pathlib import Path

# r128 is the last release whose builds are plain <script> globals rather than
# ES modules, which keeps the page dependency-free and openable from file://.
VENDOR = {
    "three.min.js": "https://unpkg.com/three@0.128.0/build/three.min.js",
    "OrbitControls.js":
        "https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js",
}


def ensure_vendor(root: Path) -> bool:
    vd = root / "vendor"
    vd.mkdir(parents=True, exist_ok=True)
    for name, url in VENDOR.items():
        dst = vd / name
        if dst.is_file() and dst.stat().st_size > 1000:
            continue
        print(f"fetching {name} …", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                dst.write_bytes(r.read())
        except Exception as exc:                     # noqa: BLE001
            print(f"could not fetch {url}: {exc}\n"
                  f"download it by hand into {dst}", file=sys.stderr)
            return False
    return True


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".bin": "application/octet-stream",
                      ".js": "text/javascript", ".json": "application/json"}

    def end_headers(self) -> None:
        # The blob is refetched on every scene switch; caching it is the whole
        # reason a 40 MB scene feels instant the second time.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args) -> None:
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--root", default="viz")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "index.html").is_file():
        print(f"no viewer at {root}", file=sys.stderr)
        return 1
    if not ensure_vendor(root):
        return 1
    if not (root / "data" / "index.json").is_file():
        print("warning: viz/data is empty — run scripts/export_viz.py first",
              file=sys.stderr)

    socketserver.TCPServer.allow_reuse_address = True
    handler = functools.partial(Handler, directory=str(root))
    with socketserver.ThreadingTCPServer((args.bind, args.port), handler) as srv:
        print(f"serving {root} at http://{args.bind}:{args.port}", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
