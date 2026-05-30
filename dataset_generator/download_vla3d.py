"""Download the VLA-3D Unity subset (~2 GB) — the raw INPUT for the generators.

VLA-3D hosts its scene data on CMU AirLab's public Swift/S3 bucket, not in
the GitHub repo. We only need the Unity subset (`Unity.zip`), whose 15 scenes
match the challenge's training set. This wraps the same public endpoint that
VLA-3D's own download_dataset.py uses, but with stdlib only (no boto3).

Default target is `dataset_generator/vla-3d/`, so after extraction the scenes
land at `dataset_generator/vla-3d/Unity/<scene>/...` — exactly where
vla3d_loader.DEFAULT_VLA3D_ROOT looks. Override with --dest or VLA3D_ROOT.

    uv run python dataset_generator/download_vla3d.py
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Public, unsigned endpoint + bucket from VLA-3D's download_dataset.py.
_ENDPOINT = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
_BUCKET = "vla"
UNITY_ZIP_URL = f"{_ENDPOINT}/{_BUCKET}/Unity.zip"

# Where the loader expects the data: dataset_generator/vla-3d/Unity/
HERE = Path(__file__).parent
DEFAULT_DEST = HERE / "vla-3d"


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n       ->  {target}")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 1024 * 1024
        with target.open("wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = 100 * done / total
                    sys.stdout.write(
                        f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)"
                    )
                else:
                    sys.stdout.write(f"\r  {done / 1e6:8.1f} MB")
                sys.stdout.flush()
    sys.stdout.write("\n")


def _extract(zip_path: Path, dest: Path) -> None:
    print(f"Extracting {zip_path.name} -> {dest}/")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help=f"Where to extract Unity/ (default: {DEFAULT_DEST})",
    )
    ap.add_argument(
        "--zip", type=Path, default=None,
        help="Where to keep the downloaded Unity.zip (default: <dest>/Unity.zip)",
    )
    ap.add_argument(
        "--keep-zip", action="store_true",
        help="Keep Unity.zip after extracting (default: delete it)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-download / re-extract even if the data is already present",
    )
    args = ap.parse_args()

    dest: Path = args.dest
    unity_dir = dest / "Unity"
    zip_path: Path = args.zip or (dest / "Unity.zip")

    if unity_dir.is_dir() and any(unity_dir.iterdir()) and not args.force:
        print(f"{unity_dir} already populated — nothing to do (use --force to refetch).")
        return 0

    if not zip_path.exists() or args.force:
        _download(UNITY_ZIP_URL, zip_path)
    else:
        print(f"Reusing existing {zip_path} (use --force to refetch).")

    _extract(zip_path, dest)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)

    n_scenes = sum(1 for d in unity_dir.iterdir() if d.is_dir()) if unity_dir.is_dir() else 0
    print(f"\nDone. {n_scenes} scene dirs under {unity_dir}")
    print(f"Point the generators at it with:  export VLA3D_ROOT={unity_dir}")
    if n_scenes == 0:
        print("WARNING: no scene dirs found — the zip layout may differ from "
              "expectations; check the contents of", dest)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
