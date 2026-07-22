"""Screenshot the simulator's RViz window off the X server.

Companion to ``_visualize.py``: that renders the explorer's own view of the
world, this captures what the simulator drew — the traversed path over the
scene mesh, as RViz shows it.

RViz has no service or CLI hook to dump its own render, so the only way to
get its pixels is to read the window off the X server it draws into.  The
sim renders into the host's display, so the window is an ordinary X client.

Xlib and pillow are imported inside the functions, so importing this module
never drags an X11 dependency into a headless run — the same way
``_visualize.py`` treats matplotlib.  Requires the ``[exploration]`` extra.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

DEFAULT_MATCH = r"rviz"


class CaptureError(RuntimeError):
    """No window could be grabbed (no display, no match, or a failed read)."""


def _open(display_name: str | None):
    from Xlib import display as xdisplay

    target = display_name or os.environ.get("DISPLAY") or ":0"
    try:
        return xdisplay.Display(target), target
    except Exception as exc:  # Xlib raises assorted types here
        raise CaptureError(f"cannot open X display {target!r}: {exc}") from exc


def _titled_windows(dsp):
    """Yield (window, title) for every titled window, depth-first."""
    from Xlib import error

    stack = [dsp.screen().root]
    while stack:
        win = stack.pop()
        try:
            stack.extend(win.query_tree().children)
        except error.XError:
            continue

        title = None
        try:
            prop = win.get_full_property(
                dsp.intern_atom("_NET_WM_NAME"), dsp.intern_atom("UTF8_STRING")
            )
            raw = prop.value if prop else win.get_wm_name()
            if raw:
                title = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        except error.XError:
            pass

        if title:
            yield win, title


def list_windows(display_name: str | None = None) -> list[str]:
    """Every titled window on the display — use it to find RViz's real title."""
    from Xlib import error

    dsp, _ = _open(display_name)
    try:
        out = []
        for win, title in _titled_windows(dsp):
            try:
                geom = win.get_geometry()
                out.append(f"{title}  [{geom.width}x{geom.height}]")
            except error.XError:
                out.append(title)
        return out
    finally:
        dsp.close()


def save_rviz_screenshot(
    output_path: Path | str,
    display_name: str | None = None,
    match: str = DEFAULT_MATCH,
) -> Path:
    """Grab the window whose title matches ``match`` and write it as a PNG.

    Raises CaptureError on any failure.  Callers in the tick loop treat that
    as non-fatal — and it deliberately never falls back to a full-screen
    grab, since a screenshot of the bare desktop would look like a good run.
    """
    from PIL import Image
    from Xlib import X, error

    dsp, target = _open(display_name)
    try:
        regex = re.compile(match, re.IGNORECASE)
        hits = [(w, t) for w, t in _titled_windows(dsp) if regex.search(t)]
        if not hits:
            raise CaptureError(f"no window title matched {match!r} on {target}")

        # Largest match wins — RViz spawns small tool and tooltip windows that
        # also carry "RViz" in the title; we want the main canvas.
        win, title = max(hits, key=lambda hit: _area(hit[0]))

        # X11 without a compositor does not retain the pixels of obscured
        # regions, so an overlapped window reads back as garbage. Raising it
        # first is what makes the grab reliable.
        try:
            win.configure(stack_mode=X.Above)
            dsp.sync()
            time.sleep(0.4)
        except error.XError:
            pass  # unmanaged or already topmost — grab it as-is

        try:
            geom = win.get_geometry()
            raw = win.get_image(0, 0, geom.width, geom.height, X.ZPixmap, 0xFFFFFFFF)
        except error.XError as exc:
            raise CaptureError(f"grab of window {title!r} failed: {exc}") from exc

        data = raw.data
        # 4 bytes/pixel on the usual 24-bit-depth/32-bpp visual, 3 on a packed one.
        stride = len(data) // (geom.width * geom.height)
        if stride not in (3, 4):
            raise CaptureError(f"unexpected pixel stride {stride} for {title!r}")
        image = Image.frombytes(
            "RGB", (geom.width, geom.height), data, "raw", "BGRX" if stride == 4 else "BGR"
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        return out
    finally:
        dsp.close()


def _area(win) -> int:
    from Xlib import error

    try:
        geom = win.get_geometry()
        return int(geom.width) * int(geom.height)
    except error.XError:
        return 0
