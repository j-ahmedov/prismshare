"""Fullscreen frame player for the transmitter.

    python -m prism_share.transmit.display frames --colour-depth 4 --cell-px 8 --payload-bytes 200000 --out data/runs/c4_px8
    python -m prism_share.transmit.display play data/runs/c4_px8 --reference
    python -m prism_share.transmit.display calibrate
    python -m prism_share.transmit.display verify screenshot.png --frame data/runs/c4_px8/frame_00000.png

Guarantees made by this module (the OS and panel are outside its control; see
README, "Transmitter"):

* **Nearest-neighbour, integer scaling only.** A frame is shown at the largest
  integer scale that fits the screen (or ``--scale``), each frame pixel
  replicated into an s x s block with ``np.repeat``. Nothing is ever
  interpolated, and a frame larger than the screen is refused, never shrunk.
* **No colour management, ICC profile or gamma correction.** Pixel values go
  from the PNG to the window unmodified; PNGs are read without applying any
  embedded profile, and any colour chunk in an input PNG is reported.
* **Every value written is logged.** A JSON Lines log records the session
  (screen, scale, placement, surround colour), each item's exact set of
  distinct RGB triples with pixel counts and SHA-256 hashes of the frame and
  of the composed screen buffer, and every display event with its monotonic
  timestamp.
* The same code path serves the free-running sequence, single-frame hold
  (arrow keys step), the reference frame and ``calibrate`` (solid patches).

Keys: space = play/pause, right/down = next, left/up = previous,
r = reference frame, q/Esc = quit. Stepping pauses free-running playback.

``verify`` checks a screenshot of the running player against a frame: exact
match at an integer scale proves that neither the OS nor the window system
scaled, smoothed or colour-managed the pixels on their way to the framebuffer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image

from prism_share.codec.encoder import encode_frames, save_frames
from prism_share.codec.framing import frame_capacity
from prism_share.codec.palette import palette
from prism_share.codec.params import (
    ALLOWED_COLOUR_DEPTHS,
    BLACK_RGB,
    CALIBRATION_GREY_LEVELS,
    DISPLAY_DEFAULT_INTERVAL_MS,
    DISPLAY_SURROUND_RGB,
    KEEPOUT_PX,
    WHITE_RGB,
    CodecParams,
)
from prism_share.codec.prng import keystream
from prism_share.transmit.reference import reference_frame

UInt8Array = npt.NDArray[np.uint8]
WINDOW_NAME = "prism-share"
_MIN_SCREEN_PX = 64  # a detected size below this means detection failed
_PNG_COLOUR_CHUNKS = ("icc_profile", "gamma", "srgb", "chromaticity")

# --------------------------------------------------------------------------- #
# Items and pure helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Item:
    """One thing the player can show: a frame, the reference or a calibration patch."""

    name: str
    image: UInt8Array
    """(H, W, 3) uint8 RGB. Solid patches are (1, 1, 3) and fill the whole screen."""
    kind: str = "frame"
    """'frame', 'reference' or 'patch'."""


@dataclass(frozen=True)
class Placement:
    screen_w: int
    screen_h: int
    scale: int
    x: int
    y: int


def placement(frame_w: int, frame_h: int, screen_w: int, screen_h: int, scale: int | None = None) -> Placement:
    """Centre a frame at the largest integer scale that fits (or at ``scale``)."""
    fit = min(screen_w // frame_w, screen_h // frame_h)
    if fit < 1:
        raise ValueError(f"a {frame_w}x{frame_h} frame does not fit a {screen_w}x{screen_h} screen at scale 1; refusing to shrink it")
    s = fit if scale is None else scale
    if not 1 <= s <= fit:
        raise ValueError(f"scale {s} does not fit; the largest integer scale is {fit}")
    return Placement(screen_w, screen_h, s, (screen_w - s * frame_w) // 2, (screen_h - s * frame_h) // 2)


def upscale_nearest(image: UInt8Array, scale: int) -> UInt8Array:
    """Replicate each pixel into a scale x scale block. The only scaling this project performs."""
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def compose_screen(item: Item, screen_w: int, screen_h: int, scale: int | None = None,
                   surround: tuple[int, int, int] = DISPLAY_SURROUND_RGB) -> tuple[UInt8Array, Placement | None]:
    """The exact RGB buffer handed to the window, and where the frame sits in it."""
    if item.kind == "patch":
        return np.broadcast_to(item.image.reshape(1, 1, 3), (screen_h, screen_w, 3)).copy(), None
    h, w = item.image.shape[:2]
    where = placement(w, h, screen_w, screen_h, scale)
    screen = np.empty((screen_h, screen_w, 3), dtype=np.uint8)
    screen[...] = surround
    screen[where.y : where.y + where.scale * h, where.x : where.x + where.scale * w] = upscale_nearest(item.image, where.scale)
    return screen, where


def colour_census(image: UInt8Array) -> list[dict[str, Any]]:
    """Every distinct RGB triple in ``image`` with its pixel count, sorted by RGB."""
    colours, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    return [{"rgb": [int(v) for v in c], "pixels": int(n)} for c, n in zip(colours, counts, strict=True)]


def image_sha256(image: UInt8Array) -> str:
    """SHA-256 over shape and raw RGB bytes."""
    h = hashlib.sha256(repr(image.shape).encode())
    h.update(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
    return h.hexdigest()


def load_png(path: Path) -> tuple[UInt8Array, list[str]]:
    """Read a PNG's stored RGB values verbatim; return them and any colour-management chunks found."""
    with Image.open(path) as im:
        found = [k for k in _PNG_COLOUR_CHUNKS if k in im.info]
        if im.mode != "RGB":
            raise ValueError(f"{path}: expected an 8-bit RGB PNG, got mode {im.mode}")
        return np.array(im, dtype=np.uint8), found


def load_frames(directory: Path) -> list[Item]:
    paths = sorted(directory.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"no frame_*.png in {directory}")
    items = []
    for p in paths:
        image, chunks = load_png(p)
        if chunks:
            print(f"warning: {p.name} carries {chunks}; ignored, values shown as stored", file=sys.stderr)
        items.append(Item(p.name, image, "frame"))
    return items


def calibration_items() -> list[Item]:
    """Solid patches: black, white, the RGB primaries and secondaries, every palette
    colour of every depth, and an evenly spaced grey ramp. Duplicates removed, order kept."""
    top = max(WHITE_RGB)
    named: list[tuple[str, tuple[int, int, int]]] = [
        ("black", BLACK_RGB), ("white", WHITE_RGB),
        ("red", (top, 0, 0)), ("green", (0, top, 0)), ("blue", (0, 0, top)),
        ("cyan", (0, top, top)), ("magenta", (top, 0, top)), ("yellow", (top, top, 0)),
    ]
    for depth in ALLOWED_COLOUR_DEPTHS:
        named += [(f"palette{depth}[{i}]", tuple(int(v) for v in c)) for i, c in enumerate(palette(depth))]  # type: ignore[misc]
    levels = np.rint(np.linspace(0, top, CALIBRATION_GREY_LEVELS)).astype(int)
    named += [(f"grey{v}", (int(v), int(v), int(v))) for v in levels]
    seen: set[tuple[int, int, int]] = set()
    items = []
    for name, rgb in named:
        if rgb in seen:
            continue
        seen.add(rgb)
        items.append(Item(f"{name} {rgb}", np.array(rgb, dtype=np.uint8).reshape(1, 1, 3), "patch"))
    return items


# --------------------------------------------------------------------------- #
# Keys and player state
# --------------------------------------------------------------------------- #

# waitKeyEx codes differ per HighGUI backend: Windows, GTK, Qt, macOS Cocoa.
_KEYCODES: dict[str, set[int]] = {
    "right": {2555904, 65363, 0x01000014, 63235},
    "left": {2424832, 65361, 0x01000012, 63234},
    "up": {2490368, 65362, 0x01000013, 63232},
    "down": {2621440, 65364, 0x01000015, 63233},
    "space": {ord(" ")},
    "quit": {27, ord("q"), ord("Q")},
    "reference": {ord("r"), ord("R")},
}


def normalise_key(code: int) -> str | None:
    """Map a backend key code to 'right', 'left', 'up', 'down', 'space', 'quit', 'reference' or None."""
    if code < 0:
        return None
    for name, codes in _KEYCODES.items():
        if code in codes:
            return name
    return None


@dataclass
class Player:
    """Pure playback state: which item is shown and whether playback runs."""

    n_items: int
    paused: bool = False
    loop: bool = True
    index: int = 0
    reference_index: int | None = None
    finished: bool = False

    def advance(self) -> None:
        """Free-running step to the next frame (skipping the reference)."""
        nxt = self.index + 1
        if nxt == self.reference_index:
            nxt += 1
        if nxt >= self.n_items:
            if not self.loop:
                self.finished = True
                return
            nxt = 0 if self.reference_index != 0 else 1
        self.index = nxt

    def on_key(self, key: str | None) -> None:
        if key == "quit":
            self.finished = True
        elif key == "space":
            self.paused = not self.paused
        elif key in ("right", "down"):
            self.paused = True
            self.index = min(self.index + 1, self.n_items - 1)
        elif key in ("left", "up"):
            self.paused = True
            self.index = max(self.index - 1, 0)
        elif key == "reference" and self.reference_index is not None:
            self.paused = True
            self.index = self.reference_index


# --------------------------------------------------------------------------- #
# Log
# --------------------------------------------------------------------------- #


@dataclass
class DisplayLog:
    """JSON Lines log. One 'session' record, one 'item' record per item, one 'show' per display."""

    path: Path | None
    records: list[dict[str, Any]] = field(default_factory=list)

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class Backend(Protocol):
    def open(self) -> tuple[int, int]: ...
    def show(self, screen_rgb: UInt8Array) -> None: ...
    def wait_key(self, timeout_ms: int) -> str | None: ...
    def close(self) -> None: ...


class OpenCVBackend:
    """A borderless fullscreen OpenCV HighGUI window.

    The buffer given to ``show`` is always exactly the window size, so HighGUI
    has nothing to rescale. ``screen`` overrides size detection, which on HiDPI
    (Retina) displays may report points rather than pixels - verify with a
    screenshot (``verify``) before trusting a new machine.
    """

    def __init__(self, screen: tuple[int, int] | None = None, windowed: bool = False) -> None:
        self._screen = screen
        self._windowed = windowed

    def open(self) -> tuple[int, int]:
        import cv2

        flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
        cv2.namedWindow(WINDOW_NAME, flags)
        if not self._windowed:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if self._screen is not None:
            if self._windowed:
                cv2.resizeWindow(WINDOW_NAME, *self._screen)
            return self._screen
        # Let the window system settle, then read the client area size.
        probe = np.zeros((_MIN_SCREEN_PX, _MIN_SCREEN_PX, 3), dtype=np.uint8)
        for _ in range(10):
            cv2.imshow(WINDOW_NAME, probe)
            cv2.waitKey(30)
        _, _, w, h = cv2.getWindowImageRect(WINDOW_NAME)
        if w < _MIN_SCREEN_PX or h < _MIN_SCREEN_PX:
            raise RuntimeError(f"could not detect the screen size (got {w}x{h}); pass --screen WIDTHxHEIGHT")
        return int(w), int(h)

    def show(self, screen_rgb: UInt8Array) -> None:
        import cv2

        cv2.imshow(WINDOW_NAME, np.ascontiguousarray(screen_rgb[:, :, ::-1]))  # HighGUI expects BGR; values untouched

    def wait_key(self, timeout_ms: int) -> str | None:
        import cv2

        return normalise_key(cv2.waitKeyEx(max(1, int(timeout_ms))))

    def close(self) -> None:
        import cv2

        cv2.destroyWindow(WINDOW_NAME)


class NullBackend:
    """No window: records what would be shown. For --dry-run and tests."""

    def __init__(self, screen: tuple[int, int], keys: Sequence[str | None] = (), max_waits: int | None = None) -> None:
        self.screen = screen
        self.keys = list(keys)
        self.max_waits = max_waits
        self.shown: list[UInt8Array] = []
        self.waits = 0

    def open(self) -> tuple[int, int]:
        return self.screen

    def show(self, screen_rgb: UInt8Array) -> None:
        self.shown.append(screen_rgb)

    def close(self) -> None:
        pass

    def wait_key(self, timeout_ms: int) -> str | None:
        self.waits += 1
        if self.max_waits is not None and self.waits > self.max_waits:
            return "quit"
        return self.keys.pop(0) if self.keys else None


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def run(
    items: list[Item],
    backend: Backend,
    log: DisplayLog,
    *,
    mode: str,
    interval_ms: int = DISPLAY_DEFAULT_INTERVAL_MS,
    start_paused: bool = False,
    loop: bool = True,
    scale: int | None = None,
    reference_ms: int = 0,
    session_extra: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Player:
    """Show ``items``; returns the final player state. The reference, if present, must be items[0]."""
    screen_w, screen_h = backend.open()
    ref_index = 0 if items and items[0].kind == "reference" else None
    player = Player(len(items), paused=start_paused, loop=loop, reference_index=ref_index)
    if ref_index is not None and not start_paused:
        player.index = 0  # the reference opens the session, then playback moves on

    buffers: dict[int, UInt8Array] = {}
    session = {
        "type": "session",
        "mode": mode,
        "started": dt.datetime.now(dt.timezone.utc).isoformat(),
        "screen_px": [screen_w, screen_h],
        "surround_rgb": list(DISPLAY_SURROUND_RGB),
        "interval_ms": interval_ms,
        "reference_ms": reference_ms,
        "scaling": "nearest-neighbour, integer only",
        "colour_management": "none (values written as stored)",
        **(session_extra or {}),
    }
    log.write(session)
    for i, item in enumerate(items):
        buf, where = compose_screen(item, screen_w, screen_h, scale)
        buffers[i] = buf
        log.write(
            {
                "type": "item",
                "index": i,
                "name": item.name,
                "kind": item.kind,
                "image_sha256": image_sha256(item.image),
                "screen_sha256": image_sha256(buf),
                "placement": None if where is None else {"scale": where.scale, "x": where.x, "y": where.y},
                "rgb_values": colour_census(item.image),
            }
        )

    # Free-running deadlines follow a fixed schedule (each item is due when the
    # previous one's time is up), so drawing time does not accumulate as drift.
    due: float | None = None
    try:
        while not player.finished:
            idx = player.index
            backend.show(buffers[idx])
            shown_at = clock()
            log.write({"type": "show", "index": idx, "name": items[idx].name, "t_monotonic_s": shown_at, "paused": player.paused})
            hold = reference_ms if (idx == ref_index and not player.paused) else interval_ms
            deadline = (due if due is not None else shown_at) + hold / 1000.0
            due = None
            while True:
                if player.paused:
                    key = backend.wait_key(1000)
                    before = (player.index, player.paused)
                    player.on_key(key)
                    if player.finished or (player.index, player.paused) != before:
                        break
                    continue
                remaining_ms = int((deadline - clock()) * 1000)
                key = backend.wait_key(max(1, remaining_ms))
                if key is not None:
                    player.on_key(key)
                    break
                if clock() >= deadline:
                    player.advance()
                    # Keep the schedule unless we have fallen a whole interval behind (then resynchronise).
                    due = deadline if clock() - deadline < interval_ms / 1000.0 else None
                    break
    finally:
        backend.close()
        log.write({"type": "end", "t_monotonic_s": clock()})
    return player


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VerifyResult:
    exact: bool
    scale: int | None
    x: int | None
    y: int | None
    mismatched_pixels: int
    max_abs_diff: int
    mean_diff_rgb: tuple[float, float, float]
    message: str


def verify_screenshot(screenshot: UInt8Array, frame: UInt8Array) -> VerifyResult:
    """Find ``frame`` in ``screenshot`` at an integer scale and compare every pixel."""
    import cv2

    fh, fw = frame.shape[:2]
    sh, sw = screenshot.shape[:2]
    patch = min(KEEPOUT_PX, fh, fw)  # the top-left keep-out square: identical in every frame, unique in the frame
    best: tuple[float, int, int, int] | None = None
    for s in range(1, min(sh // fh, sw // fw) + 1):
        corner = upscale_nearest(frame[:patch, :patch], s)
        res = cv2.matchTemplate(screenshot.astype(np.float32), corner.astype(np.float32), cv2.TM_SQDIFF)
        _, _, loc, _ = cv2.minMaxLoc(res)
        x, y = loc
        if y + s * fh > sh or x + s * fw > sw:
            continue
        region = screenshot[y : y + s * fh, x : x + s * fw].astype(np.int64)
        diff = region - upscale_nearest(frame, s).astype(np.int64)
        score = float(np.abs(diff).mean())
        if best is None or score < best[0]:
            best = (score, s, x, y)
    if best is None:
        return VerifyResult(False, None, None, None, -1, -1, (0.0, 0.0, 0.0), "frame not found at any integer scale")
    _, s, x, y = best
    region = screenshot[y : y + s * fh, x : x + s * fw].astype(np.int64)
    diff = region - upscale_nearest(frame, s).astype(np.int64)
    bad = int((diff != 0).any(axis=2).sum())
    mean = tuple(float(v) for v in diff.reshape(-1, 3).mean(axis=0))
    if bad == 0:
        msg = f"EXACT: every pixel matches at integer scale {s}, offset ({x}, {y})"
    elif np.abs(diff).max() <= 2 and bad > region.shape[0] * region.shape[1] // 2:
        msg = "values differ slightly everywhere: colour management or gamma adjustment suspected"
    else:
        msg = "localised mismatches: interpolated (non-integer or smoothed) scaling or overlay suspected"
    return VerifyResult(bad == 0, s, x, y, bad, int(np.abs(diff).max()), mean, msg)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _screen_arg(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    w, h = (int(v) for v in text.lower().split("x"))
    return w, h


def _default_log(mode: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("data") / "display_logs" / f"{stamp}_{mode}.jsonl"


def _backend(args: argparse.Namespace) -> Backend:
    screen = _screen_arg(args.screen)
    if args.dry_run is not None:
        return NullBackend(screen or (1920, 1080), max_waits=args.dry_run)
    return OpenCVBackend(screen, windowed=args.windowed)


def cmd_frames(args: argparse.Namespace) -> None:
    params = CodecParams(colour_depth=args.colour_depth, cell_px=args.cell_px, seed=args.seed)
    if args.payload_file:
        payload = Path(args.payload_file).read_bytes()
        source = {"file": str(args.payload_file), "sha256": hashlib.sha256(payload).hexdigest()}
    else:
        payload = keystream(params.seed, "run-payload", args.payload_bytes)
        source = {"keystream": {"seed": params.seed, "domain": "run-payload", "bytes": args.payload_bytes}}
    frames = encode_frames(payload, params, n_frames=args.n_frames)
    out = Path(args.out)
    paths = save_frames([f.image for f in frames], out)
    doc = {
        "params": params.to_dict(),
        "params_label": params.label,
        "params_fingerprint": params.fingerprint(),
        "payload": {"length": len(payload), **source},
        "block_bytes": frame_capacity(params).block_bytes,
        "frames": [{"file": p.name, "block_id": f.header.block_id, "image_sha256": image_sha256(f.image)} for p, f in zip(paths, frames, strict=True)],
    }
    (out / "frames.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {len(paths)} frames and frames.json to {out}")


def cmd_play(args: argparse.Namespace) -> None:
    directory = Path(args.directory)
    items = load_frames(directory)
    extra: dict[str, Any] = {"source": str(directory)}
    meta = directory / "frames.json"
    if meta.exists():
        extra["frames_json"] = json.loads(meta.read_text())
    if args.reference:
        h = items[0].image.shape[0]
        items = [Item("reference", np.asarray(reference_frame(h)), "reference"), *items]
    log = DisplayLog(Path(args.log) if args.log else _default_log("play"))
    run(items, _backend(args), log, mode="play", interval_ms=args.interval_ms, start_paused=args.hold,
        loop=not args.once, scale=args.scale, reference_ms=args.reference_ms, session_extra=extra)
    print(f"log: {log.path}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    log = DisplayLog(Path(args.log) if args.log else _default_log("calibrate"))
    run(calibration_items(), _backend(args), log, mode="calibrate", interval_ms=args.interval_ms,
        start_paused=not args.auto, loop=not args.once)
    print(f"log: {log.path}")


def cmd_verify(args: argparse.Namespace) -> int:
    with Image.open(args.screenshot) as im:  # screenshots are often RGBA; convert() applies no ICC profile
        screenshot = np.array(im.convert("RGB"), dtype=np.uint8)
    frame, _ = load_png(Path(args.frame))
    result = verify_screenshot(screenshot, frame)
    print(result.message)
    print(f"scale={result.scale} offset=({result.x}, {result.y}) mismatched_pixels={result.mismatched_pixels} "
          f"max_abs_diff={result.max_abs_diff} mean_diff_rgb={tuple(round(v, 3) for v in result.mean_diff_rgb)}")
    return 0 if result.exact else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def window_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--interval-ms", type=int, default=DISPLAY_DEFAULT_INTERVAL_MS, help="time per item when free-running")
        p.add_argument("--once", action="store_true", help="stop after the last item instead of looping")
        p.add_argument("--screen", default=None, metavar="WxH", help="screen size in pixels (overrides detection)")
        p.add_argument("--windowed", action="store_true", help="debug only: a normal window instead of fullscreen")
        p.add_argument("--dry-run", type=int, default=None, metavar="WAITS", help="no window; stop after WAITS key polls")
        p.add_argument("--log", default=None, help="JSON Lines log path (default data/display_logs/...)")

    f = sub.add_parser("frames", help="encode a run's frames to PNG + frames.json")
    f.add_argument("--colour-depth", type=int, required=True)
    f.add_argument("--cell-px", type=int, required=True)
    f.add_argument("--seed", type=int, default=0)
    src = f.add_mutually_exclusive_group(required=True)
    src.add_argument("--payload-bytes", type=int, help="deterministic pseudo-random payload of this length")
    src.add_argument("--payload-file", help="encode this file")
    f.add_argument("--n-frames", type=int, default=None, help="frames to generate (default: source + repair)")
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_frames)

    p = sub.add_parser("play", help="show a directory of frame_*.png")
    p.add_argument("directory")
    p.add_argument("--hold", action="store_true", help="start paused (single-frame hold)")
    p.add_argument("--scale", type=int, default=None, help="integer scale (default: largest that fits)")
    p.add_argument("--reference", action="store_true", help="open with the reference frame")
    p.add_argument("--reference-ms", type=int, default=3000, help="how long the reference opens the session")
    window_options(p)
    p.set_defaults(func=cmd_play)

    c = sub.add_parser("calibrate", help="full-screen solid RGB patches (arrow keys step)")
    c.add_argument("--auto", action="store_true", help="advance automatically every --interval-ms")
    window_options(c)
    c.set_defaults(func=cmd_calibrate)

    v = sub.add_parser("verify", help="check a screenshot shows a frame pixel-exactly")
    v.add_argument("screenshot")
    v.add_argument("--frame", required=True)
    v.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    result = args.func(args)
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
