"""Step 4: the transmitter's guarantees, tested without opening a window."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from prism_share.codec.encoder import encode, write_png
from prism_share.codec.palette import palette
from prism_share.codec.params import ALLOWED_COLOUR_DEPTHS, CALIBRATION_GREY_LEVELS, DISPLAY_SURROUND_RGB, CodecParams
from prism_share.transmit import display
from prism_share.transmit.display import DisplayLog, Item, NullBackend, Player

P = CodecParams(colour_depth=4, cell_px=5)
FRAMES = encode(b"display tests", P, n_frames=3)


class FakeClock:
    """Monotonic clock that advances a fixed step per call, so runs never sleep."""

    def __init__(self, step_s: float = 0.05) -> None:
        self.t = 0.0
        self.step = step_s

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _items() -> list[Item]:
    return [Item(f"frame_{i:05d}.png", f) for i, f in enumerate(FRAMES)]


# --------------------------------------------------------------------------- scaling


def test_placement_largest_integer_scale_centred() -> None:
    where = display.placement(1024, 1024, 3840, 2160)
    assert (where.scale, where.x, where.y) == (2, (3840 - 2048) // 2, (2160 - 2048) // 2)
    assert display.placement(1024, 1024, 1920, 1080).scale == 1


def test_placement_refuses_to_shrink_or_overscale() -> None:
    with pytest.raises(ValueError):
        display.placement(1024, 1024, 1366, 768)
    with pytest.raises(ValueError):
        display.placement(1024, 1024, 1920, 1080, scale=2)


def test_upscale_is_pixel_replication() -> None:
    img = np.random.default_rng(0).integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    assert np.array_equal(display.upscale_nearest(img, 3), np.kron(img, np.ones((3, 3, 1), dtype=np.uint8)))


@pytest.mark.parametrize("screen", [(1920, 1080), (2560, 2160)])
def test_composed_screen_is_the_frame_verbatim(screen: tuple[int, int]) -> None:
    item = _items()[0]
    buf, where = display.compose_screen(item, *screen)
    assert where is not None and buf.shape == (screen[1], screen[0], 3) and buf.dtype == np.uint8
    s = where.scale
    region = buf[where.y : where.y + s * 1024, where.x : where.x + s * 1024]
    assert np.array_equal(region, display.upscale_nearest(item.image, s))
    # No value appears that was not in the frame or the surround: nothing interpolated, nothing gamma-adjusted.
    frame_values = {tuple(v) for v in np.unique(item.image.reshape(-1, 3), axis=0)}
    assert {tuple(v) for v in np.unique(buf.reshape(-1, 3), axis=0)} <= frame_values | {DISPLAY_SURROUND_RGB}
    outside = np.ones(buf.shape[:2], dtype=bool)
    outside[where.y : where.y + s * 1024, where.x : where.x + s * 1024] = False
    assert (buf[outside] == DISPLAY_SURROUND_RGB).all()


def test_patch_fills_the_screen() -> None:
    buf, where = display.compose_screen(Item("red", np.array([[[255, 0, 0]]], dtype=np.uint8), "patch"), 64, 32)
    assert where is None and buf.shape == (32, 64, 3) and (buf == [255, 0, 0]).all()


# --------------------------------------------------------------------------- logging helpers


def test_colour_census_is_exact() -> None:
    census = display.colour_census(FRAMES[0])
    values = {tuple(c["rgb"]) for c in census}
    expected = {tuple(int(v) for v in c) for c in palette(P.colour_depth)} | {(0, 0, 0), (255, 255, 255)}
    assert values == expected
    assert sum(c["pixels"] for c in census) == 1024 * 1024


def test_image_hash_depends_on_shape_and_values() -> None:
    a = np.zeros((2, 3, 3), dtype=np.uint8)
    assert display.image_sha256(a) != display.image_sha256(a.reshape(3, 2, 3))
    b = a.copy()
    b[0, 0, 0] = 1
    assert display.image_sha256(a) != display.image_sha256(b)


def test_load_png_reports_colour_chunks_and_keeps_values(tmp_path: Path) -> None:
    img = FRAMES[0][:32, :32]
    tagged = tmp_path / "frame_00000.png"
    Image.fromarray(img).save(tagged, icc_profile=b"not really a profile")
    values, chunks = display.load_png(tagged)
    assert np.array_equal(values, img) and chunks == ["icc_profile"]
    Image.fromarray(img[:, :, 0]).save(tmp_path / "grey.png")
    with pytest.raises(ValueError):
        display.load_png(tmp_path / "grey.png")


def test_calibration_patches() -> None:
    items = display.calibration_items()
    rgbs = [tuple(int(v) for v in it.image.reshape(3)) for it in items]
    assert len(rgbs) == len(set(rgbs)), "duplicate patch"
    assert rgbs[:2] == [(0, 0, 0), (255, 255, 255)]
    for depth in ALLOWED_COLOUR_DEPTHS:
        assert {tuple(int(v) for v in c) for c in palette(depth)} <= set(rgbs)
    greys = {r for r in rgbs if r[0] == r[1] == r[2]}
    assert len(greys) == CALIBRATION_GREY_LEVELS
    assert all(it.kind == "patch" for it in items)


# --------------------------------------------------------------------------- keys and state


def test_key_normalisation_covers_backends() -> None:
    assert display.normalise_key(63235) == "right"  # macOS Cocoa
    assert display.normalise_key(65361) == "left"  # GTK
    assert display.normalise_key(2490368) == "up"  # Windows
    assert display.normalise_key(0x01000015) == "down"  # Qt
    assert display.normalise_key(27) == display.normalise_key(ord("q")) == "quit"
    assert display.normalise_key(ord(" ")) == "space"
    assert display.normalise_key(-1) is None and display.normalise_key(ord("x")) is None


def test_player_free_running_loops_and_skips_reference() -> None:
    p = Player(n_items=4, reference_index=0)
    seen = []
    for _ in range(7):
        p.advance()
        seen.append(p.index)
    assert seen == [1, 2, 3, 1, 2, 3, 1]


def test_player_once_finishes() -> None:
    p = Player(n_items=2, loop=False)
    p.advance()
    p.advance()
    assert p.finished


def test_player_keys_step_and_pause() -> None:
    p = Player(n_items=3, reference_index=None)
    p.on_key("right")
    assert (p.index, p.paused) == (1, True)
    p.on_key("right")
    p.on_key("right")
    assert p.index == 2  # clamps at the end
    p.on_key("left")
    assert p.index == 1
    p.on_key("space")
    assert not p.paused
    p.on_key("reference")  # no reference: ignored
    assert p.index == 1
    p.on_key("quit")
    assert p.finished


# --------------------------------------------------------------------------- run loop


def test_free_running_run_logs_every_value_written() -> None:
    backend = NullBackend((1920, 1080), max_waits=40)
    log = DisplayLog(None)
    display.run(_items(), backend, log, mode="play", interval_ms=100, clock=FakeClock())
    kinds = [r["type"] for r in log.records]
    assert kinds[0] == "session" and kinds[-1] == "end" and kinds.count("item") == 3
    session = log.records[0]
    assert session["screen_px"] == [1920, 1080] and session["surround_rgb"] == list(DISPLAY_SURROUND_RGB)
    items = [r for r in log.records if r["type"] == "item"]
    for rec, frame in zip(items, FRAMES, strict=True):
        assert rec["image_sha256"] == display.image_sha256(frame)
        assert rec["rgb_values"] == display.colour_census(frame)
        assert rec["placement"] == {"scale": 1, "x": 448, "y": 28}
    shows = [r["index"] for r in log.records if r["type"] == "show"]
    assert shows[:6] == [0, 1, 2, 0, 1, 2]
    # Every buffer shown is exactly the logged screen buffer.
    by_index = {r["index"]: r["screen_sha256"] for r in items}
    for idx, buf in zip(shows, backend.shown, strict=False):
        assert display.image_sha256(buf) == by_index[idx]
    times = [r["t_monotonic_s"] for r in log.records if r["type"] == "show"]
    assert times == sorted(times)


def test_single_frame_hold_steps_with_keys() -> None:
    backend = NullBackend((1920, 1080), keys=["right", None, "right", "left", "quit"])
    log = DisplayLog(None)
    display.run(_items(), backend, log, mode="play", start_paused=True, clock=FakeClock())
    assert [r["index"] for r in log.records if r["type"] == "show"] == [0, 1, 2, 1]


def test_reference_opens_the_session_then_codes_loop() -> None:
    ref = Item("reference", np.zeros((1024, 1024, 3), dtype=np.uint8), "reference")
    backend = NullBackend((1920, 1080), max_waits=60)
    log = DisplayLog(None)
    display.run([ref, *_items()], backend, log, mode="play", interval_ms=100, reference_ms=300, clock=FakeClock())
    shows = [r["index"] for r in log.records if r["type"] == "show"]
    assert shows[0] == 0 and 0 not in shows[1:]
    assert shows[1:7] == [1, 2, 3, 1, 2, 3]


def test_log_file_is_json_lines(tmp_path: Path) -> None:
    log = DisplayLog(tmp_path / "logs" / "session.jsonl")
    display.run(display.calibration_items()[:3], NullBackend((64, 64), max_waits=3), log, mode="calibrate",
                start_paused=True, clock=FakeClock())
    lines = [json.loads(line) for line in (tmp_path / "logs" / "session.jsonl").read_text().splitlines()]
    assert lines[0]["mode"] == "calibrate"
    assert [r["rgb_values"][0]["rgb"] for r in lines if r["type"] == "item"] == [[0, 0, 0], [255, 255, 255], [255, 0, 0]]


# --------------------------------------------------------------------------- verify


@pytest.mark.parametrize("screen", [(1920, 1080), (2200, 2100)])
def test_verify_accepts_an_exact_screenshot(screen: tuple[int, int]) -> None:
    buf, where = display.compose_screen(_items()[1], *screen)
    result = display.verify_screenshot(buf, FRAMES[1])
    assert result.exact and where is not None and (result.scale, result.x, result.y) == (where.scale, where.x, where.y)


def test_verify_detects_colour_management() -> None:
    buf, _ = display.compose_screen(_items()[1], 1920, 1080)
    # Gamut mapping to a display profile: pure primaries and white come out slightly different.
    matrix = np.array([[0.94, 0.05, 0.01], [0.02, 0.97, 0.01], [0.01, 0.03, 0.96]])
    managed = np.clip(np.rint(buf.astype(np.float64) @ matrix.T), 0, 255).astype(np.uint8)
    result = display.verify_screenshot(managed, FRAMES[1])
    assert not result.exact and result.mismatched_pixels > 0


def test_verify_detects_smoothed_scaling() -> None:
    buf, _ = display.compose_screen(_items()[1], 1920, 1080)
    # A HiDPI compositor that upsamples 2x with interpolation.
    smoothed = cv2.resize(buf, (3840, 2160), interpolation=cv2.INTER_LINEAR)
    result = display.verify_screenshot(smoothed, FRAMES[1])
    assert not result.exact and "interpolated" in result.message


# --------------------------------------------------------------------------- CLI


def test_cli_frames_then_dry_run_play(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert display.main(["frames", "--colour-depth", "2", "--cell-px", "8", "--payload-bytes", "9000",
                         "--out", str(run_dir)]) == 0
    meta = json.loads((run_dir / "frames.json").read_text())
    assert meta["params"]["colour_depth"] == 2 and meta["payload"]["length"] == 9000
    assert len(meta["frames"]) == len(list(run_dir.glob("frame_*.png")))

    log_path = tmp_path / "play.jsonl"
    assert display.main(["play", str(run_dir), "--reference", "--dry-run", "5", "--hold", "--log", str(log_path)]) == 0
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[0]["frames_json"]["params_label"] == meta["params_label"]
    items = [r for r in records if r["type"] == "item"]
    assert items[0]["kind"] == "reference" and {tuple(c["rgb"]) for c in items[0]["rgb_values"]} == {(0, 0, 0), (255, 255, 255)}


def test_cli_verify(tmp_path: Path) -> None:
    buf, _ = display.compose_screen(_items()[0], 1920, 1080)
    shot, frame = tmp_path / "shot.png", tmp_path / "frame.png"
    Image.fromarray(np.dstack([buf, np.full(buf.shape[:2], 255, np.uint8)])).save(shot)  # RGBA, like macOS screenshots
    write_png(frame, FRAMES[0])
    assert display.main(["verify", str(shot), "--frame", str(frame)]) == 0
    Image.fromarray(buf[::-1]).save(shot)  # an upside-down screen is not the frame
    assert display.main(["verify", str(shot), "--frame", str(frame)]) == 1
