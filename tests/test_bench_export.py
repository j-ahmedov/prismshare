"""The decode-benchmark bundle: complete, self-describing, and identical to what display.py shows.

The core test reads ground_truth.json the way a second implementation would:
it uses only the JSON and the PNG files, rebuilds the cell order from the
written rule, redraws every cell and reads the index band from the written
encoding. If the prose in the JSON and the code disagree, it fails.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from prism_share.codec.decoder import read_index_band, read_symbols
from prism_share.codec.framing import frame_capacity
from prism_share.codec.params import BENCH_CONFIGURATIONS, FRAME_FORMAT_VERSION, CodecParams
from prism_share.export import bench
from prism_share.transmit import display

N_FRAMES = 2


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bench") / "bench"
    assert bench.main(["--out", str(out), "--frames", str(N_FRAMES)]) == 0
    return out


def _folders(out: Path) -> list[Path]:
    return [out / CodecParams(colour_depth=d, cell_px=c).label for d, c in BENCH_CONFIGURATIONS]


def test_bundle_layout(bundle: Path) -> None:
    assert BENCH_CONFIGURATIONS == ((1, 4), (16, 4))
    for folder in _folders(bundle):
        names = sorted(p.name for p in folder.iterdir())
        assert names == [f"frame_{i:04d}.png" for i in range(N_FRAMES)] + ["ground_truth.json", "params.json"]
    assert (bundle / "README.md").exists()


def test_params_json_is_the_flat_codec_params(bundle: Path) -> None:
    for (depth, cell), folder in zip(BENCH_CONFIGURATIONS, _folders(bundle), strict=True):
        doc = json.loads((folder / "params.json").read_text())
        assert all(not isinstance(v, dict | list) for v in doc.values())
        assert CodecParams.from_dict(doc) == CodecParams(colour_depth=depth, cell_px=cell)


def _cells_from_rule(order: dict) -> tuple[list[int], list[int]]:
    """Re-derive the cell order from the prose fields only."""
    g = order["grid"]
    gap, n = g["cell_gap_px"], g["cell_px"]
    xs, ys = [], []
    for row in range(g["cells_per_side"]):
        for col in range(g["cells_per_side"]):
            x, y = g["origin_px"] + col * g["pitch_px"], g["origin_px"] + row * g["pitch_px"]
            if not any(x - gap < r["x"] + r["width"] and x + n + gap > r["x"] and y - gap < r["y"] + r["height"] and y + n + gap > r["y"]
                       for r in order["reserved_rectangles"]):
                xs.append(x)
                ys.append(y)
    return xs, ys


def test_second_implementation_rebuilds_every_frame_from_the_json(bundle: Path) -> None:
    static: np.ndarray | None = None
    for folder in _folders(bundle):
        gt = json.loads((folder / "ground_truth.json").read_text())
        assert gt["frame_format_version"] == FRAME_FORMAT_VERSION
        order, book, band = gt["cell_order"], gt["codebook"], gt["index_band"]
        n = order["grid"]["cell_px"]

        xs, ys = _cells_from_rule(order)
        assert xs == order["cell_x"] and ys == order["cell_y"] and len(xs) == gt["symbols"]["n_cells"]

        glyphs = np.array(book["glyphs"], dtype=np.uint8)
        palette = np.array(book["palette_rgb"], dtype=np.uint8)
        background = np.array(book["background_rgb"], dtype=np.uint8)
        assert len(gt["frames"]) == N_FRAMES
        for k, frame in enumerate(gt["frames"]):
            path = folder / frame["file"]
            assert frame["file"] == f"frame_{k:04d}.png"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == frame["png_sha256"]
            with Image.open(path) as im:
                assert im.mode == "RGB" and im.size == (gt["image"]["width"], gt["image"]["height"])
                img = np.array(im)
            covered = np.zeros(img.shape[:2], dtype=bool)

            # Cells: redraw each from glyph_ids / colour_ids and the codebook.
            gids, cids = frame["glyph_ids"], frame["colour_ids"]
            assert len(gids) == len(cids) == len(xs)
            for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
                expected = np.where(glyphs[gids[i]][:, :, None] == 1, palette[cids[i]], background)
                assert np.array_equal(img[y : y + n, x : x + n], expected), (folder.name, k, i)
                covered[y : y + n, x : x + n] = True

            # Index band: bit of block p is bit (bits - 1 - p mod bits) of the value, white = 1.
            value = frame["index_band"]
            assert value == k + 1 and frame["header"]["block_id"] == k
            for p in range(band["blocks"]):
                bit = (value >> (band["bits"] - 1 - p % band["bits"])) & 1
                x0, y0, b = band["x"] + p * band["block_px"], band["y"], band["block_px"]
                assert np.all(img[y0 : y0 + b, x0 : x0 + b] == (255 if bit else 0)), (folder.name, k, p)
                covered[y0 : y0 + b, x0 : x0 + b] = True

            # Everything else is the same static region in every frame of every configuration.
            rest = np.where(covered[:, :, None], 0, img)
            if static is None:
                static = rest
            assert np.array_equal(rest, static)


def test_ground_truth_matches_the_decoder(bundle: Path) -> None:
    for (depth, cell), folder in zip(BENCH_CONFIGURATIONS, _folders(bundle), strict=True):
        params = CodecParams(colour_depth=depth, cell_px=cell)
        gt = json.loads((folder / "ground_truth.json").read_text())
        for frame in gt["frames"]:
            with Image.open(folder / frame["file"]) as im:
                img = np.array(im)
            readout = read_symbols(img, params)
            assert readout.glyphs.tolist() == frame["glyph_ids"] and readout.colours.tolist() == frame["colour_ids"]
            assert read_index_band(img, params).index == frame["index_band"]


def test_frames_are_exactly_what_display_frames_writes(bundle: Path, tmp_path: Path) -> None:
    for (depth, cell), folder in zip(BENCH_CONFIGURATIONS, _folders(bundle), strict=True):
        params = CodecParams(colour_depth=depth, cell_px=cell)
        out = tmp_path / params.label
        display.main(["frames", "--colour-depth", str(depth), "--cell-px", str(cell), "--out", str(out),
                      "--payload-bytes", str(N_FRAMES * frame_capacity(params).block_bytes), "--n-frames", str(N_FRAMES)])
        shown = sorted(out.glob("frame_*.png"))
        assert len(shown) == N_FRAMES
        for i, path in enumerate(shown):
            assert path.read_bytes() == (folder / f"frame_{i:04d}.png").read_bytes()


def test_readme_states_version_payload_and_adb_command(bundle: Path) -> None:
    text = (bundle / "README.md").read_text()
    assert f"**Frame format version:** {FRAME_FORMAT_VERSION}" in text
    for depth, cell in BENCH_CONFIGURATIONS:
        params = CodecParams(colour_depth=depth, cell_px=cell)
        assert f"| `{params.label}` |" in text and f"**{frame_capacity(params).block_bytes}**" in text
    phone = "/sdcard/Android/data/io.github.ahmedov.prismshare.capture/files/bench"
    assert f"adb shell mkdir -p {phone}\n" in text
    push = next(line for line in text.splitlines() if line.startswith("adb push "))
    assert push.endswith(f" {phone}/") and all(f.name in push for f in _folders(bundle))


def test_refuses_to_mix_into_an_existing_bundle(bundle: Path) -> None:
    with pytest.raises(FileExistsError):
        bench.export(bundle, N_FRAMES)
    assert bench.main(["--out", str(bundle), "--frames", "1"]) == 2


def test_frame_count_must_give_distinct_band_values(tmp_path: Path) -> None:
    for bad in (0, 256):
        with pytest.raises(ValueError):
            bench.bench_frames(CodecParams(colour_depth=1, cell_px=10), bad)
