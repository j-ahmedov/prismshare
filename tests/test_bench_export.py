"""The decode-benchmark bundle: complete as data, enumerable, and identical to what display.py shows.

``test_consumer_redraws_every_frame_from_data_alone`` plays the Kotlin decoder.
It starts from manifest.json and uses only the bundle's JSON and PNGs, with no
layout rule, no formula and no import of the layout code. Every pixel it
cannot account for from the data is a test failure.
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
from prism_share.codec.layout import black_reference_mask, grid_layout, white_reference_mask
from prism_share.codec.params import BENCH_CONFIGURATIONS, FRAME_FORMAT_VERSION, CodecParams
from prism_share.export import bench
from prism_share.transmit import display

N_FRAMES = 2
#: Deliberately not the capture app: the package must come from the argument, never a built-in default.
PACKAGE = "org.example.bench_consumer2"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bench") / "bench"
    assert bench.main(["--out", str(out), "--frames", str(N_FRAMES), "--package", PACKAGE]) == 0
    return out


def _manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text())


def _load_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        assert im.mode == "RGB"
        return np.array(im)


def test_manifest_enumerates_everything_and_nothing_else(bundle: Path) -> None:
    m = _manifest(bundle)
    assert m["frame_format_version"] == FRAME_FORMAT_VERSION and m["frames_per_configuration"] == N_FRAMES
    assert [(c["colour_depth"], c["cell_px"]) for c in m["configurations"]] == list(BENCH_CONFIGURATIONS) == [(1, 4), (16, 4)]
    assert [c["folder"] for c in m["configurations"]] == ["c1_px4", "c16_px4"]
    assert sorted(p.name for p in bundle.iterdir()) == sorted(["manifest.json", "README.md", "c1_px4", "c16_px4"])
    for c in m["configurations"]:
        listed = {c["params"], c["codec"], c["ground_truth"], *c["frames"]}
        assert {p.name for p in (bundle / c["folder"]).iterdir()} == listed
        assert c["frames"] == [f"frame_{i:04d}.png" for i in range(N_FRAMES)]
        params = CodecParams(colour_depth=c["colour_depth"], cell_px=c["cell_px"])
        assert c["payload_bytes_per_frame"] == frame_capacity(params).block_bytes
        assert c["params_fingerprint"] == params.fingerprint()


def test_params_json_is_the_flat_codec_params(bundle: Path) -> None:
    for c in _manifest(bundle)["configurations"]:
        doc = json.loads((bundle / c["folder"] / c["params"]).read_text())
        assert all(isinstance(v, int) for v in doc.values())
        assert CodecParams.from_dict(doc) == CodecParams(colour_depth=c["colour_depth"], cell_px=c["cell_px"])


def test_consumer_redraws_every_frame_from_data_alone(bundle: Path) -> None:
    m = _manifest(bundle)
    for c in m["configurations"]:
        folder = bundle / c["folder"]
        codec = json.loads((folder / c["codec"]).read_text())
        truth = json.loads((folder / c["ground_truth"]).read_text())
        glyphs = np.array(codec["glyphs"], dtype=np.uint8)
        palette = np.array(codec["palette"], dtype=np.uint8)
        background = np.array(codec["background"], dtype=np.uint8)
        cells = codec["cells"]
        assert len(glyphs) == codec["glyph_count"] == 16 and len(palette) == codec["colour_depth"] == c["colour_depth"]
        assert all(cell["width"] == cell["height"] == glyphs.shape[1] == glyphs.shape[2] == c["cell_px"] for cell in cells)
        assert (glyphs.reshape(len(glyphs), -1).sum(axis=1) == codec["glyph_ink_pixels"]).all()
        assert len(cells) == codec["n_cells"] == c["n_cells"]
        assert [f["file"] for f in truth["frames"]] == c["frames"]

        for name, frame in zip(c["frames"], truth["frames"], strict=True):
            path = folder / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == frame["png_sha256"]
            img = _load_png(path)
            assert img.shape == (m["frame_height"], m["frame_width"], 3)
            covered = np.zeros(img.shape[:2], dtype=np.uint8)

            gids, cids = frame["glyph_ids"], frame["colour_ids"]
            assert len(gids) == len(cids) == len(cells)
            for i, cell in enumerate(cells):
                x, y, w, h = cell["x"], cell["y"], cell["width"], cell["height"]
                expected = np.where(glyphs[gids[i]][:, :, None] == 1, palette[cids[i]], background)
                assert np.array_equal(img[y : y + h, x : x + w], expected), (c["folder"], name, i)
                covered[y : y + h, x : x + w] += 1

            band = codec["index_band"]
            assert len(band["blocks"]) == band["bits"] * band["repeats"]
            for block in band["blocks"]:
                x, y, w, h = block["x"], block["y"], block["width"], block["height"]
                bit = (frame["index_band"] >> block["bit"]) & 1
                assert np.all(img[y : y + h, x : x + w] == (255 if bit else 0)), (c["folder"], name, block)
                covered[y : y + h, x : x + w] += 1

            refs = codec["level_references"]
            for level, value in (("white", 255), ("black", 0)):
                for r in refs[level]:
                    assert np.all(img[r["y"] : r["y"] + r["height"], r["x"] : r["x"] + r["width"]] == value), (level, r)
                    covered[r["y"] : r["y"] + r["height"], r["x"] : r["x"] + r["width"]] += 1
            assert covered.max() == 1  # cells, band blocks and references never overlap one another


def test_cells_are_the_decoders_cells_in_the_decoders_order(bundle: Path) -> None:
    for c in _manifest(bundle)["configurations"]:
        params = CodecParams(colour_depth=c["colour_depth"], cell_px=c["cell_px"])
        folder = bundle / c["folder"]
        codec = json.loads((folder / c["codec"]).read_text())
        layout = grid_layout(params)
        assert [(cell["x"], cell["y"]) for cell in codec["cells"]] == list(zip(layout.cell_x.tolist(), layout.cell_y.tolist(), strict=True))
        truth = json.loads((folder / c["ground_truth"]).read_text())
        for frame in truth["frames"]:
            img = _load_png(folder / frame["file"])
            readout = read_symbols(img, params)
            assert readout.glyphs.tolist() == frame["glyph_ids"] and readout.colours.tolist() == frame["colour_ids"]
            assert read_index_band(img, params).index == frame["index_band"]


def test_level_reference_rectangles_are_exactly_the_decoders_masks(bundle: Path) -> None:
    c = _manifest(bundle)["configurations"][0]
    codec = json.loads((bundle / c["folder"] / c["codec"]).read_text())
    for level, mask in (("white", white_reference_mask(codec["frame_width"])), ("black", black_reference_mask(codec["frame_width"]))):
        drawn = np.zeros(mask.shape, dtype=np.uint8)
        for r in codec["level_references"][level]:
            drawn[r["y"] : r["y"] + r["height"], r["x"] : r["x"] + r["width"]] += 1
        assert drawn.max() == 1 and np.array_equal(drawn.astype(bool), mask)


def test_mask_rectangles_is_an_exact_disjoint_cover() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        mask = rng.random((13, 17)) < 0.4
        mask[3:9, 2:12] = True
        drawn = np.zeros(mask.shape, dtype=np.uint8)
        for r in bench.mask_rectangles(mask):
            drawn[r["y"] : r["y"] + r["height"], r["x"] : r["x"] + r["width"]] += 1
        assert drawn.max() <= 1 and np.array_equal(drawn.astype(bool), mask)


def test_frames_are_exactly_what_display_frames_writes(bundle: Path, tmp_path: Path) -> None:
    for c in _manifest(bundle)["configurations"]:
        params = CodecParams(colour_depth=c["colour_depth"], cell_px=c["cell_px"])
        out = tmp_path / c["folder"]
        display.main(["frames", "--colour-depth", str(params.colour_depth), "--cell-px", str(params.cell_px), "--out", str(out),
                      "--payload-bytes", str(N_FRAMES * frame_capacity(params).block_bytes), "--n-frames", str(N_FRAMES)])
        shown = sorted(out.glob("frame_*.png"))
        assert len(shown) == N_FRAMES
        for path, name in zip(shown, c["frames"], strict=True):
            assert path.read_bytes() == (bundle / c["folder"] / name).read_bytes()


def test_readme_states_version_payload_and_adb_commands(bundle: Path) -> None:
    text = (bundle / "README.md").read_text()
    assert f"**Frame format version: {FRAME_FORMAT_VERSION}.**" in text
    for c in _manifest(bundle)["configurations"]:
        assert f"| `{c['folder']}` |" in text and f"**{c['payload_bytes_per_frame']}** |" in text
    phone = f"/sdcard/Android/data/{PACKAGE}/files/bench"
    lines = text.splitlines()
    assert "prismshare.capture/" not in text and "prismshare.testspike" not in text  # no package but the one given
    assert f"--package {PACKAGE}`" in text  # the regeneration command reproduces it
    assert f"adb shell mkdir -p {phone}" in lines
    push = next(line for line in lines if line.startswith("adb push "))
    for name in ("manifest.json", "README.md", "c1_px4", "c16_px4"):
        assert f"{bundle.as_posix()}/{name} " in push
    assert push.endswith(f" {phone}/")
    # The chmod is generated, not hand-added, and comes after the push, with its reason.
    chmod = f"adb shell chmod -R o+rX {phone}"
    assert chmod in lines and lines.index(chmod) == lines.index(push) + 1
    assert "`ext_data_rw`" in text and "not in `ext_data_rw`" in text and "unreadable to the app" in text
    commands = [line for line in lines if line.startswith("adb ")]
    assert len(commands) == 3 and all(f"/sdcard/Android/data/{PACKAGE}/files/bench" in line for line in commands)


def test_package_is_required(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        bench.main(["--out", str(tmp_path / "b"), "--frames", "1"])
    assert exc.value.code == 2 and "--package" in capsys.readouterr().err
    assert not (tmp_path / "b").exists()


@pytest.mark.parametrize("bad", [
    "", "testspike", "io.github.ahmedov.prismshare.", ".io.github", "io..github", "io.1github", "io.github-ahmedov",
    "io.github.ahmedov/prismshare", "io.github ahmedov", "io.github.ahmedov.prismshare.testspike\n", "_io.github",
    "io.github.é",
])
def test_invalid_application_ids_are_refused_before_writing(tmp_path: Path, bad: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="Android application ID"):
        bench.validate_package(bad)
    out = tmp_path / "b"
    assert bench.main(["--out", str(out), "--frames", "1", "--package", bad]) == 2
    assert "Android application ID" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("good", ["io.github.ahmedov.prismshare.testspike", "a.b", "com.Example_1.app2", "x.y_.Z9"])
def test_valid_application_ids_are_accepted(good: str) -> None:
    assert bench.validate_package(good) == good
    assert bench.phone_bench_dir(good) == f"/sdcard/Android/data/{good}/files/bench"


def test_refuses_to_write_over_an_existing_bundle(bundle: Path) -> None:
    with pytest.raises(FileExistsError):
        bench.export(bundle, N_FRAMES, PACKAGE)
    assert bench.main(["--out", str(bundle), "--frames", "1", "--package", PACKAGE]) == 2


def test_frame_count_must_give_distinct_band_values() -> None:
    for bad in (0, 256):
        with pytest.raises(ValueError):
            bench.bench_frames(CodecParams(colour_depth=1, cell_px=10), bad)
