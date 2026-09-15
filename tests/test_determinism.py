"""Hard requirement 2: same payload + same params -> byte-identical PNGs on any machine."""

from __future__ import annotations

import hashlib
import io
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from prism_share.codec.encoder import encode, png_bytes, save_frames
from prism_share.codec.params import CodecParams

PAYLOAD = b"prism-share golden frame"

# SHA-256 of frame 1 (a fountain repair frame) of encode(PAYLOAD, params, n_frames=2).
# These pin the complete pipeline - glyphs, palette, layout, framing, RS, fountain,
# whitening and the PNG writer. A mismatch on another machine means the output is
# NOT portable; a mismatch after an intentional codec change means these must be
# updated and every earlier capture regenerated.
GOLDEN = {
    CodecParams(): "0792eb588203b9bfa2b99cb6af793c59f673b3fe6d7506284aede77b716cb6a5",
    CodecParams(colour_depth=1, cell_px=4, seed=7): "cabaff7a554b162f84606ad92ec0f5af08cab3491a3291e5e196c5e0163cf8a9",
}


@pytest.mark.parametrize("params", list(GOLDEN), ids=lambda p: p.label)
def test_golden_png_hash(params: CodecParams) -> None:
    frame = encode(PAYLOAD, params, n_frames=2)[1]
    assert hashlib.sha256(png_bytes(frame)).hexdigest() == GOLDEN[params]


def test_repeat_encoding_is_byte_identical(tmp_path: Path) -> None:
    p = CodecParams(colour_depth=8, cell_px=5, seed=3)
    a = save_frames(encode(PAYLOAD * 500, p), tmp_path / "a")
    b = save_frames(encode(PAYLOAD * 500, p), tmp_path / "b")
    assert len(a) == len(b) > 1
    for pa, pb in zip(a, b, strict=True):
        assert pa.read_bytes() == pb.read_bytes()


def test_seed_changes_output() -> None:
    a = encode(PAYLOAD, CodecParams(seed=0), n_frames=1)[0]
    b = encode(PAYLOAD, CodecParams(seed=1), n_frames=1)[0]
    assert not np.array_equal(a, b)


def test_fresh_interpreter_with_different_hash_seed_matches() -> None:
    """Rules out dependence on PYTHONHASHSEED, dict/set order or in-process caches."""
    script = (
        "import hashlib, sys;"
        "from prism_share.codec.encoder import encode, png_bytes;"
        "from prism_share.codec.params import CodecParams;"
        f"f = encode({PAYLOAD!r}, CodecParams(), n_frames=2)[1];"
        "sys.stdout.write(hashlib.sha256(png_bytes(f)).hexdigest())"
    )
    for hash_seed in ("1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": hash_seed, "PATH": ""},
        )
        assert out.stdout == GOLDEN[CodecParams()]


def test_cold_glyph_cache_gives_identical_frames(tmp_path: Path) -> None:
    """A cache miss must regenerate exactly the committed glyphs, not merely *some* glyphs.

    Runs in a fresh interpreter with the glyph cache pointed at an empty
    directory, so every glyph set used is regenerated from scratch.
    """
    script = (
        "import hashlib, sys; from pathlib import Path;"
        "import prism_share.codec.glyphs as g;"
        f"g.CACHE_DIR = Path({str(tmp_path)!r});"
        "from prism_share.codec.encoder import encode, png_bytes;"
        "from prism_share.codec.params import CodecParams;"
        f"ps = [CodecParams(), CodecParams(colour_depth=1, cell_px=4, seed=7)];"
        f"sys.stdout.write(' '.join(hashlib.sha256(png_bytes(encode({PAYLOAD!r}, p, n_frames=2)[1])).hexdigest() for p in ps))"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.split() == list(GOLDEN.values())
    regenerated = sorted(p.name for p in tmp_path.iterdir())
    assert regenerated == ["glyphs_g16_px4.json", "glyphs_g16_px8.json"], "cache was not actually cold"


def test_png_is_lossless_in_standard_readers() -> None:
    frame = encode(PAYLOAD, CodecParams(colour_depth=16, cell_px=4), n_frames=1)[0]
    data = png_bytes(frame)
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), frame)
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), frame)


def test_png_has_no_metadata_chunks() -> None:
    data = png_bytes(encode(PAYLOAD, CodecParams(), n_frames=1)[0])
    for chunk in (b"tEXt", b"tIME", b"iCCP", b"gAMA", b"sRGB", b"cHRM", b"pHYs"):
        assert chunk not in data
