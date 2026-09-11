"""The pilot pipeline runs end to end on a tiny configuration and writes a well-formed report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prism_share.codec.decoder import DecoderOptions, read_symbols
from prism_share.codec.encoder import encode_frames
from prism_share.codec.params import CodecParams
from prism_share.sim import pilot

TINY = """
seed: 0
frames_per_condition: 1
colour_depths: [1, 4]
cell_px: [8]
ecc_lengths: [155, 255]
decoders:
  - {shape_channel: max, colour_estimator: mean}
headline_decoder: {shape_channel: max, colour_estimator: mean}
series:
  - {name: blur, degradation: blur, severities: [1.0]}
  - {name: chroma_nearest, degradation: chroma, options: {upsample: nearest}, severities: [2]}
  - {name: chroma_bilinear, degradation: chroma, options: {upsample: bilinear}, severities: [2]}
  - {name: noise, degradation: noise, severities: [20]}
  - {name: perspective, degradation: perspective, severities: [0.1]}
  - {name: white_balance, degradation: white_balance, severities: [1.0]}
"""


@pytest.mark.parametrize(
    "options",
    [DecoderOptions(shape_channel=s, colour_estimator=c) for s in ("max", "luma") for c in ("mean", "saturated")],
    ids=lambda o: o.label,
)
def test_every_decoder_variant_is_exact_on_clean_frames(options: DecoderOptions) -> None:
    for depth in (1, 4, 16):
        p = CodecParams(colour_depth=depth, cell_px=4)
        fr = encode_frames(b"variants", p, n_frames=1)[0]
        r = read_symbols(fr.image, p, options)
        assert np.array_equal(r.glyphs, fr.glyphs) and np.array_equal(r.colours, fr.colours)


def test_invalid_decoder_options() -> None:
    with pytest.raises(ValueError):
        DecoderOptions(shape_channel="green")
    with pytest.raises(ValueError):
        DecoderOptions(colour_estimator="median")


def test_pilot_end_to_end(tmp_path: Path) -> None:
    config = tmp_path / "tiny.yaml"
    config.write_text(TINY)
    report = tmp_path / "pilot.md"
    pilot.main(["--config", str(config), "--out", str(tmp_path / "out"), "--report", str(report),
                "--data", str(tmp_path / "data"), "--jobs", "1"])
    text = report.read_text()
    assert text.splitlines()[0].startswith("**These are simulated results")
    assert "predict" in text.splitlines()[0]
    for name in ("fig_best_goodput", "fig_error_decomposition", "fig_goodput_surface"):
        assert (tmp_path / "out" / f"{name}.pdf").stat().st_size > 0
    summary = (tmp_path / "out" / "pilot_summary.csv").read_text().splitlines()
    assert len(summary) == 1 + 2 * 7  # header + 2 configs x (clean + 6 conditions)

    # The hand-written interpretation block survives regeneration.
    report.write_text(text.replace("_(interpretation not yet written)_", "KEEP ME"))
    pilot.main(["--config", str(config), "--out", str(tmp_path / "out"), "--report", str(report),
                "--data", str(tmp_path / "data"), "--reuse"])
    assert "KEEP ME" in report.read_text()


def test_committed_pilot_config_loads() -> None:
    cfg = pilot.load_config(Path(__file__).parent.parent / "experiments" / "pilot.yaml")
    assert len(cfg.configs()) == 25
    assert cfg.headline_decoder in cfg.decoders
    assert {s.degradation for s in cfg.series} == {"blur", "noise", "perspective", "white_balance", "chroma"}
