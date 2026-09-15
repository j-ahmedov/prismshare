"""The pilot pipeline runs end to end on a tiny configuration and writes a well-formed report."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from prism_share.codec.decoder import ALL_DECODERS, HEADLINE_DECODER, SENSITIVITY_DECODERS, DecoderOptions, read_symbols
from prism_share.codec.encoder import encode_frames
from prism_share.codec.params import NEAR_TIE_MARGIN_PCT, CodecParams
from prism_share.analysis import ecc_sim
from prism_share.sim import pilot

TINY = """
seed: 0
frames_per_condition: 2
colour_depths: [1, 4]
cell_px: [8]
ecc_lengths: [155, 255]
decoders:
  - {shape_channel: max, colour_estimator: mean}
  - {shape_channel: luma, colour_estimator: saturated}
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
    assert len(summary) == 1 + 2 * 7 * 2  # header + 2 configs x (clean + 6 conditions) x 2 decoders

    # The hand-written interpretation block survives regeneration.
    report.write_text(text.replace("_(interpretation not yet written)_", "KEEP ME"))
    pilot.main(["--config", str(config), "--out", str(tmp_path / "out"), "--report", str(report),
                "--data", str(tmp_path / "data"), "--reuse"])
    assert "KEEP ME" in report.read_text()


def test_committed_pilot_config_loads() -> None:
    cfg = pilot.load_config(Path(__file__).parent.parent / "experiments" / "pilot.yaml")
    assert len(cfg.configs()) == 25
    assert cfg.headline_decoder == HEADLINE_DECODER == cfg.decoders[0]
    assert set(cfg.decoders) == set(ALL_DECODERS)
    assert {s.degradation for s in cfg.series} == {"blur", "noise", "perspective", "white_balance", "chroma"}


def test_headline_decoder_is_pre_registered_not_configured(tmp_path: Path) -> None:
    assert HEADLINE_DECODER == DecoderOptions(shape_channel="luma", colour_estimator="saturated")
    assert ALL_DECODERS[0] == HEADLINE_DECODER and HEADLINE_DECODER not in SENSITIVITY_DECODERS
    assert len(set(ALL_DECODERS)) == 4

    overridden = tmp_path / "override.yaml"
    overridden.write_text(TINY + "headline_decoder: {shape_channel: max, colour_estimator: mean}\n")
    with pytest.raises(ValueError, match="pre-registered"):
        pilot.load_config(overridden)

    missing = tmp_path / "missing.yaml"
    missing.write_text(TINY.replace("  - {shape_channel: luma, colour_estimator: saturated}\n", ""))
    with pytest.raises(ValueError):
        pilot.load_config(missing)


def _summary(rows: list[tuple[int, int, float]], decoder: str = "luma/saturated") -> pd.DataFrame:
    return pd.DataFrame(
        [{"series": "blur", "severity": 1.0, "decoder": decoder, "colour_depth": d, "cell_px": c, "goodput_mbps": g,
          "goodput_band_credited_mbps": g, "best_rs": "RS(255,223)"} for d, c, g in rows]
    )


def test_margins_and_near_tie_flags() -> None:
    assert pilot.margin_pct(10.5, 10.0) == pytest.approx(5.0)
    assert pilot.margin_pct(0.0, 0.0) == 0.0 and pilot.margin_pct(1.0, 0.0) == float("inf")

    # Colour wins by 1.4 % over the best monochrome; the runner-up is another colour configuration.
    w = pilot.winners(_summary([(4, 6, 10.14), (4, 8, 10.12), (1, 4, 10.0), (1, 6, 8.0)]), "luma/saturated").iloc[0]
    assert w["winner"].startswith("depth 4") and w["runner_up"].startswith("depth 4")
    assert w["margin_pct"] == pytest.approx(100 * 0.02 / 10.12) and w["near_tie"]
    assert w["other_class_best"].startswith("depth 1") and w["class_margin_pct"] == pytest.approx(1.4)
    assert w["colour_wins"] and w["class_near_tie"]

    # Just over the threshold on both margins: decided.
    edge = 1 + (NEAR_TIE_MARGIN_PCT + 0.1) / 100
    w = pilot.winners(_summary([(1, 8, 10 * edge), (4, 4, 10.0)]), "luma/saturated").iloc[0]
    assert not w["colour_wins"] and not w["near_tie"] and not w["class_near_tie"]

    # Exact tie: the simpler code (fewer colours) wins, flagged as a near-tie.
    w = pilot.winners(_summary([(4, 8, 5.0), (1, 8, 5.0)]), "luma/saturated").iloc[0]
    assert not w["colour_wins"] and w["class_near_tie"]

    table = pilot.margin_table(pilot.winners(_summary([(4, 6, 10.14), (1, 4, 10.0)]), "luma/saturated"))
    assert "class near-tie" in table and "1.4 %" in table


def test_goodput_is_scored_out_of_sample(tmp_path: Path) -> None:
    config = tmp_path / "tiny.yaml"
    config.write_text(TINY)
    cfg = pilot.load_config(config)
    frames = pilot.run(cfg, jobs=1)
    summary = pilot.summarise(frames, cfg)
    assert (summary["rs_selection"] == "out_of_sample").all()
    assert (summary["n_selection_frames"] == 1).all() and (summary["n_evaluation_frames"] == 1).all()
    for r in summary.itertuples():
        g = frames[(frames["colour_depth"] == r.colour_depth) & (frames["cell_px"] == r.cell_px) & (frames["series"] == r.series)
                   & (frames["severity"] == r.severity) & (frames["decoder"] == r.decoder)]
        params = CodecParams(colour_depth=r.colour_depth, cell_px=r.cell_px, seed=cfg.seed)
        chosen = ecc_sim.best_code({n: g.loc[g["frame"] == 0, f"max_cw_errors_n{n}"] for n in cfg.ecc_lengths}, params)
        assert r.best_rs == f"RS({chosen.n},{chosen.k})"  # chosen on frame 0 only
        scored = ecc_sim.evaluate(g.loc[g["frame"] == 1, f"max_cw_errors_n{chosen.n}"], params, chosen.n, chosen.k)
        assert r.best_yield == scored.frame_yield  # scored on frame 1 only


def test_added_conditions_match_a_full_simulation(tmp_path: Path) -> None:
    """--add-missing is sound: a condition's rows do not depend on which other conditions are simulated."""
    config = tmp_path / "tiny.yaml"
    config.write_text(TINY)
    cfg = pilot.load_config(config)
    full = pilot.run(cfg, jobs=1)
    keep = full[full["series"] != "chroma_nearest"]
    missing = pilot.missing_conditions(keep, cfg)
    assert missing == {("chroma_nearest", 2.0)}
    added = pilot.run(dataclasses.replace(cfg, only_conditions=missing), jobs=1)
    key = ["colour_depth", "cell_px", "series", "severity", "decoder", "frame"]
    expected = full[full["series"] == "chroma_nearest"].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(added.sort_values(key).reset_index(drop=True), expected)
    assert pilot.missing_conditions(full, cfg) == frozenset()
    with pytest.raises(ValueError, match="partially"):
        pilot.missing_conditions(full.drop(index=full.index[full["series"] == "noise"][:1]), cfg)
    with pytest.raises(ValueError, match="does not"):
        pilot.missing_conditions(full, dataclasses.replace(cfg, series=cfg.series[:-1]))
