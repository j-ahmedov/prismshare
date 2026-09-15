"""Step 6: sweep over captured runs, three-outcome yield, provenance on every figure."""

from __future__ import annotations

import dataclasses

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from capture_fixture import RunSpec, build_run, stub_front_end

from prism_share.analysis import plots, sweep
from prism_share.analysis.ecc_sim import payload_bytes_per_frame
from prism_share.analysis.metrics import FrameOutcome, frame_outcome, yield_breakdown
from prism_share.codec.decoder import read_index_band
from prism_share.codec.decoder import ALL_DECODERS, HEADLINE_DECODER, DecoderOptions
from prism_share.codec.params import ASSUMED_FPS, CodecParams
from prism_share.ingest.ingest import ingest_run

# --------------------------------------------------------------------------- metrics


def test_three_outcomes_never_two() -> None:
    assert frame_outcome(False, 0, 155, 125) is FrameOutcome.DETECTION_FAILED  # no errors, still not detected
    assert frame_outcome(True, 15, 155, 125) is FrameOutcome.RECOVERED
    assert frame_outcome(True, 16, 155, 125) is FrameOutcome.NOT_RECOVERABLE
    bd = yield_breakdown([FrameOutcome.RECOVERED] * 6 + [FrameOutcome.NOT_RECOVERABLE] * 3 + [FrameOutcome.DETECTION_FAILED])
    assert (bd.recovered, bd.not_recoverable, bd.detection_failed, bd.n_frames) == (6, 3, 1, 10)
    assert bd.frame_yield == 0.6
    assert sum(bd.fractions().values()) == pytest.approx(1.0)


def test_a_capture_says_which_frame_it_is() -> None:
    """Identification comes from the index band, never from how well the data decoded."""
    from prism_share.codec.encoder import encode_frames
    from prism_share.codec.params import CodecParams

    params = CodecParams(colour_depth=16, cell_px=4)
    frames = encode_frames(b"index band", params, n_frames=3)
    for frame in frames:
        assert read_index_band(frame.image, params).index == frame.frame_index


def test_index_survives_a_capture_that_cannot_be_decoded() -> None:
    """The band still reads on a frame whose cells are destroyed - the case that matters."""
    import cv2

    from prism_share.codec.decoder import read_symbols
    from prism_share.codec.encoder import encode_frames
    from prism_share.codec.params import CodecParams

    params = CodecParams(colour_depth=16, cell_px=4)
    frame = encode_frames(b"wrecked", params, n_frames=1)[0]
    wrecked = cv2.GaussianBlur(frame.image.astype(np.float64), (0, 0), 3.0)  # cells unreadable
    readout = read_symbols(wrecked, params)
    assert np.mean(readout.glyphs != frame.glyphs) > 0.5
    assert read_index_band(wrecked, params).index == frame.frame_index


# --------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def processed(tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    root = tmp_path_factory.mktemp("sweep")
    # 1 reference capture + 7 code captures of the run's ONE static frame (block 1), except capture 5,
    # which shows block 2 - a protocol slip the band must catch. (Taint handling is tested in
    # test_ingest.py; any taint in 8 frames would exceed 1 % and refuse the run.)
    spec = RunSpec(run_id="d05_lab", colour_depth=4, cell_px=8, n_code_frames=7, displayed_block=1,
                   stray_captures={5: 2}, reference_captures=(0,), condition={"distance_m": 0.5, "illuminance_lux": 200})
    built = build_run(root, spec)
    cfg = sweep.SweepConfig(sources=("y", "rgb", "yuv_nearest", "yuv_bilinear"),
                            decoders=(DecoderOptions(), DecoderOptions(shape_channel="luma", colour_estimator="saturated")),
                            flat_field=True)
    run = ingest_run(built.run_dir, built.runs_dir)
    # Capture 3 (0-based) fails detection: the front end is told to fail on its 4th call.
    result = sweep.process_run(run, cfg, front_end=stub_front_end(built.h, fail_stems={3}), overlay_dir=root / "overlays")
    return result.frames, sweep.summarise(result.frames, cfg), (built, result, cfg, root)


def test_reference_capture_is_recognised_and_excluded(processed: tuple) -> None:
    frames, summary, _ = processed
    first = frames[frames["file_stem"] == "frame_0000"]
    assert (first["stimulus_kind"] == "reference").all() and (first["excluded"] == "reference").all()
    assert (first["band_index"] == 0).all()  # structural: index 0 is reserved for the reference
    assert (summary["n_reference"] == 1).all()


def test_every_code_frame_has_exactly_one_of_three_outcomes(processed: tuple) -> None:
    _, summary, _ = processed
    # Code captures have indices 1..7: even indices (2, 4, 6) choose the RS code, odd (1, 3, 5, 7) are scored.
    assert (summary["n_code_frames"] == 7).all()
    assert (summary["rs_selection"] == "out_of_sample").all()
    assert (summary["n_selection_frames"] == 3).all() and (summary["n_evaluation_frames"] == 4).all()
    for prefix in ("fixed", "best"):
        total = summary[f"{prefix}_n_detection_failed"] + summary[f"{prefix}_n_not_recoverable"] + summary[f"{prefix}_n_recovered"]
        assert (total == summary["n_evaluation_frames"]).all()
    assert (summary["best_n_detection_failed"] == 1).all()  # capture 3 is in the evaluation half
    assert (summary["best_n_not_recoverable"] >= 1).all()  # at least the stray capture 5, also evaluation


def test_stimuli_are_identified_by_their_band(processed: tuple) -> None:
    frames, summary, _ = processed
    code = frames[(frames["stimulus_kind"] == "code") & frames["identified"].astype(bool)]
    assert len(code) > 0
    assert (code["stimulus_block_id"] == 1).all() and (code["band_index"] == 2).all()  # block 1 carries index 2
    assert (code["band_agreement"] == 1.0).all()


def test_a_capture_of_the_wrong_frame_is_never_matched(processed: tuple) -> None:
    """The run holds one frame; a capture whose band names another is not recoverable, never excluded."""
    frames, summary, _ = processed
    stray = frames[frames["file_stem"] == "frame_0005"]
    assert (stray["band_index"] == 3).all() and not stray["identified"].astype(bool).any()
    assert stray["excluded"].isna().all()  # counted, not dropped
    assert (summary["n_band_index_unknown"] == 1).all()


def test_clean_synthetic_captures_decode_error_free(processed: tuple) -> None:
    _, summary, _ = processed
    colour_sources = summary[summary["pixel_source"].isin(["rgb", "yuv_nearest", "yuv_bilinear"])]
    assert (colour_sources["shape_ser"] == 0).all()
    # Colour through 4:2:0: the luma/saturated decoder recovers everything at this magnification.
    best_decoder = colour_sources[colour_sources["decoder"] == "luma/saturated"]
    assert (best_decoder["colour_ser"] == 0).all()
    assert (best_decoder["best_n_recovered"] == 2).all()  # evaluation captures 1, 3, 5, 7 - detection failure 3 - stray 5


def test_y_source_only_for_monochrome(processed: tuple) -> None:
    frames, _, _ = processed
    assert "y" not in set(frames["pixel_source"])  # this run is 4-colour


def test_goodput_is_derived_from_the_formula(processed: tuple) -> None:
    _, summary, _ = processed
    row = summary.iloc[0]
    params = CodecParams(colour_depth=4, cell_px=8)
    n, k = (int(v) for v in re.match(r"RS\((\d+),(\d+)\)", row["best_rs"]).groups())
    expected = payload_bytes_per_frame(params, n, k) * row["best_yield"] * ASSUMED_FPS * 8 / 1e6
    assert row["best_goodput_mbps"] == pytest.approx(expected)
    assert row["best_yield"] == pytest.approx(row["best_n_recovered"] / row["n_evaluation_frames"])


def test_in_sample_selection_is_explicit_warned_and_labelled(processed: tuple) -> None:
    frames, _, (_, _, cfg, _) = processed
    with pytest.warns(sweep.InSampleWarning, match="IN-SAMPLE"):
        biased = dataclasses.replace(cfg, rs_selection=sweep.IN_SAMPLE)
    summary = sweep.summarise(frames, biased)
    assert (summary["rs_selection"] == "in_sample").all()
    assert (summary["n_evaluation_frames"] == 7).all()
    luma = summary[(summary["decoder"] == "luma/saturated") & (summary["pixel_source"] != "y")]
    assert (luma["best_n_recovered"] == 5).all()  # all 7 code captures - detection failure - stray
    assert "IN-SAMPLE" in plots.provenance_text(summary, "best")
    with pytest.raises(ValueError):
        sweep.SweepConfig(rs_selection="whatever")


def test_default_provenance_says_out_of_sample(processed: tuple) -> None:
    _, summary, _ = processed
    text = plots.provenance_text(summary, "best")
    assert "chosen out of sample" in text and "IN-SAMPLE" not in text


def test_run_without_a_selection_half_reports_no_goodput(processed: tuple) -> None:
    frames, _, (_, _, cfg, _) = processed
    odd_only = frames[(frames["frame_index"] % 2 == 1) | (frames["excluded"] == "reference")]
    summary = sweep.summarise(odd_only, cfg)
    assert (summary["n_selection_frames"] == 0).all()
    assert summary["best_goodput_mbps"].isna().all()  # NaN, never an in-sample number
    assert set(sweep._outcome_column_names("best")) <= set(summary.columns)
    _, full, _ = processed
    assert [c for c in full.columns if c.startswith("best_")] == sweep._outcome_column_names("best")


def test_band_credited_goodput_is_reported_alongside(processed: tuple) -> None:
    """Same code, same yield, the band's cells counted as data: the apparatus is priced, not hidden."""
    from prism_share.codec.framing import frame_capacity, frame_capacity_band_credited
    from prism_share.codec.layout import band_cell_cost

    _, summary, _ = processed
    params = CodecParams(colour_depth=4, cell_px=8)
    for prefix in ("fixed", "best"):
        row = summary.iloc[0]
        n, k = (int(v) for v in re.match(r"RS\((\d+),(\d+)\)", row[f"{prefix}_rs"]).groups())
        credited = payload_bytes_per_frame(params, n, k, band_credited=True)
        assert row[f"{prefix}_payload_bytes_band_credited"] == credited > row[f"{prefix}_payload_bytes"]
        assert row[f"{prefix}_goodput_band_credited_mbps"] == pytest.approx(credited * row[f"{prefix}_yield"] * ASSUMED_FPS * 8 / 1e6)
    assert (summary["band_cells_lost"] == band_cell_cost(params)).all()
    total = frame_capacity(params).n_cells + band_cell_cost(params)
    assert np.allclose(summary["band_cell_cost_pct"], 100 * band_cell_cost(params) / total)
    assert frame_capacity_band_credited(params).n_cells == total


def test_winner_check_across_both_columns() -> None:
    """If the band's cost changed which configuration wins, the table says so."""
    base = {"device_model": "SM-A346E", "condition_distance_m": 0.5, "pixel_source": "yuv_bilinear", "decoder": "max/mean"}
    rows = [
        {**base, "run_id": "a", "colour_depth": 1, "cell_px": 4, "best_goodput_mbps": 4.40, "best_goodput_band_credited_mbps": 4.50},
        {**base, "run_id": "b", "colour_depth": 2, "cell_px": 10, "best_goodput_mbps": 4.45, "best_goodput_band_credited_mbps": 4.48},
    ]
    table = sweep.winners_both_columns(pd.DataFrame(rows))
    assert table.loc[0, "winner_measured"] == "2 colours, 10 px"
    assert table.loc[0, "winner_band_credited"] == "1 colours, 4 px"
    assert not table.loc[0, "same_winner"]


def test_rows_carry_provenance(processed: tuple) -> None:
    _, summary, _ = processed
    for column in ("run_id", "device_model", "device_manufacturer", "capture_width", "yuv_matrix", "yuv_range",
                   "yuv_range_source", "condition_distance_m", "condition_illuminance_lux", "kernel", "flat_field"):
        assert column in summary.columns
    assert set(summary["run_id"]) == {"d05_lab"} and set(summary["device_model"]) == {"SM-A346E"}
    assert summary["flat_field"].all()


def test_overlays_written_for_failures(processed: tuple) -> None:
    _, _, (_, result, _, root) = processed
    assert root / "overlays" / "d05_lab" / "frame_0003_overlay.png" in result.overlays


def test_resolution_filter_excludes_and_counts(processed: tuple) -> None:
    frames, _, (_, _, cfg, _) = processed
    strict = sweep.SweepConfig(sources=cfg.sources, decoders=cfg.decoders, min_source_px_per_cell=1e6)
    refiltered = frames.copy()
    refiltered.loc[refiltered["excluded"].isna() & refiltered["detected"].astype(bool), "excluded"] = "below_resolution"
    summary = sweep.summarise(refiltered, strict)
    assert (summary["n_below_resolution"] == 6).all() and (summary["n_code_frames"] == 1).all()


def test_figures_carry_run_id_device_and_counts(processed: tuple, tmp_path: Path) -> None:
    _, summary, _ = processed
    paths = plots.captured_figures(summary, tmp_path / "fig")
    assert paths and all(p.exists() for p in paths)
    for svg in (tmp_path / "fig").rglob("*.svg"):
        text = svg.read_text()
        assert "d05_lab" in text, svg.name
        assert "SM-A346E" in text, svg.name
        assert "n=" in text, svg.name
        assert "assumed" in text, svg.name  # the YUV range source is on the figure too
    goodput_svgs = list((tmp_path / "fig").rglob("captured_goodput_*.svg")) + list((tmp_path / "fig").rglob("captured_devices_*.svg"))
    assert goodput_svgs
    for svg in goodput_svgs:
        text = svg.read_text()
        assert "as measured" in text and "credited back" in text, svg.name
    assert all("band " in p.read_text() for p in (tmp_path / "fig").rglob("captured_goodput_*.svg"))  # band cost per cell

    # Headline and sensitivity figures are separated on disk and labelled in the footer.
    headline = list((tmp_path / "fig" / "headline").rglob("*.svg"))
    sensitivity = list((tmp_path / "fig" / "sensitivity").rglob("*.svg"))
    assert headline and sensitivity and len(headline) + len(sensitivity) == len(list((tmp_path / "fig").rglob("*.svg")))
    assert all("luma/saturated (pre-registered headline)" in p.read_text() for p in headline)
    assert all("(sensitivity)" in p.read_text() and "pre-registered" not in p.read_text() for p in sensitivity)


def test_sweep_decodes_with_every_decoder_headline_first() -> None:
    assert sweep.SweepConfig().decoders == ALL_DECODERS and sweep.SweepConfig().decoders[0] == HEADLINE_DECODER


# --------------------------------------------------------------------------- multi-run driver


def test_sweep_skips_and_reports_rejected_runs(tmp_path: Path) -> None:
    good = build_run(tmp_path / "g", RunSpec(run_id="good", n_code_frames=2))
    bad = build_run(tmp_path / "b", RunSpec(run_id="bad", n_code_frames=2, images=False,
                                            manifest_overrides={"schema_version": 7}))
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for built in (good, bad):
        for f in built.runs_dir.glob("*.yaml"):
            (runs_dir / f.name).write_text(f.read_text())
    cfg = sweep.SweepConfig(sources=("yuv_bilinear",))
    frames, summary, rejected = sweep.sweep([good.run_dir, bad.run_dir], cfg, runs_dir=runs_dir,
                                            front_end=stub_front_end(good.h))
    assert set(summary["run_id"]) == {"good"}
    assert list(rejected) == [str(bad.run_dir)] and "schema_version" in rejected[str(bad.run_dir)]


def test_sweep_cli_without_front_end_explains(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    built = build_run(tmp_path, RunSpec(n_code_frames=1))
    code = sweep.main([str(built.run_dir), "--runs-dir", str(built.runs_dir), "--out", str(tmp_path / "out")])
    assert code == 3 and "front end not implemented" in capsys.readouterr().err


def test_sweep_cli_in_sample_flag_prints_a_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    built = build_run(tmp_path, RunSpec(n_code_frames=1))
    sweep.main([str(built.run_dir), "--runs-dir", str(built.runs_dir), "--out", str(tmp_path / "out"), "--in-sample"])
    assert "WARNING: IN-SAMPLE RS SELECTION" in capsys.readouterr().err
    assert sweep.SweepConfig().rs_selection == sweep.OUT_OF_SAMPLE
