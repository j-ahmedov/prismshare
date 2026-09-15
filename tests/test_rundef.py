"""Laptop-side run definitions: one static frame per run, and the clip-check pair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from capture_fixture import RunSpec, build_run

from prism_share.codec.decoder import decode_frame, read_index_band
from prism_share.codec.params import CodecParams
from prism_share.ingest import rundef
from prism_share.ingest.manifest import RunRejected
from prism_share.transmit import display
from prism_share.transmit.reference import extreme_configurations, sweep_configurations


def _rewrite(built, **changes: object) -> None:  # noqa: ANN001
    path = built.runs_dir / "test_run.yaml"
    doc = yaml.safe_load(path.read_text())
    for key, value in changes.items():
        if value is None:
            doc.pop(key, None)
        else:
            doc[key] = value
    path.write_text(yaml.safe_dump(doc))


def test_a_run_holds_exactly_one_frame(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False, displayed_block=2))
    definition = rundef.load_run_definition("test_run", built.runs_dir)
    assert definition.frame.block_id == 2 and definition.frame.frame_index == 3
    assert list(definition.by_index) == [3]


@pytest.mark.parametrize("displayed", [[0, 1], "all", [], [True]])
def test_anything_but_one_frame_is_rejected(tmp_path: Path, displayed: object) -> None:
    built = build_run(tmp_path, RunSpec(images=False, stray_captures={1: 1}))  # frames 0 and 1 exist
    _rewrite(built, displayed=displayed)
    with pytest.raises(RunRejected, match="one static frame|exactly one"):
        rundef.load_run_definition("test_run", built.runs_dir)


def test_a_bare_index_is_accepted(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False))
    _rewrite(built, displayed=0)
    assert rundef.load_run_definition("test_run", built.runs_dir).frame.block_id == 0


@pytest.mark.parametrize("key", ["displayed", "clip_check"])
def test_required_keys(tmp_path: Path, key: str) -> None:
    built = build_run(tmp_path, RunSpec(images=False))
    _rewrite(built, **{key: None})
    with pytest.raises(RunRejected, match=key):
        rundef.load_run_definition("test_run", built.runs_dir)


def test_clip_check_needs_both_extremes(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False))
    doc = yaml.safe_load((built.runs_dir / "test_run.yaml").read_text())
    _rewrite(built, clip_check={"brightest": doc["clip_check"]["brightest"]})
    with pytest.raises(RunRejected, match="brightest.*darkest"):
        rundef.load_run_definition("test_run", built.runs_dir)


def test_extreme_configurations_come_from_the_luminance_table() -> None:
    from prism_share.transmit.reference import luminance_table, reference_frame

    rows = luminance_table(reference_frame(), sweep_configurations())
    brightest, darkest = extreme_configurations()
    assert brightest.frame_mean == max(r.frame_mean for r in rows)
    assert darkest.frame_mean == min(r.frame_mean for r in rows)
    assert brightest.stops_vs_reference > 0 > darkest.stops_vs_reference


def test_new_run_definition_emits_the_clip_check_pair(tmp_path: Path) -> None:
    root = tmp_path
    frames = root / "data" / "runs" / "c4_px8"
    display.main(["frames", "--colour-depth", "4", "--cell-px", "8", "--payload-bytes", "9000", "--n-frames", "3",
                  "--out", str(frames)])
    runs_dir = root / "experiments" / "runs"
    code = rundef.main(["new", "--run-id", "d07_lux200", "--frames", str(frames), "--block", "2",
                        "--condition", "distance_m=0.7", "--condition", "display=oled",
                        "--runs-dir", str(runs_dir), "--clip-dir", str(root / "data" / "clip_check")])
    assert code == 0
    doc = yaml.safe_load((runs_dir / "d07_lux200.yaml").read_text())
    assert doc["displayed"] == [2] and doc["condition"] == {"distance_m": 0.7, "display": "oled"}
    brightest, darkest = extreme_configurations()
    for role, row in (("brightest", brightest), ("darkest", darkest)):
        entry = doc["clip_check"][role]
        assert (entry["colour_depth"], entry["cell_px"]) == (row.colour_depth, row.cell_px)
        folder = Path(entry["frames"]) if Path(entry["frames"]).is_absolute() else root / entry["frames"]
        meta = json.loads((folder / "frames.json").read_text())
        params = CodecParams.from_dict(meta["params"])
        image, _ = display.load_png(folder / "frame_00000.png")
        assert decode_frame(image, params).ok  # a real, valid frame of that configuration
    definition = rundef.load_run_definition("d07_lux200", runs_dir)
    assert definition.frame.block_id == 2 and len(definition.clip_check) == 2
    with pytest.raises(FileExistsError):  # run ids are join keys: never silently reused
        rundef.new_run_definition("d07_lux200", frames, 0, {}, runs_dir=runs_dir, clip_dir=root / "data" / "clip_check")


def test_measure_sequence_locks_checks_then_holds(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False, displayed_block=1))
    definition = rundef.load_run_definition("test_run", built.runs_dir)
    items = display.measure_items(definition)
    assert [i.kind for i in items] == ["reference", "frame", "frame", "frame"]
    assert "brightest" in items[1].name and "darkest" in items[2].name and "RUN FRAME" in items[3].name
    params = definition.params
    assert read_index_band(items[0].image, params).index == 0
    assert read_index_band(items[3].image, params).index == definition.frame.frame_index
    # Space never starts playback, nothing advances by itself, and there is nothing after the run frame.
    backend = display.NullBackend((1920, 1080), keys=["right", "space", None, None, "right", "right", "right", "right", "quit"])
    log = display.DisplayLog(None)
    display.run(items, backend, log, mode="measure", hold_only=True, loop=False)
    shows = [r["index"] for r in log.records if r["type"] == "show"]
    assert shows == [0, 1, 2, 3]
    assert log.records[0]["measurement"] is True and log.records[0]["interval_ms"] is None


def test_play_is_marked_as_a_demonstrator(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    frames = tmp_path / "frames"
    display.main(["frames", "--colour-depth", "1", "--cell-px", "10", "--payload-bytes", "100", "--n-frames", "2", "--out", str(frames)])
    log = tmp_path / "play.jsonl"
    display.main(["play", str(frames), "--dry-run", "2", "--log", str(log)])
    assert "DEMONSTRATOR" in capsys.readouterr().err
    session = json.loads(log.read_text().splitlines()[0])
    assert session["measurement"] is False and "not a measurement path" in session["warning"]


def test_band_cost_equals_the_band_free_grid() -> None:
    from prism_share.codec.framing import frame_capacity, frame_capacity_band_credited
    from prism_share.codec.layout import _grid, band_cell_cost

    for params in (CodecParams(cell_px=4), CodecParams(cell_px=10)):
        without = _grid(params, reserve_band=False).n_cells
        assert frame_capacity_band_credited(params).n_cells == without
        assert frame_capacity(params).n_cells == without - band_cell_cost(params)
    assert band_cell_cost(CodecParams(cell_px=4)) == 1078 and band_cell_cost(CodecParams(cell_px=10)) == 280
    assert np.isclose(100 * 1078 / 38048, 2.83, atol=0.01) and np.isclose(100 * 280 / 7844, 3.57, atol=0.01)
