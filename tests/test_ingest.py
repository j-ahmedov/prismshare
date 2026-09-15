"""Step 5b: ingest implements docs/capture-format.md exactly, including its six rules."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from capture_fixture import RunSpec, build_run, stub_front_end
from PIL import Image

from prism_share.ingest import manifest as mf
from prism_share.ingest import pull
from prism_share.ingest.ingest import detect_run, ingest_run, load_planes, main, pixel_source
from prism_share.ingest.manifest import ContractViolation, RunRejected


def _rewrite_manifest(run_dir: Path, **changes: object) -> None:
    doc = json.loads((run_dir / "manifest.json").read_text())
    for dotted, value in changes.items():
        node = doc
        *parents, last = dotted.split(".")
        for p in parents:
            node = node[p]
        node[last] = value
    (run_dir / "manifest.json").write_text(json.dumps(doc))


def _rewrite_line(run_dir: Path, index: int, **changes: object) -> None:
    lines = [json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines()]
    lines[index].update(changes)
    (run_dir / "frames.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")


def test_contract_constants_match_the_document() -> None:
    doc = (Path(__file__).parent.parent / "docs" / "capture-format.md").read_text()
    assert mf.PHONE_CAPTURES_DIR in doc
    for reason in mf.TAINT_REASONS:
        assert f"`{reason}`" in doc
    for suffix in mf.PLANE_SUFFIXES.values():
        assert suffix in doc
    assert "0.01" in doc and mf.MAX_TAINTED_FRACTION == 0.01


def test_clean_run_is_ingested(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False, n_code_frames=10))
    run = ingest_run(built.run_dir, built.runs_dir)
    assert run.report.kept == 10 and run.report.dropped == 0 and run.report.warnings == ()
    assert run.report.yuv_range_source == "assumed"
    assert run.definition.params == built.params


# --------------------------------------------------------------------------- rule 1


def test_rule1_unknown_schema_version_rejected(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False, manifest_overrides={"schema_version": 2}))
    with pytest.raises(RunRejected, match="schema_version 2 is unknown"):
        ingest_run(built.run_dir, built.runs_dir)


# --------------------------------------------------------------------------- rule 2


def test_rule2_run_without_definition_rejected(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False, write_definition=False))
    with pytest.raises(RunRejected, match="typos"):
        ingest_run(built.run_dir, built.runs_dir)


def test_rule2_typo_in_run_id_rejected(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False))
    _rewrite_manifest(built.run_dir, run_id="test_rnu")  # typed by hand at 11pm
    with pytest.raises(RunRejected, match="test_run"):  # the message lists the known ids
        ingest_run(built.run_dir, built.runs_dir)


# --------------------------------------------------------------------------- rules 3 and 4


def test_rule3_tainted_frames_dropped_and_counted_by_reason(tmp_path: Path) -> None:
    spec = RunSpec(images=False, n_code_frames=200, tainted={3: ["ae_state"], 50: ["gains_drift", "exposure_drift"]})
    built = build_run(tmp_path, spec)
    run = ingest_run(built.run_dir, built.runs_dir)
    assert run.report.dropped == 2 and run.report.kept == 198
    assert run.report.dropped_by_reason == {"ae_state": 1, "awb_state": 0, "sensitivity_drift": 0,
                                            "exposure_drift": 1, "gains_drift": 1}
    assert all(not r.tainted for r in run.frames)
    assert {r.index for r in run.frames}.isdisjoint({3, 50})


def test_rule4_exactly_one_percent_is_accepted(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False, n_code_frames=100, tainted={7: ["awb_state"]}))
    assert ingest_run(built.run_dir, built.runs_dir).report.tainted_fraction == 0.01


def test_rule4_more_than_one_percent_refuses_the_run(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False, n_code_frames=100, tainted={7: ["awb_state"], 9: ["ae_state"]}))
    with pytest.raises(RunRejected, match="must be repeated"):
        ingest_run(built.run_dir, built.runs_dir)


# --------------------------------------------------------------------------- rule 5


def test_rule5_written_differs_from_requested_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    built = build_run(tmp_path, RunSpec(images=False, n_code_frames=10, requested=12))
    run = ingest_run(built.run_dir, built.runs_dir)
    assert run.report.warnings == ("frames.written (10) != frames.requested (12)",)
    assert "frames.requested" in capsys.readouterr().err


# --------------------------------------------------------------------------- rule 6


def test_rule6_resolution_is_read_from_the_manifest_and_checked(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(n_code_frames=1))
    run = ingest_run(built.run_dir, built.runs_dir)
    assert run.manifest.capture_resolution == (1560, 1530)
    planes = load_planes(run, run.frames[0])
    assert planes["y"].shape == (1530, 1560) and planes["u"].shape == (765, 780) and planes["rgb"].shape == (1530, 1560, 3)
    _rewrite_manifest(built.run_dir, **{"camera.capture_resolution": [1530, 1560]})  # swapped: must not pass silently
    run = ingest_run(built.run_dir, built.runs_dir)
    with pytest.raises(ContractViolation, match="capture_resolution"):
        load_planes(run, run.frames[0])


def test_rule6_yuv_matrix_and_range_come_from_the_manifest(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False, manifest_overrides={"color.yuv_matrix": "BT709",
                                                                        "color.yuv_range": "full",
                                                                        "color.yuv_range_source": "read"}))
    run = ingest_run(built.run_dir, built.runs_dir)
    assert run.yuv_spec.matrix == "bt709" and run.yuv_spec.full_range
    assert run.report.yuv_range_source == "read"


@pytest.mark.parametrize("field,value", [("color.yuv_matrix", "BT2020"), ("color.yuv_range", "video"),
                                         ("color.yuv_range_source", "guessed"), ("color.pixel_format", "HEIC")])
def test_rule6_values_that_would_need_assuming_are_rejected(tmp_path: Path, field: str, value: str) -> None:
    built = build_run(tmp_path, RunSpec(images=False, manifest_overrides={field: value}))
    with pytest.raises(RunRejected):
        ingest_run(built.run_dir, built.runs_dir)


def test_jpeg_path_is_ingested_with_one_source(tmp_path: Path) -> None:
    """color.pixel_format marks the path; a JPEG run has one frame_NNNN.jpg and one pixel source."""
    built = build_run(tmp_path, RunSpec(n_code_frames=2, pixel_format="JPEG"))
    run = ingest_run(built.run_dir, built.runs_dir)
    assert run.is_jpeg and run.available_sources == ("jpeg",)
    planes = load_planes(run, run.frames[0])
    assert set(planes) == {"jpeg"} and planes["jpeg"].shape == (1530, 1560, 3)
    assert run.frames[0].path(run.run_dir, "jpeg").name.endswith(".jpg")
    assert pixel_source(run, planes, "jpeg").shape == (1530, 1560, 3)


# --------------------------------------------------------------------------- contract violations


@pytest.mark.parametrize(
    "change,match",
    [
        ({"taint_reasons": ["focus_drift"], "tainted": True}, "closed set"),
        ({"taint_reasons": [], "tainted": True}, "if and only if"),
        ({"taint_reasons": ["ae_state"], "tainted": False}, "if and only if"),
        ({"sensor_sensitivity": "320"}, "wrong type"),
    ],
)
def test_frames_jsonl_violations_are_errors(tmp_path: Path, change: dict, match: str) -> None:
    built = build_run(tmp_path, RunSpec(images=False, n_code_frames=5))
    _rewrite_line(built.run_dir, 2, **change)
    with pytest.raises(ContractViolation, match=match):
        ingest_run(built.run_dir, built.runs_dir)


def test_missing_manifest_field_is_an_error(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False))
    doc = json.loads((built.run_dir / "manifest.json").read_text())
    del doc["frames"]["written"]
    (built.run_dir / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(ContractViolation, match="frames.written"):
        ingest_run(built.run_dir, built.runs_dir)


def test_counts_that_contradict_frames_jsonl_are_errors(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(images=False, n_code_frames=200, tainted={1: ["ae_state"]}))
    _rewrite_manifest(built.run_dir, **{"frames.tainted": 0})
    with pytest.raises(ContractViolation, match="frames.tainted"):
        ingest_run(built.run_dir, built.runs_dir)


def test_files_come_from_file_stem_not_index(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(n_code_frames=2, stem="shot_{:02d}_x"))
    run = ingest_run(built.run_dir, built.runs_dir)
    assert run.frames[0].file_stem == "shot_00_x"
    assert load_planes(run, run.frames[0], ("y",))["y"].shape == (1530, 1560)


# --------------------------------------------------------------------------- pixels and merge


def test_pixel_sources_follow_the_manifest_matrix(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(n_code_frames=1))
    run = ingest_run(built.run_dir, built.runs_dir)
    planes = load_planes(run, run.frames[0])
    near, bil = pixel_source(run, planes, "yuv_nearest"), pixel_source(run, planes, "yuv_bilinear")
    assert near.shape == bil.shape == (1530, 1560, 3) and not np.array_equal(near, bil)
    assert np.abs(bil - planes["rgb"]).max() <= 1.0  # the fixture's "phone RGB" is bilinear BT.601
    assert pixel_source(run, planes, "y").shape == (1530, 1560)


def test_small_run_with_one_tainted_frame_is_refused(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(n_code_frames=3, tainted={1: ["ae_state"]}, requested=3))
    # 1 of 3 tainted is > 1 %: the run is refused before any frame is loaded or merged.
    with pytest.raises(RunRejected):
        ingest_run(built.run_dir, built.runs_dir)


def test_detect_run_merges_one_row_per_kept_frame(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(n_code_frames=3))
    run = ingest_run(built.run_dir, built.runs_dir)
    table = detect_run(run, front_end=stub_front_end(built.h, fail_stems={1}), overlay_dir=tmp_path / "ov")
    assert len(table) == 3
    assert list(table["detected"]) == [True, False, True]
    assert table.loc[1, "failure_reason"].startswith("front end")
    for column in ("run_id", "device_model", "file_stem", "ae_state", "sensor_timestamp_ns", "tainted",
                   "source_px_per_cell", "rectify_scale_k", "resampling_kernel", "flat_field_applied",
                   "reprojection_error_px", "fiducial0_x", "condition_distance_m", "yuv_range_source"):
        assert column in table.columns
    assert (tmp_path / "ov" / "frame_0001_overlay.png").exists()  # every failure gets an overlay


def test_detect_run_without_a_front_end_says_so(tmp_path: Path) -> None:
    built = build_run(tmp_path, RunSpec(n_code_frames=1))
    with pytest.raises(NotImplementedError):
        detect_run(ingest_run(built.run_dir, built.runs_dir))


def test_cli_reports_and_rejects(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = build_run(tmp_path / "a", RunSpec(images=False, n_code_frames=100, tainted={4: ["ae_state"]}))
    assert main([str(good.run_dir), "--runs-dir", str(good.runs_dir), "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "99 frames kept, 1 tainted frames dropped" in out and "ae_state=1" in out
    report = json.loads((tmp_path / "out" / "test_run" / "ingest_report.json").read_text())
    assert report["dropped_by_reason"]["ae_state"] == 1
    bad = build_run(tmp_path / "b", RunSpec(images=False, manifest_overrides={"schema_version": 9}))
    assert main([str(bad.run_dir), "--runs-dir", str(bad.runs_dir), "--out", str(tmp_path / "out")]) == 2
    assert "REJECTED" in capsys.readouterr().err


# --------------------------------------------------------------------------- adb pull


def test_pull_runs_the_documented_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        target = Path(cmd[-1]) / "d07_a30_lux200_oled"
        target.mkdir(parents=True)
        (target / "manifest.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pull.subprocess, "run", fake_run)
    local = pull.pull("d07_a30_lux200_oled", tmp_path / "data", adb="adb")
    assert calls == [["adb", "pull",
                      "/sdcard/Android/data/io.github.ahmedov.prismshare.capture/files/captures/d07_a30_lux200_oled",
                      f"{tmp_path / 'data'}/"]]
    assert local == tmp_path / "data" / "d07_a30_lux200_oled"
    with pytest.raises(pull.PullError, match="already exists"):
        pull.pull("d07_a30_lux200_oled", tmp_path / "data", adb="adb")


def test_pull_reports_adb_failure_and_unsafe_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pull.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "no devices"))
    with pytest.raises(pull.PullError, match="no devices"):
        pull.pull("run1", tmp_path, adb="adb")
    with pytest.raises(pull.PullError):
        pull.phone_path("../etc")
    monkeypatch.setattr(pull.shutil, "which", lambda name: None)
    with pytest.raises(pull.PullError, match="platform-tools"):
        pull.pull("run2", tmp_path)
