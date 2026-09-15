"""Step 5a, lens-independent half: pure geometry, no camera.

A generated frame is imaged through a KNOWN homography (camera-like: warped
at 2x supersampling, then area-averaged). The front end is replaced by a stub
returning the true marker corners perturbed by up to +/-2 px - the accuracy
the real front end must deliver. The geometry must then recover the
homography to better than 0.1 source px everywhere on the code, and the
decoded symbols must be error-free.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from prism_share.analysis import detect as det
from prism_share.analysis.detect import FlatField, FrontEndResult
from prism_share.codec.decoder import read_symbols
from prism_share.codec.encoder import encode_frames
from prism_share.codec.params import (
    ALLOWED_COLOUR_DEPTHS,
    FIDUCIAL_IDS,
    RECTIFY_KERNELS,
    SWEEP_CELL_PX,
    CodecParams,
)
from prism_share.transmit.reference import reference_frame

F = CodecParams().frame_px
FRAME_CORNERS = np.array([[-0.5, -0.5], [F - 0.5, -0.5], [F - 0.5, F - 0.5], [-0.5, F - 0.5]], np.float32)

# Named geometries: where the frame's outer corners land in the capture, and the capture size.
GEOMETRIES = {
    "frontal_m1.0": ([[100, 80], [1150, 95], [1140, 1140], [90, 1130]], (1250, 1230)),
    "oblique_rotated_m1.9": ([[400, 150], [2350, 420], [2150, 2300], [150, 1950]], (2500, 2400)),
    "frontal_m2.6": ([[60, 50], [2720, 70], [2710, 2720], [40, 2700]], (2800, 2800)),
}


def homography(name: str) -> npt.NDArray[np.float64]:
    return cv2.getPerspectiveTransform(FRAME_CORNERS, np.array(GEOMETRIES[name][0], np.float32)).astype(np.float64)


def capture(image: npt.NDArray, h: npt.NDArray[np.float64], size: tuple[int, int], *, ss: int = 2,
            gain: npt.NDArray | None = None) -> npt.NDArray[np.float64]:
    """Camera-like image of ``image`` through ``h``: supersampled warp, area-averaged, 8-bit."""
    s = np.array([[ss, 0, (ss - 1) / 2], [0, ss, (ss - 1) / 2], [0, 0, 1]])
    img = np.asarray(image, np.float32)
    if gain is not None:
        img = img * (gain[..., None] if img.ndim == 3 else gain)
    big = cv2.warpPerspective(img, s @ h, (size[0] * ss, size[1] * ss), flags=cv2.INTER_LINEAR, borderValue=(40, 40, 40))
    return np.clip(np.rint(cv2.resize(big, size, interpolation=cv2.INTER_AREA)), 0, 255)


def stub_front_end(h: npt.NDArray[np.float64], jitter: float = 2.0, seed: int = 0) -> det.FrontEnd:
    true = det.project(h, det.ideal_marker_corners(F))
    rng = np.random.default_rng(seed)
    noisy = {m: true[i] + rng.uniform(-jitter, jitter, (4, 2)) for i, m in enumerate(FIDUCIAL_IDS)}
    return lambda image, params: FrontEndResult(dict(noisy))


def max_registration_error(recovered: tuple, true: npt.NDArray[np.float64], params: CodecParams) -> float:
    pts = det.cell_centres(params)
    return float(np.linalg.norm(det.project(np.array(recovered), pts) - det.project(true, pts), axis=1).max())


# --------------------------------------------------------------------------- the geometry test


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
@pytest.mark.parametrize("params", [CodecParams(colour_depth=16, cell_px=4), CodecParams(colour_depth=1, cell_px=4),
                                    CodecParams(colour_depth=4, cell_px=10)], ids=lambda p: p.label)
def test_known_homography_is_recovered_and_symbols_are_error_free(geometry: str, params: CodecParams) -> None:
    h = homography(geometry)
    frame = encode_frames(b"geometry", params, n_frames=1)[0]
    shot = capture(frame.image, h, GEOMETRIES[geometry][1])
    result = det.detect(shot, params, front_end=stub_front_end(h))
    rec = result.record
    assert rec.detected, rec.failure_reason
    assert max_registration_error(rec.homography, h, params) < 0.1
    assert rec.reprojection_error_px < 0.1
    readout = read_symbols(result.rectified, params)
    assert np.array_equal(readout.glyphs, frame.glyphs)
    assert np.array_equal(readout.colours, frame.colours)


def test_works_on_a_single_luma_plane() -> None:
    params = CodecParams(colour_depth=1, cell_px=4)
    h = homography("oblique_rotated_m1.9")
    frame = encode_frames(b"luma", params, n_frames=1)[0]
    y_plane = capture(cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY), h, GEOMETRIES["oblique_rotated_m1.9"][1])
    result = det.detect(y_plane, params, front_end=stub_front_end(h))
    assert result.rectified is not None and result.rectified.ndim == 2
    assert max_registration_error(result.record.homography, h, params) < 0.1
    assert np.array_equal(read_symbols(result.rectified, params).glyphs, frame.glyphs)


def test_refinement_is_identical_for_every_configuration() -> None:
    """One detector, one parameter set: the same capture geometry gives bit-identical corners
    in all 25 configurations, because refinement only reads the (identical) static region."""
    h = homography("frontal_m1.0")
    size = GEOMETRIES["frontal_m1.0"][1]
    fe = stub_front_end(h)
    results = []
    for d, c in itertools.product(ALLOWED_COLOUR_DEPTHS, SWEEP_CELL_PX):
        p = CodecParams(colour_depth=d, cell_px=c)
        shot = capture(encode_frames(b"same", p, n_frames=1)[0].image, h, size)
        results.append(det.detect(shot, p, front_end=fe).refined_corners)
    assert all(np.array_equal(r, results[0]) for r in results[1:])


# --------------------------------------------------------------------------- record contents


def test_record_resolution_fields() -> None:
    params = CodecParams(colour_depth=4, cell_px=6)
    h = homography("frontal_m2.6")
    shot = capture(encode_frames(b"r", params, n_frames=1)[0].image, h, GEOMETRIES["frontal_m2.6"][1])
    rec = det.detect(shot, params, front_end=stub_front_end(h)).record
    magnification = (2720 - 60) / F  # this geometry is almost frontal
    assert rec.source_px_per_cell == pytest.approx(magnification * params.pitch_px, rel=0.01)
    assert rec.source_px_per_cell_min <= rec.source_px_per_cell
    assert rec.quad_area_px == pytest.approx(det._shoelace(np.array(rec.fiducial_centres_px)))
    assert len(rec.fiducial_centres_px) == len(FIDUCIAL_IDS)
    assert rec.resampling_kernel == "bilinear" and rec.flat_field_applied is False


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_scale_k_never_downsamples(geometry: str) -> None:
    h = homography(geometry)
    k = det.choose_scale(h, F)
    grid = np.linspace(-0.5, F - 0.5, 65)
    gx, gy = np.meshgrid(grid, grid)
    assert k >= det.local_magnification(h, np.c_[gx.ravel(), gy.ravel()]).max()
    assert k - 1 < det.local_magnification(h, np.c_[gx.ravel(), gy.ravel()]).max()  # and is the smallest such integer


@pytest.mark.parametrize("kernel", list(RECTIFY_KERNELS))
def test_every_kernel_is_selectable_and_recorded(kernel: str) -> None:
    params = CodecParams(colour_depth=2, cell_px=8)
    h = homography("frontal_m1.0")
    shot = capture(encode_frames(b"k", params, n_frames=1)[0].image, h, GEOMETRIES["frontal_m1.0"][1])
    result = det.detect(shot, params, front_end=stub_front_end(h), kernel=kernel)
    k = result.record.rectify_scale_k
    assert result.record.resampling_kernel == kernel
    assert result.rectified.shape == (k * F, k * F, 3)
    with pytest.raises(ValueError):
        det.detect(shot, params, front_end=stub_front_end(h), kernel="bicubic")


def test_explicit_scale_is_honoured() -> None:
    params = CodecParams()
    h = homography("frontal_m1.0")
    shot = capture(encode_frames(b"k", params, n_frames=1)[0].image, h, GEOMETRIES["frontal_m1.0"][1])
    result = det.detect(shot, params, front_end=stub_front_end(h), k=3)
    assert result.record.rectify_scale_k == 3 and result.rectified.shape[0] == 3 * F


def test_rectification_of_an_identity_view_reproduces_the_frame() -> None:
    frame = encode_frames(b"id", CodecParams(), n_frames=1)[0].image
    for k in (1, 2):
        out = det.rectify(frame, np.eye(3), F, k, "nearest")
        assert np.array_equal(out[::k, ::k].astype(np.uint8), frame)


def test_to_row_is_flat() -> None:
    params = CodecParams()
    h = homography("frontal_m1.0")
    shot = capture(encode_frames(b"row", params, n_frames=1)[0].image, h, GEOMETRIES["frontal_m1.0"][1])
    row = det.detect(shot, params, front_end=stub_front_end(h)).record.to_row()
    assert all(not isinstance(v, (tuple, list, dict)) for v in row.values())
    assert {"fiducial0_x", "fiducial3_y", "h22", "source_px_per_cell", "rectify_scale_k"} <= set(row)


# --------------------------------------------------------------------------- flat field


def test_flat_field_recovers_vignetting() -> None:
    params = CodecParams(colour_depth=16, cell_px=4)
    h = homography("oblique_rotated_m1.9")
    size = GEOMETRIES["oblique_rotated_m1.9"][1]
    yy, xx = np.mgrid[0:F, 0:F]
    vignetting = (1 - 0.35 * ((xx - F / 2) ** 2 + (yy - F / 2) ** 2) / (F / 2) ** 2).astype(np.float32)
    fe = stub_front_end(h, jitter=0.5)
    ref = det.detect(capture(np.asarray(reference_frame(F)), h, size, gain=vignetting), params, front_end=fe)
    flat = FlatField.from_reference(ref.rectified, params)
    error = np.abs(flat.gain[..., 1] - vignetting / np.median(vignetting))
    assert np.median(error) < 0.005 and np.percentile(error[64:-64, 64:-64], 99) < 0.02

    frame = encode_frames(b"flat", params, n_frames=1)[0]
    result = det.detect(capture(frame.image, h, size, gain=vignetting), params, front_end=fe, flat_field=flat)
    assert result.record.flat_field_applied
    readout = read_symbols(result.rectified, params)
    assert np.array_equal(readout.glyphs, frame.glyphs) and np.array_equal(readout.colours, frame.colours)


# --------------------------------------------------------------------------- failure is an outcome


def test_default_front_end_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        det.detect(np.zeros((100, 100), np.uint8), CodecParams())


def test_missing_marker_is_a_recorded_failure() -> None:
    h = homography("frontal_m1.0")
    partial = stub_front_end(h)
    def front_end(image, params):  # noqa: ANN001, ANN202
        found = partial(image, params).corners
        found.pop(FIDUCIAL_IDS[2])
        return FrontEndResult(found, failure_reason="marker 2 occluded")
    rec = det.detect(np.zeros((1250, 1230, 3)), CodecParams(), front_end=front_end).record
    assert not rec.detected and "occluded" in rec.failure_reason and str(FIDUCIAL_IDS[2]) in rec.failure_reason
    assert rec.homography is None and rec.rectify_scale_k is None


def test_featureless_image_is_a_recorded_failure() -> None:
    h = homography("frontal_m1.0")
    rec = det.detect(np.full((1250, 1230), 128.0), CodecParams(), front_end=stub_front_end(h)).record
    assert not rec.detected and "contrast" in rec.failure_reason


def test_mirrored_marker_order_is_a_recorded_failure() -> None:
    h = homography("frontal_m1.0")
    params = CodecParams()
    shot = capture(encode_frames(b"m", params, n_frames=1)[0].image, h, GEOMETRIES["frontal_m1.0"][1])
    good = stub_front_end(h)(shot, params).corners
    swapped = {FIDUCIAL_IDS[0]: good[FIDUCIAL_IDS[1]], FIDUCIAL_IDS[1]: good[FIDUCIAL_IDS[0]],
               FIDUCIAL_IDS[2]: good[FIDUCIAL_IDS[3]], FIDUCIAL_IDS[3]: good[FIDUCIAL_IDS[2]]}
    rec = det.detect(shot, params, front_end=lambda i, p: FrontEndResult(swapped)).record
    assert not rec.detected


# --------------------------------------------------------------------------- decoder infers scale


@pytest.mark.parametrize("k", [1, 2, 3])
def test_decoder_infers_rectification_scale(k: int) -> None:
    params = CodecParams(colour_depth=8, cell_px=5)
    frame = encode_frames(b"scale", params, n_frames=1)[0]
    upscaled = np.kron(frame.image, np.ones((k, k, 1)))
    readout = read_symbols(upscaled, params)
    assert np.array_equal(readout.glyphs, frame.glyphs) and np.array_equal(readout.colours, frame.colours)


def test_decoder_rejects_non_integer_scale() -> None:
    with pytest.raises(ValueError):
        read_symbols(np.zeros((1536, 1536, 3)), CodecParams())


# --------------------------------------------------------------------------- overlay


def test_overlay_is_written_for_success_and_failure(tmp_path: Path) -> None:
    params = CodecParams(colour_depth=4, cell_px=10)
    h = homography("frontal_m1.0")
    shot = capture(encode_frames(b"o", params, n_frames=1)[0].image, h, GEOMETRIES["frontal_m1.0"][1])
    ok = det.detect(shot, params, front_end=stub_front_end(h))
    det.write_overlay(tmp_path / "ok.png", shot, ok, params)
    img = cv2.imread(str(tmp_path / "ok.png"))
    assert img.shape[:2] == shot.shape[:2]
    assert not np.array_equal(img, cv2.cvtColor(shot.astype(np.uint8), cv2.COLOR_RGB2BGR))  # something was drawn
    bad = det.detect(np.full((400, 400), 128.0), params, front_end=stub_front_end(h))
    det.write_overlay(tmp_path / "bad.png", np.full((400, 400), 128.0), bad, params)
    assert (tmp_path / "bad.png").stat().st_size > 0
