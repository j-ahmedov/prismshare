from __future__ import annotations

import numpy as np
import pytest

from prism_share.colourspace import (
    YuvSpec,
    linear_luminance,
    linear_to_srgb,
    mean_linear_luminance,
    rgb_to_yuv,
    srgb_to_linear,
    subsample_chroma,
    upsample_chroma,
    yuv_to_rgb,
)

SPECS = [YuvSpec("bt601", False), YuvSpec("bt601", True), YuvSpec("bt709", False), YuvSpec("bt709", True)]


def test_srgb_reference_values() -> None:
    assert srgb_to_linear(128 / 255) == pytest.approx(0.2158, abs=1e-4)  # the "flat grey 128" of the brief
    assert srgb_to_linear(0.0) == 0.0 and srgb_to_linear(1.0) == pytest.approx(1.0)
    v = np.linspace(0, 1, 101)
    assert np.allclose(linear_to_srgb(srgb_to_linear(v)), v)


def test_linear_luminance() -> None:
    assert linear_luminance([255, 255, 255]) == pytest.approx(1.0)
    assert linear_luminance([0, 0, 0]) == 0.0
    assert linear_luminance([0, 255, 0]) == pytest.approx(0.7152)
    half = np.zeros((2, 2, 3))
    half[0] = 255
    assert mean_linear_luminance(half) == pytest.approx(0.5)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.matrix}-{'full' if s.full_range else 'limited'}")
def test_black_white_levels(spec: YuvSpec) -> None:
    y, cb, cr = rgb_to_yuv(np.array([[[0, 0, 0], [255, 255, 255]]]), spec)
    lo, hi = (0, 255) if spec.full_range else (16, 235)
    assert y.tolist() == [[lo, hi]]
    assert cb.tolist() == cr.tolist() == [[128, 128]]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.matrix}-{'full' if s.full_range else 'limited'}")
def test_round_trip_error_is_quantisation_only(spec: YuvSpec) -> None:
    rgb = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3)).astype(np.float64)
    back = yuv_to_rgb(*rgb_to_yuv(rgb, spec), spec)
    # 8-bit rounding of Y'CbCr costs at most ~1-2 code values per channel after the inverse matrix.
    assert np.abs(back - rgb).max() <= 3.0


def test_greys_survive_any_chroma_processing_exactly() -> None:
    grey = np.stack([np.array([[0, 255], [255, 0]], dtype=np.float64)] * 3, axis=-1)
    y, cb, cr = rgb_to_yuv(grey)
    cb2 = upsample_chroma(subsample_chroma(cb, 2), y.shape, "bilinear")
    cr2 = upsample_chroma(subsample_chroma(cr, 2), y.shape, "bilinear")
    assert np.array_equal(yuv_to_rgb(y, cb2, cr2), grey)


def test_matrices_differ() -> None:
    red = np.array([[[255, 0, 0]]], dtype=np.float64)
    assert rgb_to_yuv(red, YuvSpec("bt601"))[0] != rgb_to_yuv(red, YuvSpec("bt709"))[0]


def test_subsample_by_two_is_block_mean() -> None:
    plane = np.array([[10, 20, 30, 30], [30, 40, 30, 30]], dtype=np.uint8)
    assert subsample_chroma(plane, 2).tolist() == [[25, 30]]


def test_nearest_upsample_replicates() -> None:
    plane = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert np.array_equal(upsample_chroma(plane, (4, 6), "nearest"), np.kron(plane, np.ones((2, 2))))


def test_bilinear_upsample_interpolates() -> None:
    plane = np.array([[0, 100]], dtype=np.uint8)
    up = upsample_chroma(plane, (2, 4), "bilinear")
    assert up[0, 0] == 0 and up[0, 3] == 100 and 0 < up[0, 1] < up[0, 2] < 100


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        YuvSpec("bt2020")
    with pytest.raises(ValueError):
        upsample_chroma(np.zeros((2, 2), np.uint8), (4, 4), "bicubic")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        subsample_chroma(np.zeros((2, 2), np.uint8), 0.5)
