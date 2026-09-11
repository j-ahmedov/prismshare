"""Step 3: degradations are pure, seeded, composable and do what their severity says."""

from __future__ import annotations

import numpy as np
import pytest

from prism_share.codec.decoder import read_symbols
from prism_share.codec.encoder import encode_frames
from prism_share.codec.params import CodecParams
from prism_share.colourspace import rgb_to_yuv
from prism_share.sim import degrade
from prism_share.sim.degrade import DEGRADATIONS, Degradation, apply_chain

P = CodecParams(colour_depth=4, cell_px=6)
FRAME = encode_frames(b"degradation tests", P, n_frames=1)[0]
GREY = np.full((64, 64, 3), 128.0)

# (name, a mild severity) for every degradation with an identity at severity 0.
MILD = [("blur", 0.8), ("noise", 10.0), ("perspective", 0.05), ("white_balance", 1.0)]


@pytest.mark.parametrize("name,severity", [*MILD, ("chroma", 2.0), ("quantize", 0.0)])
def test_contract_shape_dtype_range_purity(name: str, severity: float) -> None:
    before = FRAME.image.copy()
    out = DEGRADATIONS[name](FRAME.image, severity, P, index=0)
    assert out.shape == FRAME.image.shape and out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 255.0
    assert np.array_equal(FRAME.image, before), "input was modified"


@pytest.mark.parametrize("name", ["blur", "noise", "perspective", "white_balance"])
def test_severity_zero_is_identity(name: str) -> None:
    assert np.array_equal(DEGRADATIONS[name](FRAME.image, 0.0, P), FRAME.image.astype(np.float64))


@pytest.mark.parametrize("name,severity", MILD)
def test_seeded_and_deterministic(name: str, severity: float) -> None:
    a = DEGRADATIONS[name](FRAME.image, severity, P, index=3)
    b = DEGRADATIONS[name](FRAME.image, severity, P, index=3)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name,severity", [("noise", 10.0), ("perspective", 0.05)])
def test_randomness_depends_on_seed_and_index(name: str, severity: float) -> None:
    base = DEGRADATIONS[name](FRAME.image, severity, P, index=0)
    assert not np.array_equal(base, DEGRADATIONS[name](FRAME.image, severity, P, index=1))
    assert not np.array_equal(base, DEGRADATIONS[name](FRAME.image, severity, P.replace(seed=1), index=0))


def test_randomness_independent_of_chain_position() -> None:
    noise = Degradation.of("noise", 10)
    identity = Degradation.of("blur", 0)
    a = apply_chain(FRAME.image, [noise], P, index=2)
    b = apply_chain(FRAME.image, [identity, noise, identity], P, index=2)
    assert np.array_equal(a, b)


def test_chains_compose_in_any_order() -> None:
    steps = [Degradation.of("blur", 0.7), Degradation.of("chroma", 2, upsample="bilinear"), Degradation.of("noise", 5)]
    forward = apply_chain(FRAME.image, steps, P)
    backward = apply_chain(FRAME.image, steps[::-1], P)
    assert forward.shape == backward.shape == FRAME.image.shape
    assert not np.array_equal(forward, backward)  # order matters physically, and both run


def test_noise_has_requested_sigma() -> None:
    out = degrade.sensor_noise(GREY, 12.0, P)
    assert abs(float((out - GREY).mean())) < 0.3
    assert float((out - GREY).std()) == pytest.approx(12.0, rel=0.03)


def test_normal_variates_are_standard() -> None:
    z = degrade._normal(P, "test", 0, (200_000,))
    assert abs(z.mean()) < 0.01 and z.std() == pytest.approx(1.0, abs=0.01)


def test_blur_spreads_an_edge() -> None:
    edge = np.zeros((32, 32, 3))
    edge[:, 16:] = 255
    out = degrade.gaussian_blur(edge, 2.0, P)
    assert 0 < out[16, 15, 0] < 128 < out[16, 16, 0] < 255


def test_perspective_is_mild_at_small_severity_and_worse_at_large() -> None:
    small = np.abs(degrade.perspective_warp(FRAME.image, 0.02, P) - FRAME.image).mean()
    large = np.abs(degrade.perspective_warp(FRAME.image, 0.3, P) - FRAME.image).mean()
    assert 0 < small < large


def test_white_balance_is_a_linear_light_gain_with_clipping() -> None:
    out = degrade.white_balance_shift(np.array([[[128.0, 128.0, 128.0], [255, 255, 255], [0, 0, 0]]]), 1.0, P)
    grey, white, black = out[0]
    assert grey[0] > 128 and grey[1] == pytest.approx(128, abs=1e-9) and grey[2] < 128
    assert white[0] == pytest.approx(255) and white[2] < 255  # red clips at full scale
    assert np.all(black == 0)


def test_white_balance_is_undone_by_decoder_normalisation() -> None:
    """A pure gain without clipping is cancelled by the reference-white normalisation."""
    shifted = degrade.quantize(degrade.white_balance_shift(FRAME.image, -1.0, P))  # cool shift: R gain < 1, B clips
    r = read_symbols(shifted, P)
    assert np.mean(r.glyphs != FRAME.glyphs) == 0.0


def test_chroma_leaves_luma_alone_except_where_rgb_clips() -> None:
    """4:2:0 never touches Y'. Converting back to RGB can leave the gamut (e.g. a black
    pixel given a neighbour's blue chroma needs negative R and G); clipping those
    channels to 0 is what then changes luma, and only there."""
    out = degrade.chroma_subsample(FRAME.image, 2.0, P, upsample="bilinear")
    y_in = rgb_to_yuv(FRAME.image)[0].astype(int)
    y_out = rgb_to_yuv(out)[0].astype(int)
    in_gamut = ((out > 0.5) & (out < 254.5)).all(axis=2)
    assert in_gamut.mean() > 0.2
    assert np.abs(y_out - y_in)[in_gamut].max() <= 2


def test_chroma_does_not_touch_monochrome() -> None:
    mono = CodecParams(colour_depth=1, cell_px=4)
    frame = encode_frames(b"mono", mono, n_frames=1)[0].image
    for upsample in ("nearest", "bilinear"):
        assert np.array_equal(degrade.chroma_subsample(frame, 4.0, mono, upsample=upsample), frame.astype(np.float64))


def test_chroma_damage_grows_with_pitch() -> None:
    p16 = CodecParams(colour_depth=16, cell_px=6)
    fr = encode_frames(b"pitch", p16, n_frames=1)[0]
    ser = []
    for pitch in (1.0, 2.0, 4.0):
        r = read_symbols(degrade.quantize(degrade.chroma_subsample(fr.image, pitch, p16)), p16)
        ser.append(np.mean(r.colours != fr.colours))
    assert ser[0] == 0.0 and ser[0] < ser[1] < ser[2]


def test_chroma_rejects_pitch_below_one() -> None:
    with pytest.raises(ValueError):
        degrade.chroma_subsample(FRAME.image, 0.5, P)


def test_degradation_dataclass() -> None:
    d = Degradation.of("chroma", 2, upsample="bilinear", matrix="bt709")
    assert d.label == "chroma(2,matrix=bt709,upsample=bilinear)"
    assert hash(d) == hash(Degradation.of("chroma", 2, matrix="bt709", upsample="bilinear"))
    with pytest.raises(ValueError):
        Degradation.of("fog", 1)


def test_quantize_rounds_and_clips() -> None:
    out = degrade.quantize(np.array([[[-3.0, 12.4, 300.0]]]))
    assert out.tolist() == [[[0.0, 12.0, 255.0]]]
