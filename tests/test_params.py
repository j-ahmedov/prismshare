from __future__ import annotations

import pytest

from prism_share.codec.params import ALLOWED_COLOUR_DEPTHS, CodecParams


def test_defaults_are_the_reference_configuration() -> None:
    p = CodecParams()
    assert (p.colour_depth, p.cell_px, p.cell_gap_px, p.glyph_count) == (4, 8, 1, 16)
    assert (p.frame_px, p.ecc_data, p.ecc_total, p.seed) == (1024, 125, 155, 0)


@pytest.mark.parametrize("depth", ALLOWED_COLOUR_DEPTHS)
def test_bits_per_cell(depth: int) -> None:
    p = CodecParams(colour_depth=depth)
    assert p.glyph_bits == 4
    assert 2**p.colour_bits == depth
    assert p.bits_per_cell == p.glyph_bits + p.colour_bits


def test_derived_quantities() -> None:
    p = CodecParams(cell_px=5, cell_gap_px=2, ecc_data=125, ecc_total=155)
    assert p.pitch_px == 7
    assert p.glyph_weight == 12
    assert p.ecc_parity == 30
    assert p.ecc_correctable == 15


@pytest.mark.parametrize(
    "changes",
    [
        {"colour_depth": 3},
        {"colour_depth": 32},
        {"cell_px": 3},
        {"cell_gap_px": -1},
        {"glyph_count": 12},
        {"glyph_count": 1},
        {"frame_px": 150},
        {"ecc_data": 155},
        {"ecc_data": 0},
        {"ecc_total": 256, "ecc_data": 200},
        {"seed": -1},
    ],
)
def test_invalid_params_rejected(changes: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        CodecParams(**changes)


def test_non_int_rejected() -> None:
    with pytest.raises(TypeError):
        CodecParams(cell_px=8.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CodecParams(seed=True)


def test_frozen() -> None:
    with pytest.raises(AttributeError):
        CodecParams().cell_px = 5  # type: ignore[misc]


def test_serialisation_round_trip() -> None:
    p = CodecParams(colour_depth=16, cell_px=5, seed=42)
    assert CodecParams.from_dict(p.to_dict()) == p
    with pytest.raises(ValueError):
        CodecParams.from_dict({**p.to_dict(), "bogus": 1})


def test_fingerprint_is_stable_and_discriminating() -> None:
    # Pinned: the fingerprint is embedded in every frame header, so it must never drift.
    assert CodecParams().fingerprint() == 2013704606
    fps = {CodecParams(colour_depth=d, cell_px=c).fingerprint() for d in (1, 2, 4) for c in (4, 5, 6)}
    assert len(fps) == 9


def test_label_is_unique_per_config() -> None:
    a, b = CodecParams(), CodecParams(seed=1)
    assert a.label != b.label
    assert "/" not in a.label and " " not in a.label
