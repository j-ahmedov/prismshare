from __future__ import annotations

import numpy as np
import pytest

from prism_share.analysis import ecc_sim
from prism_share.analysis.metrics import byte_errors, goodput_bytes_per_s, goodput_mbit_per_s, symbol_errors
from prism_share.codec.encoder import encode_frames
from prism_share.codec.framing import decode_frame_symbols, frame_capacity
from prism_share.codec.params import ASSUMED_FPS, CodecParams

P = CodecParams(colour_depth=4, cell_px=8)
FR = encode_frames(b"metrics", P, n_frames=1)[0]
CAP = frame_capacity(P)


def test_symbol_errors_are_reported_separately() -> None:
    g, c = FR.glyphs.copy(), FR.colours.copy()
    g[:10] = (g[:10] + 1) % P.glyph_count
    c[5:25] = (c[5:25] + 1) % P.colour_depth
    e = symbol_errors(FR.glyphs, FR.colours, g, c)
    assert e.shape.sum() == 10 and e.colour.sum() == 20 and (e.shape | e.colour).sum() == 25
    assert e.shape_ser == 10 / e.n_cells and e.colour_ser == 20 / e.n_cells and e.symbol_ser == 25 / e.n_cells


def test_byte_errors_locate_a_single_cell_error() -> None:
    g = FR.glyphs.copy()
    g[100] ^= 1  # flips the last glyph bit of cell 100: bit 100*6 + 3
    errs = byte_errors(FR.glyphs, FR.colours, g, FR.colours, P)
    assert np.flatnonzero(errs).tolist() == [(100 * P.bits_per_cell + 3) // 8]


def test_no_errors_no_byte_errors() -> None:
    assert not byte_errors(FR.glyphs, FR.colours, FR.glyphs, FR.colours, P).any()


def test_codeword_counts_follow_interleaving() -> None:
    errs = np.zeros(CAP.capacity_bytes, dtype=bool)
    errs[[0, CAP.n_codewords, 2 * CAP.n_codewords, 1]] = True  # three hits on codeword 0, one on codeword 1
    counts = ecc_sim.codeword_error_counts(errs, P.ecc_total)
    assert counts[0] == 3 and counts[1] == 1 and counts.sum() == 4
    assert ecc_sim.max_codeword_errors(errs, P.ecc_total) == 3


def test_recoverability_threshold() -> None:
    errs = np.zeros(CAP.capacity_bytes, dtype=bool)
    errs[np.arange(P.ecc_correctable) * CAP.n_codewords] = True  # exactly t errors in codeword 0
    assert ecc_sim.frame_recoverable(errs, P.ecc_total, P.ecc_data)
    errs[P.ecc_correctable * CAP.n_codewords] = True
    assert not ecc_sim.frame_recoverable(errs, P.ecc_total, P.ecc_data)


def test_simulated_recoverable_implies_real_decode_succeeds() -> None:
    """ecc_sim's prediction agrees with the real RS decoder on a corrupted frame."""
    rng = np.random.default_rng(0)
    g = FR.glyphs.copy()
    cells = rng.choice(len(g), size=120, replace=False)
    g[cells] = (g[cells] + 1) % P.glyph_count
    errs = byte_errors(FR.glyphs, FR.colours, g, FR.colours, P)
    predicted = ecc_sim.frame_recoverable(errs, P.ecc_total, P.ecc_data)
    assert predicted, "test setup should stay within the RS bound"
    assert decode_frame_symbols(g, FR.colours, P).ok


def test_payload_bytes_match_framing() -> None:
    assert ecc_sim.payload_bytes_per_frame(P, P.ecc_total, P.ecc_data) == CAP.block_bytes
    other = P.replace(ecc_total=255, ecc_data=223)
    assert ecc_sim.payload_bytes_per_frame(P, 255, 223) == frame_capacity(other).block_bytes


def test_evaluate_and_best_rate() -> None:
    clean = [0, 0, 0, 0]
    best = ecc_sim.best_rate(clean, P, 255)
    assert (best.k, best.frame_yield) == (254, 1.0)
    noisy = [5, 7, 6, 3]  # worst frame needs n - k >= 14
    best = ecc_sim.best_rate(noisy, P, 255)
    assert best.k == 255 - 14 and best.frame_yield == 1.0
    half = ecc_sim.evaluate(noisy, P, 255, 255 - 12)  # t = 6: frames with 5, 6, 3 pass
    assert half.frame_yield == 0.75
    assert half.goodput_bytes_per_s == pytest.approx(half.payload_bytes * 0.75 * ASSUMED_FPS)


def test_goodput_formula() -> None:
    assert goodput_bytes_per_s(1000, 0.5) == 1000 * 0.5 * ASSUMED_FPS
    assert goodput_mbit_per_s(1000, 1.0, fps=30) == pytest.approx(0.24)
    with pytest.raises(ValueError):
        goodput_bytes_per_s(1000, 1.5)


# --------------------------------------------------------------------------- out-of-sample RS selection


def test_selection_split_is_by_position_only() -> None:
    from prism_share.codec.params import RS_SELECTION_PERIOD

    assert RS_SELECTION_PERIOD == 2
    assert ecc_sim.selection_mask([0, 1, 2, 3, 10, 11]).tolist() == [True, False, True, False, True, False]


def test_out_of_sample_chooses_on_selection_and_scores_on_evaluation() -> None:
    """Selection frames are clean, evaluation frames are not: the choice cannot see the errors it is scored on."""
    p = CodecParams(colour_depth=4, cell_px=8)
    n = 255
    positions = np.arange(8)
    worst = np.where(positions % 2 == 0, 0, 3)  # evaluation frames have 3 errors in their worst codeword
    oos = ecc_sim.out_of_sample({n: worst}, positions, p)
    assert (oos.chosen.n, oos.chosen.k) == (n, n - 1)  # nothing to correct in the selection half
    assert oos.evaluated.k == n - 1 and oos.evaluated.frame_yield == 0.0 and oos.evaluated.goodput_bytes_per_s == 0.0
    assert (oos.n_selection, oos.n_evaluation) == (4, 4)
    in_sample = ecc_sim.best_code({n: worst}, p)
    assert in_sample.goodput_bytes_per_s > oos.evaluated.goodput_bytes_per_s  # the bias being removed
    assert in_sample.k == n - 2 * 3  # just enough parity for the errors it saw


def test_out_of_sample_needs_both_halves() -> None:
    p = CodecParams(colour_depth=4, cell_px=8)
    with pytest.raises(ValueError, match="both halves"):
        ecc_sim.out_of_sample({155: [0, 0]}, [0, 2], p)
    with pytest.raises(ValueError, match="both halves"):
        ecc_sim.out_of_sample({155: [0]}, [1], p)
    with pytest.raises(ValueError, match="length"):
        ecc_sim.out_of_sample({155: [0, 0, 0]}, [0, 1], p)
