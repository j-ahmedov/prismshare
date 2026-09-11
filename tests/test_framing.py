from __future__ import annotations

import random

import numpy as np
import pytest

from prism_share.codec import framing
from prism_share.codec.params import BITS_PER_BYTE, CodecParams


@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
def test_stream_symbol_round_trip(depth: int) -> None:
    p = CodecParams(colour_depth=depth, cell_px=10)
    cap = framing.frame_capacity(p)
    stream = random.Random(depth).randbytes(cap.capacity_bytes)
    glyphs, colours = framing.stream_to_symbols(stream, p)
    assert glyphs.min() >= 0 and glyphs.max() < p.glyph_count
    assert colours.min() >= 0 and colours.max() < p.colour_depth
    assert framing.symbols_to_stream(glyphs, colours, p) == stream


def test_capacity_accounting() -> None:
    p = CodecParams()
    cap = framing.frame_capacity(p)
    assert cap.capacity_bits == cap.n_cells * p.bits_per_cell
    assert cap.capacity_bytes == cap.capacity_bits // BITS_PER_BYTE
    assert cap.stream_bytes == cap.n_codewords * p.ecc_total <= cap.capacity_bytes
    assert cap.stream_bytes + p.ecc_total > cap.capacity_bytes  # no room for another codeword
    assert cap.block_bytes == cap.n_codewords * p.ecc_data - framing.HEADER_BYTES - framing.CRC_BYTES
    assert 0 <= cap.pad_bits < BITS_PER_BYTE


def test_interleave_positions_match_interleave() -> None:
    n_cw, n = 7, 11
    codewords = [bytes([j] * n) for j in range(n_cw)]
    stream = framing.interleave(codewords)
    pos = framing.codeword_stream_positions(n_cw, n)
    for j in range(n_cw):
        assert all(stream[q] == j for q in pos[j])
    assert framing.deinterleave(stream, n_cw, n) == codewords


def test_adjacent_cells_land_in_different_codewords() -> None:
    p = CodecParams()
    cap = framing.frame_capacity(p)
    owner = np.full(cap.capacity_bytes, -1)
    for j, row in enumerate(framing.codeword_stream_positions(cap.n_codewords, p.ecc_total)):
        owner[row] = j
    # A run of cells shorter than n_codewords bytes touches each codeword at most once.
    spans = framing.cell_stream_bytes(p)
    run = [b for cell in range(0, 40) for b in spans[cell]]
    owners = owner[sorted(set(run))]
    assert len(owners) == len(set(owners.tolist()))


def test_header_round_trip() -> None:
    h = framing.FrameHeader(1, 2, 3, 0xDEADBEEF)
    assert framing.FrameHeader.unpack(h.pack()) == h


def _frame(p: CodecParams) -> tuple[framing.FrameHeader, bytes]:
    cap = framing.frame_capacity(p)
    return framing.FrameHeader(5, 9, 1234, p.fingerprint()), random.Random(0).randbytes(cap.block_bytes)


def test_frame_symbols_round_trip() -> None:
    p = CodecParams()
    header, block = _frame(p)
    g, c = framing.encode_frame_symbols(header, block, p)
    result = framing.decode_frame_symbols(g, c, p)
    assert result.ok and result.header == header and result.block == block


def test_symbol_errors_within_rs_bound_are_corrected() -> None:
    p = CodecParams()
    header, block = _frame(p)
    g, c = framing.encode_frame_symbols(header, block, p)
    # Corrupt one cell in every 40: well under t = 15 byte errors per codeword.
    g = g.copy()
    g[::40] ^= 1
    result = framing.decode_frame_symbols(g, c, p)
    assert result.ok and result.block == block


def test_heavy_corruption_fails_cleanly() -> None:
    p = CodecParams()
    header, block = _frame(p)
    g, c = framing.encode_frame_symbols(header, block, p)
    g = (g + 1) % p.glyph_count
    result = framing.decode_frame_symbols(g, c, p)
    assert not result.ok and result.block is None
    assert result.codewords_failed > 0


def test_erasures_extend_correction() -> None:
    p = CodecParams()
    header, block = _frame(p)
    g, c = framing.encode_frame_symbols(header, block, p)
    cap = framing.frame_capacity(p)
    # Destroy a contiguous band of cells hitting each codeword ~20 times (> t = 15).
    bad = np.zeros(cap.n_cells, dtype=bool)
    n_bad = 20 * cap.n_codewords * framing.BITS_PER_BYTE // p.bits_per_cell
    bad[1000 : 1000 + n_bad] = True
    g2 = np.where(bad, (g + 3) % p.glyph_count, g)
    assert not framing.decode_frame_symbols(g2, c, p).ok
    assert framing.decode_frame_symbols(g2, c, p, erased_cells=bad).ok


def test_wrong_params_rejected_by_fingerprint() -> None:
    p = CodecParams()
    header, block = _frame(p)
    wrong = framing.FrameHeader(header.block_id, header.n_source_blocks, header.payload_len, 1)
    g, c = framing.encode_frame_symbols(wrong, block, p)
    result = framing.decode_frame_symbols(g, c, p)
    assert result.crc_ok and not result.fingerprint_ok and not result.ok
