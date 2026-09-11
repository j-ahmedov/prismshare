"""Framing: one frame's bytes <-> one frame's cell symbols.

Encoding a frame, for CodecParams p with RS(n, k) = (p.ecc_total, p.ecc_data):

1. **Frame data** (``data_bytes = n_codewords * k`` bytes)::

       header (16 B) | block (block_bytes) | CRC-32 of header+block (4 B)

   Header = block_id, n_source_blocks, payload_len, params fingerprint.
2. **RS encoding.** Frame data is cut into n_codewords consecutive k-byte
   messages; message j becomes codeword j (n bytes, systematic).
3. **Interleaving.** Codeword j symbol s goes to stream byte ``s * n_codewords
   + j``. Consecutive stream bytes - which are spatially adjacent cells - hence
   belong to different codewords, and every codeword samples the whole frame.
4. **Padding.** The stream is zero-padded to ``capacity_bytes``, the whole
   bytes the cells can carry.
5. **Whitening.** The stream is XORed with the keystream ("whitening", seed
   p.seed). This makes the glyph and colour distribution uniform whatever the
   payload, so frame statistics do not depend on content, and it is what lets
   ecc_sim treat parity and data symbols alike.
6. **Bits to cells.** The stream is unpacked MSB-first, followed by
   ``pad_bits`` (< 8) keystream bits so every cell is filled. Cell i takes bits
   ``[i*b, (i+1)*b)`` with b = bits_per_cell: the first glyph_bits are the glyph
   index, the remaining colour_bits the colour index (both MSB-first).

The byte <-> cell mapping (steps 3-6) does not depend on k, and on n only
through the interleaving depth, so ecc_sim can re-group a recorded byte error
pattern into codewords of any RS(n', k').
"""

from __future__ import annotations

import functools
import struct
import zlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from prism_share.codec.ecc import rs_decode, rs_encode
from prism_share.codec.layout import grid_layout
from prism_share.codec.params import BITS_PER_BYTE, CRC_FORMAT, HEADER_FORMAT, CodecParams
from prism_share.codec.prng import keystream_array

UInt8Array = npt.NDArray[np.uint8]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

HEADER_BYTES = struct.calcsize(HEADER_FORMAT)
CRC_BYTES = struct.calcsize(CRC_FORMAT)
_CRC_MASK = 0xFFFFFFFF


@dataclass(frozen=True)
class FrameCapacity:
    """How the cells of one frame are spent. All counts are per frame."""

    n_cells: int
    capacity_bits: int
    """n_cells * bits_per_cell."""
    capacity_bytes: int
    """Whole bytes the cells carry."""
    n_codewords: int
    stream_bytes: int
    """Bytes occupied by RS codewords: n_codewords * n."""
    data_bytes: int
    """RS message bytes: n_codewords * k."""
    block_bytes: int
    """Payload bytes per frame: data_bytes - header - CRC."""
    pad_bytes: int
    """capacity_bytes - stream_bytes (whitened zeros, carry nothing)."""
    pad_bits: int
    """capacity_bits - 8 * capacity_bytes (keystream bits, carry nothing)."""


@dataclass(frozen=True)
class FrameHeader:
    block_id: int
    n_source_blocks: int
    payload_len: int
    params_fingerprint: int

    def pack(self) -> bytes:
        return struct.pack(
            HEADER_FORMAT, self.block_id, self.n_source_blocks, self.payload_len, self.params_fingerprint
        )

    @classmethod
    def unpack(cls, data: bytes) -> FrameHeader:
        return cls(*struct.unpack(HEADER_FORMAT, data[:HEADER_BYTES]))


@dataclass(frozen=True)
class FrameDecodeResult:
    ok: bool
    """All codewords decoded, CRC matches and the fingerprint is ours."""
    header: FrameHeader | None
    block: bytes | None
    n_codewords: int
    codewords_failed: int
    crc_ok: bool
    fingerprint_ok: bool


# --------------------------------------------------------------------------- #
# Capacity
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=None)
def frame_capacity(params: CodecParams) -> FrameCapacity:
    n_cells = grid_layout(params).n_cells
    capacity_bits = n_cells * params.bits_per_cell
    capacity_bytes = capacity_bits // BITS_PER_BYTE
    n_codewords = capacity_bytes // params.ecc_total
    data_bytes = n_codewords * params.ecc_data
    block_bytes = data_bytes - HEADER_BYTES - CRC_BYTES
    if block_bytes <= 0:
        raise ValueError(f"{params.label}: frame too small for header + CRC ({n_codewords} codewords)")
    return FrameCapacity(
        n_cells=n_cells,
        capacity_bits=capacity_bits,
        capacity_bytes=capacity_bytes,
        n_codewords=n_codewords,
        stream_bytes=n_codewords * params.ecc_total,
        data_bytes=data_bytes,
        block_bytes=block_bytes,
        pad_bytes=capacity_bytes - n_codewords * params.ecc_total,
        pad_bits=capacity_bits - capacity_bytes * BITS_PER_BYTE,
    )


# --------------------------------------------------------------------------- #
# Interleaving and whitening
# --------------------------------------------------------------------------- #


def codeword_stream_positions(n_codewords: int, n: int) -> IntArray:
    """(n_codewords, n): stream byte index of every codeword symbol."""
    s = np.arange(n, dtype=np.int64)[None, :]
    j = np.arange(n_codewords, dtype=np.int64)[:, None]
    return s * n_codewords + j


def interleave(codewords: list[bytes]) -> bytes:
    arr = np.array([np.frombuffer(c, dtype=np.uint8) for c in codewords], dtype=np.uint8)
    return arr.T.tobytes()  # row-major over (symbol, codeword)


def deinterleave(stream: bytes, n_codewords: int, n: int) -> list[bytes]:
    arr = np.frombuffer(stream[: n_codewords * n], dtype=np.uint8).reshape(n, n_codewords)
    return [arr[:, j].tobytes() for j in range(n_codewords)]


def _whitening(params: CodecParams) -> UInt8Array:
    """capacity_bytes of stream whitening followed by one byte for the pad bits."""
    cap = frame_capacity(params)
    return keystream_array(params.seed, "whitening", cap.capacity_bytes + 1)


# --------------------------------------------------------------------------- #
# Stream <-> symbols
# --------------------------------------------------------------------------- #


def stream_to_symbols(stream: bytes, params: CodecParams) -> tuple[IntArray, IntArray]:
    """Whitened on-screen stream (capacity_bytes) -> (glyph, colour) index per cell."""
    cap = frame_capacity(params)
    if len(stream) != cap.capacity_bytes:
        raise ValueError(f"stream must be {cap.capacity_bytes} bytes, got {len(stream)}")
    pad = np.unpackbits(_whitening(params)[cap.capacity_bytes :])[: cap.pad_bits]
    bits = np.concatenate([np.unpackbits(np.frombuffer(stream, dtype=np.uint8)), pad])
    bits = bits.reshape(cap.n_cells, params.bits_per_cell).astype(np.int64)
    return _bits_to_int(bits[:, : params.glyph_bits]), _bits_to_int(bits[:, params.glyph_bits :])


def symbols_to_stream(glyphs: IntArray, colours: IntArray, params: CodecParams) -> bytes:
    """(glyph, colour) per cell -> whitened on-screen stream (capacity_bytes); pad bits dropped."""
    cap = frame_capacity(params)
    glyphs, colours = np.asarray(glyphs, dtype=np.int64), np.asarray(colours, dtype=np.int64)
    if glyphs.shape != (cap.n_cells,) or colours.shape != (cap.n_cells,):
        raise ValueError(f"expected {cap.n_cells} symbols")
    bits = np.concatenate(
        [_int_to_bits(glyphs, params.glyph_bits), _int_to_bits(colours, params.colour_bits)], axis=1
    ).ravel()
    return np.packbits(bits[: cap.capacity_bytes * BITS_PER_BYTE].astype(np.uint8)).tobytes()


def _bits_to_int(bits: IntArray) -> IntArray:
    weights = 1 << np.arange(bits.shape[1] - 1, -1, -1, dtype=np.int64)
    return bits @ weights


def _int_to_bits(values: IntArray, width: int) -> IntArray:
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return (values[:, None] >> shifts[None, :]) & 1


def cell_stream_bytes(params: CodecParams) -> list[range]:
    """For each cell, the range of stream byte indices its bits fall in."""
    b = params.bits_per_cell
    return [range(i * b // BITS_PER_BYTE, ((i + 1) * b - 1) // BITS_PER_BYTE + 1) for i in range(frame_capacity(params).n_cells)]


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #


def encode_frame_symbols(header: FrameHeader, block: bytes, params: CodecParams) -> tuple[IntArray, IntArray]:
    """Header + block -> (glyph, colour) index per cell."""
    cap = frame_capacity(params)
    if len(block) != cap.block_bytes:
        raise ValueError(f"block must be {cap.block_bytes} bytes, got {len(block)}")
    body = header.pack() + block
    data = body + struct.pack(CRC_FORMAT, zlib.crc32(body) & _CRC_MASK)
    n, k = params.ecc_total, params.ecc_data
    codewords = [rs_encode(data[j * k : (j + 1) * k], n, k) for j in range(cap.n_codewords)]
    coded = np.zeros(cap.capacity_bytes, dtype=np.uint8)
    coded[: cap.stream_bytes] = np.frombuffer(interleave(codewords), dtype=np.uint8)
    stream = coded ^ _whitening(params)[: cap.capacity_bytes]
    return stream_to_symbols(stream.tobytes(), params)


def decode_frame_symbols(
    glyphs: IntArray,
    colours: IntArray,
    params: CodecParams,
    erased_cells: BoolArray | None = None,
) -> FrameDecodeResult:
    """(glyph, colour) per cell -> frame contents, using RS error+erasure decoding.

    ``erased_cells`` marks cells whose symbols are unreliable; every stream byte
    they touch is passed to RS as an erasure.
    """
    cap = frame_capacity(params)
    n, k = params.ecc_total, params.ecc_data
    stream = np.frombuffer(symbols_to_stream(glyphs, colours, params), dtype=np.uint8)
    coded = (stream ^ _whitening(params)[: cap.capacity_bytes]).tobytes()

    erased_bytes = np.zeros(cap.capacity_bytes, dtype=bool)
    if erased_cells is not None:
        spans = cell_stream_bytes(params)
        for cell in np.flatnonzero(erased_cells):
            span = spans[cell]
            erased_bytes[span.start : min(span.stop, cap.capacity_bytes)] = True
    positions = codeword_stream_positions(cap.n_codewords, n)

    messages: list[bytes] = []
    failed = 0
    for j, codeword in enumerate(deinterleave(coded, cap.n_codewords, n)):
        erasures = np.flatnonzero(erased_bytes[positions[j]]).tolist()
        message = rs_decode(codeword, n, k, erasures)
        if message is None:
            failed += 1
            message = codeword[:k]  # systematic: best guess, kept only for diagnostics
        messages.append(message)

    data = b"".join(messages)
    body, crc = data[:-CRC_BYTES], data[-CRC_BYTES:]
    crc_ok = struct.unpack(CRC_FORMAT, crc)[0] == (zlib.crc32(body) & _CRC_MASK)
    header = FrameHeader.unpack(body)
    fingerprint_ok = header.params_fingerprint == params.fingerprint()
    ok = failed == 0 and crc_ok and fingerprint_ok
    return FrameDecodeResult(
        ok=ok,
        header=header if ok else None,
        block=body[HEADER_BYTES:] if ok else None,
        n_codewords=cap.n_codewords,
        codewords_failed=failed,
        crc_ok=crc_ok,
        fingerprint_ok=fingerprint_ok,
    )
