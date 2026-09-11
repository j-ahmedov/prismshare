"""Platform-independent pseudo-random streams.

Every random-looking byte in the codec comes from here. The generator is
SHA-256 in counter mode, so its output is fixed by the SHA-256 specification
alone: it cannot drift with the numpy version, the CPU or the OS, which is what
"byte-identical PNGs on any machine" requires. numpy's Generator is avoided on
purpose because numpy only guarantees stream stability for its bit generators,
not for the methods built on them.

Streams are separated by a ``domain`` string (what the bytes are for) and an
``index`` (e.g. a frame or candidate number), so no two uses ever share bytes.
"""

from __future__ import annotations

import hashlib

import numpy as np
import numpy.typing as npt

_DIGEST_BYTES = hashlib.sha256().digest_size


def keystream(seed: int, domain: str, n_bytes: int, *, index: int = 0) -> bytes:
    """Return ``n_bytes`` pseudo-random bytes determined by (seed, domain, index)."""
    if n_bytes < 0:
        raise ValueError("n_bytes must be >= 0")
    blocks = -(-n_bytes // _DIGEST_BYTES)
    out = bytearray()
    for counter in range(blocks):
        message = f"prism-share|{domain}|seed={seed}|index={index}|block={counter}"
        out += hashlib.sha256(message.encode("ascii")).digest()
    return bytes(out[:n_bytes])


def keystream_array(seed: int, domain: str, n_bytes: int, *, index: int = 0) -> npt.NDArray[np.uint8]:
    """``keystream`` as a read-only uint8 array."""
    arr = np.frombuffer(keystream(seed, domain, n_bytes, index=index), dtype=np.uint8)
    return arr


def keystream_bits(seed: int, domain: str, n_bits: int, *, index: int = 0) -> npt.NDArray[np.bool_]:
    """``n_bits`` pseudo-random bits (MSB-first unpacking of ``keystream``)."""
    n_bytes = -(-n_bits // np.iinfo(np.uint8).bits)
    bits = np.unpackbits(keystream_array(seed, domain, n_bytes, index=index))
    return bits[:n_bits].astype(bool)
