from __future__ import annotations

import random

import numpy as np
import pytest

from prism_share.codec import fountain

B = 64


def _source(k: int, seed: int = 0) -> np.ndarray:
    return fountain.split_payload(random.Random(seed).randbytes(k * B - 7), B)


def test_split_pads_last_block() -> None:
    blocks = fountain.split_payload(b"\x01" * (B + 1), B)
    assert blocks.shape == (2, B)
    assert blocks[1, 0] == 1 and not blocks[1, 1:].any()


def test_empty_payload_is_one_block() -> None:
    assert fountain.n_source_blocks(0, B) == 1


def test_systematic_blocks_are_the_source() -> None:
    src = _source(5)
    for i in range(5):
        assert fountain.encode_block(src, i, seed=0) == src[i].tobytes()


def test_coefficients_deterministic_and_seeded() -> None:
    a = fountain.coefficients(40, 30, seed=1)
    assert np.array_equal(a, fountain.coefficients(40, 30, seed=1))
    assert not np.array_equal(a, fountain.coefficients(40, 30, seed=2))
    assert a.any()


def test_decode_from_source_only() -> None:
    src = _source(10)
    received = {i: fountain.encode_block(src, i, 0) for i in range(10)}
    out = fountain.decode_blocks(received, 10, B, 0)
    assert out is not None and np.array_equal(out, src)


@pytest.mark.parametrize("lost", [1, 3, 10])
def test_repair_blocks_replace_lost_source_blocks(lost: int) -> None:
    k = 20
    src = _source(k, seed=lost)
    rng = random.Random(lost)
    kept_source = rng.sample(range(k), k - lost)
    repair = range(k, k + lost + 6)  # a few extra for rank
    received = {i: fountain.encode_block(src, i, 3) for i in [*kept_source, *repair]}
    out = fountain.decode_blocks(received, k, B, 3)
    assert out is not None and np.array_equal(out, src)


def test_insufficient_blocks_returns_none() -> None:
    src = _source(10)
    received = {i: fountain.encode_block(src, i, 0) for i in range(9)}
    assert fountain.decode_blocks(received, 10, B, 0) is None


def test_default_frame_count_has_repair() -> None:
    assert fountain.default_frame_count(1) > 1
    assert fountain.default_frame_count(100) >= 125
