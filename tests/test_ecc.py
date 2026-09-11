from __future__ import annotations

import random

import pytest

from prism_share.codec.ecc import is_recoverable, rs_decode, rs_encode

N, K = 155, 125
T = (N - K) // 2


def _message(seed: int) -> bytes:
    return random.Random(seed).randbytes(K)


def _corrupt(codeword: bytes, positions: list[int]) -> bytes:
    out = bytearray(codeword)
    for p in positions:
        out[p] ^= 0xA5
    return bytes(out)


def test_systematic() -> None:
    msg = _message(0)
    cw = rs_encode(msg, N, K)
    assert len(cw) == N and cw[:K] == msg


@pytest.mark.parametrize("n_errors", [0, 1, T])
def test_corrects_up_to_t_errors(n_errors: int) -> None:
    msg = _message(n_errors)
    positions = random.Random(1).sample(range(N), n_errors)
    assert rs_decode(_corrupt(rs_encode(msg, N, K), positions), N, K) == msg


def test_erasures_double_capacity() -> None:
    msg = _message(2)
    positions = random.Random(2).sample(range(N), N - K)
    assert rs_decode(_corrupt(rs_encode(msg, N, K), positions), N, K, erasures=positions) == msg


def test_mixed_errors_and_erasures_at_bound() -> None:
    msg = _message(3)
    rng = random.Random(3)
    positions = rng.sample(range(N), 10 + 10)
    erased, errors = positions[:10], positions[10:]  # 2*10 + 10 = 30 = n - k
    assert is_recoverable(len(errors), len(erased), N, K)
    assert rs_decode(_corrupt(rs_encode(msg, N, K), positions), N, K, erasures=erased) == msg


def test_far_beyond_bound_does_not_return_the_message() -> None:
    msg = _message(4)
    positions = random.Random(4).sample(range(N), 3 * T)
    assert rs_decode(_corrupt(rs_encode(msg, N, K), positions), N, K) != msg


def test_is_recoverable_bound() -> None:
    assert is_recoverable(T, 0, N, K)
    assert not is_recoverable(T + 1, 0, N, K)
    assert is_recoverable(0, N - K, N, K)
    assert not is_recoverable(0, N - K + 1, N, K)


def test_validation() -> None:
    with pytest.raises(ValueError):
        rs_encode(b"short", N, K)
    with pytest.raises(ValueError):
        rs_encode(b"x" * 10, 256, 10)
