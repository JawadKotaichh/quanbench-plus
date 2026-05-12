from __future__ import annotations

from itertools import permutations
from math import log2
from typing import Iterable, Mapping, Sequence

import numpy as np


def as_prob_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("probabilities must be a 1-D vector")
    if len(arr) == 0 or len(arr) & (len(arr) - 1):
        raise ValueError(f"probability length must be a power of 2, got {len(arr)}")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("probabilities sum to zero")
    return arr / total


def num_bits(values: Sequence[float]) -> int:
    return int(log2(len(values)))


def bitstrings_for_probs(values: Sequence[float]) -> list[str]:
    n_bits = num_bits(values)
    return [format(i, f"0{n_bits}b") for i in range(len(values))]


def bitstring_index(bitstring: str) -> int:
    return int(bitstring, 2)


def kl_divergence(probs: Sequence[float], expected: Sequence[float], eps: float = 1e-12) -> float:
    p = np.clip(as_prob_array(probs), eps, 1)
    q = np.clip(as_prob_array(expected), eps, 1)
    if len(p) != len(q):
        raise ValueError(f"shape mismatch: model len {len(p)}, expected len {len(q)}")
    return float(np.sum(p * np.log(p / q)))


def distribution_from_support(
    support: Iterable[str],
    n_bits: int,
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    out = np.zeros(2**n_bits, dtype=float)
    support_list = list(support)
    if not support_list:
        raise ValueError("support cannot be empty")

    if weights is None:
        for bitstring in support_list:
            out[bitstring_index(bitstring)] = 1.0 / len(support_list)
        return out

    for bitstring in support_list:
        out[bitstring_index(bitstring)] = float(weights[bitstring])
    total = float(out.sum())
    if total <= 0:
        raise ValueError("support weights sum to zero")
    return out / total


def observed_support(probs: np.ndarray, tau: float) -> frozenset[str]:
    bitstrings = bitstrings_for_probs(probs)
    return frozenset(bit for bit, probability in zip(bitstrings, probs) if probability > tau)


def support_matches(
    observed: frozenset[str],
    expected: frozenset[str],
    *,
    permutation_invariant: bool,
) -> tuple[bool, tuple[int, ...] | None]:
    if observed == expected:
        return True, None
    if not permutation_invariant or not expected:
        return False, None

    n_bits = len(next(iter(expected)))
    for perm in permutations(range(n_bits)):
        permuted = frozenset("".join(bit[i] for i in perm) for bit in expected)
        if observed == permuted:
            return True, perm
    return False, None


def top_k_bitstrings(probs: np.ndarray, top_k: int) -> list[str]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    bitstrings = bitstrings_for_probs(probs)
    order = sorted(range(len(probs)), key=lambda i: (-float(probs[i]), bitstrings[i]))
    return [bitstrings[i] for i in order[:top_k]]


def project_and_normalize(probs: np.ndarray, bitstrings: Iterable[str]) -> np.ndarray:
    out = np.asarray([probs[bitstring_index(bit)] for bit in bitstrings], dtype=float)
    total = float(out.sum())
    return out / total if total > 0 else out
