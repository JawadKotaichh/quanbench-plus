from __future__ import annotations

from typing import Any

import numpy as np


def unitaries_equivalent(
    candidate: Any,
    expected: Any,
    *,
    tolerance: float,
    ignore_global_phase: bool,
) -> tuple[bool, float]:
    if candidate is None or expected is None:
        return False, float("inf")
    cand = np.asarray(candidate, dtype=complex)
    exp = np.asarray(expected, dtype=complex)
    if cand.shape != exp.shape:
        return False, float("inf")

    if ignore_global_phase:
        inner = np.vdot(exp.reshape(-1), cand.reshape(-1))
        phase = inner / abs(inner) if abs(inner) > 0 else 1.0
        diff = cand - phase * exp
    else:
        diff = cand - exp

    distance = float(np.linalg.norm(diff) / np.sqrt(diff.size))
    return distance <= tolerance, distance


def u_gate_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    return np.asarray(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ],
        dtype=complex,
    )
