from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import log2
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class GradeContext:
    probabilities: Sequence[float]
    canonical_probabilities: Sequence[float] | None = None
    candidate_unitary: Any | None = None
    canonical_unitary: Any | None = None
    target_unitary: Any | None = None
    metadata: Mapping[str, Any] | None = None
    code: str | None = None


def _as_prob_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("probabilities must be a 1-D vector")
    if len(arr) == 0 or len(arr) & (len(arr) - 1):
        raise ValueError(f"probability length must be a power of 2, got {len(arr)}")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("probabilities sum to zero")
    return arr / total


def _num_bits(values: Sequence[float]) -> int:
    return int(log2(len(values)))


def _bitstrings_for_probs(values: Sequence[float]) -> list[str]:
    n_bits = _num_bits(values)
    return [format(i, f"0{n_bits}b") for i in range(len(values))]


def _idx(bitstring: str) -> int:
    return int(bitstring, 2)


def _kl(probs: Sequence[float], expected: Sequence[float], eps: float = 1e-12) -> float:
    p = np.clip(_as_prob_array(probs), eps, 1)
    q = np.clip(_as_prob_array(expected), eps, 1)
    if len(p) != len(q):
        raise ValueError(f"shape mismatch: model len {len(p)}, expected len {len(q)}")
    return float(np.sum(p * np.log(p / q)))


def _distribution_from_support(
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
            out[_idx(bitstring)] = 1.0 / len(support_list)
    else:
        for bitstring in support_list:
            out[_idx(bitstring)] = float(weights[bitstring])
        total = float(out.sum())
        if total <= 0:
            raise ValueError("support weights sum to zero")
        out /= total
    return out


def _observed_support(probs: np.ndarray, tau: float) -> frozenset[str]:
    bitstrings = _bitstrings_for_probs(probs)
    return frozenset(bit for bit, p in zip(bitstrings, probs) if p > tau)


def _permute_bitstring(bitstring: str, perm: tuple[int, ...]) -> str:
    return "".join(bitstring[i] for i in perm)


def _support_matches(
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
        permuted = frozenset(_permute_bitstring(bit, perm) for bit in expected)
        if observed == permuted:
            return True, perm
    return False, None


def _top_k_bitstrings(probs: np.ndarray, top_k: int) -> list[str]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    bitstrings = _bitstrings_for_probs(probs)
    order = sorted(range(len(probs)), key=lambda i: (-float(probs[i]), bitstrings[i]))
    return [bitstrings[i] for i in order[:top_k]]


def _project_and_normalize(probs: np.ndarray, bitstrings: Iterable[str]) -> np.ndarray:
    bits = list(bitstrings)
    out = np.asarray([probs[_idx(bit)] for bit in bits], dtype=float)
    total = float(out.sum())
    if total <= 0:
        return out
    return out / total


def _unitaries_equivalent(
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


def _grade_exact_distribution(
    spec: Mapping[str, Any], context: GradeContext, probs: np.ndarray
) -> dict[str, Any]:
    if spec.get("comparison") == "unitary":
        expected_unitary = (
            context.target_unitary
            if context.target_unitary is not None
            else context.canonical_unitary
        )
        passed, distance = _unitaries_equivalent(
            context.candidate_unitary,
            expected_unitary,
            tolerance=float(spec.get("tolerance", 1e-8)),
            ignore_global_phase=bool(spec.get("ignore_global_phase", True)),
        )
        return {
            "passed": passed,
            "metric": distance,
            "metric_name": "unitary_distance",
            "grader_type": "exact_distribution",
            "comparison": "unitary",
            "tolerance": float(spec.get("tolerance", 1e-8)),
        }

    expected = spec.get("expected_distribution")
    if expected is None:
        expected = context.canonical_probabilities
    if expected is None:
        raise ValueError("exact_distribution requires expected_distribution or canonical_probabilities")

    threshold = float(spec.get("threshold", spec.get("epsilon", 1e-3)))
    kl_value = _kl(probs, expected)
    return {
        "passed": kl_value < threshold,
        "kl_value": kl_value,
        "grader_type": "exact_distribution",
        "threshold": threshold,
    }


def _grade_deterministic_dominant(
    spec: Mapping[str, Any], context: GradeContext, probs: np.ndarray
) -> dict[str, Any]:
    expected = set(spec["expected_dominants"])
    min_p = float(spec.get("min_dominant_probability", 0.95))
    bitstrings = _bitstrings_for_probs(probs)
    max_idx = max(range(len(probs)), key=lambda i: float(probs[i]))
    dominant = bitstrings[max_idx]
    dominant_p = float(probs[max_idx])
    passed = dominant in expected and dominant_p >= min_p
    min_non_measure_ops = spec.get("min_non_measure_ops")
    non_measure_ops = None
    if min_non_measure_ops is not None:
        op_counts = dict((context.metadata or {}).get("operation_counts") or {})
        non_measure_ops = sum(count for name, count in op_counts.items() if name != "measure")
        passed = passed and non_measure_ops >= int(min_non_measure_ops)

    out = {
        "passed": passed,
        "grader_type": "deterministic_dominant",
        "dominant_bitstring": dominant,
        "dominant_probability": dominant_p,
        "accepted": sorted(expected),
        "min_required": min_p,
    }
    if min_non_measure_ops is not None:
        out["min_non_measure_ops"] = int(min_non_measure_ops)
        out["non_measure_ops"] = int(non_measure_ops or 0)
    return out


def _grade_support_match(spec: Mapping[str, Any], probs: np.ndarray) -> dict[str, Any]:
    expected = frozenset(spec.get("canonical_support") or spec.get("expected_support") or [])
    if not expected:
        raise ValueError("support_match requires canonical_support or expected_support")
    tau = float(spec.get("tau", 0.01))
    observed = _observed_support(probs, tau)
    matched, perm = _support_matches(
        observed,
        expected,
        permutation_invariant=bool(spec.get("permutation_invariant", False)),
    )
    out: dict[str, Any] = {
        "passed": matched,
        "grader_type": "support_match",
        "observed_support": sorted(observed),
        "expected_support": sorted(expected),
        "tau": tau,
    }
    if perm is not None:
        out["matched_permutation"] = list(perm)
    return out


def _grade_peak_match(
    spec: Mapping[str, Any], context: GradeContext, probs: np.ndarray
) -> dict[str, Any]:
    top_k = int(spec.get("top_k", 1))
    observed_top = _top_k_bitstrings(probs, top_k)

    accepted_peak_sets = spec.get("accepted_peak_sets")
    if accepted_peak_sets is not None:
        accepted = [set(bits) for bits in accepted_peak_sets]
        observed_set = set(observed_top)
        match_mode = str(spec.get("match_mode", "exact"))
        if match_mode == "subset":
            passed_set = any(observed_set.issubset(bits) for bits in accepted)
        else:
            passed_set = any(observed_set == bits for bits in accepted)
        return {
            "passed": passed_set,
            "grader_type": "peak_match",
            "observed_top": observed_top,
            "accepted_peak_sets": [sorted(bits) for bits in accepted],
            "top_k": top_k,
            "match_mode": match_mode,
        }

    expected_peaks = spec.get("expected_peaks")
    if expected_peaks is None:
        if context.canonical_probabilities is None:
            raise ValueError("peak_match requires expected_peaks or canonical_probabilities")
        expected_probs = _as_prob_array(context.canonical_probabilities)
        expected_peaks = _top_k_bitstrings(expected_probs, top_k)
    else:
        expected_probs = None

    observed_set = set(observed_top)
    expected_set = set(expected_peaks)
    passed_set = observed_set == expected_set
    if not passed_set and spec.get("permutation_invariant"):
        matched, _ = _support_matches(
            frozenset(observed_set),
            frozenset(expected_set),
            permutation_invariant=True,
        )
        passed_set = matched

    kl_value = None
    threshold = spec.get("threshold")
    if expected_probs is not None and threshold is not None:
        ordered = list(expected_peaks)
        kl_value = _kl(
            _project_and_normalize(probs, ordered),
            _project_and_normalize(expected_probs, ordered),
        )
        passed = passed_set and kl_value < float(threshold)
    else:
        min_peak_probability = float(spec.get("min_peak_probability", 0.0))
        passed = passed_set and all(float(probs[_idx(bit)]) >= min_peak_probability for bit in expected_peaks)

    out = {
        "passed": passed,
        "grader_type": "peak_match",
        "observed_top": observed_top,
        "expected_peaks": sorted(expected_set),
        "top_k": top_k,
    }
    if threshold is not None:
        out["threshold"] = float(threshold)
    if kl_value is not None:
        out["kl_value"] = kl_value
    return out


def _grade_support_uniformity(
    spec: Mapping[str, Any], probs: np.ndarray
) -> dict[str, Any]:
    n_bits = _num_bits(probs)
    support = spec.get("support") or spec.get("canonical_support") or spec.get("expected_support")
    if support in (None, "all"):
        support_bits = [format(i, f"0{n_bits}b") for i in range(2**n_bits)]
    else:
        support_bits = list(support)

    threshold = float(spec.get("threshold", spec.get("epsilon", 0.02)))
    outside_tolerance = float(spec.get("outside_support_tolerance", 0.01))
    expected = _distribution_from_support(support_bits, n_bits)
    kl_value = _kl(probs, expected)
    support_indices = {_idx(bit) for bit in support_bits}
    outside_mass = float(sum(p for i, p in enumerate(probs) if i not in support_indices))
    return {
        "passed": kl_value < threshold and outside_mass <= outside_tolerance,
        "grader_type": "support_uniformity",
        "kl_value": kl_value,
        "threshold": threshold,
        "support": sorted(support_bits),
        "outside_support_mass": outside_mass,
        "outside_support_tolerance": outside_tolerance,
    }


def _grade_structural(
    spec: Mapping[str, Any], context: GradeContext, probs: np.ndarray
) -> dict[str, Any]:
    name = str(spec.get("structural_name", ""))
    metadata = dict(context.metadata or {})
    code = context.code or ""

    if name != "vqe_z2_ansatz":
        raise ValueError(f"unknown structural grader: {name}")

    forbidden_terms = tuple(spec.get("forbidden_terms", ["minimize(", "Estimator(", "Sampler("]))
    no_forbidden_terms = not any(term in code for term in forbidden_terms)
    min_entangling = int(spec.get("min_entangling_gates", 0))
    no_measurements = int(metadata.get("measurement_count", 0)) == 0
    checks = {
        "num_qubits_is_2": int(metadata.get("num_qubits", -1)) == 2,
        "no_measurements": no_measurements,
        "min_entangling_gates": int(metadata.get("entangling_gate_count", 0)) >= min_entangling,
        "no_classical_optimizer_call": no_forbidden_terms,
        "probability_shape_is_2q": len(probs) == 4,
    }
    return {
        "passed": all(checks.values()),
        "grader_type": "structural",
        "structural_name": name,
        "checks": checks,
        "metadata": metadata,
    }


def grade(spec: Mapping[str, Any], context: GradeContext) -> dict[str, Any]:
    probs = _as_prob_array(context.probabilities)
    grader_type = str(spec["type"])
    if grader_type == "exact_distribution":
        return _grade_exact_distribution(spec, context, probs)
    if grader_type == "deterministic_dominant":
        return _grade_deterministic_dominant(spec, context, probs)
    if grader_type == "support_match":
        return _grade_support_match(spec, probs)
    if grader_type == "peak_match":
        return _grade_peak_match(spec, context, probs)
    if grader_type == "support_uniformity":
        return _grade_support_uniformity(spec, probs)
    if grader_type == "structural":
        return _grade_structural(spec, context, probs)
    raise ValueError(f"unknown grader type: {grader_type}")
