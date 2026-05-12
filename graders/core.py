from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from graders.contracts import GradeContext
from graders.probabilities import (
    as_prob_array,
    bitstring_index,
    bitstrings_for_probs,
    distribution_from_support,
    kl_divergence,
    num_bits,
    observed_support,
    project_and_normalize,
    support_matches,
    top_k_bitstrings,
)
from graders.structural import grade_structural
from graders.unitaries import unitaries_equivalent


def _non_measurement_operation_count(metadata: Mapping[str, Any]) -> int:
    explicit = metadata.get("non_measurement_operation_count")
    if explicit is not None:
        return int(explicit)

    measurement_names = {"m", "measure", "measurement", "measurementgate"}
    op_counts = dict(metadata.get("operation_counts") or {})
    return sum(
        int(count)
        for name, count in op_counts.items()
        if str(name).lower() not in measurement_names
    )


def _grade_exact_distribution(
    spec: Mapping[str, Any], context: GradeContext, probs: np.ndarray
) -> dict[str, Any]:
    if spec.get("comparison") == "unitary":
        expected_unitary = (
            context.target_unitary
            if context.target_unitary is not None
            else context.canonical_unitary
        )
        passed, distance = unitaries_equivalent(
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
    kl_value = kl_divergence(probs, expected)
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
    bitstrings = bitstrings_for_probs(probs)
    max_idx = max(range(len(probs)), key=lambda i: float(probs[i]))
    dominant = bitstrings[max_idx]
    dominant_p = float(probs[max_idx])
    passed = dominant in expected and dominant_p >= min_p
    min_non_measure_ops = spec.get("min_non_measure_ops")
    non_measure_ops = None
    if min_non_measure_ops is not None:
        non_measure_ops = _non_measurement_operation_count(context.metadata or {})
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
    observed = observed_support(probs, tau)
    matched, perm = support_matches(
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
    observed_top = top_k_bitstrings(probs, top_k)

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
        expected_probs = as_prob_array(context.canonical_probabilities)
        expected_peaks = top_k_bitstrings(expected_probs, top_k)
    else:
        expected_probs = None

    observed_set = set(observed_top)
    expected_set = set(expected_peaks)
    passed_set = observed_set == expected_set
    if not passed_set and spec.get("permutation_invariant"):
        matched, _ = support_matches(
            frozenset(observed_set),
            frozenset(expected_set),
            permutation_invariant=True,
        )
        passed_set = matched

    kl_value = None
    threshold = spec.get("threshold")
    if expected_probs is not None and threshold is not None:
        ordered = list(expected_peaks)
        kl_value = kl_divergence(
            project_and_normalize(probs, ordered),
            project_and_normalize(expected_probs, ordered),
        )
        passed = passed_set and kl_value < float(threshold)
    else:
        min_peak_probability = float(spec.get("min_peak_probability", 0.0))
        passed = passed_set and all(
            float(probs[bitstring_index(bit)]) >= min_peak_probability for bit in expected_peaks
        )

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
    n_bits = num_bits(probs)
    support = spec.get("support") or spec.get("canonical_support") or spec.get("expected_support")
    if support in (None, "all"):
        support_bits = [format(i, f"0{n_bits}b") for i in range(2**n_bits)]
    else:
        support_bits = list(support)

    threshold = float(spec.get("threshold", spec.get("epsilon", 0.02)))
    outside_tolerance = float(spec.get("outside_support_tolerance", 0.01))
    expected = distribution_from_support(support_bits, n_bits)
    kl_value = kl_divergence(probs, expected)
    support_indices = {bitstring_index(bit) for bit in support_bits}
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


def grade(spec: Mapping[str, Any], context: GradeContext) -> dict[str, Any]:
    probs = as_prob_array(context.probabilities)
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
        return grade_structural(spec, metadata=context.metadata, probabilities=probs, code=context.code)
    raise ValueError(f"unknown grader type: {grader_type}")
