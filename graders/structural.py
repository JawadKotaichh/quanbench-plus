from __future__ import annotations

from typing import Any, Mapping, Sequence


def grade_structural(
    spec: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None,
    probabilities: Sequence[float],
    code: str | None,
) -> dict[str, Any]:
    name = str(spec.get("structural_name", ""))
    if name != "vqe_z2_ansatz":
        raise ValueError(f"unknown structural grader: {name}")

    metadata_dict = dict(metadata or {})
    forbidden_terms = tuple(spec.get("forbidden_terms", ["minimize(", "Estimator(", "Sampler("]))
    checks = {
        "num_qubits_is_2": int(metadata_dict.get("num_qubits", -1)) == 2,
        "no_measurements": int(metadata_dict.get("measurement_count", 0)) == 0,
        "min_entangling_gates": int(metadata_dict.get("entangling_gate_count", 0))
        >= int(spec.get("min_entangling_gates", 0)),
        "no_classical_optimizer_call": not any(term in (code or "") for term in forbidden_terms),
        "probability_shape_is_2q": len(probabilities) == 4,
    }
    return {
        "passed": all(checks.values()),
        "grader_type": "structural",
        "structural_name": name,
        "checks": checks,
        "metadata": metadata_dict,
    }
