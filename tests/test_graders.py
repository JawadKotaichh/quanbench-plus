import numpy as np
from qiskit import QuantumCircuit

from graders.core import GradeContext, grade
from graders.qiskit_execution import exact_probabilities, circuit_unitary


def test_deterministic_dominant_accepts_endianness_class():
    result = grade(
        {
            "type": "deterministic_dominant",
            "expected_dominants": ["011", "110"],
            "min_dominant_probability": 0.95,
        },
        GradeContext(probabilities=[0, 0, 0, 1, 0, 0, 0, 0]),
    )

    assert result["passed"] is True
    assert result["dominant_bitstring"] == "011"


def test_support_uniformity_rejects_outside_support_mass():
    result = grade(
        {
            "type": "support_uniformity",
            "support": ["00", "11"],
            "threshold": 0.02,
            "outside_support_tolerance": 0.01,
        },
        GradeContext(probabilities=[0.45, 0.1, 0.0, 0.45]),
    )

    assert result["passed"] is False
    assert result["outside_support_mass"] == 0.1


def test_peak_match_uses_canonical_peaks_when_not_explicit():
    result = grade(
        {"type": "peak_match", "top_k": 2},
        GradeContext(
            probabilities=[0.05, 0.45, 0.45, 0.05],
            canonical_probabilities=[0.0, 0.5, 0.5, 0.0],
        ),
    )

    assert result["passed"] is True
    assert result["expected_peaks"] == ["01", "10"]


def test_unitary_comparison_ignores_global_phase():
    qc = QuantumCircuit(1)
    qc.x(0)
    candidate = circuit_unitary(qc)
    expected = -1 * np.asarray(candidate)

    result = grade(
        {
            "type": "exact_distribution",
            "comparison": "unitary",
            "ignore_global_phase": True,
            "tolerance": 1e-8,
        },
        GradeContext(probabilities=[0.0, 1.0], candidate_unitary=candidate, target_unitary=expected),
    )

    assert result["passed"] is True


def test_qiskit_exact_probabilities_follow_count_key_order():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.measure([0, 1], [0, 1])

    probs = exact_probabilities(qc)

    assert probs.tolist() == [0.0, 1.0, 0.0, 0.0]
