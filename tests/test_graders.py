import numpy as np
from qiskit import QuantumCircuit

from graders.contracts import GradeContext
from graders.core import grade
from graders.qiskit_execution import circuit_unitary, exact_probabilities


def test_deterministic_dominant_accepts_endianness_class():
    # Arrange
    spec = {
        "type": "deterministic_dominant",
        "expected_dominants": ["011", "110"],
        "min_dominant_probability": 0.95,
    }
    context = GradeContext(probabilities=[0, 0, 0, 1, 0, 0, 0, 0])

    # Act
    result = grade(spec, context)

    # Assert
    assert result["passed"] is True
    assert result["dominant_bitstring"] == "011"


def test_support_uniformity_rejects_outside_support_mass():
    # Arrange
    spec = {
        "type": "support_uniformity",
        "support": ["00", "11"],
        "threshold": 0.02,
        "outside_support_tolerance": 0.01,
    }
    context = GradeContext(probabilities=[0.45, 0.1, 0.0, 0.45])

    # Act
    result = grade(spec, context)

    # Assert
    assert result["passed"] is False
    assert result["outside_support_mass"] == 0.1


def test_peak_match_uses_canonical_peaks_when_not_explicit():
    # Arrange
    spec = {"type": "peak_match", "top_k": 2}
    context = GradeContext(
        probabilities=[0.05, 0.45, 0.45, 0.05],
        canonical_probabilities=[0.0, 0.5, 0.5, 0.0],
    )

    # Act
    result = grade(spec, context)

    # Assert
    assert result["passed"] is True
    assert result["expected_peaks"] == ["01", "10"]


def test_unitary_comparison_ignores_global_phase():
    # Arrange
    circuit = QuantumCircuit(1)
    circuit.x(0)
    candidate = circuit_unitary(circuit)
    expected = -1 * np.asarray(candidate)
    spec = {
        "type": "exact_distribution",
        "comparison": "unitary",
        "ignore_global_phase": True,
        "tolerance": 1e-8,
    }
    context = GradeContext(probabilities=[0.0, 1.0], candidate_unitary=candidate, target_unitary=expected)

    # Act
    result = grade(spec, context)

    # Assert
    assert result["passed"] is True


def test_qiskit_exact_probabilities_follow_count_key_order():
    # Arrange
    circuit = QuantumCircuit(2, 2)
    circuit.x(0)
    circuit.measure([0, 1], [0, 1])

    # Act
    probabilities = exact_probabilities(circuit)

    # Assert
    assert probabilities.tolist() == [0.0, 1.0, 0.0, 0.0]
