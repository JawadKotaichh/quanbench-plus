import json

from pass_at_k_pipeline.cirq_pip.get_cirq_results import main as cirq_main
from pass_at_k_pipeline.cirq_pip.paths import CANONICAL_SOLUTIONS_DIR as CIRQ_CANONICAL
from pass_at_k_pipeline.defaults import GLOBAL_INPUTS
from pass_at_k_pipeline.pennylane_pip.get_pennylane_results import main as pennylane_main
from pass_at_k_pipeline.pennylane_pip.paths import CANONICAL_SOLUTIONS_DIR as PENNYLANE_CANONICAL
from pass_at_k_pipeline.qiskit_pip.get_qiskit_results import main as qiskit_main
from pass_at_k_pipeline.qiskit_pip.paths import CANONICAL_SOLUTIONS_DIR as QISKIT_CANONICAL


def test_qiskit_v2_result_file_accepts_unitary_solution(tmp_path):
    # Arrange
    raw_path = tmp_path / "qiskit_raw.json"
    response_path = tmp_path / "qiskit_response.json"
    result_path = tmp_path / "qiskit_result.json"
    code = """
from qiskit import QuantumCircuit

def Toffoli_gate_decompose() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    return qc
"""
    raw_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "43",
                    "entry_point": "Toffoli_gate_decompose",
                    "category": "unitary",
                    "version": 1,
                    "code": code,
                }
            ]
        ),
        encoding="utf-8",
    )

    # Act
    results = qiskit_main(
        raw_path,
        "smoke_qiskit",
        response_path,
        result_path,
        QISKIT_CANONICAL,
        GLOBAL_INPUTS,
        benchmark_version="v2",
    )

    # Assert
    assert results[0]["any_passed"] is True


def test_cirq_v2_result_file_accepts_bell_solution(tmp_path):
    # Arrange
    raw_path = tmp_path / "cirq_raw.json"
    response_path = tmp_path / "cirq_response.json"
    result_path = tmp_path / "cirq_result.json"
    code = """
import cirq

def Bell_State():
    q = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.measure(q[0], q[1], key="result"))
"""
    raw_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "16",
                    "entry_point": "Bell_State",
                    "category": "state",
                    "version": 1,
                    "code": code,
                }
            ]
        ),
        encoding="utf-8",
    )

    # Act
    results = cirq_main(
        raw_path,
        "smoke_cirq",
        response_path,
        result_path,
        CIRQ_CANONICAL,
        GLOBAL_INPUTS,
        benchmark_version="v2",
    )

    # Assert
    assert results[0]["any_passed"] is True


def test_pennylane_v2_result_file_accepts_bell_solution(tmp_path):
    # Arrange
    raw_path = tmp_path / "pennylane_raw.json"
    response_path = tmp_path / "pennylane_response.json"
    result_path = tmp_path / "pennylane_result.json"
    code = """
import pennylane as qml

def Bell_State():
    dev = qml.device("default.qubit", wires=2, shots=None)
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])
    return circuit()
"""
    raw_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "16",
                    "entry_point": "Bell_State",
                    "category": "state",
                    "version": 1,
                    "code": code,
                }
            ]
        ),
        encoding="utf-8",
    )

    # Act
    results = pennylane_main(
        raw_path,
        "smoke_pennylane",
        response_path,
        result_path,
        PENNYLANE_CANONICAL,
        GLOBAL_INPUTS,
        benchmark_version="v2",
    )

    # Assert
    assert results[0]["any_passed"] is True
