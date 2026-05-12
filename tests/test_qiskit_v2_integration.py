import json
from pathlib import Path

from graders.qiskit_v2_specs import QiskitV2Evaluator, load_qiskit_v2_tasks
from utils.get_canonical_results import GLOBAL_INPUTS, task_6_input


GLOBAL_INPUTS["06"] = task_6_input()


def test_qiskit_v2_specs_cover_all_qiskit_tasks():
    # Arrange
    prompt_path = Path("prompts/qiskit_v2.jsonl")

    # Act
    task_ids = {
        json.loads(line)["task_id"]
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    # Assert
    assert len(task_ids) == 42


def test_task_36_accepts_true_uncompute_distribution():
    # Arrange
    evaluator = QiskitV2Evaluator(load_qiskit_v2_tasks(), GLOBAL_INPUTS)
    code = """
from qiskit import QuantumCircuit

def reverse_state_preparation_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    return qc
"""

    # Act
    _, details = evaluator.grade_code(
        task_id="36",
        code=code,
        entry_point="reverse_state_preparation_bell",
    )

    # Assert
    assert details["passed"] is True
    assert details["dominant_bitstring"] == "00"


def test_task_36_rejects_trivial_identity():
    # Arrange
    evaluator = QiskitV2Evaluator(load_qiskit_v2_tasks(), GLOBAL_INPUTS)
    code = """
from qiskit import QuantumCircuit

def reverse_state_preparation_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])
    return qc
"""

    # Act
    _, details = evaluator.grade_code(
        task_id="36",
        code=code,
        entry_point="reverse_state_preparation_bell",
    )

    # Assert
    assert details["passed"] is False
    assert details["non_measure_ops"] == 0


def test_task_43_accepts_textbook_toffoli():
    # Arrange
    evaluator = QiskitV2Evaluator(load_qiskit_v2_tasks(), GLOBAL_INPUTS)
    code = """
from qiskit import QuantumCircuit

def Toffoli_gate_decompose() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    return qc
"""

    # Act
    _, details = evaluator.grade_code(
        task_id="43",
        code=code,
        entry_point="Toffoli_gate_decompose",
    )

    # Assert
    assert details["passed"] is True


def test_task_43_rejects_identity():
    # Arrange
    evaluator = QiskitV2Evaluator(load_qiskit_v2_tasks(), GLOBAL_INPUTS)
    code = """
from qiskit import QuantumCircuit

def Toffoli_gate_decompose() -> QuantumCircuit:
    return QuantumCircuit(3)
"""

    # Act
    _, details = evaluator.grade_code(
        task_id="43",
        code=code,
        entry_point="Toffoli_gate_decompose",
    )

    # Assert
    assert details["passed"] is False
