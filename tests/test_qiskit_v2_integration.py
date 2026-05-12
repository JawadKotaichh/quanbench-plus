import json
from pathlib import Path

from graders.qiskit_v2_specs import QiskitV2Evaluator, load_qiskit_v2_tasks
from utils.get_canonical_results import GLOBAL_INPUTS, task_6_input


GLOBAL_INPUTS["06"] = task_6_input()


def test_qiskit_v2_specs_cover_all_qiskit_tasks():
    task_ids = {
        json.loads(line)["task_id"]
        for line in Path("prompts/qiskit_v2.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    assert len(task_ids) == 42


def test_task_36_accepts_true_uncompute_distribution():
    tasks = load_qiskit_v2_tasks()
    evaluator = QiskitV2Evaluator(tasks, GLOBAL_INPUTS)
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

    _, details = evaluator.grade_code(
        task_id="36",
        code=code,
        entry_point="reverse_state_preparation_bell",
    )

    assert details["passed"] is True
    assert details["dominant_bitstring"] == "00"


def test_task_36_rejects_trivial_identity():
    tasks = load_qiskit_v2_tasks()
    evaluator = QiskitV2Evaluator(tasks, GLOBAL_INPUTS)
    code = """
from qiskit import QuantumCircuit

def reverse_state_preparation_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])
    return qc
"""

    _, details = evaluator.grade_code(
        task_id="36",
        code=code,
        entry_point="reverse_state_preparation_bell",
    )

    assert details["passed"] is False
    assert details["non_measure_ops"] == 0


def test_task_43_accepts_textbook_toffoli_and_rejects_identity():
    tasks = load_qiskit_v2_tasks()
    evaluator = QiskitV2Evaluator(tasks, GLOBAL_INPUTS)
    ccx_code = """
from qiskit import QuantumCircuit

def Toffoli_gate_decompose() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    return qc
"""
    identity_code = """
from qiskit import QuantumCircuit

def Toffoli_gate_decompose() -> QuantumCircuit:
    return QuantumCircuit(3)
"""

    _, ccx_details = evaluator.grade_code(
        task_id="43",
        code=ccx_code,
        entry_point="Toffoli_gate_decompose",
    )
    _, identity_details = evaluator.grade_code(
        task_id="43",
        code=identity_code,
        entry_point="Toffoli_gate_decompose",
    )

    assert ccx_details["passed"] is True
    assert identity_details["passed"] is False
