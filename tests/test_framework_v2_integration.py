import json
from pathlib import Path

from graders.framework_v2 import FrameworkV2Evaluator, load_framework_v2_tasks
from pass_at_k_pipeline.cirq_pip.get_cirq_results import task_6_input as cirq_task_6_input
from pass_at_k_pipeline.defaults import GLOBAL_INPUTS
from pass_at_k_pipeline.pennylane_pip.get_pennylane_results import (
    task_6_input as pennylane_task_6_input,
)


def test_cirq_v2_accepts_bell_state():
    # Arrange
    inputs = dict(GLOBAL_INPUTS)
    inputs["06"] = cirq_task_6_input()
    evaluator = FrameworkV2Evaluator("cirq", load_framework_v2_tasks("cirq"), inputs)
    code = """
import cirq

def Bell_State():
    q = cirq.LineQubit.range(2)
    return cirq.Circuit(
        cirq.H(q[0]),
        cirq.CNOT(q[0], q[1]),
        cirq.measure(q[0], q[1], key="result"),
    )
"""

    # Act
    _, details = evaluator.grade_code(task_id="16", code=code, entry_point="Bell_State")

    # Assert
    assert details["passed"] is True


def test_cirq_v2_rejects_measurement_only_oracle():
    # Arrange
    inputs = dict(GLOBAL_INPUTS)
    inputs["06"] = cirq_task_6_input()
    evaluator = FrameworkV2Evaluator("cirq", load_framework_v2_tasks("cirq"), inputs)
    code = """
import cirq

def grover_search_oracle_00():
    q = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.measure(q[0], q[1], key="result"))
"""

    # Act
    _, details = evaluator.grade_code(
        task_id="01",
        code=code,
        entry_point="grover_search_oracle_00",
    )

    # Assert
    assert details["passed"] is False
    assert details["non_measure_ops"] == 0


def test_cirq_v2_unitary_rejects_identity_toffoli():
    # Arrange
    inputs = dict(GLOBAL_INPUTS)
    inputs["06"] = cirq_task_6_input()
    evaluator = FrameworkV2Evaluator("cirq", load_framework_v2_tasks("cirq"), inputs)
    code = """
import cirq

def Toffoli_gate_decompose():
    return cirq.Circuit()
"""

    # Act
    _, details = evaluator.grade_code(
        task_id="43",
        code=code,
        entry_point="Toffoli_gate_decompose",
    )

    # Assert
    assert details["passed"] is False


def test_framework_v2_prompts_do_not_ship_qiskit_canonicals():
    # Arrange
    prompt_paths = (Path("prompts/cirq_v2.jsonl"), Path("prompts/pennylane_v2.jsonl"))

    # Act
    prompt_tasks = [
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in prompt_paths
    ]

    # Assert
    assert all(len(tasks) == 42 for tasks in prompt_tasks)
    assert all("canonical_solution" not in task for tasks in prompt_tasks for task in tasks)


def test_pennylane_v2_accepts_exact_probs_bell_state():
    # Arrange
    inputs = dict(GLOBAL_INPUTS)
    inputs["06"] = pennylane_task_6_input()
    evaluator = FrameworkV2Evaluator("pennylane", load_framework_v2_tasks("pennylane"), inputs)
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

    # Act
    _, details = evaluator.grade_code(task_id="16", code=code, entry_point="Bell_State")

    # Assert
    assert details["passed"] is True


def test_pennylane_v2_unitary_rejects_identity_toffoli():
    # Arrange
    inputs = dict(GLOBAL_INPUTS)
    inputs["06"] = pennylane_task_6_input()
    evaluator = FrameworkV2Evaluator("pennylane", load_framework_v2_tasks("pennylane"), inputs)
    code = """
import pennylane as qml

def Toffoli_gate_decompose():
    dev = qml.device("default.qubit", wires=3, shots=None)

    @qml.qnode(dev)
    def circuit():
        return qml.probs(wires=[0, 1, 2])

    return circuit()
"""

    # Act
    _, details = evaluator.grade_code(
        task_id="43",
        code=code,
        entry_point="Toffoli_gate_decompose",
    )

    # Assert
    assert details["passed"] is False
