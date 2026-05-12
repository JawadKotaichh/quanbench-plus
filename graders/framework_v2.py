from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import cirq
import json
import numpy as np

from graders.cirq_execution import execute_cirq_task
from graders.contracts import ExecutionResult, GradeContext
from graders.core import grade
from graders.pennylane_execution import execute_pennylane_task
from graders.qiskit_v2_specs import (
    QISKIT_V2_SPECS,
    QiskitV2Evaluator,
    load_qiskit_v2_tasks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CIRQ_V2_JSONL = REPO_ROOT / "prompts" / "cirq_v2.jsonl"
PENNYLANE_V2_JSONL = REPO_ROOT / "prompts" / "pennylane_v2.jsonl"


FRAMEWORK_V2_NOTES = {
    "cirq": """

QuanBench+ v2 grading note:
- Preserve the exact function signature and return a cirq.Circuit.
- Use a measurement key named 'result' when measuring.
- Measure exactly the register requested by the prompt. Do not measure ancillas unless explicitly requested.
- If the prompt asks for a decomposition/state without measurement, return the unmeasured circuit.
""",
    "pennylane": """

QuanBench+ v2 grading note:
- Preserve the exact function signature.
- Prefer analytic probabilities: return qml.probs(wires=[...]) from a QNode on default.qubit with shots=None.
- Return probabilities for exactly the register requested by the prompt. Do not include ancillas unless explicitly requested.
- If using qml.sample, the evaluator captures the QNode tape and grades exact probabilities from the operations.
""",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def with_framework_v2_fields(task: dict[str, Any], framework: str) -> dict[str, Any]:
    task_id = str(task["task_id"]).zfill(2)
    if task_id not in QISKIT_V2_SPECS:
        raise KeyError(f"missing v2 spec for task {task_id}")
    out = deepcopy(task)
    out["task_id"] = task_id
    out.pop("canonical_solution", None)
    out["canonical_class"] = deepcopy(QISKIT_V2_SPECS[task_id])
    out["prompt_v2"] = task["complete_prompt"].rstrip() + FRAMEWORK_V2_NOTES[framework]
    return out


def write_framework_v2_jsonl(framework: str) -> Path:
    if framework not in {"cirq", "pennylane"}:
        raise ValueError("framework must be cirq or pennylane")
    source_path = REPO_ROOT / "prompts" / f"{framework}.jsonl"
    output_path = REPO_ROOT / "prompts" / f"{framework}_v2.jsonl"
    tasks = read_jsonl(source_path)
    task_ids = {str(task["task_id"]).zfill(2) for task in tasks}
    if task_ids != set(QISKIT_V2_SPECS):
        raise ValueError(
            f"v2 spec/task mismatch for {framework}: "
            f"missing={sorted(set(QISKIT_V2_SPECS) - task_ids)} "
            f"extra={sorted(task_ids - set(QISKIT_V2_SPECS))}"
        )
    output_path.write_text(
        "\n".join(json.dumps(with_framework_v2_fields(task, framework), ensure_ascii=False) for task in tasks) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_framework_v2_tasks(framework: str) -> dict[str, dict[str, Any]]:
    path = REPO_ROOT / "prompts" / f"{framework}_v2.jsonl"
    return {str(task["task_id"]).zfill(2): task for task in read_jsonl(path)}


def _u_gate_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
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


def _cirq_target_unitary(task_id: str, spec: dict[str, Any], inputs: dict[str, Any]) -> np.ndarray | None:
    target = spec.get("target_unitary")
    if not target:
        return None
    if target == "u_gate":
        return _u_gate_matrix(*inputs[task_id])

    q = cirq.LineQubit.range(3)
    if target == "controlled_h":
        circuit = cirq.Circuit(cirq.H(q[1]).controlled_by(q[0]))
    elif target == "ccx":
        circuit = cirq.Circuit(cirq.CCX(q[0], q[1], q[2]))
    elif target == "cx":
        circuit = cirq.Circuit(cirq.CNOT(q[0], q[1]))
    else:
        raise ValueError(f"unknown target_unitary for task {task_id}: {target}")
    return np.asarray(cirq.unitary(circuit), dtype=complex)


def _pennylane_target_unitary(task_id: str, spec: dict[str, Any], inputs: dict[str, Any]) -> np.ndarray | None:
    target = spec.get("target_unitary")
    if not target:
        return None
    if target == "u_gate":
        return _u_gate_matrix(*inputs[task_id])

    import pennylane as qml

    if target == "controlled_h":
        ops = [qml.ctrl(qml.Hadamard, control=0)(wires=1)]
        wire_order = [0, 1]
    elif target == "ccx":
        ops = [qml.Toffoli(wires=[0, 1, 2])]
        wire_order = [0, 1, 2]
    elif target == "cx":
        ops = [qml.CNOT(wires=[0, 1])]
        wire_order = [0, 1]
    else:
        raise ValueError(f"unknown target_unitary for task {task_id}: {target}")
    return np.asarray(qml.matrix(ops, wire_order=wire_order), dtype=complex)


class FrameworkV2Evaluator:
    def __init__(self, framework: str, tasks: dict[str, dict[str, Any]], inputs: dict[str, Any]):
        if framework not in {"cirq", "pennylane"}:
            raise ValueError("framework must be cirq or pennylane")
        self.framework = framework
        self.tasks = tasks
        self.inputs = inputs
        self._canonical = QiskitV2Evaluator(load_qiskit_v2_tasks(), inputs)

    @property
    def _execute(self) -> Callable[..., ExecutionResult]:
        return execute_cirq_task if self.framework == "cirq" else execute_pennylane_task

    def _target_unitary(self, task_id: str, spec: dict[str, Any]) -> np.ndarray | None:
        if self.framework == "cirq":
            return _cirq_target_unitary(task_id, spec, self.inputs)
        return _pennylane_target_unitary(task_id, spec, self.inputs)

    def grade_execution(
        self,
        *,
        task_id: str,
        execution: ExecutionResult,
        code: str | None = None,
    ) -> dict[str, Any]:
        task_id = str(task_id).zfill(2)
        task = self.tasks[task_id]
        spec = task["canonical_class"]
        canonical = self._canonical._canonical_execution(task_id)
        return grade(
            spec,
            GradeContext(
                probabilities=execution.probabilities,
                canonical_probabilities=None if canonical is None else canonical.probabilities,
                candidate_unitary=execution.unitary,
                target_unitary=self._target_unitary(task_id, spec),
                metadata=execution.metadata,
                code=code,
            ),
        )

    def grade_code(self, *, task_id: str, code: str, entry_point: str) -> tuple[ExecutionResult, dict[str, Any]]:
        execution = self._execute(
            task_id=str(task_id).zfill(2),
            code=code,
            entry_point=entry_point,
            inputs=self.inputs,
        )
        return execution, self.grade_execution(task_id=task_id, execution=execution, code=code)
