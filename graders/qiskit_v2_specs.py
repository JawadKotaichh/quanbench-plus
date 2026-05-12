from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import json
import numpy as np
from qiskit import QuantumCircuit

from graders.contracts import ExecutionResult, GradeContext
from graders.core import grade
from graders.qiskit_execution import execute_qiskit_task
from graders.qiskit_v2_data import (
    QISKIT_V2_CANONICAL_SOLUTION_OVERRIDES,
    QISKIT_V2_PROMPT_NOTE,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
QISKIT_V2_JSONL = REPO_ROOT / "prompts" / "qiskit_v2.jsonl"
QISKIT_JSONL = REPO_ROOT / "prompts" / "qiskit.jsonl"


QISKIT_V2_SPECS: dict[str, dict[str, Any]] = {
    "01": {
        "type": "deterministic_dominant",
        "expected_dominants": ["00"],
        "min_dominant_probability": 0.95,
        "min_non_measure_ops": 1,
    },
    "02": {"type": "peak_match", "expected_peaks": ["011", "100"], "top_k": 2, "min_peak_probability": 0.45},
    "03": {"type": "support_uniformity", "support": ["010", "110"], "threshold": 0.02},
    "04": {"type": "peak_match", "top_k": 8, "threshold": 0.08},
    "06": {"type": "exact_distribution", "threshold": 1e-3},
    "07": {"type": "support_uniformity", "support": ["000", "010", "100", "110"], "threshold": 0.03},
    "08": {"type": "support_uniformity", "support": "all", "threshold": 0.02},
    "09": {"type": "peak_match", "expected_peaks": ["000", "100"], "top_k": 2, "min_peak_probability": 0.40},
    "10": {"type": "exact_distribution", "threshold": 1e-3},
    "11": {
        "type": "deterministic_dominant",
        "expected_dominants": ["0000", "0011", "1100", "1111"],
        "min_dominant_probability": 0.95,
    },
    "12": {"type": "deterministic_dominant", "expected_dominants": ["0000"], "min_dominant_probability": 0.95},
    "13": {"type": "support_uniformity", "support": ["00", "11"], "threshold": 0.02},
    "14": {"type": "support_uniformity", "support": ["000", "001", "110", "111"], "threshold": 0.03},
    "15": {"type": "exact_distribution", "threshold": 1e-3},
    "16": {"type": "support_uniformity", "support": ["00", "11"], "threshold": 0.02},
    "17": {"type": "support_uniformity", "support": ["000", "111"], "threshold": 0.02},
    "18": {"type": "support_uniformity", "support": ["0", "1"], "threshold": 0.02},
    "19": {"type": "peak_match", "expected_peaks": ["00101", "00110"], "top_k": 2, "min_peak_probability": 0.45},
    "20": {"type": "deterministic_dominant", "expected_dominants": ["011", "110"], "min_dominant_probability": 0.95},
    "21": {
        "type": "deterministic_dominant",
        "expected_dominants": ["000"],
        "min_dominant_probability": 0.95,
        "min_non_measure_ops": 1,
    },
    "22": {"type": "deterministic_dominant", "expected_dominants": ["0110"], "min_dominant_probability": 0.95},
    "23": {"type": "deterministic_dominant", "expected_dominants": ["001", "100"], "min_dominant_probability": 0.95},
    "24": {"type": "support_uniformity", "support": ["0001", "0010", "0100", "1000"], "threshold": 0.05},
    "25": {
        "type": "peak_match",
        "accepted_peak_sets": [["001", "111"], ["010", "110"]],
        "top_k": 2,
    },
    "26": {"type": "exact_distribution", "threshold": 0.02},
    "27": {"type": "exact_distribution", "comparison": "unitary", "target_unitary": "cx", "tolerance": 1e-8},
    "28": {
        "type": "deterministic_dominant",
        "expected_dominants": ["0000"],
        "min_dominant_probability": 0.95,
        "min_non_measure_ops": 1,
    },
    "29": {"type": "exact_distribution", "threshold": 0.02},
    "30": {
        "type": "deterministic_dominant",
        "expected_dominants": ["0"],
        "min_dominant_probability": 0.95,
        "min_non_measure_ops": 1,
    },
    "31": {
        "type": "peak_match",
        "expected_peaks": ["0000", "0100", "1000", "1100"],
        "top_k": 4,
        "min_peak_probability": 0.20,
    },
    "32": {"type": "support_uniformity", "support": ["00000000", "01000000", "10000000", "11000000"], "threshold": 0.05},
    "33": {"type": "deterministic_dominant", "expected_dominants": ["01", "10"], "min_dominant_probability": 0.95},
    "34": {"type": "deterministic_dominant", "expected_dominants": ["001", "100"], "min_dominant_probability": 0.95},
    "35": {
        "type": "deterministic_dominant",
        "expected_dominants": ["0"],
        "min_dominant_probability": 0.95,
        "min_non_measure_ops": 1,
    },
    "36": {
        "type": "deterministic_dominant",
        "expected_dominants": ["00"],
        "min_dominant_probability": 0.95,
        "min_non_measure_ops": 2,
    },
    "37": {"type": "exact_distribution", "comparison": "unitary", "target_unitary": "controlled_h", "tolerance": 1e-8},
    "39": {"type": "exact_distribution", "threshold": 1e-3},
    "40": {"type": "exact_distribution", "threshold": 1e-3},
    "41": {
        "type": "structural",
        "structural_name": "vqe_z2_ansatz",
        "min_entangling_gates": 0,
        "forbidden_terms": ["minimize(", "Estimator(", "Sampler("],
    },
    "42": {"type": "exact_distribution", "comparison": "unitary", "target_unitary": "u_gate", "tolerance": 1e-8},
    "43": {"type": "exact_distribution", "comparison": "unitary", "target_unitary": "ccx", "tolerance": 1e-8},
    "44": {"type": "exact_distribution", "comparison": "unitary", "target_unitary": "cx", "tolerance": 1e-8},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def with_qiskit_v2_fields(task: dict[str, Any]) -> dict[str, Any]:
    task_id = str(task["task_id"]).zfill(2)
    if task_id not in QISKIT_V2_SPECS:
        raise KeyError(f"missing v2 spec for task {task_id}")
    out = deepcopy(task)
    out["task_id"] = task_id
    out["canonical_class"] = deepcopy(QISKIT_V2_SPECS[task_id])
    if task_id in QISKIT_V2_CANONICAL_SOLUTION_OVERRIDES:
        out["canonical_solution"] = QISKIT_V2_CANONICAL_SOLUTION_OVERRIDES[task_id]
    out["prompt_v2"] = task["complete_prompt"].rstrip() + QISKIT_V2_PROMPT_NOTE
    return out


def write_qiskit_v2_jsonl(
    source_path: Path = QISKIT_JSONL,
    output_path: Path = QISKIT_V2_JSONL,
) -> Path:
    tasks = read_jsonl(source_path)
    if {str(task["task_id"]).zfill(2) for task in tasks} != set(QISKIT_V2_SPECS):
        missing = set(QISKIT_V2_SPECS) - {str(task["task_id"]).zfill(2) for task in tasks}
        extra = {str(task["task_id"]).zfill(2) for task in tasks} - set(QISKIT_V2_SPECS)
        raise ValueError(f"v2 spec/task mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    output_path.write_text(
        "\n".join(json.dumps(with_qiskit_v2_fields(task), ensure_ascii=False) for task in tasks) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_qiskit_v2_tasks(path: Path = QISKIT_V2_JSONL) -> dict[str, dict[str, Any]]:
    return {str(task["task_id"]).zfill(2): task for task in read_jsonl(path)}


def target_unitary_for_task(task_id: str, spec: dict[str, Any], inputs: dict[str, Any]) -> np.ndarray | None:
    target = spec.get("target_unitary")
    if not target:
        return None
    if target == "controlled_h":
        qc = QuantumCircuit(2)
        qc.ch(0, 1)
    elif target == "u_gate":
        theta, phi, lam = inputs[task_id]
        qc = QuantumCircuit(1)
        qc.u(theta, phi, lam, 0)
    elif target == "ccx":
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
    elif target == "cx":
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
    else:
        raise ValueError(f"unknown target_unitary for task {task_id}: {target}")
    from qiskit.quantum_info import Operator

    return np.asarray(Operator(qc).data, dtype=complex)


class QiskitV2Evaluator:
    def __init__(self, tasks: dict[str, dict[str, Any]], inputs: dict[str, Any]):
        self.tasks = tasks
        self.inputs = inputs
        self._canonical_cache: dict[str, ExecutionResult] = {}

    def _canonical_execution(self, task_id: str) -> ExecutionResult | None:
        if task_id in self._canonical_cache:
            return self._canonical_cache[task_id]
        task = self.tasks[task_id]
        spec = task["canonical_class"]
        needs_canonical_probs = spec["type"] in {"exact_distribution", "peak_match"} and not (
            spec.get("expected_distribution")
            or spec.get("expected_peaks")
            or spec.get("accepted_peak_sets")
            or spec.get("comparison") == "unitary"
        )
        needs_canonical_unitary = spec.get("comparison") == "unitary" and not spec.get("target_unitary")
        if not needs_canonical_probs and not needs_canonical_unitary:
            return None
        execution = execute_qiskit_task(
            task_id=task_id,
            code=task["canonical_solution"],
            entry_point=task["entry_point"],
            inputs=self.inputs,
        )
        self._canonical_cache[task_id] = execution
        return execution

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
        canonical = self._canonical_execution(task_id)
        target_unitary = target_unitary_for_task(task_id, spec, self.inputs)
        return grade(
            spec,
            GradeContext(
                probabilities=execution.probabilities,
                canonical_probabilities=None if canonical is None else canonical.probabilities,
                candidate_unitary=execution.unitary,
                canonical_unitary=None if canonical is None else canonical.unitary,
                target_unitary=target_unitary,
                metadata=execution.metadata,
                code=code,
            ),
        )

    def grade_code(self, *, task_id: str, code: str, entry_point: str) -> tuple[ExecutionResult, dict[str, Any]]:
        execution = execute_qiskit_task(
            task_id=str(task_id).zfill(2),
            code=code,
            entry_point=entry_point,
            inputs=self.inputs,
        )
        return execution, self.grade_execution(task_id=task_id, execution=execution, code=code)
