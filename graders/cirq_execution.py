from __future__ import annotations

from typing import Any

import cirq
import numpy as np

from graders.contracts import ExecutionResult
from utils.common import get_handler


def _bitstrings_to_array(counts: dict[str, float]) -> np.ndarray:
    cleaned = {"".join(str(key).split()): float(value) for key, value in counts.items()}
    if not cleaned:
        raise ValueError("counts dictionary is empty")
    n_bits = len(next(iter(cleaned)))
    out = np.array(
        [cleaned.get(format(i, f"0{n_bits}b"), 0.0) for i in range(2**n_bits)],
        dtype=float,
    )
    total = float(out.sum())
    return out / total if total > 0 else out


def _measurement_qubits(circuit: cirq.Circuit) -> list[cirq.Qid]:
    ops = list(circuit.all_operations())
    measurement_ops = [op for op in ops if cirq.is_measurement(op)]
    result_ops = [
        op
        for op in measurement_ops
        if "result" in cirq.measurement_key_names(op)
    ]
    selected = result_ops or measurement_ops
    qubits: list[cirq.Qid] = []
    for op in selected:
        qubits.extend(op.qubits)
    return qubits


def _all_qubits(circuit: cirq.Circuit) -> list[cirq.Qid]:
    return sorted(circuit.all_qubits())


def exact_probabilities(circuit: cirq.Circuit) -> np.ndarray:
    qubit_order = _all_qubits(circuit)
    measured = _measurement_qubits(circuit) or qubit_order
    measured_positions = [qubit_order.index(q) for q in measured]
    state = cirq.final_state_vector(
        circuit,
        qubit_order=qubit_order,
        ignore_terminal_measurements=True,
        dtype=np.complex128,
    )
    basis_probs = np.abs(np.asarray(state, dtype=complex)) ** 2
    out = np.zeros(2 ** len(measured), dtype=float)
    n_qubits = len(qubit_order)
    for basis_index, probability in enumerate(basis_probs):
        measured_index = 0
        for position in measured_positions:
            bit = (basis_index >> (n_qubits - position - 1)) & 1
            measured_index = (measured_index << 1) | bit
        out[measured_index] += float(probability)
    total = float(out.sum())
    return out / total if total > 0 else out


def circuit_unitary(circuit: cirq.Circuit) -> np.ndarray | None:
    try:
        return np.asarray(cirq.unitary(cirq.drop_terminal_measurements(circuit)), dtype=complex)
    except Exception:
        return None


def circuit_metadata(circuit: cirq.Circuit) -> dict[str, Any]:
    ops = list(circuit.all_operations())
    op_counts: dict[str, int] = {}
    non_measurement_ops = 0
    for op in ops:
        name = str(op.gate).split("(")[0] if op.gate is not None else type(op).__name__
        op_counts[name] = op_counts.get(name, 0) + 1
        if not cirq.is_measurement(op):
            non_measurement_ops += 1
    return {
        "num_qubits": len(circuit.all_qubits()),
        "measurement_count": sum(1 for op in ops if cirq.is_measurement(op)),
        "non_measurement_operation_count": non_measurement_ops,
        "operation_counts": op_counts,
        "entangling_gate_count": sum(
            1 for op in ops if not cirq.is_measurement(op) and len(op.qubits) >= 2
        ),
        "measurement_qubits": [str(q) for q in _measurement_qubits(circuit)],
    }


def execute_cirq_task(
    *,
    task_id: str,
    code: str,
    entry_point: str,
    inputs: dict[str, Any],
) -> ExecutionResult:
    result = get_handler(task_id, code, entry_point, inputs)
    if isinstance(result, dict):
        return ExecutionResult(
            probabilities=_bitstrings_to_array(result).tolist(),
            metadata={"returned_counts": True},
            unitary=None,
            circuit=None,
        )
    if not isinstance(result, cirq.Circuit):
        raise TypeError(f"Expected Cirq Circuit or dict, got {type(result)} instead.")
    return ExecutionResult(
        probabilities=exact_probabilities(result).tolist(),
        metadata=circuit_metadata(result),
        unitary=circuit_unitary(result),
        circuit=result,
    )
