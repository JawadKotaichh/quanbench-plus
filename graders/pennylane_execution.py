from __future__ import annotations

from typing import Any

import numpy as np
import pennylane as qml

from graders.contracts import ExecutionResult
from utils.common import get_handler


def _probabilities_from_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.floating):
        total = float(np.sum(arr))
        if total > 0 and np.all(arr >= -1e-12):
            return np.asarray(arr, dtype=float) / total
    if arr.ndim == 1:
        counts = np.zeros(2, dtype=float)
        for sample in arr.tolist():
            counts[1 if sample > 0 else 0] += 1
    elif arr.ndim == 2:
        counts = np.zeros(2 ** arr.shape[1], dtype=float)
        for row in arr.astype(int).tolist():
            idx = 0
            for bit in row:
                idx = (idx << 1) | int(bit)
            counts[idx] += 1
    else:
        raise TypeError(f"Expected 1-D or 2-D PennyLane array, got shape {arr.shape}")
    total = float(np.sum(counts))
    return counts / total if total > 0 else counts


def _measurement_wires(tape: Any) -> list[Any]:
    for measurement in tape.measurements:
        wires = list(measurement.wires)
        if wires:
            return wires
    return list(tape.wires)


def _probabilities_from_tape(tape: Any) -> np.ndarray:
    wire_order = list(tape.wires)
    measured = _measurement_wires(tape)
    unitary = np.asarray(qml.matrix(tape.operations, wire_order=wire_order), dtype=complex)
    state = np.zeros(2 ** len(wire_order), dtype=complex)
    state[0] = 1.0
    final_state = unitary @ state
    basis_probs = np.abs(final_state) ** 2
    measured_positions = [wire_order.index(wire) for wire in measured]
    out = np.zeros(2 ** len(measured), dtype=float)
    n_wires = len(wire_order)
    for basis_index, probability in enumerate(basis_probs):
        measured_index = 0
        for position in measured_positions:
            bit = (basis_index >> (n_wires - position - 1)) & 1
            measured_index = (measured_index << 1) | bit
        out[measured_index] += float(probability)
    total = float(out.sum())
    return out / total if total > 0 else out


def _unitary_from_tape(tape: Any) -> np.ndarray | None:
    try:
        return np.asarray(qml.matrix(tape.operations, wire_order=list(tape.wires)), dtype=complex)
    except Exception:
        return None


def _metadata_from_tape(tape: Any | None) -> dict[str, Any]:
    if tape is None:
        return {}
    op_counts: dict[str, int] = {}
    for op in tape.operations:
        name = getattr(op, "name", type(op).__name__)
        op_counts[name] = op_counts.get(name, 0) + 1
    return {
        "num_qubits": len(tape.wires),
        "measurement_count": len(tape.measurements),
        "non_measurement_operation_count": len(tape.operations),
        "measurement_wires": [str(wire) for wire in _measurement_wires(tape)],
        "operation_counts": op_counts,
        "entangling_gate_count": sum(1 for op in tape.operations if len(op.wires) >= 2),
    }


def execute_pennylane_task(
    *,
    task_id: str,
    code: str,
    entry_point: str,
    inputs: dict[str, Any],
) -> ExecutionResult:
    tapes: list[Any] = []
    original_call = qml.QNode.__call__

    def recording_call(self, *args, **kwargs):
        tape = self.construct(args, kwargs)
        tapes.append(tape)
        return original_call(self, *args, **kwargs)

    qml.QNode.__call__ = recording_call
    try:
        result = get_handler(task_id, code, entry_point, inputs)
    finally:
        qml.QNode.__call__ = original_call

    tape = tapes[-1] if tapes else None
    probabilities = _probabilities_from_tape(tape) if tape is not None else _probabilities_from_array(result)
    return ExecutionResult(
        probabilities=probabilities.tolist(),
        metadata=_metadata_from_tape(tape),
        unitary=None if tape is None else _unitary_from_tape(tape),
        circuit=tape,
    )
