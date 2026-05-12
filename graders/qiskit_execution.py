from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, Statevector

from utils.common import get_handler


@dataclass
class QiskitExecution:
    probabilities: list[float]
    metadata: dict[str, Any]
    unitary: Any | None = None
    circuit: QuantumCircuit | None = None


def counts_to_array(counts: dict[str, int] | list[dict[str, int]]) -> np.ndarray:
    if isinstance(counts, list):
        counts = counts[0]
    if not isinstance(counts, dict):
        raise TypeError(f"Expected dict or list of dicts, got {type(counts)}")

    cleaned: dict[str, float] = {}
    for key, value in counts.items():
        clean_key = str(key).split()[0]
        cleaned[clean_key] = cleaned.get(clean_key, 0.0) + float(value)

    if not cleaned:
        raise ValueError("counts dictionary is empty")

    n_bits = len(next(iter(cleaned)))
    out = np.array(
        [cleaned.get(format(i, f"0{n_bits}b"), 0.0) for i in range(2**n_bits)],
        dtype=float,
    )
    total = float(out.sum())
    return out / total if total > 0 else out


def circuit_without_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    stripped = QuantumCircuit(*circuit.qregs)
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            continue
        stripped.append(instruction.operation, instruction.qubits, [])
    return stripped


def measurement_pairs(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        qubit_index = circuit.find_bit(instruction.qubits[0]).index
        clbit_index = circuit.find_bit(instruction.clbits[0]).index
        pairs.append((qubit_index, clbit_index))
    return pairs


def exact_probabilities(circuit: QuantumCircuit) -> np.ndarray:
    pairs = measurement_pairs(circuit)
    if not pairs:
        pairs = [(i, i) for i in range(circuit.num_qubits)]
        n_bits = circuit.num_qubits
    else:
        n_bits = max(clbit for _, clbit in pairs) + 1

    stripped = circuit_without_measurements(circuit)
    state = Statevector.from_instruction(stripped)
    basis_probs = np.asarray(state.probabilities(), dtype=float)
    out = np.zeros(2**n_bits, dtype=float)

    for basis_index, probability in enumerate(basis_probs):
        if probability == 0:
            continue
        classical_index = 0
        for qubit_index, clbit_index in pairs:
            if (basis_index >> qubit_index) & 1:
                classical_index |= 1 << clbit_index
        out[classical_index] += float(probability)

    total = float(out.sum())
    return out / total if total > 0 else out


def circuit_unitary(circuit: QuantumCircuit) -> np.ndarray | None:
    try:
        return np.asarray(Operator(circuit_without_measurements(circuit)).data, dtype=complex)
    except Exception:
        return None


def circuit_metadata(circuit: QuantumCircuit) -> dict[str, Any]:
    pairs = measurement_pairs(circuit)
    op_counts: dict[str, int] = {}
    entangling = 0
    for instruction in circuit.data:
        name = instruction.operation.name
        op_counts[name] = op_counts.get(name, 0) + 1
        if name != "measure" and len(instruction.qubits) >= 2:
            entangling += 1

    return {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "measurement_count": len(pairs),
        "measurement_pairs": [[q, c] for q, c in pairs],
        "operation_counts": op_counts,
        "entangling_gate_count": entangling,
        "has_measurements": bool(pairs),
    }


def execute_qiskit_task(
    *,
    task_id: str,
    code: str,
    entry_point: str,
    inputs: dict[str, Any],
) -> QiskitExecution:
    result = get_handler(task_id, code, entry_point, inputs)
    if isinstance(result, dict):
        return QiskitExecution(
            probabilities=counts_to_array(result).tolist(),
            metadata={"returned_counts": True},
            unitary=None,
            circuit=None,
        )
    if not isinstance(result, QuantumCircuit) and not hasattr(result, "data"):
        raise TypeError(f"Expected QuantumCircuit or dict, got {type(result)} instead.")

    circuit = result
    metadata = circuit_metadata(circuit)
    try:
        probabilities = exact_probabilities(circuit)
        metadata["probability_method"] = "statevector"
    except Exception as exc:
        sim = AerSimulator(seed_simulator=42)
        compiled = transpile(circuit, sim)
        counts = sim.run(compiled, shots=2048, seed_simulator=42).result().get_counts()
        probabilities = counts_to_array(counts)
        metadata["probability_method"] = "qasm_fallback"
        metadata["statevector_error"] = f"{type(exc).__name__}: {exc}"
    return QiskitExecution(
        probabilities=probabilities.tolist(),
        metadata=metadata,
        unitary=circuit_unitary(circuit),
        circuit=circuit,
    )
