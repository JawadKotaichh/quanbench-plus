import numpy as np
import cirq


def task_6_input_cirq():
    q = cirq.LineQubit(0)
    qc = cirq.Circuit(
        cirq.H(q),
        cirq.rz((25 * np.pi) / 54)(q),
    )
    return qc


def get_probs_dictionnary_cirq(circuit, shots):
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=shots)

    # Measurement matrix with shape (shots, num_measured_qubits)
    data = result.measurements["result"]

    # Convert each row (bit array) into the correct bitstring
    bitstrings = ["".join(str(bit) for bit in row) for row in data]

    # Count occurrences
    unique, counts = np.unique(bitstrings, return_counts=True)

    # Convert to probabilities
    return {u: c / shots for u, c in zip(unique, counts)}


def counts_to_array_cirq(counts, outcomes=None, normalize=True):
    if isinstance(counts, list):
        counts = counts[0]
    if not isinstance(counts, dict):
        raise TypeError(f"Expected dict or list of dicts, got {type(counts)}")
    counts = {"".join(k.split()): v for k, v in counts.items()}
    if outcomes is None:
        outcomes = sorted(counts.keys())

    n_bits = len(outcomes[0])
    all_outcomes = [format(i, f"0{n_bits}b") for i in range(2**n_bits)]

    arr = np.array([counts.get(k, 0) for k in all_outcomes], dtype=float)
    if normalize:
        total = arr.sum()
        if total > 0:
            arr /= total
    return arr
