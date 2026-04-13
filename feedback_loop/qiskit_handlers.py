from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
import numpy as np


def task_6_input_qiskit():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz((25 * np.pi) / 54, 0)
    return qc


def get_probs_dictionnary_qiskit(qc, shots):
    qc = qc.copy()
    if not any(inst.operation.name == "measure" for inst in qc.data):
        qc.measure_all()

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    job = sim.run(compiled, shots=shots)
    result = job.result()
    return result.get_counts()


def counts_to_array_qiskit(counts, outcomes=None, normalize=True):
    if isinstance(counts, list):
        counts = counts[0]
    if not isinstance(counts, dict):
        raise TypeError(f"Expected dict or list of dicts, got {type(counts)}")

    # Drop everything after the first space + merge duplicates
    cleaned = {}
    for k, v in counts.items():
        clean_key = k.split()[0]
        cleaned[clean_key] = cleaned.get(clean_key, 0) + v
    counts = cleaned

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
