from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from graders.qiskit_execution import execute_qiskit_task
from pass_at_k_pipeline.defaults import NUMBER_OF_SHOTS
from pass_at_k_pipeline.qiskit_pip.paths import (
    MODEL_RESPONSES_DIR,
    QISKIT_JSONL,
    QISKIT_V2_JSONL,
)
from pass_at_k_pipeline.save_responses import SaveResponsesConfig, save_framework_responses
from utils.common import get_handler


def get_probs_dictionnary(qc, shots):
    qc = qc.copy()
    if not any(inst.operation.name == "measure" for inst in qc.data):
        qc.measure_all()

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    return sim.run(compiled, shots=shots).result().get_counts()


def counts_to_array(counts, outcomes=None, normalize=True):
    if isinstance(counts, list):
        counts = counts[0]
    if not isinstance(counts, dict):
        raise TypeError(f"Expected dict or list of dicts, got {type(counts)}")

    cleaned: dict[str, float] = {}
    for key, value in counts.items():
        clean_key = key.split()[0]
        cleaned[clean_key] = cleaned.get(clean_key, 0.0) + float(value)

    outcomes = outcomes or sorted(cleaned.keys())
    n_bits = len(outcomes[0])
    arr = np.array(
        [cleaned.get(format(index, f"0{n_bits}b"), 0.0) for index in range(2**n_bits)],
        dtype=float,
    )
    total = arr.sum()
    return arr / total if normalize and total > 0 else arr


def get_probs(task_id, solution, entry_point, shots, inputs):
    circuit_or_counts = get_handler(task_id, solution, entry_point, inputs)
    if isinstance(circuit_or_counts, dict):
        counts = circuit_or_counts
    elif hasattr(circuit_or_counts, "name"):
        counts = get_probs_dictionnary(circuit_or_counts, shots)
    else:
        raise TypeError(
            f"Expected QuantumCircuit or dict, got {type(circuit_or_counts)} instead."
        )
    return counts_to_array(counts)


def save_qiskit_responses(
    file: Path,
    response_path: Path,
    output_dir: Path = Path("./model_results"),
    inputss: dict[str, Any] | None = None,
    benchmark_version: str = "v1",
):
    config = SaveResponsesConfig(
        framework="qiskit",
        model_responses_dir=MODEL_RESPONSES_DIR,
        prompts_path=Path(QISKIT_JSONL),
        v2_prompts_path=Path(QISKIT_V2_JSONL),
        shots=NUMBER_OF_SHOTS,
        v1_executor=get_probs,
        v2_executor=execute_qiskit_task,
    )
    return save_framework_responses(
        config=config,
        file=file,
        response_path=response_path,
        output_dir=output_dir,
        inputs=inputss,
        benchmark_version=benchmark_version,
    )
