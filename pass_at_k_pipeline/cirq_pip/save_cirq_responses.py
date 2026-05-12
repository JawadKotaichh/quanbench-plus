from __future__ import annotations

from pathlib import Path
from typing import Any

import cirq
import numpy as np

from graders.cirq_execution import execute_cirq_task
from pass_at_k_pipeline.cirq_pip.paths import CIRQ_JSONL, CIRQ_V2_JSONL, MODEL_RESPONSES_DIR
from pass_at_k_pipeline.defaults import NUMBER_OF_SHOTS
from pass_at_k_pipeline.save_responses import SaveResponsesConfig, save_framework_responses
from utils.common import get_handler


def get_probs_dictionnary(circuit, shots):
    sim = cirq.Simulator()
    data = sim.run(circuit, repetitions=shots).measurements["result"]
    bitstrings = ["".join(str(bit) for bit in row) for row in data]
    unique, counts = np.unique(bitstrings, return_counts=True)
    return {bitstring: count / shots for bitstring, count in zip(unique, counts)}


def counts_to_array(counts, outcomes=None, normalize=True):
    if isinstance(counts, list):
        counts = counts[0]
    if not isinstance(counts, dict):
        raise TypeError(f"Expected dict or list of dicts, got {type(counts)}")

    cleaned = {"".join(key.split()): value for key, value in counts.items()}
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
    elif isinstance(circuit_or_counts, cirq.Circuit):
        counts = get_probs_dictionnary(circuit_or_counts, shots)
    else:
        raise TypeError(
            f"Expected CirqCircuit or dict, got {type(circuit_or_counts)} instead."
        )
    return counts_to_array(counts)


def save_cirq_responses(
    file: Path,
    response_path: Path,
    output_dir: Path = Path("./model_results"),
    inputss: dict[str, Any] | None = None,
    benchmark_version: str = "v1",
):
    config = SaveResponsesConfig(
        framework="cirq",
        model_responses_dir=MODEL_RESPONSES_DIR,
        prompts_path=Path(CIRQ_JSONL),
        v2_prompts_path=Path(CIRQ_V2_JSONL),
        shots=NUMBER_OF_SHOTS,
        v1_executor=get_probs,
        v2_executor=execute_cirq_task,
    )
    return save_framework_responses(
        config=config,
        file=file,
        response_path=response_path,
        output_dir=output_dir,
        inputs=inputss,
        benchmark_version=benchmark_version,
    )
