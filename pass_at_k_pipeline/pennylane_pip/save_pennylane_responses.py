from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from graders.pennylane_execution import execute_pennylane_task
from pass_at_k_pipeline.defaults import NUMBER_OF_SHOTS
from pass_at_k_pipeline.pennylane_pip.paths import (
    MODEL_RESPONSES_DIR,
    PENNYLANE_JSONL,
    PENNYLANE_V2_JSONL,
)
from pass_at_k_pipeline.save_responses import SaveResponsesConfig, save_framework_responses
from utils.common import get_handler


def binary_array_to_decimal(bits):
    value = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError("All elements must be 0 or 1")
        value = value * 2 + bit
    return value


def get_probs(task_id, solution, entry_point, shots, inputs):
    circuit_or_counts = get_handler(task_id, solution, entry_point, inputs)
    if not isinstance(circuit_or_counts, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(circuit_or_counts)} instead.")

    values = circuit_or_counts.tolist()
    if isinstance(values[0], list):
        counts = [0] * (2 ** len(values[0]))
        for sample in values:
            counts[binary_array_to_decimal(sample)] += 1
        return np.array(counts, dtype=float) / len(values)
    if isinstance(values[0], float):
        raise TypeError("Model return expected value or sampled on a specified basis, wrong return type")

    counts = [0, 0]
    for sample in values:
        counts[1 if sample > 0 else 0] += 1
    return np.array(counts, dtype=float) / len(values)


def save_pennylane_responses(
    file: Path,
    response_path: Path,
    output_dir: Path = Path("./model_results"),
    inputss: dict[str, Any] | None = None,
    benchmark_version: str = "v1",
):
    config = SaveResponsesConfig(
        framework="pennylane",
        model_responses_dir=MODEL_RESPONSES_DIR,
        prompts_path=Path(PENNYLANE_JSONL),
        v2_prompts_path=Path(PENNYLANE_V2_JSONL),
        shots=NUMBER_OF_SHOTS,
        v1_executor=get_probs,
        v2_executor=execute_pennylane_task,
    )
    return save_framework_responses(
        config=config,
        file=file,
        response_path=response_path,
        output_dir=output_dir,
        inputs=inputss,
        benchmark_version=benchmark_version,
    )
