from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from feedback_loop.framework_paths.paths_cirq import CIRQ_JSONL
from feedback_loop.framework_paths.paths_cirq import MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_CIRQ
from feedback_loop.framework_paths.paths_pennylane import PENNYLANE_JSONL
from feedback_loop.framework_paths.paths_pennylane import MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_PENNYLANE
from feedback_loop.framework_paths.paths_qiskit import QISKIT_JSONL
from feedback_loop.framework_paths.paths_qiskit import MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_QISKIT


def get_jsonl_path(framework: str) -> str:
    if framework == "cirq":
        return CIRQ_JSONL
    if framework == "pennylane":
        return PENNYLANE_JSONL
    return QISKIT_JSONL


def get_model_responses_dir(framework: str) -> Path:
    if framework == "cirq":
        return MODEL_RESPONSES_DIR_CIRQ
    if framework == "pennylane":
        return MODEL_RESPONSES_DIR_PENNYLANE
    return MODEL_RESPONSES_DIR_QISKIT


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
