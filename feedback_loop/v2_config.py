from __future__ import annotations

from feedback_loop.framework_paths.paths_cirq import CIRQ_JSONL
from feedback_loop.framework_paths.paths_cirq import MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_CIRQ
from feedback_loop.framework_paths.paths_pennylane import MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_PENNYLANE
from feedback_loop.framework_paths.paths_pennylane import PENNYLANE_JSONL
from feedback_loop.framework_paths.paths_qiskit import MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_QISKIT
from feedback_loop.framework_paths.paths_qiskit import QISKIT_JSONL
from graders.framework_v2 import CIRQ_V2_JSONL, PENNYLANE_V2_JSONL
from graders.qiskit_v2_specs import QISKIT_V2_JSONL


PROMPT_PATHS = {
    ("cirq", "v1"): CIRQ_JSONL,
    ("cirq", "v2"): str(CIRQ_V2_JSONL),
    ("pennylane", "v1"): PENNYLANE_JSONL,
    ("pennylane", "v2"): str(PENNYLANE_V2_JSONL),
    ("qiskit", "v1"): QISKIT_JSONL,
    ("qiskit", "v2"): str(QISKIT_V2_JSONL),
}

MODEL_RESPONSE_DIRS = {
    "cirq": MODEL_RESPONSES_DIR_CIRQ,
    "pennylane": MODEL_RESPONSES_DIR_PENNYLANE,
    "qiskit": MODEL_RESPONSES_DIR_QISKIT,
}
