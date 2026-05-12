from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import traceback

import cirq
import numpy as np

from feedback_loop.cirq_handlers import counts_to_array_cirq, get_probs_dictionnary_cirq, task_6_input_cirq
from feedback_loop.defaults import GLOBAL_INPUTS, NUMBER_OF_SHOTS
from feedback_loop.pennylane_handlers import binary_array_to_decimal_pennylane, task_6_input_pennylane
from feedback_loop.qiskit_handlers import counts_to_array_qiskit, get_probs_dictionnary_qiskit, task_6_input_qiskit
from utils.common import get_handler
from utils.get_kl_div import get_kl_div


@dataclass
class EvalResult:
    compiled: bool
    ran: bool
    kl_div_bool: bool
    kl_div_result: float | None = None
    error: str | None = None
    output: Any | None = None


def load_global_inputs(framework: str) -> dict[str, Any]:
    if framework == "cirq":
        GLOBAL_INPUTS["06"] = task_6_input_cirq()
    elif framework == "pennylane":
        GLOBAL_INPUTS["06"] = task_6_input_pennylane()
    else:
        GLOBAL_INPUTS["06"] = task_6_input_qiskit()
    return GLOBAL_INPUTS


def get_probs_cirq(task_id: str, solution: str, entry_point: str, shots: int, inputs: dict[str, Any]) -> np.ndarray:
    circuit_or_counts = get_handler(task_id, solution, entry_point, inputs)
    if isinstance(circuit_or_counts, dict):
        counts = circuit_or_counts
    elif isinstance(circuit_or_counts, cirq.Circuit):
        counts = get_probs_dictionnary_cirq(circuit_or_counts, shots)
    else:
        raise TypeError(f"Expected CirqCircuit or dict, got {type(circuit_or_counts)} instead.")
    return counts_to_array_cirq(counts)


def get_probs_pennylane(task_id: str, solution: str, entry_point: str, shots: int, inputs: dict[str, Any]) -> np.ndarray:
    del shots
    circuit_or_counts = get_handler(task_id, solution, entry_point, inputs)
    if not isinstance(circuit_or_counts, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(circuit_or_counts)} instead.")
    values = circuit_or_counts.tolist()
    if type(values[0]) is list:
        counts = [0] * (2 ** len(values[0]))
        for sample in values:
            counts[binary_array_to_decimal_pennylane(sample)] += 1
        return np.array([count / len(values) for count in counts])
    if type(values[0]) is float:
        raise TypeError("Model return expected value or sampled on a specified basis, wrong return type")
    counts = [0, 0]
    for value in values:
        counts[1 if value > 0 else 0] += 1
    return np.array([count / len(values) for count in counts])


def get_probs_qiskit(task_id: str, solution: str, entry_point: str, shots: int, inputs: dict[str, Any]) -> np.ndarray:
    circuit_or_counts = get_handler(task_id, solution, entry_point, inputs)
    if isinstance(circuit_or_counts, dict):
        counts = circuit_or_counts
    elif hasattr(circuit_or_counts, "name"):
        counts = get_probs_dictionnary_qiskit(circuit_or_counts, shots)
    else:
        raise TypeError(f"Expected QuantumCircuit or dict, got {type(circuit_or_counts)} instead.")
    return counts_to_array_qiskit(counts)


def evaluate_generated_code(
    task_id: str,
    entry_point: str,
    code: str,
    framework: str,
    canonical_by_task: dict[str, dict[str, Any]],
    inputss: dict[str, Any] = GLOBAL_INPUTS,
) -> EvalResult:
    try:
        output = _run_framework(task_id, entry_point, code, framework, inputss)
    except SyntaxError:
        return EvalResult(compiled=False, ran=False, kl_div_bool=False, error=traceback.format_exc())
    except Exception:
        return EvalResult(compiled=True, ran=False, kl_div_bool=False, error=traceback.format_exc())
    return _compare_output(task_id, output, canonical_by_task)


def _run_framework(
    task_id: str,
    entry_point: str,
    code: str,
    framework: str,
    inputs: dict[str, Any],
) -> np.ndarray:
    if framework == "cirq":
        return get_probs_cirq(task_id, code, entry_point, NUMBER_OF_SHOTS, inputs)
    if framework == "pennylane":
        return get_probs_pennylane(task_id, code, entry_point, NUMBER_OF_SHOTS, inputs)
    if framework == "qiskit":
        return get_probs_qiskit(task_id, code, entry_point, NUMBER_OF_SHOTS, inputs)
    raise ValueError(f"Unknown framework '{framework}'. Expected one of: cirq|pennylane|qiskit")


def _compare_output(task_id: str, output: np.ndarray, canonical_by_task: dict[str, dict[str, Any]]) -> EvalResult:
    canonical_probs = canonical_by_task[task_id]["canonical_output"]
    try:
        if len(output) != len(canonical_probs):
            return EvalResult(
                compiled=True,
                ran=True,
                kl_div_bool=False,
                output=output,
                error=f"shape mismatch: model_probs len {len(output)},canonical_probs len {len(canonical_probs)}",
            )
        kl_value, passed = get_kl_div(probs=output, expected_probs=canonical_probs)
        return EvalResult(compiled=True, ran=True, kl_div_result=kl_value, kl_div_bool=passed, output=output)
    except Exception:
        return EvalResult(
            compiled=True,
            ran=True,
            kl_div_bool=False,
            output=output,
            error="Failed while comparing output to expected:\n" + traceback.format_exc(),
        )


def build_feedback_message(eval_res: EvalResult) -> str:
    if not eval_res.compiled:
        return (
            "Your previous code did not compile.\n\n"
            "Here is the error:\n"
            f"{eval_res.error}\n\n"
            "Please fix the code and respond with the FULL corrected code."
        )
    if not eval_res.ran:
        return (
            "Your code compiled but failed at runtime.\n\n"
            "Here is the error:\n"
            f"{eval_res.error}\n\n"
            "Please fix the issue and respond with the FULL corrected code."
        )
    return (
        "Your code ran, but the output does NOT match the expected canonical output.\n\n"
        f"Got (stringified): {str(eval_res.output)[:2000]}\n\n"
        "Please try again and respond with the FULL corrected code."
    )
