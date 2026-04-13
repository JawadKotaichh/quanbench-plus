import cirq
import numpy as np
import json
import traceback
from pathlib import Path
from pass_at_k_pipeline.cirq_pip.paths import MODEL_RESPONSES_DIR, CIRQ_JSONL
from pass_at_k_pipeline.defaults import NUMBER_OF_SHOTS
from utils.common import (
    normalize_task_id,
    load_prompts_jsonl_as_dict,
    add_header_if_missing,
    get_handler,
    _to_jsonable,
    _extract_token_fields,
    save_json,
)


def get_probs_dictionnary(circuit, shots):
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


def counts_to_array(counts, outcomes=None, normalize=True):
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


def _resolve_model_responses_path(file_path):
    path = Path(file_path)
    if path.is_absolute():
        return path

    parts = path.parts
    if parts and parts[0] == "model_responses":
        parts = parts[1:]

    return MODEL_RESPONSES_DIR.joinpath(*parts)


def read_json(file_path):
    resolved_path = _resolve_model_responses_path(file_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model responses file not found: {resolved_path}\n"
            f"Expected location: {resolved_path}\n"
            f"Please ensure the file exists or run api.py first to generate it."
        )
    out = []
    with open(resolved_path, "r", encoding="utf-8") as file:
        out = json.load(file)
    return out


def save_cirq_responses(
    file: Path,
    response_path: Path,
    output_dir: Path = Path("./model_results"),
    inputss=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = response_path
    include_errors_as_records = True
    outputs = []
    failures = []
    successes = []

    data = read_json(file)
    prompts = load_prompts_jsonl_as_dict(Path(CIRQ_JSONL))

    if not data:
        print(f"!! No data loaded from {file}")
        return [], None

    for i, task in enumerate(data):
        raw_task_id = task["task_id"]
        print(f"\n--- Processing task {i + 1}/{len(data)}: task_id={raw_task_id} ---")

        try:
            if "code" not in task:
                raise KeyError("Missing key: 'code'")
            if "entry_point" not in task:
                raise KeyError("Missing key: 'entry_point'")
            tid = normalize_task_id(raw_task_id)
            prompt_header = prompts.get(tid, {}).get("header", "")
            task["code"] = add_header_if_missing(task["code"], prompt_header)
            output = get_probs(
                raw_task_id,
                task["code"],
                task["entry_point"],
                NUMBER_OF_SHOTS,
                inputss,
            )

            outputs.append(
                {
                    "task_id": raw_task_id,
                    "category": task.get("category"),
                    "version": task.get("version"),
                    **_extract_token_fields(task),
                    "output": _to_jsonable(output),
                }
            )
            successes.append(raw_task_id)

        except Exception as e:
            print(f"!! Error in task {raw_task_id}: {type(e).__name__}: {e}")
            tb_str = "".join(traceback.format_exc())
            if include_errors_as_records:
                outputs.append(
                    {
                        "task_id": raw_task_id,
                        "category": task.get("category"),
                        "version": task.get("version"),
                        **_extract_token_fields(task),
                        "output": None,
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "stacktrace": tb_str[-4000:],
                        },
                    }
                )
            failures.append(
                {"task_id": raw_task_id, "type": type(e).__name__, "message": str(e)}
            )

    saved_path = save_json(outputs, out_path) if outputs else None

    print("\n=== Summary ===")
    print(f"Total tasks: {len(data)}")
    print(f"  ✓ Successes: {len(successes)}")
    print(f"  ✗ Failures:  {len(failures)}")
    if failures:
        for f in failures:
            print(f"  - {f['task_id']} ({f['type']}: {f['message']})")
    if saved_path:
        print(f"\n✅ Results saved to: {saved_path}")

    return outputs, saved_path
