import numpy as np
import json
import traceback
from pathlib import Path
from pass_at_k_pipeline.defaults import NUMBER_OF_SHOTS
from pass_at_k_pipeline.pennylane_pip.paths import MODEL_RESPONSES_DIR, PENNYLANE_JSONL
from utils.common import (
    normalize_task_id,
    load_prompts_jsonl_as_dict,
    add_header_if_missing,
    get_handler,
    _to_jsonable,
    save_json,
)


def _resolve_model_responses_path(file_path):
    path = Path(file_path)
    if path.is_absolute():
        return path

    parts = path.parts
    if parts and parts[0] == "model_responses":
        parts = parts[1:]

    return MODEL_RESPONSES_DIR.joinpath(*parts)


def binary_array_to_decimal(bits):
    """
    bits: list like [1, 0, 1] representing the binary number 101
    returns: decimal integer (here, 5)
    """
    value = 0
    for b in bits:
        # optional: basic validation
        if b not in (0, 1):
            raise ValueError("All elements must be 0 or 1")
        value = value * 2 + b
    return value


def get_probs(task_id, solution, entry_point, shots, inputs):
    circuit_or_counts = get_handler(task_id, solution, entry_point, inputs)
    if isinstance(circuit_or_counts, np.ndarray):
        batta = circuit_or_counts.tolist()
        if type(batta[0]) is list:
            counts = [0] * (2 ** len(batta[0]))
            for sample in batta:
                counts[binary_array_to_decimal(sample)] += 1
            for j in range(len(counts)):
                counts[j] /= len(batta)
        elif type(batta[0]) is float:
            raise TypeError(
                "Model return expected value or sampled on a specified basis, wrong return type"
            )
        else:
            counts = [0, 0]
            for i in range(len(batta)):
                if batta[i] > 0:
                    counts[1] += 1
                else:
                    counts[0] += 1
            counts[0] /= len(batta)
            counts[1] /= len(batta)

    else:
        raise TypeError(f"Expected numpy array, got {type(circuit_or_counts)} instead.")
    return np.array(counts)


def read_json(file_path):
    resolved_path = _resolve_model_responses_path(file_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model responses file not found: {resolved_path}\n"
            f"Expected location: {resolved_path}\n"
            f"Please ensure the file exists or run the api.py first to generate it."
        )
    out = []
    with open(resolved_path, "r", encoding="utf-8") as file:
        out = json.load(file)
    return out


def _extract_token_fields(task):
    return {
        "prompt_tokens": task.get("prompt_tokens"),
        "completion_tokens": task.get("completion_tokens"),
        "total_tokens": task.get("total_tokens"),
        "reasoning_tokens": task.get("reasoning_tokens"),
        "accepted_prediction_tokens": task.get("accepted_prediction_tokens"),
        "rejected_prediction_tokens": task.get("rejected_prediction_tokens"),
        "cached_tokens": task.get("cached_tokens"),
        "cache_write_tokens": task.get("cache_write_tokens"),
    }


def save_pennylane_responses(
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
    prompts = load_prompts_jsonl_as_dict(Path(PENNYLANE_JSONL))

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
