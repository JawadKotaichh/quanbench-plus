from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import traceback

from graders.contracts import ExecutionResult
from utils.common import (
    _extract_token_fields,
    _to_jsonable,
    add_header_if_missing,
    load_prompts_jsonl_as_dict,
    normalize_task_id,
    save_json,
)


V1Executor = Callable[[str, str, str, int, dict[str, Any]], Any]
V2Executor = Callable[..., ExecutionResult]


@dataclass(frozen=True)
class SaveResponsesConfig:
    framework: str
    model_responses_dir: Path
    prompts_path: Path
    v2_prompts_path: Path
    shots: int
    v1_executor: V1Executor
    v2_executor: V2Executor


def _resolve_model_responses_path(file_path: Path, model_responses_dir: Path) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        return path

    parts = path.parts
    if parts and parts[0] == "model_responses":
        parts = parts[1:]
    return model_responses_dir.joinpath(*parts)


def read_model_responses(file_path: Path, model_responses_dir: Path) -> list[dict[str, Any]]:
    resolved_path = _resolve_model_responses_path(file_path, model_responses_dir)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model responses file not found: {resolved_path}\n"
            f"Expected location: {resolved_path}\n"
            f"Please ensure the file exists or run api.py first to generate it."
        )
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def _execute_task(
    task: dict[str, Any],
    *,
    task_id: str,
    code: str,
    config: SaveResponsesConfig,
    inputs: dict[str, Any],
    benchmark_version: str,
) -> tuple[Any, dict[str, Any] | None]:
    if benchmark_version == "v2":
        execution = config.v2_executor(
            task_id=task_id,
            code=code,
            entry_point=task["entry_point"],
            inputs=inputs,
        )
        return execution.probabilities, execution.metadata

    output = config.v1_executor(
        task["task_id"],
        code,
        task["entry_point"],
        config.shots,
        inputs,
    )
    return output, None


def _success_record(
    task: dict[str, Any],
    *,
    code: str,
    output: Any,
    execution_metadata: dict[str, Any] | None,
    benchmark_version: str,
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "entry_point": task.get("entry_point"),
        "category": task.get("category"),
        "version": task.get("version"),
        **_extract_token_fields(task),
        "code": code,
        "output": _to_jsonable(output),
        "benchmark_version": benchmark_version,
        "execution_metadata": execution_metadata,
    }


def _error_record(task: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "category": task.get("category"),
        "version": task.get("version"),
        **_extract_token_fields(task),
        "output": None,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "stacktrace": "".join(traceback.format_exc())[-4000:],
        },
    }


def _process_task(
    task: dict[str, Any],
    *,
    prompts: dict[str, dict[str, Any]],
    config: SaveResponsesConfig,
    inputs: dict[str, Any],
    benchmark_version: str,
) -> dict[str, Any]:
    if "code" not in task:
        raise KeyError("Missing key: 'code'")
    if "entry_point" not in task:
        raise KeyError("Missing key: 'entry_point'")

    task_id = normalize_task_id(task["task_id"])
    code = add_header_if_missing(task["code"], prompts.get(task_id, {}).get("header", ""))
    output, metadata = _execute_task(
        task,
        task_id=task_id,
        code=code,
        config=config,
        inputs=inputs,
        benchmark_version=benchmark_version,
    )
    return _success_record(
        task,
        code=code,
        output=output,
        execution_metadata=metadata,
        benchmark_version=benchmark_version,
    )


def save_framework_responses(
    *,
    config: SaveResponsesConfig,
    file: Path,
    response_path: Path,
    output_dir: Path = Path("./model_results"),
    inputs: dict[str, Any] | None = None,
    benchmark_version: str = "v1",
) -> tuple[list[dict[str, Any]], str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = read_model_responses(file, config.model_responses_dir)
    prompts_path = config.v2_prompts_path if benchmark_version == "v2" else config.prompts_path
    prompts = load_prompts_jsonl_as_dict(prompts_path)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, task in enumerate(data, start=1):
        task_id = task.get("task_id")
        print(f"\n--- Processing task {index}/{len(data)}: task_id={task_id} ---")
        try:
            records.append(
                _process_task(
                    task,
                    prompts=prompts,
                    config=config,
                    inputs=inputs or {},
                    benchmark_version=benchmark_version,
                )
            )
        except Exception as exc:
            print(f"Error in task {task_id}: {type(exc).__name__}: {exc}")
            records.append(_error_record(task, exc))
            failures.append({"task_id": str(task_id), "type": type(exc).__name__, "message": str(exc)})

    saved_path = save_json(records, response_path) if records else None
    print("\n=== Summary ===")
    print(f"Total tasks: {len(data)}")
    print(f"  Successes: {len(data) - len(failures)}")
    print(f"  Failures:  {len(failures)}")
    for failure in failures:
        print(f"  - {failure['task_id']} ({failure['type']}: {failure['message']})")
    if saved_path:
        print(f"\nResults saved to: {saved_path}")
    return records, saved_path
