from __future__ import annotations

from pathlib import Path
from typing import Any

import cirq
import numpy as np

from graders.framework_v2 import FrameworkV2Evaluator, load_framework_v2_tasks
from pass_at_k_pipeline.cirq_pip.paths import (
    CANONICAL_SOLUTIONS_DIR,
    MODEL_RESPONSES_DIR,
    RESPONSES_OUTPUT_DIR,
    RESULTS_OUTPUT_DIR,
)
from pass_at_k_pipeline.cirq_pip.save_cirq_responses import save_cirq_responses
from pass_at_k_pipeline.defaults import DEFAULT_MODELS, GLOBAL_INPUTS
from pass_at_k_pipeline.results import evaluate_model_responses
from utils.evaluation_summary import print_evaluation_summary


def task_6_input():
    q = cirq.LineQubit(0)
    return cirq.Circuit(cirq.H(q), cirq.rz((25 * np.pi) / 54)(q))


GLOBAL_INPUTS["06"] = task_6_input()


def build_paths(models: list[str]) -> tuple[list[tuple[Path, str]], list[Path], list[Path]]:
    model_names = [name.replace("/", "_") for name in models]
    model_response_paths = [(MODEL_RESPONSES_DIR / f"{name}_cirq.json", name) for name in model_names]
    response_paths = [RESPONSES_OUTPUT_DIR / f"{name}.json" for name in model_names]
    result_paths = [RESULTS_OUTPUT_DIR / f"{name}.json" for name in model_names]
    return model_response_paths, response_paths, result_paths


def main(
    model_responses_path: Path,
    model_name: str,
    response_path: Path,
    result_path: Path,
    canonical_solutions_path: Path,
    global_inputs: dict[str, Any],
    benchmark_version: str = "v1",
) -> list[dict[str, Any]]:
    v2_tasks = load_framework_v2_tasks("cirq") if benchmark_version == "v2" else {}
    evaluator = FrameworkV2Evaluator("cirq", v2_tasks, global_inputs) if benchmark_version == "v2" else None
    return evaluate_model_responses(
        model_responses_path=model_responses_path,
        model_name=model_name,
        response_path=response_path,
        result_path=result_path,
        canonical_solutions_path=canonical_solutions_path,
        global_inputs=global_inputs,
        benchmark_version=benchmark_version,
        save_responses=save_cirq_responses,
        v2_tasks=v2_tasks,
        v2_evaluator=evaluator,
    )


def get_cirq_results(models: list[str], passk: int, benchmark_version: str = "v1") -> None:
    RESPONSES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_response_paths, response_paths, result_paths = build_paths(models)
    all_results: dict[str, list[dict[str, Any]]] = {}

    for (model_response_path, model_name), response_path, result_path in zip(
        model_response_paths, response_paths, result_paths
    ):
        all_results[model_name] = main(
            model_responses_path=model_response_path,
            model_name=model_name,
            response_path=response_path,
            result_path=result_path,
            canonical_solutions_path=CANONICAL_SOLUTIONS_DIR,
            global_inputs=GLOBAL_INPUTS,
            benchmark_version=benchmark_version,
        )

    print_evaluation_summary(all_results=all_results, passk=passk)


def parse_cli_args(argv: list[str]) -> tuple[list[str], int, str]:
    if not argv:
        return DEFAULT_MODELS, 1, "v1"
    benchmark_version = "v1"
    args = list(argv)
    if "--benchmark-version" in args:
        idx = args.index("--benchmark-version")
        benchmark_version = args[idx + 1]
        del args[idx : idx + 2]
    return args[:-1] or DEFAULT_MODELS, int(args[-1]), benchmark_version


if __name__ == "__main__":
    import sys

    cli_models, cli_passk, cli_benchmark_version = parse_cli_args(sys.argv[1:])
    get_cirq_results(cli_models, passk=cli_passk, benchmark_version=cli_benchmark_version)
