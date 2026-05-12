from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from pass_at_k_pipeline.api_client import parse_requests, process_requests_pass_k
from pass_at_k_pipeline.cirq_pip.paths import CIRQ_JSONL, CIRQ_V2_JSONL
from pass_at_k_pipeline.cirq_pip.paths import MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_CIRQ
from pass_at_k_pipeline.defaults import DEFAULT_MODELS
from pass_at_k_pipeline.pennylane_pip.paths import MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_PENNYLANE
from pass_at_k_pipeline.pennylane_pip.paths import PENNYLANE_JSONL, PENNYLANE_V2_JSONL
from pass_at_k_pipeline.qiskit_pip.paths import MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_QISKIT
from pass_at_k_pipeline.qiskit_pip.paths import QISKIT_JSONL, QISKIT_V2_JSONL

load_dotenv()


PROMPT_PATHS = {
    ("cirq", "v1"): CIRQ_JSONL,
    ("cirq", "v2"): CIRQ_V2_JSONL,
    ("pennylane", "v1"): PENNYLANE_JSONL,
    ("pennylane", "v2"): PENNYLANE_V2_JSONL,
    ("qiskit", "v1"): QISKIT_JSONL,
    ("qiskit", "v2"): QISKIT_V2_JSONL,
}

MODEL_RESPONSE_DIRS = {
    "cirq": MODEL_RESPONSES_DIR_CIRQ,
    "pennylane": MODEL_RESPONSES_DIR_PENNYLANE,
    "qiskit": MODEL_RESPONSES_DIR_QISKIT,
}


def get_jsonl_path(framework: str, benchmark_version: str = "v1"):
    return PROMPT_PATHS[(framework, benchmark_version)]


def get_model_reponses_dir(framework: str) -> Path:
    return MODEL_RESPONSE_DIRS[framework]


def save_model_results(results: list[dict], models: list[str], framework: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        model_name = model.replace("/", "_")
        model_results = [result for result in results if result.get("model") == model]
        output_file = output_dir / f"{model_name}_{framework}.json"
        output_file.write_text(json.dumps(model_results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   Saved {len(model_results)} results to {output_file}")


def main(models: list[str], framework: str, pass_k: int = 1, benchmark_version: str = "v1") -> None:
    jsonl_path = get_jsonl_path(framework=framework, benchmark_version=benchmark_version)
    model_response_dir = get_model_reponses_dir(framework=framework)
    print("Starting API requests...")
    requests, tasks_info = parse_requests(jsonl_path, models, benchmark_version=benchmark_version)

    print(f"   Generated {len(requests)} requests across {len(models)} models")
    results = process_requests_pass_k(requests, tasks_info, pass_k)
    print(f"Completed {len(results)} responses")

    print(f"Saving results to file: {model_response_dir}...")
    save_model_results(results, models, framework, model_response_dir)
    print(f"All done! Saved {len(results)} total responses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM evaluation on quantum computing tasks")
    parser.add_argument("models", nargs="*", help="Model names to evaluate")
    parser.add_argument(
        "--framework",
        type=str,
        default="cirq",
        choices=["cirq", "pennylane", "qiskit"],
        help="Framework to use: cirq, pennylane, or qiskit (default: cirq)",
    )
    parser.add_argument("--pass_k", type=int, default=1, help="Number of pass@k samples")
    parser.add_argument("--benchmark-version", choices=["v1", "v2"], default="v1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.models or DEFAULT_MODELS, args.framework, args.pass_k, args.benchmark_version)
