from __future__ import annotations

import argparse

from pass_at_k_pipeline.defaults import DEFAULT_MODELS
from pass_at_k_pipeline.qiskit_pip.get_qiskit_results import get_qiskit_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved QuanBench+ model responses.")
    parser.add_argument("models", nargs="*", help="Model names whose saved responses should be evaluated.")
    parser.add_argument("--framework", choices=["qiskit"], default="qiskit")
    parser.add_argument("--pass_k", type=int, default=1)
    parser.add_argument("--benchmark-version", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    get_qiskit_results(models=models, passk=args.pass_k, benchmark_version=args.benchmark_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
