from typing import Any, Dict, List

from utils.pass_at_k import pass_at_k


def print_evaluation_summary(
    all_results: Dict[str, List[Dict[str, Any]]], passk: int
) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for model_name, results in all_results.items():
        total_tasks = len(results)
        if total_tasks == 0:
            continue

        # any_compiled is the pass@k aggregate for compilation success
        compiled = sum(1 for r in results if r.get("any_compiled", False))

        # pass@k accuracy uses the unbiased estimator per task
        pass_k_estimates = []
        for r in results:
            n = int(r.get("pass_k", passk))
            c = int(r.get("passed_versions", 0))
            pass_k_estimates.append(pass_at_k(n=n, c=c, k=passk))

        expected_passed = sum(pass_k_estimates)

        compile_rate = (compiled / total_tasks) * 100
        accuracy = (expected_passed / total_tasks) * 100

        print(f"\nModel: {model_name}     pass@{passk}")
        print("-" * 50)
        print(f"{'Total Tasks:':<20} {total_tasks}")
        print(f"{'Compiled@k:':<20} {compiled}/{total_tasks} ({compile_rate:.1f}%)")
        print(
            f"{'Pass@k (est):':<20} "
            f"{expected_passed:.2f}/{total_tasks} ({accuracy:.1f}%)"
        )
        print("-" * 50)

    print("=" * 60 + "\n")
