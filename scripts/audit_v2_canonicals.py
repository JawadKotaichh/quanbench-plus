from __future__ import annotations

import argparse
import json
import signal
import traceback
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graders.qiskit_execution import execute_qiskit_task  # noqa: E402
from graders.qiskit_v2_specs import QiskitV2Evaluator, load_qiskit_v2_tasks  # noqa: E402
from utils.get_canonical_results import GLOBAL_INPUTS, task_6_input  # noqa: E402


GLOBAL_INPUTS["06"] = task_6_input()


class AuditTimeout(Exception):
    pass


def _raise_timeout(_signum, _frame):
    raise AuditTimeout("task audit timed out")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Qiskit v2 canonical classes.")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("prompts/qiskit_v2.jsonl"),
        help="Qiskit v2 prompt JSONL.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=30)
    args = parser.parse_args()

    tasks = load_qiskit_v2_tasks(args.prompts)
    evaluator = QiskitV2Evaluator(tasks, GLOBAL_INPUTS)
    records = []

    for task_id in sorted(tasks):
        task = tasks[task_id]
        try:
            signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(args.timeout_seconds)
            execution = execute_qiskit_task(
                task_id=task_id,
                code=task["canonical_solution"],
                entry_point=task["entry_point"],
                inputs=GLOBAL_INPUTS,
            )
            details = evaluator.grade_execution(
                task_id=task_id,
                execution=execution,
                code=task["canonical_solution"],
            )
            records.append(
                {
                    "task_id": task_id,
                    "entry_point": task["entry_point"],
                    "grader_type": task["canonical_class"]["type"],
                    "passed": bool(details["passed"]),
                    "details": details,
                }
            )
            signal.alarm(0)
        except Exception as exc:
            signal.alarm(0)
            records.append(
                {
                    "task_id": task_id,
                    "entry_point": task["entry_point"],
                    "grader_type": task["canonical_class"]["type"],
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                }
            )

    print(json.dumps(records, indent=2, ensure_ascii=False))
    failed = [record["task_id"] for record in records if not record["passed"]]
    if failed:
        print(f"\nKnown canonical mismatches or audit failures: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
