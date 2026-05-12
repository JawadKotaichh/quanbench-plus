from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graders.qiskit_v2_specs import write_qiskit_v2_jsonl  # noqa: E402


if __name__ == "__main__":
    print(write_qiskit_v2_jsonl())
