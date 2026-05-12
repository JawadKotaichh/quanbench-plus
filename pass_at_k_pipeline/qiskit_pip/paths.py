from pathlib import Path
import sys

qiskit = Path(__file__).resolve().parent
PASS_AT_K_DIR = qiskit.parents[0]
REPO_ROOT = qiskit.parents[1]
for path in (REPO_ROOT, PASS_AT_K_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

CANONICAL_SOLUTIONS_DIR = REPO_ROOT / "canonical_results" / "canonical_solutions.json"
QISKIT_JSONL = str(REPO_ROOT / "prompts" / "qiskit.jsonl")
QISKIT_V2_JSONL = str(REPO_ROOT / "prompts" / "qiskit_v2.jsonl")
RESPONSES_OUTPUT_DIR = REPO_ROOT / "responses" / "qiskit" / "pass_at_k_new_models"
RESULTS_OUTPUT_DIR = REPO_ROOT / "results" / "qiskit" / "pass_at_k_new_models"
MODEL_RESPONSES_DIR = REPO_ROOT / "model_responses" / "qiskit" / "pass_at_k_new_models"
PROMPTS_DIR = REPO_ROOT / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
RESPONSES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
