from pathlib import Path
import sys

FRAMEWORK_PATHS = Path(__file__).resolve().parent
FEEDBACK_LOOP_DIR = FRAMEWORK_PATHS.parents[0]
REPO_ROOT = FRAMEWORK_PATHS.parents[1]

for path in (REPO_ROOT, FEEDBACK_LOOP_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

CANONICAL_SOLUTIONS_DIR = REPO_ROOT / "canonical_results" / "canonical_solutions.json"
QISKIT_JSONL = str(REPO_ROOT / "prompts" / "qiskit.jsonl")
RESPONSES_OUTPUT_DIR = REPO_ROOT / "responses" / "qiskit" / "feedback_loop"
RESULTS_OUTPUT_DIR = REPO_ROOT / "results" / "qiskit" / "feedback_loop"
MODEL_RESPONSES_DIR = REPO_ROOT / "model_responses" / "qiskit" / "feedback_loop"
PROMPTS_DIR = REPO_ROOT / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
RESPONSES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
