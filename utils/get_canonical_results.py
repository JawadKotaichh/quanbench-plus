from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import json
import traceback
from datetime import datetime
from pathlib import Path

import networkx as nx

from utils.common import get_handler
from utils.serialization import save_json, to_jsonable

# To edit the inputs of the tasks edit the dictionary GLOBAL_INPUTS
TASK4_GRAPH = [[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]]

# Create the graph for task 4
G = nx.Graph()
G.add_edges_from(TASK4_GRAPH)


def task_6_input():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz((25 * np.pi) / 54, 0)
    return qc


GLOBAL_INPUTS = {
    "04": [
        G,
        [((25 * np.pi) / 54) for i in range(5)],
        [((25 * np.pi) / 54) for i in range(5)],
    ],
    "06": task_6_input(),
    "29": [1, 0],
    "39": [((25 * np.pi) / 54), ((25 * np.pi) / 54)],
    "40": [((25 * np.pi) / 54) for i in range(8)],
    "41": [((25 * np.pi) / 54) for i in range(8)],
    "42": [((25 * np.pi) / 54), ((25 * np.pi) / 54), ((25 * np.pi) / 54)],
}


NUMBER_OF_SHOTS = 1000


def get_probs_dictionnary_canonical_solution(qc, shots):
    qc = qc.copy()
    if not any(inst.operation.name == "measure" for inst in qc.data):
        qc.measure_all()

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    job = sim.run(compiled, shots=shots)
    result = job.result()
    return result.get_counts()


def counts_to_array(counts, outcomes=None, normalize=True):
    if isinstance(counts, list):
        counts = counts[0]
    if not isinstance(counts, dict):
        raise TypeError(f"Expected dict or list of dicts, got {type(counts)}")

    cleaned = {}
    for k, v in counts.items():
        clean_key = k.split()[0]
        cleaned[clean_key] = cleaned.get(clean_key, 0) + v
    counts = cleaned

    if outcomes is None:
        outcomes = sorted(counts.keys())

    n_bits = len(outcomes[0])
    all_outcomes = [format(i, f"0{n_bits}b") for i in range(2**n_bits)]

    arr = np.array([counts.get(k, 0) for k in all_outcomes], dtype=float)

    if normalize:
        total = arr.sum()
        if total > 0:
            arr /= total

    return arr



def get_probs_canonical_solution(
    task_id, canonical_solution, entry_point, shots, inputs
):
    circuit_or_counts = get_handler(task_id, canonical_solution, entry_point, inputs)
    if isinstance(circuit_or_counts, dict):
        counts = circuit_or_counts
    elif hasattr(circuit_or_counts, "name"):
        counts = get_probs_dictionnary_canonical_solution(circuit_or_counts, shots)
    else:
        raise TypeError(
            f"Expected QuantumCircuit or dict, got {type(circuit_or_counts)} instead."
        )
    return counts_to_array(counts)


def read_jsonl(file_path):
    out = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[read_jsonl] Skipping bad line: {e}")
    return out


def _timestamp():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def load_canonical_solutions(path: str = "./canonical_results/canonical_solutions.json"):
    """Load canonical solutions as a dict keyed by task_id."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["task_id"]): item for item in data}


def save_canonical_outputs_json(
    canonical_file: str,
    output_dir: str = "canonical_results",
    output_filename: str = None,
    include_errors_as_records: bool = True,
    inputss=None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / (
        output_filename or f"canonical_outputs_{_timestamp()}.json"
    )

    canonical_outputs = []
    failures = []
    successes = []

    data = read_jsonl(canonical_file)
    if not data:
        print(f"No data loaded from {canonical_file}")
        return [], None

    for i, task in enumerate(data):
        raw_task_id = str(task.get("task_id", f"task_{i + 1}")).zfill(2)
        print(f"\n--- Processing task {i + 1}/{len(data)}: task_id={raw_task_id} ---")

        try:
            if "canonical_solution" not in task:
                raise KeyError("Missing key: 'canonical_solution'")
            if "entry_point" not in task:
                raise KeyError("Missing key: 'entry_point'")

            canonical_output = get_probs_canonical_solution(
                raw_task_id,
                task["canonical_solution"],
                task["entry_point"],
                NUMBER_OF_SHOTS,
                inputss,
            )

            canonical_outputs.append(
                {
                    "task_id": raw_task_id,
                    "canonical_output": to_jsonable(canonical_output),
                }
            )
            successes.append(raw_task_id)

        except Exception as e:
            print(f"Error in task {raw_task_id}: {type(e).__name__}: {e}")
            tb_str = "".join(traceback.format_exc())
            if include_errors_as_records:
                canonical_outputs.append(
                    {
                        "task_id": raw_task_id,
                        "canonical_output": None,
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

    saved_path = save_json(canonical_outputs, out_path) if canonical_outputs else None

    print("\n=== Summary ===")
    print(f"Total tasks: {len(data)}")
    print(f"  ✓ Successes: {len(successes)}")
    print(f"  ✗ Failures:  {len(failures)}")
    if failures:
        for f in failures:
            print(f"  - {f['task_id']} ({f['type']}: {f['message']})")
    if saved_path:
        print(f"\n✅ Results saved to: {saved_path}")

    return canonical_outputs, saved_path


if __name__ == "__main__":
    save_canonical_outputs_json(
        canonical_file="./prompts/qiskit.jsonl",
        output_dir="canonical_results",
        output_filename="canonical_solutions.json",
        inputss=GLOBAL_INPUTS,
    )
