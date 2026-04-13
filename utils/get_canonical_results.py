from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.exceptions import QiskitError
import numpy as np
import json
import traceback
from datetime import datetime
from pathlib import Path

import networkx as nx

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


def execute_code_with_args(
    code: str, entry_point: str, arg1=None, arg2=None, arg3=None
):
    """Execute the canonical solution code and run the entry point safely with up to three arguments."""
    ns = {}
    try:
        exec(code, ns, ns)
    except SyntaxError as e:
        print(f"\n[SyntaxError] line={e.lineno}, msg={e.msg}")
        print("----- CODE CAUSING ERROR -----")
        print(code)
        print("------------------------------")
        raise

    func = ns.get(entry_point)
    if not callable(func):
        raise RuntimeError(f"Entry point '{entry_point}' not found or not callable.")

    args = [x for x in (arg1, arg2, arg3) if x is not None]
    return func(*args)


def handle_task_04(code, entry_point, graph, betta, gamma):
    return execute_code_with_args(code, entry_point, graph, betta, gamma)


def handle_task_06(code, entry_point, gate_list):
    return execute_code_with_args(code, entry_point, gate_list)


def handle_task_29(code, entry_point, alice, bob):
    return execute_code_with_args(code, entry_point, alice, bob)


def handle_task_39(code, entry_point, array):
    return execute_code_with_args(code, entry_point, array)


def handle_task_40(code, entry_point, params):
    return execute_code_with_args(code, entry_point, params)


def handle_task_41(code, entry_point, params):
    return execute_code_with_args(code, entry_point, params)


def handle_task_42(code, entry_point, theta, phi, lam):
    return execute_code_with_args(code, entry_point, theta, phi, lam)


def get_handler(x, code: str, entry_point: str, inputs: dict):
    handlers = {
        "04": lambda: handle_task_04(
            code, entry_point, inputs[x][0], inputs[x][1], inputs[x][2]
        ),
        "06": lambda: handle_task_06(code, entry_point, inputs[x]),
        "29": lambda: handle_task_29(code, entry_point, inputs[x][0], inputs[x][1]),
        "39": lambda: handle_task_39(code, entry_point, inputs[x]),
        "40": lambda: handle_task_40(code, entry_point, inputs[x]),
        "41": lambda: handle_task_41(code, entry_point, inputs[x]),
        "42": lambda: handle_task_42(
            code, entry_point, inputs[x][0], inputs[x][1], inputs[x][2]
        ),
    }

    func = handlers.get(x, lambda: execute_code_with_args(code, entry_point))
    return func()


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


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def save_json(records, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return str(out_path.resolve())


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
                    "canonical_output": _to_jsonable(canonical_output),
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
