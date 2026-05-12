# QuanBench+ V2 Implementation Spec

V2 keeps the QuanBench+ task set, Pass@k bookkeeping, and feedback protocol intact. It changes the Qiskit grader from sampled KL against one canonical output to deterministic ideal-probability grading against a per-task `canonical_class`.

## Inputs

- `prompts/qiskit.jsonl` remains the v1 prompt source.
- `prompts/qiskit_v2.jsonl` is derived from v1 and adds:
  - `prompt_v2`: original prompt plus measurement and Qiskit bit-order clarifications.
  - `canonical_class`: grader type and thresholds.
- Existing model response JSON shape is preserved. V2 adds optional `benchmark_version`, `execution_metadata`, `canonical_class`, `grader_type`, and `grader_details`.

## Execution

Qiskit v2 execution uses statevector probabilities, not sampled counts.

1. Execute candidate code with the existing task harness.
2. If the result is a `QuantumCircuit`, remove measurements for simulation.
3. Build the statevector exactly.
4. Marginalize probabilities onto the measured classical bits, preserving Qiskit count-key order.
5. If no measurements exist, grade probabilities over all qubits.
6. For unitary-decomposition tasks, compare the candidate unitary to the target unitary up to global phase.

Returned count dictionaries are still supported, but they cannot be unitary-graded.

## Grader Types

- `exact_distribution`: KL to exact canonical probabilities, or unitary distance when `comparison="unitary"`.
- `deterministic_dominant`: dominant bitstring must be in an accepted equivalence class.
- `support_match`: observed support above `tau` must match expected support.
- `peak_match`: top-k peaks must match explicit peaks, accepted peak sets, or canonical peaks.
- `support_uniformity`: distribution must be uniform over a support set with little outside mass.
- `structural`: task-specific predicate, currently for `VQE_Z2`.

## CLI

Generate v2 prompt file:

```bash
PYENV_VERSION=3.12.11 python scripts/build_qiskit_v2_prompts.py
```

Run pass@k generation and scoring:

```bash
PYENV_VERSION=3.12.11 bash pass_at_k_pipeline/runner.sh \
  --framework qiskit \
  --benchmark-version v2 \
  --pass_k 5 \
  coda-local
```

Evaluate saved responses only:

```bash
PYENV_VERSION=3.12.11 python evaluate.py \
  --framework qiskit \
  --benchmark-version v2 \
  --pass_k 5 \
  coda-local
```

Audit canonical classes:

```bash
PYENV_VERSION=3.12.11 python scripts/audit_v2_canonicals.py
```

## Coda Local Harness

Model aliases `coda`, `coda-local`, and `coda/local` route to a local Coda agent server instead of OpenRouter.

Environment:

```bash
export CODA_AGENTS_URL=http://127.0.0.1:8000
export CODA_AGENTS_FAST=false
```

The client posts to `/agents`, reads SSE events, stops at the `completed` event, and extracts `structured_response.code`. This matches the Coda server contract where `/agents` streams tokens but terminates once `output_generation_node` emits structured code.

## Current Scope

V2 grading is implemented for Qiskit. Cirq and PennyLane remain on v1 until their framework-specific exact execution paths are added.
