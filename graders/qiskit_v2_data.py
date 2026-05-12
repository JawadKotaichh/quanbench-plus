QISKIT_V2_PROMPT_NOTE = """

QuanBench+ v2 grading note:
- Preserve the exact function signature and return a Qiskit QuantumCircuit.
- Use Qiskit's native count-key convention: qubit 0 is the least-significant bit.
- Measure exactly the register requested by the prompt. Do not measure ancillas unless explicitly requested.
- If the prompt says "without measure", return an unmeasured circuit.
"""


QISKIT_V2_CANONICAL_SOLUTION_OVERRIDES: dict[str, str] = {
    "03": """from qiskit import QuantumCircuit

def grover_3SAT() -> QuantumCircuit:
    qc = QuantumCircuit(3, 3)
    qc.x(1)
    qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc
""",
}
