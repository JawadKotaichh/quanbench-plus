import pennylane as qml
import numpy as np


def task_6_input_pennylane():
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.RZ((25 * np.pi) / 54, wires=0)
        return qml.state()

    return circuit


def binary_array_to_decimal_pennylane(bits):
    """
    bits: list like [1, 0, 1] representing the binary number 101
    returns: decimal integer (here, 5)
    """
    value = 0
    for b in bits:
        if b not in (0, 1):
            raise ValueError("All elements must be 0 or 1")
        value = value * 2 + b
    return value
