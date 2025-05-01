import pennylane as qml
from pennylane import numpy as np

def quantum_relu_circuit(params, wires):
    """Implements a QReLU-style entangled activation function using tensor products."""
    for i, wire in enumerate(wires):
        qml.RY(params[i], wires=wire)
    for i in range(0, len(wires)-1, 2):
        qml.CNOT(wires=[wires[i], wires[i+1]])
