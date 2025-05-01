import pennylane as qml

def circuit_d(params, wires):
    """A representative entangling circuit with CRZ gates (Circuit-D)."""
    n_qubits = len(wires)
    for i in range(n_qubits):
        qml.RZ(params[i], wires=wires[i])
        qml.RY(params[i], wires=wires[i])
    for i in range(n_qubits - 1):
        qml.CRX(params[i + n_qubits], wires=[wires[i], wires[i + 1]])
    qml.CRZ(params[-1], wires=[wires[0], wires[-1]])
