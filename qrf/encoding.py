import pennylane as qml
from pennylane import numpy as np

def angle_encoding(features, wires):
    """Angle encoding: encode one feature per qubit using RY."""
    for x, w in zip(features, wires):
        qml.RY(x, wires=w)

def dense_angle_encoding(features, wires):
    """Dense angle encoding: encode two features per qubit using RY and RZ."""
    assert len(features) == 2 * len(wires), "Each qubit requires two features."
    for i, wire in enumerate(wires):
        theta = np.pi * features[2 * i]
        phi = 2 * np.pi * features[2 * i + 1]
        qml.RY(theta, wires=wire)
        qml.RZ(phi, wires=wire)

def wavefunction_encoding(features, wires):
    """Wavefunction encoding (simplified version): prepare a basis state."""
    assert len(features) <= 2 ** len(wires), "Too many features for given qubits."
    norm = np.linalg.norm(features)
    state = np.array(features) / norm
    qml.MottonenStatePreparation(state, wires=wires)

def general_qubit_encoding(features, wires):
    """General encoding using RX, RY, RZ."""
    assert len(features) == 3 * len(wires), "Requires 3 features per qubit."
    for i, wire in enumerate(wires):
        qml.RX(features[3*i], wires=wire)
        qml.RY(features[3*i + 1], wires=wire)
        qml.RZ(features[3*i + 2], wires=wire)
