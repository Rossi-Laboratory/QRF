import pennylane as qml

def apply_circuit_a(params, wires):
    for i in range(3):  # x, y, z
        qml.RX(params[i], wires=wires[i])
        qml.RY(params[i+3], wires=wires[i])
    qml.RX(params[6], wires=wires[3])  # depth
    qml.RX(params[7], wires=wires[4])  # theta
    qml.RX(params[8], wires=wires[5])  # phi
    qml.RY(params[9], wires=wires[4])
    qml.RY(params[10], wires=wires[5])
    qml.CNOT(wires=[wires[0], wires[3]])
    qml.CNOT(wires=[wires[1], wires[3]])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.CNOT(wires=[wires[3], wires[4]])
    qml.CNOT(wires=[wires[4], wires[5]])

def apply_circuit_b(params, wires):
    for i in range(3):
        qml.RX(params[i], wires=wires[i])
        qml.RY(params[i+3], wires=wires[i])
    qml.RX(params[6], wires=wires[3])
    qml.RX(params[7], wires=wires[4])
    qml.RX(params[8], wires=wires[5])
    qml.RY(params[9], wires=wires[4])
    qml.RY(params[10], wires=wires[5])
    qml.CZ(wires=[wires[0], wires[1]])
    qml.CZ(wires=[wires[1], wires[2]])
    qml.CZ(wires=[wires[2], wires[3]])
    qml.CZ(wires=[wires[3], wires[4]])
    qml.CZ(wires=[wires[4], wires[5]])
    for i in range(4):
        qml.RZ(params[11 + i], wires=wires[i])

def apply_circuit_c(params, wires):
    for i in range(6):
        qml.RX(params[i], wires=wires[i])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.CNOT(wires=[wires[3], wires[4]])
    qml.CNOT(wires=[wires[4], wires[5]])
    for i in range(6):
        qml.RY(params[6+i], wires=wires[i])
    qml.CNOT(wires=[wires[5], wires[4]])
    qml.CNOT(wires=[wires[4], wires[3]])
    qml.CNOT(wires=[wires[3], wires[2]])
    for i in range(4):
        qml.RZ(params[12+i], wires=wires[i])

def apply_circuit_d(params, wires):
    for i in range(6):
        qml.RX(params[i], wires=wires[i])
        qml.RY(params[6+i], wires=wires[i])
    qml.RZ(params[12], wires=wires[0])
    qml.RZ(params[13], wires=wires[1])
    qml.RZ(params[14], wires=wires[2])
    qml.CZ(wires=[wires[0], wires[3]])
    qml.CZ(wires=[wires[1], wires[4]])
    qml.CZ(wires=[wires[2], wires[5]])
    for i in range(4):
        qml.RZ(params[15+i], wires=wires[i])
