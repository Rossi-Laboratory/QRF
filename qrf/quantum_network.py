import pennylane as qml
from qrf.encoding import (
    angle_encoding,
    dense_angle_encoding,
    wavefunction_encoding,
    general_qubit_encoding
)
from qrf.quantum_circuit import circuit_d  # default
from qrf.qrelu import quantum_relu_circuit

ENCODING_FUNCTIONS = {
    'angle': angle_encoding,
    'dense_angle': dense_angle_encoding,
    'wavefunction': wavefunction_encoding,
    'general': general_qubit_encoding
}

def qrf_model(x, params, dev, circuit_fn=circuit_d, encoding_type='dense_angle'):
    n_qubits = dev.num_wires
    encode_fn = ENCODING_FUNCTIONS[encoding_type]
    @qml.qnode(dev, interface="autograd")
    def circuit():
        encode_fn(x, wires=range(n_qubits))
        circuit_fn(params[:2*n_qubits+1], wires=range(n_qubits))
        quantum_relu_circuit(params[2*n_qubits+1:], wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
    return circuit()
