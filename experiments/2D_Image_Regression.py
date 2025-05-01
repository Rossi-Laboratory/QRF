import pennylane as qml
from pennylane import numpy as np
from qrf.quantum_network import qrf_model
from qrf.quantum_circuit import circuit_d

def generate_dataset(n_samples=50):
    # Synthetic: 2D (x, y) → RGB values in [0, 1]
    x_vals = np.random.uniform(0, 1, size=(n_samples, 2))
    y_vals = np.stack([x_vals[:, 0], x_vals[:, 1], 1 - x_vals[:, 0]], axis=1)
    return list(zip(x_vals, y_vals))

def train_2d_image_regression():
    n_qubits = 3
    dev = qml.device("default.qubit", wires=n_qubits)
    dataset = generate_dataset(100)
    params = np.random.uniform(0, np.pi, size=(3*n_qubits,), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)

    for step in range(50):
        loss = 0
        for x, y in dataset:
            x_input = np.concatenate([x, x])
            def cost(p):
                out = qrf_model(x_input, p, dev, circuit_fn=circuit_d)
                return np.sum((out - y[:len(out)])**2)
            params = opt.step(cost, params)
            loss += cost(params)
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss/len(dataset):.4f}")

if __name__ == "__main__":
    train_2d_image_regression()
