import pennylane as qml
from pennylane import numpy as np
from qrf.quantum_network import qrf_model
from qrf.quantum_circuit import circuit_d
from qrf.volume_rendering import accumulate_volume

def sample_ray_positions(n=10):
    return np.linspace(0.1, 1.0, n)

def simulate_3d_scene_rendering():
    n_qubits = 3
    dev = qml.device("default.qubit", wires=n_qubits)
    params = np.random.uniform(0, np.pi, size=(3*n_qubits,), requires_grad=True)

    for pixel_id in range(5):
        ray_samples = sample_ray_positions()
        sigmas = np.random.uniform(0.1, 1.0, len(ray_samples))
        colors = [qrf_model(np.random.rand(6), params, dev, circuit_fn=circuit_d) for _ in ray_samples]
        rgb = accumulate_volume(ray_samples, sigmas, colors, params)
        print(f"Pixel {pixel_id}: RGB = {rgb}")

if __name__ == "__main__":
    simulate_3d_scene_rendering()
