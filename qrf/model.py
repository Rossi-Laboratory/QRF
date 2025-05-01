import pennylane as qml
from pennylane import numpy as np
from .quantum_circuit import apply_circuit_a, apply_circuit_b, apply_circuit_c, apply_circuit_d
from .encoding import (
    angle_encoding, dense_angle_encoding,
    wavefunction_encoding, general_qubit_encoding
)
from .quantum_network import qrf_model
from .volume_rendering import sample_ray, render_one_ray, accumulate_volume_rgb

CIRCUITS = {
    "circuit_a": apply_circuit_a,
    "circuit_b": apply_circuit_b,
    "circuit_c": apply_circuit_c,
    "circuit_d": apply_circuit_d
}

ENCODINGS = {
    "angle": angle_encoding,
    "dense_angle": dense_angle_encoding,
    "wavefunction": wavefunction_encoding,
    "general": general_qubit_encoding
}

class QRFForward:
    """
    QRFForward encapsulates the full QRF pipeline:
    - Encoding (positional + view)
    - Quantum circuit application
    - Quantum activation & measurement
    - Volume rendering integration
    """
    def __init__(self, circuit="circuit_a", encoding="dense_angle", output_map=None, device=None):
        self.circuit_fn = CIRCUITS[circuit]
        self.encoding_type = encoding
        # Default mapping: qubit 0->R,1->G,2->B,3->sigma
        self.output_map = output_map or {"R": [0], "G": [1], "B": [2], "sigma": [3]}
        self.device = device or qml.device("default.qubit", wires=6)
        self.params = None

    def set_params(self, params):
        """Set the model parameters."""
        self.params = params

    def encode_input(self, xyz, view_dir, delta):
        """Combine inputs into a single feature vector."""
        return np.concatenate([xyz, [delta], view_dir])

    def infer_point(self, xyz, view_dir, delta):
        """Quantum inference for a single point."""
        assert self.params is not None, "Parameters not set."
        input_vec = self.encode_input(xyz, view_dir, delta)
        return qrf_model(input_vec, self.params, self.device,
                         circuit_fn=self.circuit_fn, encoding_type=self.encoding_type)

    def render_ray(self, origin, direction, n_samples=32, near=0.1, far=1.0, visualize_sigma=False):
        """Full NeRF-style rendering for one ray."""
        points = sample_ray(origin, direction, n_samples, near, far)
        t_vals = np.linspace(near, far, n_samples)
        deltas = np.diff(t_vals, prepend=t_vals[0])
        rgb = accumulate_volume_rgb(
            points, direction[:2], deltas, self.params,
            self.device, self.circuit_fn, self.encoding_type,
            visualize_sigma=visualize_sigma
        )
        return rgb
