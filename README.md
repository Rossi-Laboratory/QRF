# Quantum Radiance Fields (QRF)

A fully quantum-native framework for photorealistic neural rendering, using PennyLane and the Borealis quantum backend.

## 🌌 Key Modules

### `qrf/encoding.py`
Implements 4 major quantum encoding strategies:
- `angle_encoding`: One feature per qubit (RY rotation)
- `dense_angle_encoding`: Two features per qubit using RY + RZ
- `wavefunction_encoding`: Normalized vector state loading (Mottonen)
- `general_qubit_encoding`: RX, RY, RZ for 3-feature mapping

### `qrf/quantum_circuit.py`
Implements 4 parameterized quantum circuits used in the QRF paper:
- `circuit_a`: RX + RZ + CNOT chain
- `circuit_b`: RY + CRZ
- `circuit_c`: RY + CZ
- `circuit_d`: RZ + RY + CRX + CRZ (default)

### `qrf/qrelu.py`
Quantum ReLU activation function using entanglement-based tensor product representations of ReLU and LeakyReLU.

### `qrf/quantum_network.py`
Builds the QRF pipeline from encoding → PQC → activation → measurement.
Supports flexible encoding and circuit choices.

### `qrf/volume_rendering.py`
Simulates Grover-based volume rendering:
- Oracle function to simulate ray energy
- Grover-like selection of high-energy rays
- Accumulation for rendering color

### `qrf/train.py`
Training loop using Adam optimizer and PennyLane's autograd support.

---

## 📁 Experiments

### `2D_Image_Regression.py`
Reproduces QNN regression results on synthetic 2D → RGB mappings (Fig. 6).

### `3D_Scene_Reconstruction.py`
Renders photorealistic 3D views using quantum-native NeRF architecture.

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python experiments/2D_Image_Regression.py
python experiments/3D_Scene_Reconstruction.py
```

---

## 🧠 Backend

This implementation defaults to the `default.qubit` device for simulation. To use the **Borealis** backend, modify the device configuration in `train.py` or experiment scripts.

---

© 2025 YuanFu Yang. All rights reserved.
