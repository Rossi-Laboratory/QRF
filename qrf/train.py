import pennylane as qml
from pennylane import numpy as np
from qrf.quantum_network import qrf_model
from qrf.quantum_circuit import apply_circuit_a, apply_circuit_b, apply_circuit_c, apply_circuit_d
from qrf.volume_rendering import render_one_ray
import os
import argparse
import json
from datetime import datetime

CIRCUITS = {
    "circuit_a": apply_circuit_a,
    "circuit_b": apply_circuit_b,
    "circuit_c": apply_circuit_c,
    "circuit_d": apply_circuit_d
}

def log_metrics(log_path, metrics):
    with open(log_path, 'a') as f:
        f.write(json.dumps(metrics) + '\n')

def save_checkpoint(params, path):
    np.savez(path, params=params)

def train_2d_image_regression(config):
    n_qubits = config["n_qubits"]
    dev = qml.device("default.qubit", wires=n_qubits)
    circuit_fn = CIRCUITS[config["circuit"]]
    encoding_type = config["encoding"]
    n_epochs = config["epochs"]
    stepsize = config["lr"]
    save_path = config["save_path"]

    params = np.random.uniform(0, np.pi, size=(20,), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=stepsize)

    def gen_2d_data(n=100):
        x = np.random.rand(n, 2)
        y = np.stack([x[:, 0], x[:, 1], 1 - x[:, 0]], axis=1)
        return list(zip(x, y))

    data = gen_2d_data(config["batch_size"])
    log_path = os.path.join(save_path, "2d_log.txt")

    for epoch in range(n_epochs):
        loss_total = 0
        for x, y in data:
            def cost(p):
                input_vec = np.concatenate([x, [0.01], [0.5, 0.5]])
                out = qrf_model(input_vec, p, dev, circuit_fn=circuit_fn, encoding_type=encoding_type)
                return np.sum((out[:3] - y) ** 2)
            params = opt.step(cost, params)
            loss_total += cost(params)
        avg_loss = float(loss_total) / len(data)
        print(f"[2D] Epoch {epoch} | Loss: {avg_loss:.4f}")
        log_metrics(log_path, {"epoch": epoch, "loss": avg_loss})
        if epoch % config["save_interval"] == 0:
            save_checkpoint(params, os.path.join(save_path, f"2d_epoch{epoch}.npz"))

def train_3d_scene_representation(config):
    n_qubits = 6
    dev = qml.device("default.qubit", wires=n_qubits)
    circuit_fn = CIRCUITS[config["circuit"]]
    encoding_type = config["encoding"]
    n_epochs = config["epochs"]
    stepsize = config["lr"]
    save_path = config["save_path"]

    params = np.random.uniform(0, np.pi, size=(24,), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=stepsize)

    def gen_rays(n=32):
        origins = np.random.rand(n, 3)
        dirs = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
        targets = np.random.rand(n, 3)
        return list(zip(origins, dirs, targets))

    rays = gen_rays(config["batch_size"])
    log_path = os.path.join(save_path, "3d_log.txt")

    for epoch in range(n_epochs):
        loss_total = 0
        for origin, dir, target_color in rays:
            def cost(p):
                rgb = render_one_ray(origin, dir, config["n_samples"], delta=0.01,
                                     view_dir=np.array([0.5, 0.5]), params=p, dev=dev,
                                     circuit_fn=circuit_fn, encoding_type=encoding_type)
                return np.sum((rgb - target_color) ** 2)
            params = opt.step(cost, params)
            loss_total += cost(params)
        avg_loss = float(loss_total) / len(rays)
        print(f"[3D] Epoch {epoch} | Loss: {avg_loss:.4f}")
        log_metrics(log_path, {"epoch": epoch, "loss": avg_loss})
        if epoch % config["save_interval"] == 0:
            save_checkpoint(params, os.path.join(save_path, f"3d_epoch{epoch}.npz"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["2d", "3d"], required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config["save_path"] = config.get("save_path", f"experiments/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(config["save_path"], exist_ok=True)

    if args.task == "2d":
        train_2d_image_regression(config)
    elif args.task == "3d":
        train_3d_scene_representation(config)
