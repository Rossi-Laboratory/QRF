import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from qrf.quantum_network import qrf_model

def sample_ray(origin: np.ndarray, direction: np.ndarray, n_samples: int,
               near: float = 0.1, far: float = 1.0) -> List[np.ndarray]:
    """
    Generate sample points along a ray.
    Args:
        origin: Ray origin (3,).
        direction: Ray direction (3,), assumed normalized.
        n_samples: Number of samples along ray.
        near: Near plane distance.
        far: Far plane distance.
    Returns:
        List of 3D points along ray.
    """
    # t values either uniform or variable step size based on distance
    t_vals = np.linspace(near, far, n_samples)
    return [origin + t * direction * (1 + 0.5 * np.sin(2 * np.pi * t)) for t in t_vals]

def qrf_evaluate_point(point: np.ndarray, view_dir: np.ndarray, delta: float,
                       params: np.ndarray, dev, circuit_fn, encoding_type: str) -> Tuple[float,float,float,float]:
    """
    Evaluate QRF model at a single 3D point.
    Returns:
        R, G, B, sigma
    """
    input_data = np.concatenate([point, [delta], view_dir])
    out = qrf_model(input_data, params, dev, circuit_fn=circuit_fn, encoding_type=encoding_type)
    # Expect out = [R, G, B, sigma]
    return out[0], out[1], out[2], out[3]

def accumulate_volume_rgb(points: List[np.ndarray], view_dir: np.ndarray, deltas: List[float],
                          params: np.ndarray, dev, circuit_fn, encoding_type: str,
                          bg_color: np.ndarray = np.array([1.0,1.0,1.0]),
                          visualize_sigma: bool = False) -> np.ndarray:
    """
    Perform volume rendering (NeRF-style) with quantum model outputs.
    Args:
        points: sample points along ray.
        view_dir: viewing direction (theta, phi).
        deltas: distances between samples.
        params: QRF model parameters.
        dev: quantum device.
        circuit_fn: selected circuit function.
        encoding_type: encoding key.
        bg_color: background color for remaining transparency.
        visualize_sigma: whether to plot sigma heatmap.
    Returns:
        Final RGB color.
    """
    T = 1.0
    rgb_final = np.zeros(3)
    sigmas = []
    for i, point in enumerate(points):
        delta = deltas[i]
        R, G, B, sigma = qrf_evaluate_point(point, view_dir, delta,
                                            params, dev, circuit_fn, encoding_type)
        sigmas.append(sigma)
        alpha = 1.0 - np.exp(-sigma * delta)
        weight = T * alpha
        rgb_final += weight * np.array([R, G, B])
        T *= np.exp(-sigma * delta)
    # add background color weighted by remaining transparency
    rgb_final += T * bg_color

    if visualize_sigma:
        _visualize_sigma_map(sigmas, deltas)

    return rgb_final

def _visualize_sigma_map(sigmas: List[float], deltas: List[float]):
    """
    Plot sigma values along ray as heatmap/bar chart.
    """
    positions = np.cumsum([0] + deltas[:-1])
    plt.figure(figsize=(6,2))
    plt.bar(positions, sigmas, width=deltas, color='hot', alpha=0.7)
    plt.xlabel('Distance along ray')
    plt.ylabel('Sigma (density)')
    plt.title('Sigma distribution along ray')
    plt.show()

def render_one_ray(origin: np.ndarray, direction: np.ndarray, n_samples: int,
                   params: np.ndarray, dev, circuit_fn, encoding_type: str,
                   near: float = 0.1, far: float = 1.0,
                   visualize_sigma: bool = False) -> np.ndarray:
    """
    Wrapper to sample ray, compute deltas, and render color.
    """
    points = sample_ray(origin, direction, n_samples, near, far)
    # variable deltas: differences between successive t*direction magnitude
    t_vals = np.linspace(near, far, n_samples)
    deltas = np.diff(t_vals, prepend=t_vals[0])
    return accumulate_volume_rgb(points, direction[:2], deltas, params, dev,
                                 circuit_fn, encoding_type, visualize_sigma=visualize_sigma)
