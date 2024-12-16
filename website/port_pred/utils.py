import numpy as np
import pickle
import ast
from csv import DictReader

def read_files(pred_file: str, true_file: str, ports_file: str, idx: int) -> tuple[list, list, list]:
    """
    Read the file containing trajectory predictions, true trajectories, and ports
    """
    with open(pred_file, "rb") as f:
        data = pickle.load(f)[idx]

    with open(true_file, "rb") as f:
        true = pickle.load(f)[idx]

    ports = []
    with open(ports_file, "r") as f:
        rows = DictReader(f)
        for port in rows:
            coords = ast.literal_eval(port["coords"])
            ports.append(coords)
    return data, true, ports

# Function originating from: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
def haversine_np(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the haversine distance between two coordinates
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def port_dist(coord: tuple, kaier: list) -> tuple[float, tuple]:
    """
    Finds and returns the distance and coordinates of the closest port to the given coordinates
    """
    dist = float("inf")
    port = None
    for k in kaier:
        d = haversine_np(coord[1], coord[0], k[1], k[0])
        if d < dist:
            dist = d
            port = k
    return dist, tuple(port)

def mov_avg(traj: np.ndarray, window_size: int) -> list:
    """
    Smooth out the trajectory by applying a moving average filter,
    averaging out the coordinates of the trajectory with a factor of the window size
    """
    ext_traj = np.pad(traj, ((window_size//2, window_size//2), (0, 0)), mode='edge')
    weights = np.ones(window_size) / window_size
    x_coords = np.convolve(ext_traj[:, 0], weights, mode='valid')
    y_coords = np.convolve(ext_traj[:, 1], weights, mode='valid')
    smooth_traj = [(x_coords[i], y_coords[i]) for i in range(len(x_coords))]
    return smooth_traj
