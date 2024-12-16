import numpy as np
import pickle
import ast
from csv import DictReader

def read_files(pred_file: str, true_file: str, ports_file: str) -> tuple[list, list, list]:
    """
    Read the file containing trajectory predictions, true trajectories, and ports
    """
    with open(pred_file, "rb") as f:
        data = pickle.load(f)

    with open(true_file, "rb") as f:
        true = pickle.load(f)

    ports = []
    with open(ports_file, "r") as f:
        rows = DictReader(f)
        for port in rows:
            coords = ast.literal_eval(port["coords"])
            ports.append(coords)
    return data, true, ports

def angle_between_points(c1: tuple, c2: tuple, c3: tuple) -> float:
    """
    Calculates the angle between the vectors c1->c2 and c2->c3
    """
    # Calculate the dot product and vector norm
    vec1 = np.array(c2) - np.array(c1)
    vec2 = np.array(c3) - np.array(c2)
    dot_prod = np.dot(vec1, vec2)
    norm_prod = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm_prod == 0:
        return None

    # Calculate the angle in radians
    angle = dot_prod / norm_prod
    angle = np.arccos(max(min(angle, 1), -1))
    return angle

def is_heading_towards(c1: tuple, c2: tuple, c3: tuple) -> bool:
    """
    Checks of coordinate 3 (the port) is within the predefined angle of the
    vessel's trajectory based on its last two coordinates.
    This is intended to check if the vessel is heading towards the port (coordinate 3).
    This function is currently not in use, but were tested as a possible solution
    """
    angle = angle_between_points(c1, c2, c3)
    if angle is None:
        return False
    # If the angle is less than 45 degrees, the vessel is heading towards the port
    return angle < np.pi / 4

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