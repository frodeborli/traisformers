import numpy as np
from ais import AISSmall
from utils import port_dist, mov_avg
from constants import MAX_RADIUS, MIN_RADIUS, SPEED_LIMIT

def predict(pred_trajs: list, ports: list) -> tuple[tuple, float, dict, list]:
    """
    Predict the destination port of a vessel based on its predicted trajectories

    Returns a tuple containing the following:
        - Coordinates of the predicted port
        - The confidence in it being the destination port
        - Dictionary with all non-zero probability ports
        - The predicteds trajectories, but cut off if they reached a port
    """
    pred_ports, trajs = {}, []
    # Predict destination ports for each predicted trajectory
    for i, pred in enumerate(pred_trajs):
        port, idx = find_port(pred, ports, MIN_RADIUS)

        # Cut trajectory if it arrived at a port
        trajs.append(pred_trajs[i][0][:idx])
        # Count the number of times a port was predicted
        if port == None:
            continue
        elif port in pred_ports:
            pred_ports[port] += 1
        else:
            pred_ports[port] = 1

    # If no ports were predicted, return None
    if len(pred_ports) == 0:
        return None, 0, {}, trajs

    # Calculate the predicted probabilities of each non-zero probability port
    total = sum(pred_ports.values())
    port = max(pred_ports, key=pred_ports.get)
    prob = pred_ports[port] / total
    for p in pred_ports:
        pred_ports[p] = pred_ports[p] / total
    return port, prob, pred_ports, trajs


def _find_port(ais: list, track: list, ports: list, radius: int) -> tuple[tuple|None, int]:
    """
    Check if the vessel at some point is within the radius of any port,
    with a predicted speed less than the speed limit.
    If so, predict that port as this trajectory's destination port.
    Return the coordinates of the port and the index of the AIS message arriving at the port, else None
    """
    for idx, msg in enumerate(ais):
        if msg.sog > SPEED_LIMIT:
            continue
        dist, port = port_dist(track[idx], ports)
        if dist > radius:
            continue
        return port, idx
    return None, len(track)

def find_port(pred: list, ports: list, radius: int):
    """
    Recursively find the destination port of a vessel based on the given predicted trajecotry,
    increasing the radius for each recursive call until a port is found, or the maximum radius is reached.
    Returns the coordinates of the predicted port and the index of the AIS message arriving at the port, or None
    """
    if radius > MAX_RADIUS:
        return None, len(pred[0])

    ais = [AISSmall(*x) for x in pred[0]]
    track = np.array([x.coords for x in ais])
    track = mov_avg(track, 5)

    port, idx = _find_port(ais, track, ports, 5)
    if port == None:
        return find_port(pred, ports, radius+1)
    return port, idx