from tqdm import tqdm
from predict import find_port
from constants import MIN_RADIUS

def gen_test(data: list, true: list, ports: list) -> tuple[int, int, int]:
    """
    Count the number of tracks from the test set where the correct port
    is found for at least one of the predicted trajectories.
    This essentially counts the cases where the probability given
    to the actual destination port is non-zero
    """
    correct, wrong, not_found = 0, 0, 0
    # Iterate over the testset
    for idx, testcase in enumerate(tqdm(data, total=len(data), desc="Testcases")):
        corr_port = true[idx].get("port")[0]
        no_port, found = 0, False
        # Get the port predicted for each trajectory
        for _, pred in enumerate(testcase):
            port, _ = find_port(pred, ports, MIN_RADIUS)
            # Count if the predicted port is correct, wrong, or not predicted
            if port == corr_port:
                correct += 1
                found = True
                break
            elif port == None:
                no_port += 1
        # No ports were predicted by any tracks
        if found == False:
            if no_port == len(testcase):
                not_found += 1
            else:
                wrong += 1
    return correct, wrong, not_found

def acc_test(data: list, true: list, ports: list) -> tuple[int, int, int]:
    """
    Count the number of trajectories form the test set
    where the correct destination port is predicted
    """
    correct, wrong, not_found = 0, 0, 0
    # Iterate over the test set
    for idx, testcase in enumerate(tqdm(data, total=len(data), desc="Testcases")):
        corr_port = true[idx].get("port")[0]
        # Get the port predicted for each trajectory,
        # and check if the prediction is correct
        for _, pred in enumerate(testcase):
            port, _ = find_port(pred, ports, MIN_RADIUS)
            if port == None:
                not_found += 1
            elif port == corr_port:
                correct += 1
            else:
                wrong += 1
    return correct, wrong, not_found

def top_test(data: list, true: list, ports: list) -> tuple[int, int, int]:
    """
    Counts the number of times the predicted port was
    the correct port, wrong port, or no port was predicted.
    This test measures how accurate the port prediction algorithm is
    when predicting a single destination port given 16 predicted trajectories for an input
    """
    correct, wrong, not_found = 0, 0, 0
    # Iterate over the test set
    for idx, testcase in enumerate(tqdm(data, total=len(data), desc="Testcases")):
        corr_port = true[idx].get("port")[0]
        pred_ports = {}
        # Get the port predicted for each trajectory
        for _, pred in enumerate(testcase):
            port, _ = find_port(pred, ports, MIN_RADIUS)
            if port == None:
                continue
            elif port in pred_ports:
                pred_ports[port] += 1
            else:
                pred_ports[port] = 1
        # Not ports were predicted by any track
        if len(pred_ports) == 0:
            not_found += 1
            continue
        top_port = max(pred_ports, key=pred_ports.get)
        if top_port == corr_port:
            correct += 1
        else:
            wrong += 1
    return correct, wrong, not_found

def top_three_test(data: list, true: list, ports: list) -> tuple[int, int, int]:
    """
    Count the number of times the predicted top three most probable ports
    include the actual destination port, based on the predicted trajectories in the test set
    """
    correct, wrong, not_found = 0, 0, 0
    # Iterate over the test set
    for idx, testcase in enumerate(tqdm(data, total=len(data), desc="Testcases")):
        corr_port = true[idx].get("port")[0]
        pred_ports = {}
        # Get the port predicted for each trajectory
        for _, pred in enumerate(testcase):
            port, _ = find_port(pred, ports, MIN_RADIUS)
            if port == None:
                continue
            elif port in pred_ports:
                pred_ports[port] += 1
            else:
                pred_ports[port] = 1
        # No ports were predicted by any tracks
        if len(pred_ports) == 0:
            not_found += 1
            continue
        top_three_ports = sorted(pred_ports, key=pred_ports.get, reverse=True)[:3]
        if corr_port in top_three_ports:
            correct += 1
        else:
            wrong += 1
    return correct, wrong, not_found