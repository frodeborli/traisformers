import pickle
import numpy as np
import argparse

def load_datasets(dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the train and test datasets
    """
    with open(f"{dir}train.pkl", "rb") as f:
        train = np.array(pickle.load(f))

    with open(f"{dir}test.pkl", "rb") as f:
        test = np.array(pickle.load(f))
    return train, test

def load_model() -> dict:
    """
    Load the model from disk
    """
    print("[+] Loading model")
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def store_model(model: dict) -> None:
    """
    Store the model to disk
    """
    print("[+] Storing model!")
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

def train_model(dataset: np.ndarray) -> dict:
    """
    Train the baseline model on the given dataset
    """
    pre_model, model = {}, {}
    # Count occurances at each port for each vessel
    for track in dataset:
        mmsi = track.get("mmsi")
        port_coords = track.get("port")[0]

        # Count number of visits to each port for each vessel
        if not mmsi in pre_model:
            pre_model[mmsi] = {}
        if port_coords in pre_model.get(mmsi):
            pre_model[mmsi][port_coords] += 1
        else:
            pre_model[mmsi][port_coords] = 1

    # Find probabilities in descending order for each port for each vessel
    for mmsi in pre_model.keys():
        ports = pre_model.get(mmsi)
        total_trips = sum(ports.values())

        # Creat list on the format [(lat, lon), probability] for each port for each mmsi
        port_probs = list(ports.items())
        port_probs.sort(key=lambda x: x[1], reverse=True)
        port_probs = np.array(port_probs, dtype=object)
        port_probs[:,1] /= total_trips
        model[mmsi] = port_probs
    return model

def evaluate_model(model: dict, dataset: np.ndarray) -> None:
    """
    Evaluate the baseline model on the given dataset
    """
    single_correct, triple_correct = 0, 0
    total = len(dataset)
    # Measure the accuracy of the model
    # with top prediction and top three predictions
    for track in dataset:
        mmsi = track.get("mmsi")
        true_port_coords = track.get("port")[0]
        probs = model.get(mmsi)

        # If new mmsi the model can't make a prediction
        if probs is None or len(probs) == 0:
            continue

        # Check if top probability port is correct (also check if in top three)
        pred_port_coords, top_three_ports_pred = predict(model, mmsi)
        pred_port_coords = pred_port_coords[0]
        if true_port_coords == pred_port_coords:
            single_correct += 1
            triple_correct += 1
            continue

        # Check if true port is in the top three probability ports of the model
        if true_port_coords in list(top_three_ports_pred[:, 0]):
            triple_correct += 1

    print(f"Accuracy single port:    {single_correct/total:.3f} -> {single_correct}/{total}")
    print(f"Accuracy top three ports {triple_correct/total:.3f} -> {triple_correct}/{total}")


def predict(model: dict, mmsi: int) -> tuple[list|None, list|None]:
    """
    Return the highest probability port and the top three probability ports. The returned value also contains the accuracies
    Returns None if the mmsi is new
    """
    probs = model.get(mmsi)
    if probs is None or len(probs) == 0:
        return None, None
    return probs[0], probs[:3]

def main(dataset_dir: str, train: bool, evaluate: bool, mmsi: int) -> None:
    train_set, test_set = load_datasets(dataset_dir)
    if train:
        print("[+] Training model")
        model = train_model(train_set)
        store_model(model)
    else:
        model = load_model()
    if evaluate:
        print("[+] Evaluating model")
        evaluate_model(model, test_set)
    elif mmsi != 0:
        print(f"[+] Predicting port for MMSI: {mmsi}")
        pred_port, top_three_ports_pred = predict(model, mmsi)
        if pred_port is None:
            print(f"MMSI {mmsi} is not in the model")
        else:
            print(f"Predicted port: {pred_port}")
            print(f"Top three ports: {top_three_ports_pred}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="data/", help="Path to directory with training and test datasets")
    parser.add_argument("-t", "--train", action="store_true", help="Train the model")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("-p", "--predict", type=int, default=0, help="Predict port for a given MMSI")
    args = parser.parse_args()

    dataset_dir = args.dir
    train = args.train
    evaluate = args.evaluate
    mmsi = args.predict

    if not dataset_dir.endswith("/"):
        dataset_dir += "/"
    main(dataset_dir, train, evaluate, mmsi)
