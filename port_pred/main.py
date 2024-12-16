from utils import read_files
from predict import predict
from tests import gen_test, acc_test, top_three_test, top_test
import argparse


def main(pred_file: str, true_traj_file: str, ports_file: str, pred_type: str, num_tracks: int):
    """
    Test the port prediction algorithm, or make a port prediction
    """
    data, true, ports = read_files(pred_file, true_traj_file, ports_file)
    if num_tracks > 0:
        data = data[:num_tracks]
        true = true[:num_tracks]

    if pred_type == "gen":
        correct, wrong, not_found = gen_test(data, true, ports)

    elif pred_type == "acc":
        correct, wrong, not_found = acc_test(data, true, ports)

    elif pred_type == "pred":
        data = data[-1]
        port, prob, _, _ = predict(data, ports)
        print(f"Predicted port: {port}")
        print(f"Probability   : {prob:.2f}")
        return

    elif pred_type == "top":
        correct, wrong, not_found = top_test(data, true, ports)

    elif pred_type == "top_three":
        correct, wrong, not_found = top_three_test(data, true, ports)
    else:
        print("Prediction type not recognized")
        exit()

    print(f"--- {pred_file} ---")
    print(f"Correct: {correct}/{correct+wrong+not_found} -> {correct/(correct+wrong+not_found):.3f}")
    print(f"Wrong: {wrong}/{correct+wrong+not_found} -> {wrong/(correct+wrong+not_found):.3f}")
    print(f"Not found: {not_found}/{correct+wrong+not_found} -> {not_found/(correct+wrong+not_found):.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/preds/12.pkl", help="Path to the pickle file with trajectory predictions")
    parser.add_argument("--true", type=str, default="data/true.pkl", help="Path to the pickle file with true trajectories")
    parser.add_argument("--ports", type=str, default="data/ports.csv", help="Path to the csv file with ports")
    parser.add_argument("--type", type=str, default="acc", help="Prediction type: acc, gen, top, top_three, pred")
    parser.add_argument("--num", type=int, default=0, help="Number of tracks to predict ports for")
    args = parser.parse_args()

    main(args.file, args.true, args.ports, args.type, args.num)
