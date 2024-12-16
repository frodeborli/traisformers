import pickle
import sys
import numpy as np
import argparse

def split_dataset(dataset_filename: str, out_dir: str):
    """
    Split the dataset into training, validation, and test sets
    """
    if out_dir[-1] != '/':
        out_dir += '/'

    with open(dataset_filename, 'rb') as f:
        dataset = pickle.load(f)

    # Deterministic shuffle
    rng = np.random.default_rng(0)
    rng.shuffle(dataset)

    # Split into 80% training, 10% validation, 10% test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size+val_size]
    test_set = dataset[train_size+val_size:]

    print(f"Training set: {len(train_set)}")
    print(f"Validation set: {len(val_set)}")
    print(f"Test set: {len(test_set)}")

    with open(out_dir + "train.pkl", "wb") as f:
        pickle.dump(train_set, f)

    with open(out_dir + "valid.pkl", "wb") as f:
        pickle.dump(val_set, f)

    with open(out_dir + "test.pkl", "wb") as f:
        pickle.dump(test_set, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into training, validation, and test sets')
    parser.add_argument('file', type=str, help='Pickle file with AIS tracks')
    parser.add_argument('out_dir', type=str, help='Output directory')
    args = parser.parse_args()

    split_dataset(args.file, args.out_dir)
