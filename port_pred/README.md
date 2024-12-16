# Port Prediction Algorithm
The code in this directory is the implementation of the port prediction algorithm. With `main.py` different test cases can be ran to evaluate the accuracy of the algorithm.

## Requirements
The following files are required to test the algorithm:
- A CSV file containing the ports and their corresponding coordinates
- A pickle file containing the true trajectory of the vessel(s) being tested
- A pickle file containing the predicted trajectory of the vessel(s) being tested

## Usage
To run the algorithm, execute the following command:
```bash
python main.py --file <traj_preds_file> --true <true_trajs_file> --ports <ports_file> --type <test_type> --num <num_tests>
```
where:
- `<traj_preds_file>` is the path to the pickle file containing the predicted trajectory of the vessel(s) being tested
- `<true_trajs_file>` is the path to the pickle file containing the true trajectory of the vessel(s) being tested
- `<ports_file>` is the path to the CSV file containing the ports and their corresponding coordinates
- `<test_type>` is the type of test to be ran. It can be: `acc`, `gen`, `top`, `top_three`, or `pred`
- `<num_tests>` is the number of testcases to use for a test, or the index of the track to make a port prediction for if the test type is `pred`

## Dataset
An example dataset can be found in the `data/` directory. The dataset contains trajectory predictions for the 20 first AIS tracks in the test set. The true trajectories are also included in this directory, for the same 20 AIS tracks. To generate for the whole test set, the `TrAISformer` model must be ran.

To use the dataset, decompress the tar archive first.
