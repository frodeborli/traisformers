# Website for Visualization of AIS tracks

## Running the website
The website will launch on `127.0.0.1:5000` when running
```bash
python3 app.py
```

## Endpoints
The following endpoints are available

- `/` - Home page with overview of the website
- `/area` - Visualization of our Region Of Interest (ROI)
- `/ports` - All ports in the ROI
- `/<input_len>/<idx>` - Predictions given the input with length `<input_len>` and for the trajectory at index `<idx>` in the test dataset

Only input lengths of 12 (one hour) are currently supported, due to the trajectory predictions currently being precomputed. It is located in the `data/preds/` directory.

## Dataset
An example dataset can be found in the `data/` directory. The dataset contains trajectory predictions for the 20 first AIS tracks in the test set. The true trajectories are also included in this directory, for the same 20 AIS tracks. To generate for the whole test set, the `TrAISformer` model must be ran.

To use the dataset, decompress the tar archive in the `data/` directory first.
