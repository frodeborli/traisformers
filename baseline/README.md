# Baseline model
The baseline model is a simple port prediction model predicting the destination port based on the past port-visit history of the given vessel (MMSI).

## Dataset
Our dataset is located in the `data/` directory, and must be decompressed before it can be used by the model.

## Training the Model
To train the model, run the following command:
```bash
python3 baseline.py -d <path_to_dataset_dir> -t
```
The model will be stored in the `model` directory

## Evaluating the Model
To evaluate the model, run the following command:
```bash
python3 baseline.py -d <path_to_dataset_dir> -e
```

## Predicting a Destination Port
To predict the destination port of a given vessel, run the following command:
```bash
python3 baseline.py -d <path_to_dataset_dir> -p <mmsi>
```
A trained model must exist for a prediction to be made.
