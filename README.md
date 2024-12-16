# Predicting the Destination Port of Fishing Vessels

## Requirements
The required libraries and their versions are listed in the `requirements.txt` file. To install the required libraries, run the following command:
```
pip install -r requirements.txt
```
Additional libraries are required to be installed to use the TrAISformer model. These are listed in the `TrAISformer/requirements.yml` file.

## Directory Structure
The directory structure is as follows:
- [baseline/](./baseline/): Contains the implementation of the baseline model
- [preprocess_data/](./preprocess_data/): Contains the code for preprocessing ERS and AIS data, and generating datasets of AIS tracks
- [TrAISformer/](./TrAISformer/): Contains the code for the TrAISformer deep learning model, cloned from the [original repository](https://github.com/CIA-Oceanix/TrAISformer). The model is unchanged, with only slight refactoring and extending of functionality in what the model can be used for (predicting for single tracks, easier evaluation of different input length, etc.)
- [port_pred/](./port_pred/): The implementation of the port prediction algorithm, and code for evaluating the performance of the algorithm through different implemented tests
- [website/](./website/): Contains the code for the web application which generates interactive maps visualizing the predicted TrAISformer trajectories, probabilities of ports, the predicted destination port, and the true destination port and trajectory

Each of the directories contains a README file with more detailed information about the contents of the directory, and how to run the code.

## Datasets
The training, validation, and test datasets used in the project are included in thie repository, and can be found in the `data/` directory for each of the models.

Examples of predicted trajectories can be found in `website/data/preds`, and contains trajectory predictions for a subset of the test set.
