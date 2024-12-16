# AIS preprocessing
The following directory contains utility files for fetching and preprocessing public AIS data.

## Requirements
This requires the files `ports.csv` and `radio2mmsi.csv` to be located in the `data/` directory. `radio2mmsi.csv` should be created by running the script `create_radio2mmsi.py` from the `utils/` directory.

## Fetching data
The `ais4area.py` script fetches AIS data within a specified area and time range. These parameters must be changed within the script if another area or time range is desired.
```bash
python3 ais4area.py <out_file>
```

## Create AIS tracks
To preprocess and create AIS tracks from the fetched data, ERS data must already be preprocessed and ready for that same time period. The `create_tracks.py` script can then be used to match the AIS and ERS datasets.
```bash
python3 create_tracks.py <ais_file> <ers_file>
```
Running this command will create `tracks.pkl` in the `data/`, containing preprocessed AIS tracks.

## Split dataset
`split_dataset.py` can be used to split the dataset in `tracks.pkl` into training, validation, and test sets, with a 80%, 10%, 10% split.
```bash
python3 split_dataset.py <file> <out_dir>
```
This will create three files in the specified output directory: `train.pkl`, `valid.pkl`, and `test.pkl`.
