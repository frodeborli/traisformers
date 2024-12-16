# ERS preprocessing

## Preprocessing data
The `preprocess_ers.py` script can be used to preprocess ERS data. It requires three separate CSV files, each containing POR, DCA, and DEP messages.
```bash
python3 preprocess_ers.py <por_file> <dca_file> <dep_file>
```
The merged dataset will be output into the `data/` in a file named `merged.csv`.