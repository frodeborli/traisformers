# Create mapping between radio callsign and MMSI
To create a mapping between the radio callsign and MMSI of a vessel `create_radio2mmsi.py` can be ran. The script will resolve the MMSI of a vessel through a public API. The file requires an unprocessed ERS POR CSV file to fetch the radio callsigns from.
```bash
python3 create_radio2mmsi.py <por_file>
```
Running the script will produce a file `radio2mmsi.csv` containing the mappings.