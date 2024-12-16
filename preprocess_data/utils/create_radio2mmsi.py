import pandas as pd
from tqdm import tqdm
from radio2mmsi import resolve_radio
import argparse

def get_callsigns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only unique radio callsigns in the dataframe
    """
    df = df.loc[:, ["Radiokallesignal (ERS)"]]
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def radio2mmsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve the MMSI of a vessel based on the radio callsign
    """
    df["mmsi"] = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Resolving MMSIs"):
        df.loc[idx, "mmsi"] = resolve_radio(row["Radiokallesignal (ERS)"])
    df = df.dropna()
    df["mmsi"] = df["mmsi"].astype(int)
    return df

def main(por_file: str) -> None:
    """
    Create mapping between the radio callsign and MMSI of a vessel
    """
    por_df = pd.read_csv(por_file, sep=";", decimal=",", low_memory=False)
    por_radio = get_callsigns(por_df)
    df = radio2mmsi(por_radio)
    df.to_csv("radio2mmsi.csv", index=False, sep=";")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a mapping from radio callsign to MMSI")
    parser.add_argument("por", type=str, help="Path to the POR dataset")
    args = parser.parse_args()
    main(args.por)