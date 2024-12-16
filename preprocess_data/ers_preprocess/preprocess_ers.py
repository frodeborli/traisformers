import pandas as pd
from pandas import DataFrame
import argparse

def convert_str2time(df: DataFrame, column: str) -> DataFrame:
    """
    Convert datetime string to datetime object
    for the given column in the dataframe
    """
    df[column] = pd.to_datetime(df[column], format="%d.%m.%Y %H:%M:%S")
    return df

def load_ers_csv(filename: str, datefield: None|str=None, sep: str=";", decimal: str=",", low_memory: bool=False) -> DataFrame:
    """
    Load ERS CSV file into a pandas dataframe
    """
    df = pd.read_csv(filename, sep=sep, decimal=decimal, low_memory=low_memory)
    # Convert column to datetime
    if datefield:
        df = convert_str2time(df, datefield)
    return df

def compress_msgs(df: DataFrame, field: str) -> DataFrame:
    """
    Compress ERS FIS messages with the same "Melding ID"

    This is done to reduce the amount of rows in the dataframe, as an ERS message is sent for each species caught,
    and we only care about the time of the last haul, regardless of what species was caught
    """
    df.sort_values(by=field, ascending=False, inplace=True)
    df = df.groupby("Melding ID").first().reset_index()
    return df

def clean_dca(dca_file: str) -> DataFrame:
        """
        Clean the ERS DCA message dataset
        """
        df: DataFrame = pd.read_csv(dca_file, sep=";", decimal=",", low_memory=False)

        # Get the relevant columns for all FIS messages
        df = df.loc[df["Aktivitet (kode)"] == "FIS"]
        df["Stopptidspunkt"] = pd.to_datetime(df["Stoppdato"] + df["Stoppklokkeslett"], format="%d.%m.%Y%H:%M")
        df = df.loc[:, ["Melding ID", "Radiokallesignal (ERS)", "Stopptidspunkt"]]

        # Remove duplicates, NANs, and duplicate ID messages
        df = df.dropna()
        df = df.drop_duplicates()
        df = compress_msgs(df, "Stopptidspunkt")
        return df

def clean_por(por_file: str, max_amount: int=0) -> DataFrame:
    """
    Clean the ERS POR message dataset
    """
    df: DataFrame = pd.read_csv(por_file, sep=";", decimal=",", low_memory=False)

    # Get the relevant columns for all POR messages
    df["Ankomsttidspunkt"] = pd.to_datetime(df["Ankomstdato"] + df["Ankomstklokkeslett"], format="%d.%m.%Y%H:%M")
    df = df.loc[:, ["Melding ID", "Radiokallesignal (ERS)", "Ankomsttidspunkt"]]

    # Remove duplicates, NANs, and duplicate ID messages
    df = df.dropna()
    df = df.drop_duplicates()
    df = compress_msgs(df, "Ankomsttidspunkt")

    if max_amount > 0:
        df = df[:max_amount]
    return df

def clean_dep(dep_file: str) -> DataFrame:
    """
    Clean the ERS DEP message dataset
    """
    df: DataFrame = pd.read_csv(dep_file, sep=";", decimal=",", low_memory=False)

    df["Avgangstidspunkt"] = pd.to_datetime(df["Avgangsdato"] + df["Avgangsklokkeslett"], format="%d.%m.%Y%H:%M")
    df = df.loc[:, ["Melding ID", "Radiokallesignal (ERS)", "Avgangstidspunkt"]]

    # Remove duplicates, NANs, and duplicate ID messages
    df = df.dropna()
    df = df.drop_duplicates()
    df = compress_msgs(df, "Avgangstidspunkt")
    return df


def merge_dca_por_dep(dca_df: str, por_df: str, dep_df: str) -> DataFrame:
    """
    Merge the DCA, POR, and DEP messages to one dataframe
    """
    # Sort the keys we are merging on
    dca_df = dca_df.sort_values("Stopptidspunkt")
    por_df = por_df.sort_values("Ankomsttidspunkt")
    dep_df = dep_df.sort_values("Avgangstidspunkt")

    # Merge last haul DCA message with POR message, and remove DCA/POR messages that doesn't have a match
    merged = pd.merge_asof(left=por_df, right=dca_df, left_on="Ankomsttidspunkt", right_on="Stopptidspunkt", by="Radiokallesignal (ERS)")
    merged = merged.dropna(subset=["Stopptidspunkt"])

    # Merge DCA/POR messages with the first DEP message that occurred after the POR, and remove messages that doesn't have a match
    merged = pd.merge_asof(left=merged, right=dep_df, left_on="Ankomsttidspunkt", right_on="Avgangstidspunkt", by="Radiokallesignal (ERS)", direction="forward")
    merged = merged.dropna(subset=["Avgangstidspunkt"])

    # Only return relevant columns (we don't need the ID field anymore)
    merged = merged.loc[:, ["Radiokallesignal (ERS)", "Stopptidspunkt", "Ankomsttidspunkt", "Avgangstidspunkt"]]
    merged.sort_values(by="Ankomsttidspunkt", ascending=True, inplace=True)
    return merged

def combine_ers(dca_file: str, por_file: str, dep_file: str, max_amount: int=0) -> None:
    """
    Preprocess the ERS DCA, POR, and DEP messages and merge them together
    """
    dca_clean = clean_dca(dca_file)
    por_clean = clean_por(por_file, max_amount=max_amount)
    dep_clean = clean_dep(dep_file)

    print(f"[*] DCA size: {len(dca_clean)}")
    print(f"[*] POR size: {len(por_clean)}")
    print(f"[*] DEP size: {len(dep_clean)}")

    merged = merge_dca_por_dep(dca_clean, por_clean, dep_clean)
    print(f"[*] Merged size: {len(merged)}")
    merged.to_csv("data/merged.csv", index=False, sep=";", decimal=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to merge ERS DCA and POR messages together into one csv-file")
    parser.add_argument("por", type=str, help="CSV file with POR messages")
    parser.add_argument("dca", type=str, help="CSV file with DCA messages")
    parser.add_argument("dep", type=str, help="CSV file with DEP messages")
    parser.add_argument("-m", "--max", type=int, default=0, help="Max amount of records in the merged CSV-file")
    args = parser.parse_args()

    por_file = args.por
    dca_file = args.dca
    dep_file = args.dep
    max_amount = args.max
    combine_ers(dca_file, por_file, dep_file, max_amount=max_amount)