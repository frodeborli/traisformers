import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from time import sleep
import argparse

radio2mmsi = pd.read_csv('data/radio2mmsi.csv', skiprows=1, delimiter=";", index_col=0).squeeze().to_dict()

def filter_msgs(messages: list) -> list:
    """
    Remove messages which doesn't match our mmsi list
    """
    filtered = []
    for msg in messages:
        if msg[0] in radio2mmsi.values():
            filtered.append(msg)
    return filtered

def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe entries
    """
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.loc[(df["sog"] > 0.5) & (df["sog"] <= 30)]
    df = df.loc[(df["cog"] >= 0) & (df["cog"] <= 360)]
    return df

def store_msgs(messages: list, filename: str) -> None:
    """
    Append to existing CSV of the other messages, or create a new file
    """
    df = pd.DataFrame(messages)
    df.columns = ["mmsi", "date", "long", "lat", "cog", "sog", "msg_num", "speed(kmh)", "sec_prev_point", "dist_prev_point"]
    df = df.loc[:, ["mmsi", "date", "long", "lat", "cog", "sog"]]
    df = filter_df(df)
    df.to_csv(filename, mode="a", header=not os.path.exists(filename), index=False)

def get_ais(filename):
    """
    Fetch AIS messages from the public API
    """
    url = "https://kystdatahuset.no/ws/api/ais/positions/within-geom-time"

    # Period to fetch AIS messages between
    # Fetching is done in 1 hour intervals
    # or else it might not return any data
    starttime = datetime(2016,10,1)
    while starttime < datetime(2017,1,1):
        messages = []
        stoptime = starttime + timedelta(minutes=59, seconds=59)
        # Set Region of Interest, period to fetch AIS messages for, and minimum speed of messages
        body = {
            "geom": "POLYGON((31.5 69.2, 31.5 73.0, 13.0 73.0, 13.0 69.2, 31.5 69.2))",
            "start": starttime.strftime("%Y-%m-%dT%H:%M:%S"),
            "end": stoptime.strftime("%Y-%m-%dT%H:%M:%S"),
            "minSpeed": 0.5
        }
        try:
            r = requests.post(url, json=body)
            res = r.json()
        except:
            print(f"[!] Could not fetch AIS data for {str(starttime)} - {str(stoptime)}")
            sleep(5)
            continue
        # Retry if an error occurred
        if not res.get("success") == True:
            print(f"[x] Success=False: {str(starttime)} - {str(stoptime)}")
            sleep(10)
            continue
        else:
            messages = res.get("data")
            print(f"[+] AIS Success: {str(starttime)} - {str(stoptime)}", end=" -> ")
            messages = filter_msgs(messages)
            print(f"{len(messages)} msgs")
            if len(messages) > 0:
                store_msgs(messages, filename)

            # Delay to avoid stressing the API too much
            sleep(5)
            starttime += timedelta(hours=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch AIS data for a given period")
    parser.add_argument("file", type=str, help="Filename to store AIS messages in")
    args = parser.parse_args()
    get_ais(args.file)