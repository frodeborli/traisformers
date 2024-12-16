import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import json
import os.path
from pyproj import Geod
import argparse

geod = Geod(ellps='WGS84')

# Min and max values for normalization
min_cog, max_cog = 0.0, 360.0
min_sog, max_sog = 0, 30.0
min_lat, max_lat = 69.2, 73.0
min_lon, max_lon = 13.0, 31.5

# Function originating from: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
def haversine_np(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

# Interpolation function from GeoTrackNet (used in original TrAISformer dataset preprocessing also):
# Url: https://github.com/CIA-Oceanix/GeoTrackNet/blob/master/utils.py#L192C1-L239C20
def interpolate(t: int, track: np.ndarray) -> np.ndarray|None:
    """
    Interpolating the AIS message of vessel at a specific "t".
    INPUT:
        - t :
        - track     : AIS track, whose structure is
                     [LAT, LON, SOG, COG, TIMESTAMP, MMSI]
    OUTPUT:
        - [LAT, LON, SOG, COG, TIMESTAMP, MMSI]
    """
    LAT, LON, SOG, COG, TIMESTAMP, MMSI = list(range(6))
    before_p = np.nonzero(t >= track[:,TIMESTAMP])[0]
    after_p = np.nonzero(t < track[:,TIMESTAMP])[0]

    if (len(before_p) > 0) and (len(after_p) > 0):
        apos = after_p[0]
        bpos = before_p[-1]
        # Interpolation
        dt_full = float(track[apos,TIMESTAMP] - track[bpos,TIMESTAMP])
        if (abs(dt_full) > 2*3600):
            return None
        dt_interp = float(t - track[bpos,TIMESTAMP])
        try:
            az, _, dist = geod.inv(track[bpos,LON], track[bpos,LAT],
                                   track[apos,LON], track[apos,LAT])
            dist_interp = dist*(dt_interp/dt_full)
            lon_interp, lat_interp, _ = geod.fwd(track[bpos,LON], track[bpos,LAT], az, dist_interp)
            speed_interp = (track[apos,SOG] - track[bpos,SOG])*(dt_interp/dt_full) + track[bpos,SOG]
            course_interp = (track[apos,COG] - track[bpos,COG] )*(dt_interp/dt_full) + track[bpos,COG]
        except:
            return None
        return np.array([lat_interp, lon_interp,
                         speed_interp, course_interp,
                         t, track[0,MMSI]])
    else:
        return None

# Preprocessing function from GeoTrackNet (used in original TrAISformer dataset preprocessing also)
# Url: https://github.com/CIA-Oceanix/GeoTrackNet/blob/master/data/dataset_preprocessing.py#L204-L222
def downsample(arr: np.array, minutes: int) -> np.ndarray:
    """
    Do interpolation on the AIS track down to the given amount of minutes between each message
    """
    TIMESTAMP = 4 # Timestamp is the fourth index in our AIS messages
    sampling_track = np.empty((0, 6))
    for t in range(int(arr[0, TIMESTAMP]), int(arr[-1, TIMESTAMP]), minutes*60):
        interpolated = interpolate(t, arr)
        if interpolated is not None:
            sampling_track = np.vstack([sampling_track, interpolated])
        else:
            sampling_track = None
            break
    return sampling_track

def close_to_port(arr: np.array, ports: pd.DataFrame) -> tuple[bool, float, tuple]:
    """
    Find the closest port to the last message in the AIS track, and return if it is within 5 km of a port
    Returns tuple with (True/False, distance to port, coordinates of closest port)
    """
    last_msg = arr[-1]
    lat = last_msg[0]
    lon = last_msg[1]
    closest_dist = float("inf")
    closest_port = None
    # Iterate over all ports and find the closest one
    for _, row in ports.iterrows():
        port_lat, port_lon = tuple(json.loads(row["coords"]))
        dist = haversine_np(lon, lat, port_lon, port_lat)
        if dist < closest_dist:
            closest_dist = dist
            closest_port = (port_lat, port_lon)

    if closest_dist < 5: # Less than 5 km from a port
        return True, closest_dist, closest_port
    return False, None, None


def filter_outlier_messages(arr: np.ndarray) -> np.ndarray:
    """
    Cut the AIS track if there is a gap larger than 2 hours between messages,
    to prevent interpolation from completely removing the track
    """
    TIMESTAMP = 4
    quarter_size = len(arr) // 4
    first_quarter_slice = 0
    last_quarter_slice = len(arr)
    # Cut track if gap is in first or last quarter of the AIS track,
    # as the track is still mostly complete
    for i in range(1, len(arr)):
        if arr[i][TIMESTAMP] - arr[i-1][TIMESTAMP] > 2*3600:
            if i <= quarter_size:
                first_quarter_slice = i
            elif i >= 3*quarter_size:
                last_quarter_slice = i
            else:
                return []
    return arr[first_quarter_slice:last_quarter_slice]


def match_ers_ais(ais_filename: str, ers_filename: str) -> None:
    """
    Match AIS- and ERS messages transmitted on the same fishing trip,
    creating AIS tracks for the vessel
    """
    print("[*] Loading Datasets")
    ais_df = pd.read_csv(ais_filename, delimiter=",")
    print("[+] Loaded AIS Dataset")
    ers_df = pd.read_csv(ers_filename, delimiter=";")
    print("[+] Loaded ERS Dataset")
    radio2mmsi = pd.read_csv('data/radio2mmsi.csv', skiprows=1, delimiter=";", index_col=0).squeeze().to_dict()
    print("[+] Loaded Radio2MMSI")
    ports = pd.read_csv("data/ports.csv", delimiter=",")
    print("[+] Loaded Ports")
    print("[+] Done loading datasets!")

    # Format datetime strings to datetime objects
    ais_df["date"] = pd.to_datetime(ais_df["date"], format="%Y-%m-%dT%H:%M:%S")
    ers_df["Stopptidspunkt"] = pd.to_datetime(ers_df["Stopptidspunkt"], format="%Y-%m-%d %H:%M:%S")
    ers_df["Ankomsttidspunkt"] = pd.to_datetime(ers_df["Ankomsttidspunkt"], format="%Y-%m-%d %H:%M:%S")
    ers_df["Avgangstidspunkt"] = pd.to_datetime(ers_df["Avgangstidspunkt"], format="%Y-%m-%d %H:%M:%S")

    # Append to existing pickle file if it exists, else create a new one
    if os.path.exists("data/tracks.pkl"):
        with open("data/tracks.pkl", "rb") as f:
            pickle_list = pickle.loads(f.read())
            print(f"[+] Loaded {len(pickle_list)} tracks!")
    else:
        pickle_list = []

    # Match AIS messages to all ERS messages
    for _, row in tqdm(ers_df.iterrows(), desc="Collecting AIS tracks", total=len(ers_df.index)):
        tmp = {}
        starttime = row["Stopptidspunkt"]
        stoptime = row["Ankomsttidspunkt"] + (row["Avgangstidspunkt"] - row["Ankomsttidspunkt"]) / 2
        mmsi = radio2mmsi.get(row["Radiokallesignal (ERS)"], 0)

        # Only match on MMSI and between the given date
        df = ais_df.loc[ais_df["mmsi"] == mmsi]
        df = df.loc[(df["date"] > starttime) & (df["date"] < stoptime)]
        df["timestamp"] = df["date"].apply(lambda x: int(x.timestamp()))
        df = df.loc[:, ["lat", "long", "sog", "cog", "timestamp", "mmsi"]]
        arr = df.to_numpy()

        # Cut the AIS track if there is a gap of more than 2 hours in the track
        arr = filter_outlier_messages(arr)

        # Remove tracks shorter than 20 messages or 4 hours
        if len(arr) < 20 or (datetime.fromtimestamp(arr[-1][-2]) - datetime.fromtimestamp(arr[0][-2]) < timedelta(hours=4)):
            continue

        # Make sure the track ends close to a port
        in_port, dist2port, port_coords = close_to_port(arr, ports)
        if not in_port:
            continue

        # Interpolat the AIS track to 5 minute intervals between messages
        arr = downsample(arr, minutes=5)
        if arr is None or len(arr) == 0:
            continue

        # Normalize values using min-max normalization
        arr[:,0] = (arr[:,0].astype(np.float32) - min_lat)/(max_lat - min_lat)
        arr[:,1] = (arr[:,1].astype(np.float32) - min_lon)/(max_lon - min_lon)
        arr[:,2] = (arr[:,2].astype(np.float32) - min_sog)/(max_sog - min_sog)
        arr[:,3] = (arr[:,3].astype(np.float32) - min_cog)/(max_cog - min_cog)
        arr[:, :4][arr[:, :4] >= 1] = 0.9999

        # Store in format required by TrAISformer model
        tmp["mmsi"] = mmsi
        tmp["traj"] = arr
        tmp["port"] = (port_coords, dist2port)
        pickle_list.append(tmp)

    print("[*] Writing to file")
    with open("data/tracks.pkl", "wb") as f:
        pickle.dump(pickle_list, f)
    print(f"[+] Saved to file: {len(pickle_list)} tracks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Match AIS and ERS messages to create AIS tracks')
    parser.add_argument('ais', type=str, help='AIS CSV file')
    parser.add_argument('ers', type=str, help='ERS CSV file')
    args = parser.parse_args()
    match_ers_ais(args.ais, args.ers)