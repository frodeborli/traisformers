from dataclasses import dataclass, field

@dataclass
class AISSmall:
    """
    Dataclass for a single AIS message
    The dataclass denormalizes the latitude, longitude, SOG and COG values
    """
    latitude: float
    longitude: float
    sog: float
    cog: float
    timestamp: float = 0
    mmsi: float = 0
    coords: (float, float) = field(init=False)

    def __post_init__(self):
        min_lat, max_lat = 69.2, 73.0
        min_lon, max_lon = 13.0, 31.5
        min_sog, max_sog = 0, 30
        min_cog, max_cog = 0, 360

        self.latitude = self.latitude * (max_lat - min_lat) + min_lat
        self.longitude = self.longitude * (max_lon - min_lon) + min_lon
        self.sog = self.sog * (max_sog - min_sog) + min_sog
        self.cog = self.cog * (max_cog - min_cog) + min_cog
        self.coords = (self.latitude, self.longitude)