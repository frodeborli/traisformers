import folium
import ast
from csv import DictReader
from ais.ais_data import AISSmall

class Map:
    def __init__(self, center: list=[68.0, 8.4], zoom: int=5):
        self.center = center
        self.zoom = zoom
        self.map: folium.Map = self._create_map()

    def render_map(self):
        """
        Render the map and return the HTML code
        """
        folium.LayerControl(collapsed=False).add_to(self.map)
        return self.map.get_root().render()

    def add_trip(self, trip: list[AISSmall], ports: bool=True, color: str="blue", show_ports=False):
        """
        Add AIS track to the map, also add ports if specified
        """
        fg = folium.FeatureGroup(name="Route", show=True)
        self._draw_markers(fg, trip, color=color)
        fg.add_to(self.map)
        if ports:
            fg_ports = folium.FeatureGroup(name="Ports", show=show_ports)
            self._mark_ports(fg_ports)
            fg_ports.add_to(self.map)
        self.map.keep_in_front(fg)

    def add_bbox(self, show_area=False):
        """
        Visualize the ROI in the map
        """
        kw = {
            "color": "orange",
            "fill": True,
            "fill_color": "yellow",
            "weight": 2,
            "fill_opacity": 0.2
        }
        fg_area = folium.FeatureGroup(name="Area", show=show_area)
        folium.Rectangle(bounds=[[69.2, 13.0], [73.0, 31.5]], **kw).add_to(fg_area)
        fg_area.add_to(self.map)

    def _draw_markers(self, fg: folium.FeatureGroup, trip: list=[AISSmall], color: str="blue"):
        """
        Draw AIS track on the map as lines
        """
        try:
            for idx, msg in enumerate(trip):
                self._draw_line(fg, msg, trip[idx+1], color=color)
        except IndexError:
            pass

    def _mark_ports(self, fg: folium.FeatureGroup) -> None:
        """
        Add markers on the map indicating the locations of ports
        """
        with open("data/ports.csv", "r") as f:
            rows = DictReader(f)
            for port in rows:
                coords = ast.literal_eval(port["coords"])
                folium.Marker(location=list(coords), tooltip=f"<p> Coords: {tuple(coords)} </p>").add_to(fg)

    def _draw_line(self, fg: folium.FeatureGroup, startpos, stoppos, color) -> None:
        """
        Draw a line between the two given points on the map
        """
        folium.PolyLine(locations=[startpos, stoppos], color=color, weight=2, opacity=1).add_to(fg)

    def _create_map(self) -> folium.Map:
        """
        Create and return a folium map
        """
        map_ = folium.Map(location=self.center, zoom_start=self.zoom, prefer_canvas=True)
        return map_