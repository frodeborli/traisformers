import folium
from flask import Flask, render_template
import numpy as np
from map_.map_ import Map
from ais.ais_data import AISSmall
from port_pred.predict import predict
from port_pred.utils import read_files, mov_avg

app = Flask(__name__)

@app.route("/<int:in_len>/<int:idx>", methods=["GET"])
def pred(in_len, idx):
    colors = ["lime", "blue", "green", "yellow", "purple", "orange", "pink", "black", "darkgreen", "brown", "cyan", "magenta", "teal", "gray", "maroon", "olive"]

    # Get the necessary data
    data_dir = "./data/"
    preds, true, ports = read_files(data_dir+f"preds/{in_len}.pkl", data_dir+"true.pkl", data_dir+"ports.csv", idx=idx)
    port, prob, pred_ports, trajs = predict(preds, ports)
    true_port = true.get("port")[0]
    map_ = Map()

    # Draw smoothened out trajectory predictions
    for i, pred in enumerate(trajs):
        ac_pred = pred[in_len:]
        ac_pred = [AISSmall(*x) for x in ac_pred]
        ac_pred = np.array([x.coords for x in ac_pred])
        ac_pred = mov_avg(ac_pred, 5)
        map_.add_trip(ac_pred, ports=False, show_ports=False, color=colors[i])

    # Draw the true trajectory of the vessel
    true_traj = [AISSmall(*x) for x in true.get("traj")[in_len:]]
    true_traj = np.array([x.coords for x in true_traj])
    map_.add_trip(true_traj, ports=True, show_ports=False, color="red")

    # Draw the input to the TrAISformer model
    true_traj = [AISSmall(*x) for x in true.get("traj")[:in_len]]
    true_traj = np.array([x.coords for x in true_traj])
    map_.add_trip(true_traj, ports=False, show_ports=False, color="white")

    # Put markers on relevant ports, with predicted probabilities attached
    if port:
        # Ports with a non-zero probability which is
        # not the most probable or true port
        for p in pred_ports:
            if p == port or p == true_port:
                continue
            tooltip = folium.Tooltip(permanent=True, text=f"{pred_ports.get(p, 0):.2f}")
            folium.Marker(location=list(p), tooltip=tooltip, icon=folium.Icon(color="gray")).add_to(map_.map)

        # Put a marker on the predicted port
        tooltip = folium.Tooltip(permanent=True, text=f"<b>{prob:.2f}</b>")
        folium.Marker(location=list(port), tooltip=tooltip, icon=folium.Icon(color="blue")).add_to(map_.map)

    # Put a marker on the true port, if we predicted wrong, with the probability attached
    if port != true_port:
        prob = pred_ports.get(true_port, 0)
        tooltip = folium.Tooltip(permanent=True, text=f"<b>{prob:.2f}</b>")
        folium.Marker(location=list(true_port), tooltip=tooltip, icon=folium.Icon(color="red")).add_to(map_.map)
        folium.Circle(list(true_port), radius=5000, color="red").add_to(map_.map)
    return map_.render_map()

# Mark all ports
@app.route("/ports", methods=["GET"])
def ports():
    map_ = Map()
    map_.add_trip([], ports=True, show_ports=True)
    map_.add_bbox()
    return map_.render_map()

# Mark our ROI
@app.route("/area", methods=["GET"])
def area():
    map_ = Map()
    map_.add_bbox(show_area=True)
    return map_.render_map()

# Home page describing how to use application
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
