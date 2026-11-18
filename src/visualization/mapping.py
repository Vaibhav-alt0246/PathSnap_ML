"""Folium visualization for genuine trip predictions."""
import folium
import pandas as pd
from src.data.loader import normalize_trip

def create_map(trip, predicted_arc_ids=None, reference_arc_ids=None, *, classifier_arc_ids=None):
    t = normalize_trip(trip)
    valid = t.track[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce").dropna()
    center = [float(valid.latitude.mean()), float(valid.longitude.mean())] if len(valid) else [0.0, 0.0]
    m = folium.Map(center, zoom_start=15, control_scale=True, tiles=None)
    m.get_root().header.add_child(folium.Element("""
    <style>
      html, body, .folium-map, .leaflet-container { opacity: 1 !important; filter: none !important; }
      .leaflet-container { background: #e8eef3 !important; }
      .leaflet-control, .leaflet-control-layers { opacity: 1 !important; color: #172033 !important; }
      .leaflet-control-layers { background: rgba(255,255,255,.98) !important; border: 1px solid #7b8794 !important; }
      .leaflet-control-zoom a { color: #172033 !important; background: #ffffff !important; }
      .leaflet-control-attribution { background: rgba(255,255,255,.96) !important; color: #172033 !important; }
    </style>
    """))
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True, show=True,
                     attr="© OpenStreetMap contributors").add_to(m)
    gps = folium.FeatureGroup(name="GPS observations", show=True)
    gps_points = []
    for _, row in t.track.iterrows():
        lat, lon = pd.to_numeric(row.latitude, errors="coerce"), pd.to_numeric(row.longitude, errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue
        point = [float(lat), float(lon)]
        gps_points.append(point)
        folium.CircleMarker(point, radius=4, color="#e76f00", weight=1,
                            fill=True, fill_color="#f4a261", fill_opacity=0.95,
                            tooltip="GPS observation").add_to(gps)
    gps.add_to(m)
    pred = set(int(x) for x in (predicted_arc_ids or [])); ref = set(int(x) for x in (reference_arc_ids or [])); clf = set(int(x) for x in (classifier_arc_ids or []))
    roads = folium.FeatureGroup(name="Road network", show=False); predicted = folium.FeatureGroup(name="HMM predicted route", show=True); classifier = folium.FeatureGroup(name="Classifier selections", show=False); reference = folium.FeatureGroup(name="Reference route", show=False)
    for aid, row in t.arcs.iterrows():
        coords = [[float(t.nodes.loc[int(row.from_node), "latitude"]), float(t.nodes.loc[int(row.from_node), "longitude"])], [float(t.nodes.loc[int(row.to_node), "latitude"]), float(t.nodes.loc[int(row.to_node), "longitude"])]]
        folium.PolyLine(coords, color="#8a97a8", weight=2, opacity=0.65, tooltip=f"Arc {int(aid)}").add_to(roads)
        if int(aid) in pred: folium.PolyLine(coords, color="#0057b8", weight=6, opacity=0.98, tooltip=f"HMM predicted arc {int(aid)}").add_to(predicted)
        if int(aid) in clf: folium.PolyLine(coords, color="#7b2cbf", weight=3, opacity=0.9, dash_array="6, 6", tooltip=f"Classifier arc {int(aid)}").add_to(classifier)
        if int(aid) in ref: folium.PolyLine(coords, color="green", weight=5, tooltip=f"Reference arc {int(aid)}").add_to(reference)
    roads.add_to(m); predicted.add_to(m); classifier.add_to(m); reference.add_to(m)
    legend = '<div style="position: fixed; bottom: 24px; left: 24px; z-index:9999; background:white; padding:10px 12px; border:1px solid #9aa5b1; border-radius:6px; color:#172033; font:13px Arial; box-shadow:0 1px 5px #777"><b>PathSnap layers</b><br><span style="color:#0057b8">━━</span> HMM predicted route<br><span style="color:#7b2cbf">╌╌</span> Classifier selections<br><span style="color:#e76f00">●</span> GPS observations<br><span style="color:#8a97a8">━━</span> Road network</div>'
    m.get_root().html.add_child(folium.Element(legend)); folium.LayerControl(collapsed=False).add_to(m)
    if gps_points:
        m.fit_bounds(gps_points, padding=(24, 24), max_zoom=17)
    return m
