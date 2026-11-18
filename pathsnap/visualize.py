from __future__ import annotations

from pathlib import Path

import folium
from folium.plugins import Fullscreen

from .data import Trip
from .geo import build_index, route_coordinates


def make_map(trip: Trip, output: Path, predicted_route=None, include_reference: bool = True):
    center = [float(trip.gps[:, 1].mean()), float(trip.gps[:, 0].mean())]
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")
    Fullscreen(position="topright").add_to(fmap)
    index = build_index(trip)
    south, north = float(trip.gps[:, 1].min()) - 0.006, float(trip.gps[:, 1].max()) + 0.006
    west, east = float(trip.gps[:, 0].min()) - 0.008, float(trip.gps[:, 0].max()) + 0.008
    roads = folium.FeatureGroup(name="Road network", show=True)
    # Clip the full per-trip graph to the observed trajectory extent.
    for src, dst in trip.arcs:
        p1, p2 = trip.nodes[int(src)], trip.nodes[int(dst)]
        if (south <= p1[1] <= north and west <= p1[0] <= east) or (south <= p2[1] <= north and west <= p2[0] <= east):
            folium.PolyLine([[float(p1[1]), float(p1[0])], [float(p2[1]), float(p2[0])]], color="#747e8b", weight=1, opacity=.38).add_to(roads)
    roads.add_to(fmap)
    gps_layer = folium.FeatureGroup(name="Observed GPS", show=True)
    gps_points = [[float(lat), float(lon)] for lon, lat, _ in trip.gps]
    folium.PolyLine(gps_points, color="#dd872e", weight=2, opacity=.65).add_to(gps_layer)
    for p in gps_points[::max(1, len(gps_points)//300)]:
        folium.CircleMarker(p, radius=2.5, color="#bd6b19", fill=True, fill_color="#ed9b4a", fill_opacity=.85).add_to(gps_layer)
    gps_layer.add_to(fmap)
    if include_reference and len(trip.route):
        ref = folium.FeatureGroup(name="Reference route (development data)", show=False)
        folium.PolyLine(route_coordinates(index, trip.route), color="#4e9563", weight=4, opacity=.85).add_to(ref)
        ref.add_to(fmap)
    if predicted_route is not None:
        pred = folium.FeatureGroup(name="Predicted route", show=True)
        folium.PolyLine(route_coordinates(index, predicted_route), color="#087f78", weight=5, opacity=.95).add_to(pred)
        pred.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[south, west], [north, east]])
    output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output))
    return output

