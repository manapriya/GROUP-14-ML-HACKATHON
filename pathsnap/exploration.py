from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .data import list_trips
from .visualize import make_map


def inspect_dataset(data_root: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for partition in ["train", "student_test"]:
        for trip in list_trips(data_root, partition):
            gps = trip.gps
            gaps = np.diff(gps[:, 2])
            invalid_coords = int(((np.abs(gps[:, 0]) > 180) | (np.abs(gps[:, 1]) > 90)).sum())
            degree = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes)) + np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes))
            rows.append({
                "partition": partition, "trip_id": trip.trip_id,
                "gps_observations": len(gps), "road_nodes": len(trip.nodes), "road_arcs": len(trip.arcs),
                "reference_route_arcs": len(trip.route), "missing_gps_values": int((~np.isfinite(gps)).sum()),
                "invalid_coordinate_rows": invalid_coords,
                "timestamp_nonincreasing_steps": int((gaps <= 0).sum()),
                "median_interval_s": float(np.median(gaps)) if len(gaps) else None,
                "max_interval_s": float(np.max(gaps)) if len(gaps) else None,
                "isolated_nodes": int((degree == 0).sum()),
                "longitude_min": float(gps[:, 0].min()), "longitude_max": float(gps[:, 0].max()),
                "latitude_min": float(gps[:, 1].min()), "latitude_max": float(gps[:, 1].max()),
            })
    out = output_dir / "dataset_inventory.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for partition in ["train", "student_test"]:
        subset = [row for row in rows if row["partition"] == partition]
        summary[partition] = {
            "trips": len(subset), "gps_observations": int(sum(r["gps_observations"] for r in subset)),
            "median_trip_observations": float(np.median([r["gps_observations"] for r in subset])) if subset else 0,
            "median_sampling_interval_s": float(np.nanmedian([r["median_interval_s"] for r in subset])) if subset else None,
        }
    import json
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    student = list_trips(data_root, "student_test")
    if student:
        make_map(student[0], output_dir / f"trip_{student[0].trip_id}_eda.html", include_reference=True)
    return rows, summary

