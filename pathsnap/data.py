from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trip:
    trip_id: str
    partition: str
    gps: np.ndarray  # longitude, latitude, timestamp
    nodes: np.ndarray  # longitude, latitude
    arcs: np.ndarray  # from-node, to-node; row index is arc ID
    route: np.ndarray


def _read(path: Path, columns: int, dtype=float) -> np.ndarray:
    values = np.loadtxt(path, dtype=dtype, ndmin=2)
    if values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(f"Expected {columns} columns in {path}, got {values.shape}")
    return values


def load_trip(path: Path, partition: str) -> Trip:
    trip_id = path.name
    gps = _read(path / f"{trip_id}.track", 3)
    nodes = _read(path / f"{trip_id}.nodes", 2)
    arcs = _read(path / f"{trip_id}.arcs", 2, dtype=np.int64)
    route_path = path / f"{trip_id}.route"
    route = np.loadtxt(route_path, dtype=np.int64, ndmin=1) if route_path.exists() else np.empty(0, dtype=np.int64)
    gps = gps[np.isfinite(gps).all(axis=1)]
    arcs = arcs[(arcs >= 0).all(axis=1) & (arcs < len(nodes)).all(axis=1)]
    route = route[(route >= 0) & (route < len(arcs))]
    if len(gps) == 0 or len(arcs) == 0:
        raise ValueError(f"Trip {trip_id} has no valid GPS observations or road arcs")
    return Trip(trip_id, partition, gps, nodes, arcs, route)


def list_trips(data_root: Path, partition: str) -> list[Trip]:
    folder = data_root / partition
    if not folder.exists():
        return []
    return [load_trip(path, partition) for path in sorted(folder.iterdir()) if path.is_dir()]

