from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np

from .data import Trip

EARTH_M = 111_320.0
FEATURE_NAMES = [
    "distance_m", "signed_cross_track_m", "along_fraction", "arc_length_m",
    "heading_error_rad", "heading_valid", "gps_speed_mps", "time_gap_s",
    "candidate_rank", "distance_margin_m", "distance_squared_m",
    "candidate_start_degree", "candidate_end_degree", "same_as_previous_best",
    "previous_gps_distance_m", "next_gps_distance_m", "previous_heading_error_rad",
    "next_heading_error_rad", "local_turn_rad", "timestamp_sin", "timestamp_cos",
]


@dataclass
class RoadIndex:
    xy: np.ndarray
    arcs: np.ndarray
    a: np.ndarray
    b: np.ndarray
    vec: np.ndarray
    length: np.ndarray
    bearing: np.ndarray
    out_degree: np.ndarray
    in_degree: np.ndarray
    buckets: dict[tuple[int, int], list[int]]
    cell: float = 100.0
    radius: float = 180.0
    reference_lat: float = 0.0
    graph: list | None = None
    path_cache: dict | None = None


def project(lonlat: np.ndarray, reference_lat: float) -> np.ndarray:
    coords = np.asarray(lonlat, dtype=float)
    return np.column_stack((coords[:, 0] * EARTH_M * math.cos(math.radians(reference_lat)), coords[:, 1] * EARTH_M))


def build_index(trip: Trip, cell: float = 100.0, radius: float = 180.0) -> RoadIndex:
    reference_lat = float(np.nanmean(trip.gps[:, 1]))
    xy = project(trip.nodes, reference_lat)
    src, dst = trip.arcs[:, 0], trip.arcs[:, 1]
    a, b = xy[src], xy[dst]
    vec = b - a
    length = np.maximum(np.linalg.norm(vec, axis=1), 0.01)
    bearing = np.arctan2(vec[:, 1], vec[:, 0])
    out_degree = np.bincount(src, minlength=len(xy))
    in_degree = np.bincount(dst, minlength=len(xy))
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    lo = np.floor((np.minimum(a, b) - radius) / cell).astype(int)
    hi = np.floor((np.maximum(a, b) + radius) / cell).astype(int)
    for edge_id, (x0, y0) in enumerate(lo):
        x1, y1 = hi[edge_id]
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                buckets[(cx, cy)].append(edge_id)
    graph: list[list[tuple[int, int, float]]] = [[] for _ in range(len(xy))]
    for edge_id, (src, dst) in enumerate(trip.arcs):
        graph[int(src)].append((int(dst), edge_id, float(length[edge_id])))
    return RoadIndex(xy, trip.arcs, a, b, vec, length, bearing, out_degree, in_degree, dict(buckets), cell, radius, reference_lat, graph, {})


def _angle_diff(a: float, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def _gps_context(track: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delta = np.diff(xy, axis=0, prepend=xy[:1])
    next_delta = np.diff(xy, axis=0, append=xy[-1:])
    gap = np.diff(track[:, 2], prepend=track[0, 2] - 1.0)
    gap = np.maximum(gap, 0.01)
    speed = np.linalg.norm(delta, axis=1) / gap
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    moving = np.linalg.norm(delta, axis=1) > 1.5
    heading[~moving] = np.nan
    next_heading = np.arctan2(next_delta[:, 1], next_delta[:, 0])
    next_heading[np.linalg.norm(next_delta, axis=1) <= 1.5] = np.nan
    return gap, speed, heading, next_heading, delta, next_delta


def observation_candidates(index: RoadIndex, gps: np.ndarray, max_candidates: int = 48) -> list[tuple[np.ndarray, np.ndarray]]:
    xy = project(gps[:, :2], float(np.nanmean(gps[:, 1])))
    gap, speed, heading, next_heading, delta, next_delta = _gps_context(gps, xy)
    output: list[tuple[np.ndarray, np.ndarray]] = []
    best_ids: list[int] = []
    best_distances: list[float] = []
    for i, point in enumerate(xy):
        cx, cy = np.floor(point / index.cell).astype(int)
        found: set[int] = set()
        reach = int(math.ceil(index.radius / index.cell)) + 1
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                found.update(index.buckets.get((cx + dx, cy + dy), ()))
        if not found:
            # Defensive fallback to a coarse bounding-box scan for tiny networks.
            found = set(range(len(index.arcs)))
        ids = np.fromiter(found, dtype=np.int64)
        v = index.vec[ids]
        rel = point - index.a[ids]
        frac = np.clip(np.einsum("ij,ij->i", rel, v) / (index.length[ids] ** 2), 0.0, 1.0)
        proj = index.a[ids] + frac[:, None] * v
        dist = np.linalg.norm(point - proj, axis=1)
        keep = dist <= index.radius
        if not np.any(keep):
            keep[np.argmin(dist)] = True
        ids, frac, dist, proj = ids[keep], frac[keep], dist[keep], proj[keep]
        order = np.argsort(dist)[:max_candidates]
        ids, frac, dist, proj = ids[order], frac[order], dist[order], proj[order]
        rank = np.arange(len(ids), dtype=float)
        start, end = index.arcs[ids, 0], index.arcs[ids, 1]
        h = heading[i]
        nh = next_heading[i]
        prev_err = _angle_diff(float(h) if np.isfinite(h) else 0.0, index.bearing[ids])
        next_err = _angle_diff(float(nh) if np.isfinite(nh) else 0.0, index.bearing[ids])
        previous_dist = float(np.linalg.norm(point - xy[max(0, i - 1)])) if i else 0.0
        next_dist = float(np.linalg.norm(point - xy[min(len(xy) - 1, i + 1)]))
        local_turn = float(_angle_diff(math.atan2(next_delta[i, 1], next_delta[i, 0]), np.array([math.atan2(delta[ i, 1], delta[i, 0])]))[0]) if i > 0 and np.linalg.norm(delta[i]) > 0 and np.linalg.norm(next_delta[i]) > 0 else 0.0
        margin = dist - dist[0]
        times = gps[i, 2] / 86400.0 * 2.0 * np.pi
        features = np.column_stack((
            dist, np.sign(index.vec[ids, 0] * (point - proj)[:, 1] - index.vec[ids, 1] * (point - proj)[:, 0]) * dist,
            frac, index.length[ids], prev_err, np.full(len(ids), float(np.isfinite(h))),
            np.full(len(ids), min(speed[i], 70.0)), np.full(len(ids), min(gap[i], 300.0)),
            rank, margin, dist**2, index.out_degree[start], index.in_degree[end],
            np.full(len(ids), float(bool(best_ids) and ids[0] == best_ids[-1])),
            np.full(len(ids), previous_dist), np.full(len(ids), next_dist), prev_err, next_err,
            np.full(len(ids), local_turn), np.full(len(ids), math.sin(times)), np.full(len(ids), math.cos(times)),
        ))
        output.append((ids.astype(np.int32), features.astype(np.float32)))
        best_ids.append(int(ids[0]))
        best_distances.append(float(dist[0]))
    return output


def label_candidates(candidates: list[list[tuple[int, np.ndarray]]], route: np.ndarray, index: RoadIndex) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    route_set = set(map(int, route))
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    misses = 0
    for cand in candidates:
        ids, feat = cand
        on_route = np.flatnonzero(np.isin(ids, list(route_set)))
        if len(on_route) == 0:
            misses += 1
            rows.append(feat)
            labels.append(np.zeros(len(ids), dtype=np.int8))
            continue
        # Label the nearest candidate edge that belongs to the reference route.
        positive = int(on_route[np.argmin(feat[on_route, 0])])
        y = np.zeros(len(ids), dtype=np.int8)
        y[positive] = 1
        rows.append(feat)
        labels.append(y)
    return rows, labels, misses


def route_coordinates(index: RoadIndex, route: np.ndarray) -> list[list[float]]:
    coords = []
    x_scale = EARTH_M * math.cos(math.radians(index.reference_lat))
    for edge_id in route:
        if 0 <= int(edge_id) < len(index.arcs):
            src, dst = index.arcs[int(edge_id)]
            coords.append([float(index.xy[src, 1] / EARTH_M), float(index.xy[src, 0] / x_scale)])
    if len(route) and int(route[-1]) < len(index.arcs):
        end = index.arcs[int(route[-1]), 1]
        coords.append([float(index.xy[end, 1] / EARTH_M), float(index.xy[end, 0] / x_scale)])
    return coords

