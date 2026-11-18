from __future__ import annotations

from functools import lru_cache
import heapq
import math

import numpy as np

from .geo import RoadIndex


def _graph(index: RoadIndex):
    if index.graph is not None:
        return index.graph
    graph: list[list[tuple[int, int, float]]] = [[] for _ in range(len(index.xy))]
    for edge_id, (src, dst) in enumerate(index.arcs):
        graph[int(src)].append((int(dst), edge_id, float(index.length[edge_id])))
    return graph


def _bounded_distances(index: RoadIndex, source: int):
    cache = index.path_cache if index.path_cache is not None else {}
    if source in cache:
        return cache[source]
    graph = _graph(index)
    dist = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if d != dist.get(node) or d > 900.0:
            continue
        for nxt, _, length in graph[node]:
            nd = d + length
            if nd <= 900.0 and nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    cache[source] = dist
    if len(cache) > 8192:
        cache.pop(next(iter(cache)))
    return dist


def viterbi_decode(candidate_lists, scores, index: RoadIndex, gps: np.ndarray, beam: int = 10) -> list[int]:
    """Decode road-arc states using emission scores and network-constrained transitions.

    The state at time t is one candidate directed arc for GPS observation t.
    Transition cost compares the implied network travel distance with observed
    displacement. If no plausible path exists, the decoder starts a new segment.
    """
    if not candidate_lists:
        return []
    graph = _graph(index)
    arcs = index.arcs
    # Keep a small beam of classifier candidates per observation.
    choices = []
    emission = []
    for cand, score in zip(candidate_lists, scores):
        candidate_ids, _ = cand
        if not len(candidate_ids):
            choices.append([]); emission.append(np.empty(0)); continue
        order = np.argsort(score)[::-1][:beam]
        choices.append(candidate_ids[order].astype(int).tolist())
        p = np.asarray(score)[order]
        p = np.clip(p, 1e-8, 1.0)
        emission.append(np.log(p))
    values: list[np.ndarray] = []
    back: list[np.ndarray] = []
    xy = gps[:, :2]
    mean_lat = float(np.mean(xy[:, 1]))
    dx_scale = 111_320.0 * math.cos(math.radians(mean_lat))
    dy_scale = 111_320.0
    for t, curr in enumerate(choices):
        if not curr:
            values.append(np.empty(0)); back.append(np.empty(0, dtype=int)); continue
        curr_emit = emission[t]
        if t == 0 or not choices[t - 1]:
            values.append(curr_emit.copy()); back.append(np.full(len(curr), -1, dtype=int)); continue
        prev = choices[t - 1]
        prev_score = values[t - 1]
        observed = math.hypot((xy[t, 0] - xy[t-1, 0]) * dx_scale, (xy[t, 1] - xy[t-1, 1]) * dy_scale)
        dt = max(1.0, float(gps[t, 2] - gps[t-1, 2]))
        max_network = min(900.0, max(250.0, observed * 4.0 + 100.0, dt * 45.0))
        curr_val = np.full(len(curr), -np.inf)
        curr_back = np.zeros(len(curr), dtype=int)
        for j, edge_b in enumerate(curr):
            bsrc, bdst = map(int, arcs[edge_b])
            for k, edge_a in enumerate(prev):
                asrc, adst = map(int, arcs[edge_a])
                if edge_a == edge_b:
                    connector = 0.0
                else:
                    connector = _bounded_distances(index, adst).get(bsrc, float("inf"))
                if connector > max_network:
                    continue
                # A small soft penalty tolerates irregular graph metadata while
                # rewarding routes whose implied speed agrees with GPS movement.
                implied = (float(index.length[edge_a]) * 0.5 + connector + float(index.length[edge_b]) * 0.5)
                scale = max(25.0, observed * 0.65 + 12.0)
                trans = -abs(implied - observed) / scale - max(0.0, implied / dt - 55.0) / 10.0
                candidate = prev_score[k] + trans + curr_emit[j]
                if candidate > curr_val[j]:
                    curr_val[j] = candidate
                    curr_back[j] = k
        if not np.isfinite(curr_val).any():
            curr_val = curr_emit.copy()
            curr_back = np.full(len(curr), -1, dtype=int)
        values.append(curr_val)
        back.append(curr_back)
    result = [choices[t][0] if choices[t] else -1 for t in range(len(choices))]
    t = len(choices) - 1
    while t >= 0:
        if not choices[t]:
            t -= 1; continue
        state = int(np.argmax(values[t]))
        result[t] = choices[t][state]
        while t > 0 and back[t][state] >= 0:
            state = int(back[t][state]); t -= 1
        t -= 1
    # Fill occasional empty candidate slots using nearest decoded neighbour.
    last = next((x for x in result if x >= 0), 0)
    for i, value in enumerate(result):
        if value < 0:
            result[i] = last
        else:
            last = value
    return result


def transition_continuity(route: list[int], index: RoadIndex, gps: np.ndarray | None = None) -> float:
    if len(route) < 2:
        return 1.0
    graph = _graph(index)
    connected = 0
    total = 0
    for i, (a, b) in enumerate(zip(route[:-1], route[1:])):
        if a < 0 or b < 0:
            continue
        total += 1
        if a == b or int(index.arcs[a, 1]) == int(index.arcs[b, 0]):
            connected += 1
            continue
        threshold = 300.0
        if gps is not None and i + 1 < len(gps):
            delta = gps[i + 1, :2] - gps[i, :2]
            mean_lat = (gps[i + 1, 1] + gps[i, 1]) / 2
            displacement = math.hypot(delta[0] * 111_320 * math.cos(math.radians(mean_lat)), delta[1] * 111_320)
            threshold = min(900.0, max(150.0, displacement * 3.0 + 50.0))
        distance = _bounded_distances(index, int(index.arcs[a, 1])).get(int(index.arcs[b, 0]), float("inf"))
        if distance <= threshold:
            connected += 1
    return connected / max(1, total)

