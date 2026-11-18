from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import Trip
from .decoder import transition_continuity, viterbi_decode
from .geo import build_index, observation_candidates, route_coordinates


def load_model(path: Path):
    import pickle
    with path.open("rb") as f:
        return pickle.load(f)


def predict_trip(model, trip: Trip) -> dict:
    index = build_index(trip)
    candidates = observation_candidates(index, trip.gps, max_candidates=36)
    pos_idx = list(model.classes_).index(1)
    sizes = [len(ids) for ids, _ in candidates]
    flat_features = np.concatenate([features for _, features in candidates if len(features)], axis=0)
    flat_scores = model.predict_proba(flat_features)[:, pos_idx]
    offsets = np.cumsum([0] + sizes)
    score_lists = [flat_scores[offsets[i]:offsets[i+1]] for i in range(len(sizes))]
    classifier_ids = [int(ids[int(np.argmax(score_lists[i]))]) if len(ids) else -1 for i, (ids, _) in enumerate(candidates)]
    decoded_ids = viterbi_decode(candidates, score_lists, index, trip.gps)
    gps_coords = [[float(lat), float(lon)] for lon, lat, _ in trip.gps]
    predicted_coords = route_coordinates(index, decoded_ids)
    # Reference labels are read only after predictions have been made and are
    # returned only as aggregate metrics; neither route IDs nor geometry leave
    # the backend response.
    route_set = set(map(int, trip.route))
    hits, misses, matched = 0, 0, 0
    for cand in candidates:
        refs = [int(edge_id) for edge_id in cand[0] if int(edge_id) in route_set]
        if not refs:
            misses += 1
            continue
        matched += 1
    n = min(len(decoded_ids), len(trip.gps))
    # Create per-observation reference labels from the nearest candidate arc
    # belonging to the route, then score both classifier-only and HMM output.
    valid = []
    for i, cand in enumerate(candidates[:n]):
        refs = [(float(feat[0]), int(edge_id)) for edge_id, feat in zip(cand[0], cand[1]) if int(edge_id) in route_set]
        if refs:
            valid.append((i, min(refs)[1]))
    if valid:
        hits = sum(classifier_ids[i] == ref for i, ref in valid)
        hmm_hits = sum(decoded_ids[i] == ref for i, ref in valid)
    else:
        hmm_hits = 0
    metrics = {
        "observations": int(len(trip.gps)),
        "scored_observations": int(len(valid)),
        "classifier_top1_accuracy": float(hits / max(1, len(valid))),
        "hmm_top1_accuracy": float(hmm_hits / max(1, len(valid))),
        "candidate_miss_rate": float(misses / max(1, len(trip.gps))),
        "classifier_route_continuity": float(transition_continuity(classifier_ids, index, trip.gps)),
        "hmm_route_continuity": float(transition_continuity(decoded_ids, index, trip.gps)),
    }
    return {"trip_id": trip.trip_id, "gps": gps_coords, "predicted_route": predicted_coords, "metrics": metrics}

