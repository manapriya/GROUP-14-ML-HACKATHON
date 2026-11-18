from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathsnap.experiment import run_experiments
from pathsnap.exploration import inspect_dataset
from pathsnap.data import list_trips
from pathsnap.geo import build_index
from pathsnap.inference import load_model, predict_trip
from pathsnap.visualize import make_map

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train and compare PathSnap route-matching models")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "Student folder")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--trials", type=int, default=3, help="Optuna Bayesian search trials for the CV-selected model")
    parser.add_argument("--eda-only", action="store_true", help="Inventory the supplied partitions and make a Folium map")
    args = parser.parse_args()
    inspect_dataset(args.data, args.output / "exploration")
    if args.eda_only:
        print(f"EDA outputs saved to {args.output / 'exploration'}")
        return
    results = run_experiments(args.data, args.output, trials=args.trials)
    selected = results["selected_model"] + "_tuned"
    best_trip = list_trips(args.data, "student_test")[0]
    model = load_model(args.output / "model.joblib")
    prediction = predict_trip(model, best_trip)
    from pathsnap.geo import observation_candidates
    from pathsnap.decoder import viterbi_decode
    idx = build_index(best_trip)
    candidates = observation_candidates(idx, best_trip.gps)
    pos_idx = list(model.classes_).index(1)
    sizes = [len(ids) for ids, _ in candidates]
    flat = model.predict_proba(__import__("numpy").concatenate([features for _, features in candidates], axis=0))[:, pos_idx]
    offsets = __import__("numpy").cumsum([0] + sizes)
    scores = [flat[offsets[i]:offsets[i+1]] for i in range(len(sizes))]
    decoded = viterbi_decode(candidates, scores, idx, best_trip.gps)
    make_map(best_trip, args.output / "maps" / f"trip_{best_trip.trip_id}_prediction.html", predicted_route=decoded, include_reference=True)
    print(json.dumps({"selected_tuned_model": selected, "student_demo_metrics": prediction["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
