from __future__ import annotations

import json
import pickle
import random
import time
import csv
from pathlib import Path

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from .data import Trip, list_trips
from .decoder import transition_continuity, viterbi_decode
from .geo import FEATURE_NAMES, build_index, label_candidates, observation_candidates
from .models import model_catalog

SEED = 2718


def prepare_trip(trip: Trip):
    index = build_index(trip)
    candidates = observation_candidates(index, trip.gps, max_candidates=36)
    rows, labels, misses = label_candidates(candidates, trip.route, index)
    observations = list(zip(candidates, labels))
    return index, candidates, observations, misses


def _fit_rows(trips_data, trip_ids, rng: np.random.Generator, negatives_per_observation: int = 4):
    xs, ys = [], []
    for trip_id in trip_ids:
        _, _, observations, _ = trips_data[trip_id]
        for candidates, labels in observations:
            positives = np.flatnonzero(labels == 1)
            if not len(positives):
                continue
            negatives = np.flatnonzero(labels == 0)
            if len(negatives) > negatives_per_observation:
                near_n = max(1, negatives_per_observation // 2)
                near = negatives[:near_n]
                remaining = negatives[near_n:]
                extra_n = negatives_per_observation - len(near)
                extra = rng.choice(remaining, min(extra_n, len(remaining)), replace=False) if extra_n and len(remaining) else np.empty(0, dtype=int)
                negatives = np.unique(np.concatenate((near, extra)))
            keep = np.concatenate((positives, negatives))
            xs.append(candidates[1][keep])
            ys.append(labels[keep])
    if not xs:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int8)
    return np.concatenate(xs, axis=0).astype(np.float32, copy=False), np.concatenate(ys).astype(np.int8, copy=False)


def _scores_for_trip(model, candidates):
    sizes = [len(ids) for ids, _ in candidates]
    x_all = np.concatenate([features for _, features in candidates if len(features)], axis=0)
    pos_idx = list(model.classes_).index(1)
    probabilities = model.predict_proba(x_all)[:, pos_idx]
    offsets = np.cumsum([0] + sizes)
    return [probabilities[offsets[i]:offsets[i + 1]] for i in range(len(sizes))]


def _evaluate_trip(model, trip: Trip, data):
    index, candidates, observations, misses = data
    valid_obs, truths, ys, candidate_ids, obs_indices = [], [], [], [], []
    for i, cand in enumerate(candidates):
        labels = observations[i][1]
        if not labels.any():
            continue
        valid_obs.append(cand)
        obs_indices.append(i)
        truths.append(int(np.argmax(labels)))
        ys.extend(labels.tolist())
        candidate_ids.append(cand[0].tolist())
    if not valid_obs:
        return {"top1_accuracy": 0.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0,
                "log_loss": None, "candidate_miss_rate": 1.0, "classifier_continuity": 0.0,
                "hmm_top1_accuracy": 0.0, "hmm_continuity": 0.0, "inference_ms": 0.0,
                "observations_scored": 0, "candidate_rows": 0}
    start = time.perf_counter()
    flat_scores = _scores_for_trip(model, valid_obs)
    elapsed = (time.perf_counter() - start) * 1000.0
    pred_positions = [int(np.argmax(s)) for s in flat_scores]
    pred_ids = [ids[pos] for ids, pos in zip(candidate_ids, pred_positions)]
    true_ids = [candidate_ids[i][truth] for i, truth in enumerate(truths)]
    all_y = np.asarray(ys, dtype=int)
    all_p = np.concatenate(flat_scores)
    all_hard = (all_p >= 0.5).astype(int)
    try:
        ll = float(log_loss(all_y, np.clip(all_p, 1e-8, 1 - 1e-8), labels=[0, 1]))
    except ValueError:
        ll = None
    evaluation_gps = trip.gps[obs_indices]
    hmm_ids = viterbi_decode(valid_obs, flat_scores, index, evaluation_gps)
    return {
        "top1_accuracy": float(np.mean(np.asarray(pred_ids) == np.asarray(true_ids))),
        "macro_f1": float(f1_score(all_y, all_hard, average="macro", zero_division=0)),
        "precision": float(precision_score(all_y, all_hard, zero_division=0)),
        "recall": float(recall_score(all_y, all_hard, zero_division=0)),
        "log_loss": ll,
        "candidate_miss_rate": float(misses / max(1, len(trip.gps))),
        "classifier_continuity": transition_continuity(pred_ids, index, evaluation_gps),
        "hmm_top1_accuracy": float(np.mean(np.asarray(hmm_ids) == np.asarray(true_ids))),
        "hmm_continuity": transition_continuity(hmm_ids, index, evaluation_gps),
        "inference_ms": float(elapsed),
        "observations_scored": len(valid_obs),
        "candidate_rows": len(all_y),
    }


def _cv_score(estimator, trips_data, trip_ids, n_splits=3, seed=SEED):
    splitter = GroupKFold(n_splits=n_splits)
    ids = np.asarray(trip_ids)
    fold_scores, fold_f1 = [], []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(ids, groups=ids), start=1):
        rng = np.random.default_rng(seed + fold)
        train_ids, valid_ids = ids[train_idx].tolist(), ids[valid_idx].tolist()
        x_train, y_train = _fit_rows(trips_data, train_ids, rng)
        model = clone(estimator)
        model.fit(x_train, y_train)
        correct, total, all_true, all_pred = 0, 0, [], []
        for trip_id in valid_ids:
            _, candidates, observations, _ = trips_data[trip_id]
            scores = _scores_for_trip(model, candidates)
            for (cand_ids, _), (_, y), p in zip(candidates, observations, scores):
                if not len(cand_ids) or not y.any():
                    continue
                correct += int(y[int(np.argmax(p))] == 1)
                total += 1
                all_true.extend(y.tolist())
                all_pred.extend((p >= 0.5).astype(int).tolist())
        fold_scores.append(correct / max(1, total))
        fold_f1.append(float(f1_score(all_true, all_pred, average="macro", zero_division=0)))
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), float(np.mean(fold_f1))


def _tuned_estimator(name, trial):
    if name == "logistic_regression":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=trial.suggest_float("C", 0.03, 10.0, log=True), max_iter=600, class_weight="balanced", random_state=SEED))
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=trial.suggest_int("n_estimators", 32, 64, step=16), max_depth=trial.suggest_int("max_depth", 8, 16, step=4), min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4), max_features=trial.suggest_categorical("max_features", ["sqrt", 0.8, 1.0]), class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=trial.suggest_int("n_estimators", 48, 96, step=24), max_depth=trial.suggest_int("max_depth", 10, 22, step=4), min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4), max_features=trial.suggest_categorical("max_features", ["sqrt", 0.8, 1.0]), class_weight="balanced", n_jobs=-1, random_state=SEED)
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=trial.suggest_int("max_iter", 100, 260, step=40), max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 47, step=8), learning_rate=trial.suggest_float("learning_rate", 0.025, 0.16, log=True), l2_regularization=trial.suggest_float("l2_regularization", 0.1, 5.0, log=True), random_state=SEED)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=trial.suggest_int("n_estimators", 64, 128, step=32), max_depth=trial.suggest_int("max_depth", 3, 8), learning_rate=trial.suggest_float("learning_rate", 0.025, 0.18, log=True), subsample=trial.suggest_float("subsample", 0.65, 1.0), colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0), min_child_weight=trial.suggest_int("min_child_weight", 1, 8), reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0, log=True), eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=SEED)
    raise ValueError(f"No search space defined for {name}")


def _plot_comparison(results, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results["models"])
    labels = [name.replace("_", " ").title() for name in names]
    metrics = [
        ("cv_top1_accuracy", "Trip-grouped CV top-1"),
        ("top1_accuracy", "Student-test top-1"),
        ("hmm_top1_accuracy", "Student-test after Viterbi"),
        ("macro_f1", "Student-test candidate macro F1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    palette = ["#007f78" if not name.endswith("_tuned") else "#d27a25" for name in names]
    for ax, (key, title) in zip(axes.flat, metrics):
        if key == "cv_top1_accuracy":
            vals = [results["models"][name][key] for name in names]
        else:
            vals = [results["models"][name]["student_test"][key] for name in names]
        bars = ax.barh(labels, vals, color=palette)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, vals):
            value = float(value)
            if value > .90:
                ax.text(value - .012, bar.get_y() + bar.get_height()/2, f"{value:.3f}", ha="right", va="center", fontsize=8)
            else:
                ax.text(value + .012, bar.get_y() + bar.get_height()/2, f"{value:.3f}", ha="left", va="center", fontsize=8)
    fig.suptitle("PathSnap model comparison · observed experiment results", fontsize=15, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_experiments(data_root: Path, output_dir: Path, trials: int = 3):
    random.seed(SEED)
    np.random.seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_trips = list_trips(data_root, "train")
    student_trips = list_trips(data_root, "student_test")
    if not train_trips or not student_trips:
        raise FileNotFoundError("Expected train/ and student_test/ trip folders under the data root")
    print(f"Preparing {len(train_trips)} train and {len(student_trips)} student-test trips", flush=True)
    train_data = {}
    for i, trip in enumerate(train_trips, start=1):
        _, candidates, observations, misses = prepare_trip(trip)
        # Training feature arrays are all CV needs; release per-trip graph and
        # spatial-index structures before collecting the remaining trips.
        train_data[trip.trip_id] = (None, candidates, observations, misses)
        if i % 10 == 0:
            print(f"  prepared {i}/{len(train_trips)} training trips", flush=True)
    student_data = {trip.trip_id: prepare_trip(trip) for trip in student_trips}
    total_obs = sum(len(t.gps) for t in train_trips)
    total_miss = sum(train_data[t.trip_id][3] for t in train_trips)
    print(f"Training GPS observations={total_obs}; candidate miss rate={total_miss/max(1,total_obs):.4f}", flush=True)
    catalog = model_catalog(SEED)
    model_results = {}
    best_name, best_cv = None, -1.0
    all_train_ids = [t.trip_id for t in train_trips]
    for name, base_model in catalog.items():
        start = time.perf_counter()
        cv_top1, cv_std, cv_macro = _cv_score(base_model, train_data, all_train_ids)
        x_fit, y_fit = _fit_rows(train_data, all_train_ids, np.random.default_rng(SEED))
        model = clone(base_model)
        model.fit(x_fit, y_fit)
        student_metrics = {trip.trip_id: _evaluate_trip(model, trip, student_data[trip.trip_id]) for trip in student_trips}
        aggregate = {}
        for key in ["top1_accuracy", "macro_f1", "precision", "recall", "log_loss", "candidate_miss_rate", "classifier_continuity", "hmm_top1_accuracy", "hmm_continuity", "inference_ms"]:
            vals = [m[key] for m in student_metrics.values() if m.get(key) is not None]
            aggregate[key] = float(np.mean(vals)) if vals else None
        duration = time.perf_counter() - start
        model_results[name] = {
            "family": name, "configuration": base_model.get_params(), "cv_top1_accuracy": cv_top1,
            "cv_top1_std": cv_std, "cv_candidate_macro_f1": cv_macro,
            "student_test": aggregate, "student_trip_metrics": student_metrics,
            "training_seconds_including_cv": duration, "fit_rows": int(len(x_fit)),
            "positive_rows": int((y_fit == 1).sum()),
        }
        print(f"{name}: CV top1={cv_top1:.4f}±{cv_std:.4f}, student top1={aggregate['top1_accuracy']:.4f}, F1={aggregate['macro_f1']:.4f}, time={duration:.1f}s", flush=True)
        if cv_top1 > best_cv:
            best_name, best_cv = name, cv_top1
            with (output_dir / "best_baseline.joblib").open("wb") as f:
                pickle.dump(model, f)
        checkpoint = {"seed": SEED, "train_trip_ids": all_train_ids,
                      "student_test_trip_ids": [t.trip_id for t in student_trips],
                      "models": model_results, "best_model_so_far": best_name}
        (output_dir / "comparison_checkpoint.json").write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")
    print(f"Selected {best_name} from group CV (student-test not used for selection)", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=2), study_name=f"pathsnap_{best_name}")
    def objective(trial):
        estimator = _tuned_estimator(best_name, trial)
        score, _, macro_f1 = _cv_score(estimator, train_data, all_train_ids, n_splits=3, seed=SEED + trial.number * 11)
        trial.set_user_attr("group_cv_top1", score)
        trial.set_user_attr("group_cv_candidate_macro_f1", macro_f1)
        return score
    tuning_started = time.perf_counter()
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    tuning_seconds = time.perf_counter() - tuning_started
    tuned_model = _tuned_estimator(best_name, optuna.trial.FixedTrial(study.best_params))
    x_fit, y_fit = _fit_rows(train_data, all_train_ids, np.random.default_rng(SEED))
    tuned_model.fit(x_fit, y_fit)
    tuned_student = {trip.trip_id: _evaluate_trip(tuned_model, trip, student_data[trip.trip_id]) for trip in student_trips}
    tuned_agg = {}
    for key in ["top1_accuracy", "macro_f1", "precision", "recall", "log_loss", "candidate_miss_rate", "classifier_continuity", "hmm_top1_accuracy", "hmm_continuity", "inference_ms"]:
        vals = [m[key] for m in tuned_student.values() if m.get(key) is not None]
        tuned_agg[key] = float(np.mean(vals)) if vals else None
    model_results[best_name + "_tuned"] = {
        "family": best_name, "configuration": tuned_model.get_params(),
        "cv_top1_accuracy": study.best_value, "cv_top1_std": None,
        "cv_candidate_macro_f1": study.best_trial.user_attrs.get("group_cv_candidate_macro_f1"), "student_test": tuned_agg,
        "student_trip_metrics": tuned_student, "training_seconds_including_cv": tuning_seconds,
        "fit_rows": int(len(x_fit)), "positive_rows": int((y_fit == 1).sum()),
    }
    trial_rows = []
    for trial in study.trials:
        row = {"number": trial.number, "state": trial.state.name, "value": trial.value,
               "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else "",
               "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else "",
               "duration_seconds": trial.duration.total_seconds() if trial.duration else ""}
        row.update({f"param_{k}": v for k, v in trial.params.items()})
        row.update({f"user_{k}": v for k, v in trial.user_attrs.items()})
        trial_rows.append(row)
    if trial_rows:
        columns = sorted({key for row in trial_rows for key in row})
        with (output_dir / "bayesian_trials.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(trial_rows)
    (output_dir / "best_params.json").write_text(json.dumps({"selected_model": best_name, "objective": "mean trip-grouped CV observation top-1 accuracy", "best_value": study.best_value, "params": study.best_params}, indent=2), encoding="utf-8")
    with (output_dir / "model.joblib").open("wb") as f:
        pickle.dump(tuned_model, f)
    results = {
        "seed": SEED,
        "train_trip_ids": all_train_ids,
        "student_test_trip_ids": [t.trip_id for t in student_trips],
        "evaluator_test_available": False,
        "features": FEATURE_NAMES,
        "candidate_settings": {"max_candidates": 36, "search_radius_m": 180.0},
        "selected_model": best_name,
        "tuned_model": best_name + "_tuned",
        "candidate_misses_train": int(total_miss),
        "gps_observations_train": int(total_obs),
        "models": model_results,
        "bayesian_trials": trials,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _plot_comparison(results, output_dir / "model_comparison.png")
    print(f"Saved model and results to {output_dir}", flush=True)
    return results
