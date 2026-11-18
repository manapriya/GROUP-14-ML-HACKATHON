from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def model_catalog(seed: int = 42):
    """Five model families, including the requested Random Forest and XGBoost."""
    from xgboost import XGBClassifier

    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=seed)),
        "random_forest": RandomForestClassifier(n_estimators=40, max_depth=12, min_samples_leaf=2, max_features=0.9, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=60, max_depth=16, min_samples_leaf=2, max_features=0.9, class_weight="balanced", n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=25, learning_rate=0.08, l2_regularization=1.0, random_state=seed),
        "xgboost": XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0, min_child_weight=3, eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=seed),
    }

