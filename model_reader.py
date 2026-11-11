"""
Lightweight reader for the trained XGBoost model (models/film_revenue_model.json).

The dashboard only needs to *predict*, not train. Reading XGBoost's JSON format
directly with NumPy means the dashboard runs on any Python version without
installing xgboost (whose newest releases require Python >= 3.12).

Predictions match xgboost's own `predict()` (checked against the notebook output).
"""
import json

import numpy as np


class XGBoostJSONModel:
    """Evaluates a regression model saved with `XGBRegressor.save_model('*.json')`."""

    def __init__(self, path):
        learner = json.loads(open(path, encoding="utf-8").read())["learner"]

        objective = learner["objective"]["name"]
        if objective != "reg:squarederror":
            raise ValueError(f"Unsupported objective: {objective}")

        # base_score is stored as "1.7686129E1" or "[1.7686129E1]" depending on version
        self.base_score = np.float32(learner["learner_model_param"]["base_score"].strip("[]"))

        self.trees = []
        for t in learner["gradient_booster"]["model"]["trees"]:
            if any(t.get("split_type", [])):
                raise ValueError("Categorical splits are not supported")
            self.trees.append((
                np.asarray(t["left_children"]),
                np.asarray(t["right_children"]),
                np.asarray(t["split_indices"]),
                np.asarray(t["split_conditions"], dtype=np.float32),  # leaf value at leaves
                np.asarray(t["default_left"], dtype=bool),            # direction for missing values
            ))

    def predict(self, X):
        """X: array-like of shape (n_samples, n_features), columns in training order."""
        X = np.asarray(X, dtype=np.float32)   # xgboost compares in float32
        out = np.full(len(X), self.base_score, dtype=np.float32)
        for left, right, feat, cond, default_left in self.trees:
            node = np.zeros(len(X), dtype=np.int64)
            active = left[node] != -1
            while active.any():
                n = node[active]
                v = X[active, feat[n]]
                go_left = np.where(np.isnan(v), default_left[n], v < cond[n])
                node[active] = np.where(go_left, left[n], right[n])
                active = left[node] != -1
            out += cond[node]
        return out.astype(np.float64)
