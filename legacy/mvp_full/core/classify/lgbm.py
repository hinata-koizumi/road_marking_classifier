from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
    LOGGER.warning("LightGBM not installed; falling back to scikit-learn gradient boosting.")
    from sklearn.ensemble import GradientBoostingClassifier
else:
    GradientBoostingClassifier = None


class HybridClassifier:
    """Wrapper that prefers LightGBM and falls back to sklearn."""

    def __init__(self, config: Dict[str, float]) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.model = None
        self.classes_: List[str] = []

    def load_or_init(self, features: np.ndarray, labels: List[str]) -> None:
        self.classes_ = sorted(set(labels or ["road_line", "stop_line", "crosswalk", "curb"]))
        if features.size == 0:
            return
        self.scaler.fit(features)
        feats = self.scaler.transform(features)
        y = np.array([self.classes_.index(lbl) for lbl in labels])
        if lgb:
            self.model = lgb.LGBMClassifier(
                max_depth=self.config.get("max_depth", 6),
                n_estimators=self.config.get("n_estimators", 150),
                learning_rate=self.config.get("learning_rate", 0.1),
                class_weight=self.config.get("class_weights"),
                random_state=self.config.get("random_state", 42),
            )
        else:
            self.model = GradientBoostingClassifier(
                max_depth=self.config.get("max_depth", 3),
                n_estimators=self.config.get("n_estimators", 50),
                random_state=self.config.get("random_state", 42),
            )
        self.model.fit(feats, y)

    def predict(self, features: np.ndarray) -> Tuple[List[str], np.ndarray]:
        if features.size == 0:
            return [], np.array([])
        feats = self.scaler.transform(features)
        if self.model is None:
            return self._heuristic(feats)
        proba = self.model.predict_proba(feats)
        labels = [self.classes_[int(idx)] for idx in np.argmax(proba, axis=1)]
        scores = np.max(proba, axis=1)
        return labels, scores

    def _heuristic(self, feats: np.ndarray) -> Tuple[List[str], np.ndarray]:
        labels = []
        probs = []
        for length, width, curvature, stripe_count, parallelism, angle, intensity in feats:
            if stripe_count >= 3:
                labels.append("crosswalk")
            elif width > 0.3 and abs(angle) > 45:
                labels.append("stop_line")
            elif curvature > 0.2:
                labels.append("curb")
            else:
                labels.append("road_line")
            probs.append(0.55)
        return labels, np.array(probs)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if lgb and isinstance(self.model, lgb.LGBMClassifier):
            self.model.booster_.save_model(str(path))
        else:
            import joblib

            joblib.dump({"model": self.model, "scaler": self.scaler, "classes": self.classes_}, path)

    def load(self, path: Path) -> None:
        if not path.exists():
            LOGGER.warning("Model path %s not found; training on the fly", path)
            return
        if lgb:
            booster = lgb.Booster(model_file=str(path))
            self.model = lgb.LGBMClassifier()
            self.model._Booster = booster
            self.classes_ = booster.pandas_categorical
        else:
            import joblib

            payload = joblib.load(path)
            self.model = payload["model"]
            self.scaler = payload["scaler"]
            self.classes_ = payload["classes"]
