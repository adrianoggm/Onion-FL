from __future__ import annotations

"""Classical baselines on the same subject roles as the federated runs (spec §12).

``evaluate`` trains one model per dataset on the training bag of a
``DataSplit`` (already imputed and scaled on that bag) and scores it on the
test and val subjects. ``cross_validate`` repeats that with each fold of
subjects as the test role, so the preprocessing is refitted per fold.
"""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, PositiveInt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from onion_fl.core.context import node_rng
from onion_fl.core.registry import Registry
from onion_fl.data.contract import SubjectData, natural_key
from onion_fl.data.roles import DataSplit, RoleOverride, RolesConfig, split_subjects

baseline_models = Registry("baseline")


class LRParams(BaseModel):
    C: float = Field(1.0, gt=0)
    max_iter: PositiveInt = 1000
    class_weight: Literal["balanced"] | None = None


@baseline_models.register(
    "lr", title="Regresión logística", description="Lineal, rápida.", params=LRParams
)
class LR:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def build(self, seed: int) -> LogisticRegression:
        return LogisticRegression(random_state=seed, **self.params)


class RFParams(BaseModel):
    n_estimators: PositiveInt = 200
    max_depth: PositiveInt | None = None
    class_weight: Literal["balanced"] | None = None


@baseline_models.register(
    "rf", title="Random forest", description="Conjunto de árboles.", params=RFParams
)
class RF:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def build(self, seed: int) -> RandomForestClassifier:
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **self.params)


class XGBParams(BaseModel):
    n_estimators: PositiveInt = 300
    max_depth: PositiveInt = 6
    learning_rate: float = Field(0.1, gt=0)


@baseline_models.register(
    "xgboost",
    title="XGBoost",
    description="Gradient boosting (extra analysis).",
    params=XGBParams,
)
class XGB:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def build(self, seed: int) -> Any:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed: pip install 'onion-fl[analysis]'"
            ) from exc
        return XGBClassifier(random_state=seed, **self.params)


def scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "n_samples": int(len(y_true)),
    }


def subject_folds(names: Sequence[str], k: int, seed: int) -> list[list[str]]:
    """``k`` disjoint folds of subjects, independent of the input order."""
    if k > len(names):
        raise ValueError(
            f"{k} folds need at least {k} subjects; got {len(names)} subjects"
        )
    ordered = sorted(names, key=natural_key)
    shuffled = [ordered[i] for i in node_rng(seed, "folds").permutation(len(ordered))]
    return [
        sorted(fold.tolist(), key=natural_key)
        for fold in np.array_split(np.array(shuffled, dtype=object), k)
    ]


def _score_on(model: Any, holders: Sequence[SubjectData]) -> dict[str, float] | None:
    if not holders:
        return None
    X = np.concatenate([h.X for h in holders])
    y = np.concatenate([h.y for h in holders])
    return scores(y, model.predict(X), holders[0].n_classes)


def evaluate(
    split: DataSplit,
    models: Sequence[str] = ("lr", "rf", "xgboost"),
    seed: int = 0,
    params: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per dataset and model: scores on the test and val subjects of ``split``."""
    out: dict[str, dict[str, Any]] = {}
    for dataset in sorted({c.dataset for c in split.clients}):
        train = [c.train for c in split.clients if c.dataset == dataset]
        X = np.concatenate([t.X for t in train])
        y = np.concatenate([t.y for t in train])
        out[dataset] = {}
        for name in models:
            model = baseline_models.create(name, (params or {}).get(name)).build(seed)
            model.fit(X, y)
            out[dataset][name] = {
                "train_samples": int(len(y)),
                "test": _score_on(
                    model, [s for s in split.test if s.dataset == dataset]
                ),
                "val": _score_on(model, [s for s in split.val if s.dataset == dataset]),
            }
    return out


def cross_validate(
    subjects: Sequence[SubjectData],
    model: str = "lr",
    k: int = 5,
    seed: int = 0,
    roles: RolesConfig | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Subject-level k-fold per dataset; preprocessing refitted on each training fold."""
    roles = roles or RolesConfig()
    out: dict[str, dict[str, Any]] = {}
    for dataset in sorted({s.dataset for s in subjects}):
        mine = [s for s in subjects if s.dataset == dataset]
        given = roles.overrides.get(dataset) or RoleOverride()
        kept = [s.subject for s in mine if s.subject not in set(given.exclude)]
        folds = []
        for fold in subject_folds(kept, k, seed):
            override = given.model_copy(update={"test": fold, "val": []})
            config = roles.model_copy(update={"overrides": {dataset: override}})
            result = evaluate(
                split_subjects(mine, config), [model], seed, {model: params or {}}
            )
            folds.append({"test_subjects": fold, **result[dataset][model]["test"]})
        metrics = ("accuracy", "balanced_accuracy", "f1_macro")
        out[dataset] = {
            "model": model,
            "folds": folds,
            "mean": {m: float(np.mean([f[m] for f in folds])) for m in metrics},
            "std": {m: float(np.std([f[m] for f in folds])) for m in metrics},
        }
    return out
