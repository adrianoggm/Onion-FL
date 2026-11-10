from __future__ import annotations

"""Explainers of a run's final model (issue #230).

An explainer gets, per dataset, a scorer and the validation subjects' rows,
and returns the importance of each feature. It runs after the run, on the
``val`` subjects (never the ``test`` ones, which stay the global evaluation),
and ``experiment.explain`` writes what it returns to ``explain.json`` next to
the run; ``run_hash`` does not cover that file.
"""

from collections.abc import Callable, Sequence
from typing import Any
from zlib import crc32

import numpy as np
from pydantic import BaseModel, Field, PositiveInt

from onion_fl.core.registry import Registry

explainers = Registry("explainer")
Scorer = Callable[[np.ndarray, np.ndarray], float]  # (X, y) -> score, higher better


class PermutationParams(BaseModel):
    metric: str = Field("macro_f1", description="Métrica con la que se puntúa")
    repeats: PositiveInt = Field(5, description="Permutaciones por feature")
    seed: int = Field(0, description="Semilla de las permutaciones")


@explainers.register(
    "permutation",
    title="Importancia por permutación",
    description=(
        "Cuánto cae la métrica en los sujetos de validación al barajar cada "
        "feature dentro de cada sujeto, por dataset."
    ),
    params=PermutationParams,
    explain=(
        "Se calcula después de la ejecución con el modelo final; los sujetos "
        "de test no se tocan. Barajar dentro de cada sujeto no mezcla sujetos."
    ),
)
class PermutationImportance:
    def __init__(
        self, metric: str = "macro_f1", repeats: int = 5, seed: int = 0
    ) -> None:
        self.metric, self.repeats, self.seed = metric, repeats, seed

    def explain(
        self,
        dataset: str,
        score: Scorer,
        X: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        features: Sequence[str],
    ) -> dict[str, Any]:
        """The baseline score and, per feature, the mean drop and its spread."""
        rng = np.random.default_rng([self.seed, crc32(dataset.encode())])
        X = np.asarray(X, np.float32)
        baseline = score(X, y)
        groups = [np.flatnonzero(subjects == s) for s in np.unique(subjects)]
        rows = []
        for j, name in enumerate(features):
            drops = []
            for _ in range(self.repeats):
                shuffled = X.copy()
                for rows_of in groups:
                    shuffled[rows_of, j] = X[rng.permutation(rows_of), j]
                drops.append(baseline - score(shuffled, y))
            rows.append(
                {
                    "feature": name,
                    "importance": float(np.mean(drops)),
                    "std": float(np.std(drops)),
                }
            )
        rows.sort(key=lambda r: (-r["importance"], r["feature"]))
        return {"metric": self.metric, "baseline": float(baseline), "features": rows}
