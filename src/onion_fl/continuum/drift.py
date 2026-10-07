from __future__ import annotations

"""Drift on a stream (spec §4.4): a statistic per window and a detector.

Each node turns what reached it since its last status into one number per kind,
and a detector watches that number for a sustained rise:

- ``data``, P(X): the mean absolute shift of the window's features from the
  edge's history, in history standard deviations;
- ``prior``, P(Y): the total variation between the classes of the labels that
  arrived and those of the history;
- ``performance``: the error rate of the stored predictions whose labels arrived.

Concept drift (the error net of a prior shift) and client drift are not
detected here; ``diagnostic.divergence_*`` already covers the client.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PositiveInt

from onion_fl.core.registry import Registry

KINDS = ("data", "prior", "performance")
detectors = Registry("detector")


class PageHinkleyParams(BaseModel):
    delta: float = Field(0.005, ge=0, description="Subida tolerada sobre la media")
    threshold: float = Field(
        0.5, gt=0, description="Subida acumulada que cuenta como deriva"
    )
    min_samples: PositiveInt = Field(3, description="Ventanas antes de poder detectar")


@detectors.register(
    "page_hinkley",
    title="Page-Hinkley",
    description="Detecta una subida sostenida del estadístico respecto a su media.",
    params=PageHinkleyParams,
    explain="Tras cada detección vuelve a empezar (Page, 1954; Hinkley, 1971).",
)
class PageHinkley:
    def __init__(
        self, delta: float = 0.005, threshold: float = 0.5, min_samples: int = 3
    ) -> None:
        self.delta, self.threshold, self.min_samples = delta, threshold, min_samples
        self.reset()

    def reset(self) -> None:
        self.n, self.mean, self.cum, self.low = 0, 0.0, 0.0, 0.0

    def update(self, value: float) -> float | None:
        """One window's value; the statistic when a rise is detected, else None."""
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.cum += value - self.mean - self.delta
        self.low = min(self.low, self.cum)
        statistic = self.cum - self.low
        if self.n >= self.min_samples and statistic > self.threshold:
            self.reset()
            return float(statistic)
        return None


class Reference:
    """What an edge's windows are compared with: its history, the rows before t₀.

    The prior counts only history rows whose label was there by t₀.
    """

    def __init__(self, stream: Any) -> None:
        d, history = stream.data, stream.history
        X = np.asarray(d.X, float)[history]
        width = np.asarray(d.X).shape[1]
        self.mean = X.mean(axis=0) if len(X) else np.zeros(width)
        std = X.std(axis=0) if len(X) > 1 else np.ones(width)
        self.std = np.where(std > 0, std, 1.0)
        known = history & (stream.label_at <= stream.available_at)
        counts = np.bincount(d.y[known], minlength=d.n_classes).astype(float)
        self.prior = (
            counts / counts.sum()
            if counts.sum()
            else np.full(d.n_classes, 1 / d.n_classes)
        )


def window(
    kind: str,
    stream: Any,
    arrived: np.ndarray,
    labelled: np.ndarray,
    logits: np.ndarray | None,
    predicted: np.ndarray | None,
    reference: Reference | None,
) -> tuple[float, int]:
    """The ``kind`` statistic over one window, and how many rows it rests on.

    ``arrived`` and ``labelled`` mark the rows that arrived and whose labels
    arrived in the window; history rows are never part of one.
    """
    d = stream.data
    if kind == "data":
        rows = arrived & ~stream.history
        if not rows.any():
            return 0.0, 0
        mean = np.asarray(d.X, float)[rows].mean(axis=0)
        shift = np.abs(mean - reference.mean) / reference.std
        return float(np.mean(shift)), int(rows.sum())
    rows = labelled & ~stream.history
    if kind == "performance":
        rows = rows & predicted
    n = int(rows.sum())
    if not n:
        return 0.0, 0
    if kind == "prior":
        p = np.bincount(d.y[rows], minlength=d.n_classes) / n
        return float(0.5 * np.abs(p - reference.prior).sum()), n
    wrong = logits[rows].argmax(axis=1) != d.y[rows]
    return float(wrong.mean()), n
