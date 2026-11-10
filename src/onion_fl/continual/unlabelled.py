from __future__ import annotations

"""Objectives on the rows whose label has not arrived (spec §9).

The edge hands an objective the features of its unlabelled pool, never their
labels. ``fit`` looks at them once, with the model as received; the trainer
then adds ``term`` over a chunk of the pool to each labelled batch's loss.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from onion_fl.core.registry import Registry
from onion_fl.learning.metrics import predict

objectives = Registry("unlabelled")


@runtime_checkable
class Objective(Protocol):
    """What a trainer uses of an objective on unlabelled rows."""

    size: int  # rows of the pool fitted

    def fit(self, model: nn.Module, X: np.ndarray) -> dict[str, Any]:
        """Look at the pool with the received model: ``seen``, ``accepted``,
        ``confidence`` (the mean max probability), and per row ``labels``,
        ``mask`` and ``row_confidence`` (``labels``/``mask`` None if it makes
        no labels)."""

    def term(self, model: nn.Module, idx: np.ndarray) -> torch.Tensor | float:
        """The extra loss over the pool rows ``idx``; 0 with nothing to add."""


class NoParams(BaseModel):
    pass


class PseudoLabelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(
        0.9, ge=0, le=1, description="Confianza mínima del modelo recibido"
    )
    weight: float = Field(
        0.5, gt=0, description="Peso de la pérdida sobre los pseudoetiquetados"
    )


class ConsistencyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    noise: float = Field(
        0.1, gt=0, description="Desviación del ruido sobre las features escaladas"
    )
    weight: float = Field(0.5, gt=0, description="Peso de la pérdida de consistencia")


class _Objective:
    def __init__(self, weight: float = 0.0) -> None:
        self.weight = weight
        self.X = torch.zeros((0, 0))
        self.size = 0

    def _fit(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """Keeps the pool; the received model's probabilities on it."""
        X = np.asarray(X, np.float32)
        self.X, self.size = torch.from_numpy(X), len(X)
        if not len(X):
            return np.zeros((0, 0))
        logits = predict(model, X)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    @staticmethod
    def _stats(p: np.ndarray, accepted: int) -> dict[str, Any]:
        confidence = p.max(axis=1) if len(p) else np.zeros(0)
        return {
            "seen": len(p),
            "accepted": accepted,
            "confidence": float(confidence.mean()) if len(p) else 0.0,
            "row_confidence": confidence,
        }


@objectives.register(
    "none",
    title="Sin datos sin etiquetar",
    description="Las filas sin etiqueta no entrenan: el edge actúa como en C3.",
    params=NoParams,
)
class NoObjective(_Objective):
    def fit(self, model: nn.Module, X: np.ndarray) -> dict[str, Any]:
        return self._stats(self._fit(model, X[:0]), 0) | {"labels": None, "mask": None}

    def term(self, model: nn.Module, idx: np.ndarray) -> torch.Tensor | float:
        return 0.0


@objectives.register(
    "pseudo_label",
    title="Pseudoetiquetas",
    description="Entrena con la clase que predice el modelo recibido cuando está seguro.",
    params=PseudoLabelParams,
    explain=(
        "Una fila sin etiqueta se acepta si la probabilidad máxima del modelo "
        "recibido llega al umbral; su clase predicha suma weight·CE al lote."
    ),
)
class PseudoLabel(_Objective):
    def __init__(self, threshold: float = 0.9, weight: float = 0.5) -> None:
        super().__init__(weight)
        self.threshold = threshold
        self.labels = torch.zeros(0, dtype=torch.long)
        self.mask = np.zeros(0, bool)

    def fit(self, model: nn.Module, X: np.ndarray) -> dict[str, Any]:
        p = self._fit(model, X)
        labels = p.argmax(axis=1) if len(p) else np.zeros(0, np.int64)
        self.mask = (p.max(axis=1) >= self.threshold) if len(p) else np.zeros(0, bool)
        self.labels = torch.from_numpy(labels.astype(np.int64))
        stats = self._stats(p, int(self.mask.sum()))
        return stats | {"labels": labels.astype(np.int64), "mask": self.mask.copy()}

    def term(self, model: nn.Module, idx: np.ndarray) -> torch.Tensor | float:
        accepted = idx[self.mask[idx]]
        if not len(accepted):
            return 0.0
        rows = torch.from_numpy(accepted)
        return self.weight * F.cross_entropy(model(self.X[rows]), self.labels[rows])


@objectives.register(
    "consistency",
    title="Consistencia",
    description="La predicción no debe cambiar al perturbar un poco las features.",
    params=ConsistencyParams,
    explain=(
        "Suma weight·KL(p(x) ‖ p(x + ε)), con p(x) fija y ε ~ N(0, noise²) "
        "sobre las features escaladas."
    ),
)
class Consistency(_Objective):
    def __init__(self, noise: float = 0.1, weight: float = 0.5) -> None:
        super().__init__(weight)
        self.noise = noise

    def fit(self, model: nn.Module, X: np.ndarray) -> dict[str, Any]:
        p = self._fit(model, X)
        return self._stats(p, len(p)) | {"labels": None, "mask": None}

    def term(self, model: nn.Module, idx: np.ndarray) -> torch.Tensor | float:
        if not len(idx):
            return 0.0
        x = self.X[torch.from_numpy(np.asarray(idx, np.int64))]
        with torch.no_grad():
            target = F.softmax(model(x), dim=1)
        # Drawn from torch's generator, which the trainer seeds per training.
        noisy = F.log_softmax(model(x + self.noise * torch.randn_like(x)), dim=1)
        return self.weight * F.kl_div(noisy, target, reduction="batchmean")
