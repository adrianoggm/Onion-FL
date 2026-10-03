from __future__ import annotations

"""Malicious edges (spec §3.6): hooks on the training data and on the update.

An attack touches only the model keys that cross the link (those the edge
received); local groups and auxiliary arrays are left alone. Label flipping
corrupts real labels to simulate an attacker; it is not synthetic training
data (docs/RULES.md).
"""

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PositiveInt

from onion_fl.core.registry import Registry
from onion_fl.learning.model import is_aux

attacks = Registry("attack")
Arrays = Mapping[str, np.ndarray]


class AttackParams(BaseModel):
    fraction: float = Field(
        0.2, ge=0, le=1, description="Fracción de edges maliciosos de cada dataset"
    )
    start_round: PositiveInt = Field(1, description="Primera ronda en la que atacan")


class Attack:
    def __init__(self, fraction: float = 0.2, start_round: int = 1) -> None:
        self.fraction, self.start_round = fraction, start_round

    def on_data(self, data: Any) -> Any:
        return data

    def on_update(
        self, arrays: Arrays, received: Arrays, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        out = dict(arrays)
        for key in arrays:
            if key in received and not is_aux(key):
                x = np.asarray(received[key], np.float64)
                y = np.asarray(arrays[key], np.float64)
                out[key] = self._poison(x, y, rng).astype(np.asarray(arrays[key]).dtype)
        return out

    def _poison(
        self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        return y


@attacks.register(
    "label_flip",
    title="Inversión de etiquetas",
    description="Entrena con y → n_clases − 1 − y.",
    params=AttackParams,
    explain="Simula a un atacante que envenena sus datos; las etiquetas son reales, no sintéticas.",
)
class LabelFlip(Attack):
    def on_data(self, data: Any) -> Any:
        flipped = copy.copy(data)
        labels = np.asarray(data.y)
        classes = int(getattr(data, "n_classes", int(labels.max()) + 1))
        object.__setattr__(flipped, "y", classes - 1 - labels)
        return flipped


class SignFlipParams(AttackParams):
    scale: float = Field(1.0, gt=0, description="Cuánto se invierte la actualización")


@attacks.register(
    "sign_flip",
    title="Inversión de signo",
    description="Envía x − s·(y − x): empuja en contra del aprendizaje.",
    params=SignFlipParams,
)
class SignFlip(Attack):
    def __init__(self, scale: float = 1.0, **params: Any) -> None:
        super().__init__(**params)
        self.scale = scale

    def _poison(self, x, y, rng):
        return x - self.scale * (y - x)


class GaussianParams(AttackParams):
    sigma: float = Field(1.0, gt=0, description="Desviación del ruido enviado")


@attacks.register(
    "gaussian",
    title="Ruido gaussiano",
    description="Envía x + N(0, σ²) en lugar de su actualización.",
    params=GaussianParams,
)
class Gaussian(Attack):
    def __init__(self, sigma: float = 1.0, **params: Any) -> None:
        super().__init__(**params)
        self.sigma = sigma

    def _poison(self, x, y, rng):
        return x + rng.normal(0.0, self.sigma, size=x.shape)


class ScaleParams(AttackParams):
    factor: float = Field(
        10.0, gt=0, description="Por cuánto multiplica su actualización"
    )


@attacks.register(
    "scale",
    title="Escalado",
    description="Envía x + f·(y − x): una actualización amplificada.",
    params=ScaleParams,
    explain="Sustitución de modelo: con f ≈ número de hijos, su modelo domina el promedio.",
)
class Scale(Attack):
    def __init__(self, factor: float = 10.0, **params: Any) -> None:
        super().__init__(**params)
        self.factor = factor

    def _poison(self, x, y, rng):
        return x + self.factor * (y - x)
