from __future__ import annotations

"""Differential privacy: the RDP accountant and the edge-side mechanism (spec §3.7)."""

from collections.abc import Mapping

import numpy as np
from pydantic import BaseModel, Field

from onion_fl.core.registry import Registry
from onion_fl.learning.model import is_aux

ORDERS = np.concatenate([np.linspace(1.01, 10.0, 400), np.arange(11.0, 513.0)])


def gaussian_epsilon(noise_multiplier: float, rounds: int, delta: float) -> float:
    """ε after ``rounds`` Gaussian mechanisms with this noise multiplier.

    RDP of order α is α/(2σ²) per round and adds up over rounds; ε = min_α
    T·α/(2σ²) + ln(1/δ)/(α − 1). No amplification by subsampling, so it is a
    conservative bound.
    """
    if rounds <= 0:
        return 0.0
    eps = rounds * ORDERS / (2 * noise_multiplier**2) + np.log(1 / delta) / (ORDERS - 1)
    return float(eps.min())


privacies = Registry("privacy")


class LocalDPParams(BaseModel):
    clip: float = Field(
        1.0, gt=0, description="Norma máxima C de la actualización del edge"
    )
    sigma: float = Field(1.0, gt=0, description="Multiplicador de ruido σ")
    delta: float = Field(1e-5, gt=0, lt=1, description="δ del presupuesto (ε, δ)")


@privacies.register(
    "local_dp",
    title="DP local",
    description="Cada edge recorta su actualización a C y le suma N(0, (σ·C)²) antes de enviarla.",
    params=LocalDPParams,
    explain=(
        "No confía en el agregador; informa ε por ronda con un contable RDP sin "
        "amplificación, con sensibilidad 2C porque cualquier actualización puede "
        "sustituirse por otra. Los grupos locales no se tocan."
    ),
)
class LocalDP:
    def __init__(
        self, clip: float = 1.0, sigma: float = 1.0, delta: float = 1e-5
    ) -> None:
        self.clip, self.sigma, self.delta = clip, sigma, delta

    def epsilon(self, rounds: int) -> float:
        # Replace-one adjacency: any update in the C-ball may become any other,
        # so the sensitivity is 2C and the noise counts as σ/2.
        return gaussian_epsilon(self.sigma / 2, rounds, self.delta)

    def on_update(
        self,
        arrays: Mapping[str, np.ndarray],
        received: Mapping[str, np.ndarray],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        keys = [k for k in arrays if k in received and not is_aux(k)]
        delta = {
            k: np.asarray(arrays[k], np.float64) - np.asarray(received[k], np.float64)
            for k in keys
        }
        norm = float(np.sqrt(sum(float((d**2).sum()) for d in delta.values())))
        scale = min(1.0, self.clip / norm) if norm > 0 else 1.0
        out = dict(arrays)
        for k in keys:
            noise = rng.normal(0.0, self.sigma * self.clip, size=delta[k].shape)
            noisy = np.asarray(received[k], np.float64) + scale * delta[k] + noise
            out[k] = noisy.astype(np.asarray(arrays[k]).dtype)
        return out
