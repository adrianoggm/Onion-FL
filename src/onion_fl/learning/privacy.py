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


class Accountant:
    """The Rényi DP spent so far at each order α of ``ORDERS``.

    Each Gaussian release adds α/(2σ²) whatever σ it used, so ε stays right
    when a continuation changes σ; ``state`` goes into the run's bundle.
    """

    rdp: np.ndarray

    def spend(self, noise_multiplier: float) -> None:
        self.rdp = self.rdp + ORDERS / (2 * noise_multiplier**2)

    def spent(self, delta: float) -> float:
        """ε at ``delta``: min_α RDP_α + ln(1/δ)/(α − 1)."""
        if not self.rdp.any():
            return 0.0
        return float((self.rdp + np.log(1 / delta) / (ORDERS - 1)).min())

    def state(self) -> dict[str, np.ndarray]:
        return {"orders": ORDERS, "rdp": self.rdp}

    def load_state(self, arrays: Mapping[str, np.ndarray]) -> None:
        if "rdp" not in arrays:
            return
        if not np.array_equal(arrays["orders"], ORDERS):
            raise ValueError("the saved privacy budget uses other RDP orders")
        self.rdp = np.array(arrays["rdp"], np.float64)


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
class LocalDP(Accountant):
    def __init__(
        self, clip: float = 1.0, sigma: float = 1.0, delta: float = 1e-5
    ) -> None:
        self.clip, self.sigma, self.delta = clip, sigma, delta
        self.rdp = np.zeros_like(ORDERS)

    def epsilon(self) -> float:
        """ε of every update this edge has released, across continuations."""
        return self.spent(self.delta)

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
        # Replace-one adjacency: any update in the C-ball may become any other,
        # so the sensitivity is 2C and the noise counts as σ/2.
        self.spend(self.sigma / 2)
        return out
