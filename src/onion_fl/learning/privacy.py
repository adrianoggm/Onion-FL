from __future__ import annotations

"""Differential privacy: the RDP accountant and the edge-side mechanism (spec §3.7)."""

import numpy as np

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
