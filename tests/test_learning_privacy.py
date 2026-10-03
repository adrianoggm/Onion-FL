"""Differential privacy: the RDP accountant and the edge-side mechanism (issue #149).

Hand-written arrays only; nothing is trained (docs/RULES.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from onion_fl.learning.privacy import gaussian_epsilon, privacies


def test_epsilon_matches_the_closed_form_optimum() -> None:
    # σ=1, T=1, δ=1e-5: min_α α/2 + ln(1e5)/(α−1) at α = 1 + sqrt(2·ln 1e5)
    assert gaussian_epsilon(1.0, 1, 1e-5) == pytest.approx(5.298, abs=0.01)


def test_epsilon_grows_with_rounds_and_shrinks_with_noise() -> None:
    assert gaussian_epsilon(1.0, 10, 1e-5) > gaussian_epsilon(1.0, 1, 1e-5)
    assert gaussian_epsilon(2.0, 10, 1e-5) < gaussian_epsilon(1.0, 10, 1e-5)
    assert gaussian_epsilon(1.0, 0, 1e-5) == 0.0


def test_local_dp_clips_and_noises_crossing_keys_only() -> None:
    dp = privacies.create("local_dp", {"clip": 1.0, "sigma": 0.0001})

    out = dp.on_update(
        {"w": np.array([3.0, 4.0]), "scaffold/w": np.full(2, 5.0)},
        {"w": np.zeros(2)},
        np.random.default_rng(0),
    )

    np.testing.assert_allclose(out["w"], [0.6, 0.8], atol=1e-3)
    np.testing.assert_allclose(out["scaffold/w"], [5.0, 5.0])
    assert dp.epsilon(3) == pytest.approx(gaussian_epsilon(0.00005, 3, 1e-5))


def test_local_dp_accounts_for_replacing_an_update() -> None:
    # Any clipped update may become any other: sensitivity 2C, so σ counts as σ/2.
    dp = privacies.create("local_dp", {"clip": 1.0, "sigma": 1.0})

    assert dp.epsilon(5) == pytest.approx(gaussian_epsilon(0.5, 5, 1e-5))
