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
    assert dp.epsilon() == pytest.approx(gaussian_epsilon(0.00005, 1, 1e-5))


def test_local_dp_accounts_for_replacing_an_update() -> None:
    # Any clipped update may become any other: sensitivity 2C, so σ counts as σ/2.
    dp = privacies.create("local_dp", {"clip": 1.0, "sigma": 1.0})
    for _ in range(5):
        dp.on_update({"w": np.ones(1)}, {"w": np.zeros(1)}, np.random.default_rng(0))

    assert dp.epsilon() == pytest.approx(gaussian_epsilon(0.5, 5, 1e-5))


def _rdp_epsilon(rdp: np.ndarray, delta: float = 1e-5) -> float:
    from onion_fl.learning.privacy import ORDERS

    return float((rdp + np.log(1 / delta) / (ORDERS - 1)).min())


def test_a_continuation_composes_the_budget_spent_under_another_sigma() -> None:
    from onion_fl.learning.privacy import ORDERS

    def release(dp, times: int) -> None:
        for _ in range(times):
            dp.on_update(
                {"w": np.ones(1)}, {"w": np.zeros(1)}, np.random.default_rng(0)
            )

    before = privacies.create("local_dp", {"sigma": 0.5})
    release(before, 10)
    after = privacies.create("local_dp", {"sigma": 1.0})
    after.load_state(before.state())
    release(after, 10)

    # replace-one: σ counts as σ/2; each release adds α/(2σ²) at every order α
    rdp = 10 * ORDERS / (2 * 0.25**2) + 10 * ORDERS / (2 * 0.5**2)
    assert after.epsilon() == pytest.approx(_rdp_epsilon(rdp))
    assert after.epsilon() > gaussian_epsilon(0.5, 20, 1e-5)  # all at σ=1 understates
