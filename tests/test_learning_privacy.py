"""Differential privacy: the RDP accountant and the edge-side mechanism (issue #149).

Hand-written arrays only; nothing is trained (docs/RULES.md).
"""

from __future__ import annotations

import pytest

from onion_fl.learning.privacy import gaussian_epsilon


def test_epsilon_matches_the_closed_form_optimum() -> None:
    # σ=1, T=1, δ=1e-5: min_α α/2 + ln(1e5)/(α−1) at α = 1 + sqrt(2·ln 1e5)
    assert gaussian_epsilon(1.0, 1, 1e-5) == pytest.approx(5.298, abs=0.01)


def test_epsilon_grows_with_rounds_and_shrinks_with_noise() -> None:
    assert gaussian_epsilon(1.0, 10, 1e-5) > gaussian_epsilon(1.0, 1, 1e-5)
    assert gaussian_epsilon(2.0, 10, 1e-5) < gaussian_epsilon(1.0, 10, 1e-5)
    assert gaussian_epsilon(1.0, 0, 1e-5) == 0.0
