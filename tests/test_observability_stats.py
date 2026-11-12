"""Tests for the paired statistics of the analysis (issue #288).

The differences are hand-written numbers, not results: they check each
statistic against ``scipy.stats`` or a value computed by hand.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from onion_fl.observability.stats import (
    cohens_dz,
    holm,
    paired_bootstrap_ci,
    paired_power,
    paired_test,
    seeds_needed,
    sign_flip_p,
    tost_p,
    verdict,
    wilcoxon_p,
)

D = np.array([0.012, 0.030, -0.004, 0.021, 0.018, 0.009, 0.025])


def test_wilcoxon_is_scipys_signed_rank_test() -> None:
    assert wilcoxon_p(D) == pytest.approx(stats.wilcoxon(D).pvalue)


def test_wilcoxon_of_no_difference_is_one() -> None:
    assert wilcoxon_p(np.zeros(5)) == 1.0


def test_the_sign_flip_test_is_exact_for_few_pairs() -> None:
    observed = abs(D.mean())
    flips = [
        abs(np.mean(D * np.array(signs))) >= observed - 1e-12
        for signs in itertools.product([-1, 1], repeat=len(D))
    ]

    # all positive, or all but the -0.004, each either way: 4 of 2**7 patterns
    assert np.mean(flips) == pytest.approx(4 / 128)
    assert sign_flip_p(D) == pytest.approx(np.mean(flips))


def test_the_sign_flip_test_is_seeded_beyond_the_exact_range() -> None:
    d = np.linspace(-0.01, 0.03, 20)

    assert sign_flip_p(d, resamples=999) == sign_flip_p(d, resamples=999)


def test_the_paired_bootstrap_resamples_the_differences() -> None:
    low, high = paired_bootstrap_ci(D)

    rng = np.random.default_rng(0)
    means = rng.choice(D, size=(2000, len(D)), replace=True).mean(axis=1)
    assert (low, high) == pytest.approx(tuple(np.percentile(means, [2.5, 97.5])))
    assert low < D.mean() < high


def test_cohens_dz_is_the_mean_over_the_sd_of_the_differences() -> None:
    assert cohens_dz(D) == pytest.approx(D.mean() / D.std(ddof=1))
    assert cohens_dz(np.zeros(3)) == 0.0


def test_tost_is_the_larger_of_two_one_sided_t_tests() -> None:
    margin = 0.03
    lower = stats.ttest_1samp(D, -margin, alternative="greater").pvalue
    upper = stats.ttest_1samp(D, margin, alternative="less").pvalue

    assert tost_p(D, margin) == pytest.approx(max(lower, upper))


def test_tost_without_spread_is_certain() -> None:
    assert tost_p(np.full(4, 0.01), 0.02) == 0.0
    assert tost_p(np.full(4, 0.03), 0.02) == 1.0


def test_tost_needs_a_positive_margin() -> None:
    with pytest.raises(ValueError, match="margin"):
        tost_p(D, 0.0)


def test_holm_adjusts_in_the_given_order() -> None:
    # sorted 0.01, 0.03, 0.04 → 3·0.01, max(0.03, 2·0.03), max(0.06, 1·0.04)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    out = holm([0.02, float("nan")])
    assert out[0] == pytest.approx(0.02) and math.isnan(out[1])


@pytest.mark.parametrize(("effect", "n"), [(0.5, 34), (0.8, 15), (1.0, 10)])
def test_seeds_needed_matches_the_paired_t_test_tables(effect: float, n: int) -> None:
    # G*Power, t-test of the mean difference of matched pairs, α = 0.05, 1-β = 0.8
    assert seeds_needed(effect * 0.02, 0.02) == n
    assert paired_power(n - 1, effect, 1.0) < 0.8 <= paired_power(n, effect, 1.0)


def test_seeds_needed_refuses_what_cannot_be_detected() -> None:
    with pytest.raises(ValueError):
        seeds_needed(0.0, 0.1)


def test_paired_test_gives_every_statistic() -> None:
    row = paired_test(D, margin=0.03)

    assert row["n_pairs"] == 7 and row["delta"] == pytest.approx(D.mean())
    assert row["sd_diff"] == pytest.approx(D.std(ddof=1))
    assert row["p_wilcoxon"] == pytest.approx(stats.wilcoxon(D).pvalue)
    assert row["p_tost"] == pytest.approx(tost_p(D, 0.03))
    assert math.isnan(paired_test(D)["p_tost"])


def test_the_verdict_puts_equivalence_first() -> None:
    assert verdict(0.01, 0.01) == "equivalent"
    assert verdict(0.01, 0.2) == "different"
    assert verdict(0.2, float("nan")) == "inconclusive"
