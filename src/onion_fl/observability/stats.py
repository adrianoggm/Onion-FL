from __future__ import annotations

"""Paired statistics for comparing arms run over the same seeds (issue #288).

Runs that share a seed share the test subjects, the partition and any seeded
degradation, so an arm is compared with a reference through the per-seed
differences ``d = arm - reference``:

- ``delta`` is the mean of ``d``, with a seeded 95 % percentile bootstrap
  interval over the pairs (the paired bootstrap);
- ``p_wilcoxon`` is the two-sided Wilcoxon signed-rank test of ``d``;
- ``p_perm`` is the two-sided sign-flip permutation test of the mean of ``d``,
  exact (all ``2**n`` sign patterns) while ``2**n <= PERMUTATIONS``;
- ``d_z`` is the paired Cohen's d, ``mean(d) / sd(d)``;
- ``p_tost`` is the two one-sided t-tests against ``±margin`` (TOST): below
  ``alpha`` the arms are equivalent within the margin.

``holm`` adjusts the p-values of several arms against one reference, and
``seeds_needed`` sizes a study from a pilot's ``sd(d)``.
"""

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats

BOOTSTRAP = 2000  # resamples of the interval of delta
PERMUTATIONS = 9999  # the sign-flip test is exact up to n = 13 pairs


def paired_bootstrap_ci(
    differences: np.ndarray, resamples: int = BOOTSTRAP, seed: int = 0
) -> tuple[float, float]:
    """95 % percentile bootstrap interval of the mean difference, resampling pairs."""
    d = np.asarray(differences, float)
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(d, size=(resamples, len(d)), replace=True)
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def wilcoxon_p(differences: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1 when every difference is 0."""
    d = np.asarray(differences, float)
    if len(d) == 0:
        return float("nan")
    if not np.any(d):
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


def sign_flip_p(
    differences: np.ndarray, resamples: int = PERMUTATIONS, seed: int = 0
) -> float:
    """Two-sided sign-flip permutation p-value of the mean difference.

    Exact when ``2**n <= resamples``; a seeded Monte Carlo test above.
    """
    d = np.asarray(differences, float)
    if len(d) == 0:
        return float("nan")
    if not np.any(d):
        return 1.0
    result = stats.permutation_test(
        (d,),
        lambda x, axis: np.mean(x, axis=axis),
        permutation_type="samples",
        n_resamples=resamples,
        alternative="two-sided",
        rng=seed,
    )
    return float(result.pvalue)


def cohens_dz(differences: np.ndarray) -> float:
    """Paired Cohen's d: the mean difference over its standard deviation."""
    d = np.asarray(differences, float)
    if len(d) < 2:
        return float("nan")
    sd = float(np.std(d, ddof=1))
    mean = float(d.mean())
    if sd == 0:
        return 0.0 if mean == 0 else math.copysign(math.inf, mean)
    return mean / sd


def tost_p(differences: np.ndarray, margin: float) -> float:
    """p-value of the TOST equivalence test of the mean difference within ±margin.

    The larger of the one-sided t-tests of ``mean > -margin`` and
    ``mean < margin``. With no spread the answer is certain: 0 inside the
    margin, 1 outside.
    """
    if margin <= 0:
        raise ValueError("the equivalence margin must be positive")
    d = np.asarray(differences, float)
    n = len(d)
    if n < 2:
        return float("nan")
    mean, sd = float(d.mean()), float(np.std(d, ddof=1))
    if sd == 0:
        return 0.0 if abs(mean) < margin else 1.0
    se = sd / math.sqrt(n)
    p_lower = stats.t.sf((mean + margin) / se, n - 1)  # H0: mean <= -margin
    p_upper = stats.t.cdf((mean - margin) / se, n - 1)  # H0: mean >= margin
    return float(max(p_lower, p_upper))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values, in the given order; NaN stays NaN."""
    p = np.asarray(pvalues, float)
    valid = np.flatnonzero(~np.isnan(p))
    out = np.full(len(p), np.nan)
    m = len(valid)
    running = 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid], kind="stable")]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def paired_power(n: int, delta: float, sd: float, alpha: float = 0.05) -> float:
    """Power of the two-sided paired t-test with ``n`` pairs to detect ``delta``."""
    if n < 2:
        return 0.0
    df = n - 1
    nc = abs(delta) / sd * math.sqrt(n)
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(crit, df, nc) + stats.nct.cdf(-crit, df, nc))


def seeds_needed(
    delta: float,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.8,
    max_n: int = 100_000,
) -> int:
    """Seeds (pairs) a paired t-test needs to detect ``delta`` with ``power``.

    ``sd`` is the standard deviation of the per-seed differences, from a pilot
    (the ``sd_diff`` column of a paired ``compare``).
    """
    if delta == 0 or sd <= 0 or not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("seeds_needed: delta != 0, sd > 0 and 0 < alpha, power < 1")
    for n in range(2, max_n + 1):
        if paired_power(n, delta, sd, alpha) >= power:
            return n
    raise ValueError(f"seeds_needed: more than {max_n} seeds")


def paired_test(
    differences: np.ndarray,
    margin: float | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Every paired statistic of one arm against the reference (unadjusted)."""
    d = np.asarray(differences, float)
    n = len(d)
    low, high = paired_bootstrap_ci(d, seed=seed)
    return {
        "n_pairs": n,
        "delta": float(d.mean()) if n else float("nan"),
        "delta_low": low,
        "delta_high": high,
        "sd_diff": float(np.std(d, ddof=1)) if n > 1 else float("nan"),
        "d_z": cohens_dz(d),
        "p_wilcoxon": wilcoxon_p(d) if n > 1 else float("nan"),
        "p_perm": sign_flip_p(d, seed=seed) if n > 1 else float("nan"),
        "p_tost": tost_p(d, margin) if margin is not None else float("nan"),
    }


def verdict(p_adj: float, p_tost: float, alpha: float = 0.05) -> str:
    """``equivalent`` within the margin, ``different``, or ``inconclusive``."""
    if p_tost < alpha:
        return "equivalent"
    if p_adj < alpha:
        return "different"
    return "inconclusive"
