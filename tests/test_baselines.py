"""Tests for the classical baselines (issue #90).

Fitting a model is learning, so it runs only on the real SWELL data and is
skipped without it (docs/RULES.md). The rest checks folds, scores and the
registry with hand-picked numbers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from onion_fl.baselines import baseline_models, scores, subject_folds
from onion_fl.core.registry import PluginError


def test_scores_on_hand_picked_predictions() -> None:
    out = scores(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), n_classes=2)

    assert out["accuracy"] == pytest.approx(0.75)
    assert out["balanced_accuracy"] == pytest.approx(0.75)
    assert out["f1_macro"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert out["n_samples"] == 4


def test_scores_count_absent_classes_in_the_macro_f1() -> None:
    out = scores(np.array([0, 0]), np.array([0, 0]), n_classes=2)

    assert out["accuracy"] == 1.0
    assert out["f1_macro"] == pytest.approx(0.5)


def test_subject_folds_partition_the_subjects() -> None:
    names = [str(i) for i in range(1, 11)]

    folds = subject_folds(names, k=3, seed=0)

    assert sorted(len(f) for f in folds) == [3, 3, 4]
    assert sorted(s for f in folds for s in f) == sorted(names)


def test_subject_folds_are_deterministic_and_seeded() -> None:
    names = [str(i) for i in range(1, 11)]

    assert subject_folds(names, 3, 0) == subject_folds(list(reversed(names)), 3, 0)
    assert subject_folds(names, 3, 0) != subject_folds(names, 3, 1)


def test_more_folds_than_subjects_is_an_error() -> None:
    with pytest.raises(ValueError, match="3 subjects"):
        subject_folds(["1", "2", "3"], k=4, seed=0)


def test_registry_lists_the_models_and_validates_params() -> None:
    assert baseline_models.names() == ["lr", "rf", "xgboost"]
    with pytest.raises(PluginError, match="n_estimators"):
        baseline_models.create("rf", {"n_estimators": 0})


def test_models_are_built_with_the_seed() -> None:
    model = baseline_models.create("rf", {"n_estimators": 10}).build(seed=7)

    assert model.random_state == 7 and model.n_estimators == 10


@pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is not None, reason="xgboost is installed"
)
def test_xgboost_without_the_package_says_how_to_install_it() -> None:
    with pytest.raises(ImportError, match="analysis"):
        baseline_models.create("xgboost").build(seed=0)


# --- real data: these fit models ------------------------------------------------------


@pytest.mark.skipif(not Path("data/SWELL").exists(), reason="data/SWELL not available")
def test_baselines_beat_chance_on_swell_test_subjects() -> None:
    from onion_fl.baselines import evaluate
    from onion_fl.data.ingest import ingest, load_spec
    from onion_fl.data.roles import RolesConfig, split_subjects

    split = split_subjects(
        ingest(load_spec("datasets/swell.yaml")), RolesConfig(test=0.2)
    )

    results = evaluate(split, models=["lr"], seed=0)

    assert results["swell"]["lr"]["test"]["balanced_accuracy"] > 0.5


@pytest.mark.skipif(not Path("data/SWELL").exists(), reason="data/SWELL not available")
def test_subject_cross_validation_on_swell() -> None:
    from onion_fl.baselines import cross_validate
    from onion_fl.data.ingest import ingest, load_spec

    out = cross_validate(ingest(load_spec("datasets/swell.yaml")), model="lr", k=5)

    folds = out["swell"]["folds"]
    assert len(folds) == 5
    assert sorted(s for f in folds for s in f["test_subjects"]) == sorted(
        {s for f in folds for s in f["test_subjects"]}
    )
    assert 0.0 <= out["swell"]["mean"]["balanced_accuracy"] <= 1.0


@pytest.mark.skipif(not Path("data/SWELL").exists(), reason="data/SWELL not available")
def test_cross_validation_keeps_the_excluded_subjects_out() -> None:
    from onion_fl.baselines import cross_validate
    from onion_fl.data.ingest import ingest, load_spec
    from onion_fl.data.roles import RoleOverride, RolesConfig

    subjects = ingest(load_spec("datasets/swell.yaml"))
    out = [s.subject for s in subjects[:2]]
    roles = RolesConfig(overrides={"swell": RoleOverride(exclude=out)})

    folds = cross_validate(subjects, model="lr", k=3, roles=roles)["swell"]["folds"]

    tested = {s for f in folds for s in f["test_subjects"]}
    assert tested == {s.subject for s in subjects} - set(out)
