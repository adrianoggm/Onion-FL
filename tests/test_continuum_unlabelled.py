"""Objectives on unlabelled rows (continuum C5, issue #160).

The models are hand-built linear layers with hand-picked logits; nothing is
trained (docs/RULES.md).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from onion_fl.continuum.unlabelled import objectives
from onion_fl.core.registry import PluginError


def fixed(logits: list[list[float]]) -> tuple[nn.Module, np.ndarray]:
    """A model whose logits on the identity rows are ``logits``."""
    n, c = len(logits), len(logits[0])
    model = nn.Linear(n, c, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(logits, dtype=torch.float32).T)
    return model, np.eye(n, dtype=np.float32)


LOGITS = [[4.0, 0.0], [0.0, 0.1], [0.0, 5.0]]  # sure, unsure, sure


def max_prob(logits: list[list[float]]) -> np.ndarray:
    return torch.softmax(torch.tensor(logits), dim=1).max(dim=1).values.numpy()


def test_pseudo_label_accepts_the_rows_above_its_threshold() -> None:
    model, X = fixed(LOGITS)
    p = max_prob(LOGITS)
    for threshold in (0.5, 0.9, 1.0):
        fit = objectives.create("pseudo_label", {"threshold": threshold}).fit(model, X)
        assert fit["mask"].tolist() == (p >= threshold).tolist()
        assert fit["accepted"] == int((p >= threshold).sum())
        assert fit["seen"] == 3
    fit = objectives.create("pseudo_label", {"threshold": 0.9}).fit(model, X)
    assert fit["labels"].tolist() == [0, 1, 1]
    assert fit["confidence"] == pytest.approx(float(p.mean()))
    assert fit["row_confidence"] == pytest.approx(p)


def test_pseudo_label_term_is_zero_without_accepted_rows() -> None:
    model, X = fixed(LOGITS)
    objective = objectives.create("pseudo_label", {"threshold": 0.9})
    objective.fit(model, X)
    assert float(objective.term(model, np.array([1]))) == 0.0  # the unsure row
    assert float(objective.term(model, np.array([], np.int64))) == 0.0
    term = objective.term(model, np.array([0, 1, 2]))
    expected = 0.5 * torch.nn.functional.cross_entropy(
        model(torch.eye(3)[[0, 2]]), torch.tensor([0, 1])
    )
    assert float(term) == pytest.approx(float(expected))


def test_an_empty_pool_fits_to_nothing() -> None:
    model, _ = fixed(LOGITS)
    fit = objectives.create("pseudo_label").fit(model, np.zeros((0, 3), np.float32))
    assert fit["seen"] == fit["accepted"] == 0
    assert fit["confidence"] == 0.0


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("pseudo_label", {"threshold": 1.01}),
        ("pseudo_label", {"threshold": -0.1}),
        ("pseudo_label", {"weight": 0}),
        ("consistency", {"noise": 0}),
        ("consistency", {"weight": -1}),
        ("consistency", {"threshold": 0.9}),
    ],
)
def test_params_are_validated(name: str, params: dict) -> None:
    with pytest.raises(PluginError):
        objectives.create(name, params)


def test_consistency_vanishes_with_tiny_noise_and_not_with_noise() -> None:
    model, X = fixed(LOGITS)
    terms = {}
    for noise in (1e-6, 1.0):
        objective = objectives.create("consistency", {"noise": noise})
        fit = objective.fit(model, X)
        assert fit["labels"] is None and fit["mask"] is None
        assert fit["seen"] == fit["accepted"] == 3
        torch.manual_seed(0)
        terms[noise] = float(objective.term(model, np.array([0, 1, 2])))
    assert terms[1e-6] == pytest.approx(0.0, abs=1e-8)
    assert terms[1.0] > 1e-3
    assert float(objective.term(model, np.array([], np.int64))) == 0.0


def test_the_objective_axis_is_exported() -> None:
    from onion_fl.experiment.config import REGISTRIES

    assert REGISTRIES["unlabelled"] is objectives
    assert {"none", "pseudo_label", "consistency"} <= set(objectives.names())
