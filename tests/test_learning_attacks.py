"""Malicious edges (issue #149).

Hand-written arrays and a tiny label array; nothing is trained. Label
flipping corrupts real labels to simulate an attacker (docs/RULES.md).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from onion_fl.learning.attacks import attacks

RECEIVED = {"w": np.zeros(2), "head.t.weight": np.zeros(1)}
TRAINED = {
    "w": np.array([1.0, 2.0]),
    "head.t.weight": np.ones(1),
    "scaffold/w": np.full(2, 5.0),
}


def test_sign_flip_reverses_the_update_of_crossing_keys_only() -> None:
    out = attacks.create("sign_flip", {"scale": 1.0}).on_update(
        TRAINED, {"w": RECEIVED["w"]}, np.random.default_rng(0)
    )

    np.testing.assert_allclose(out["w"], [-1.0, -2.0])
    np.testing.assert_allclose(out["head.t.weight"], [1.0])  # not received: local
    np.testing.assert_allclose(out["scaffold/w"], [5.0, 5.0])  # auxiliary: untouched


def test_scale_boosts_the_update() -> None:
    out = attacks.create("scale", {"factor": 10.0}).on_update(
        TRAINED, RECEIVED, np.random.default_rng(0)
    )

    np.testing.assert_allclose(out["w"], [10.0, 20.0])


def test_gaussian_replaces_the_update_with_seeded_noise() -> None:
    attack = attacks.create("gaussian", {"sigma": 1.0})
    a = attack.on_update(TRAINED, RECEIVED, np.random.default_rng(3))
    b = attack.on_update(TRAINED, RECEIVED, np.random.default_rng(3))

    np.testing.assert_array_equal(a["w"], b["w"])
    assert not np.allclose(a["w"], TRAINED["w"])


def test_label_flip_mirrors_the_labels_and_keeps_the_original() -> None:
    data = SimpleNamespace(X=np.zeros((3, 1)), y=np.array([0, 1, 1]), n_classes=2)

    flipped = attacks.create("label_flip").on_data(data)

    assert flipped.y.tolist() == [1, 0, 0] and data.y.tolist() == [0, 1, 1]


def test_every_attack_has_a_fraction_and_a_start_round() -> None:
    for name in attacks.names():
        attack = attacks.create(name, {"fraction": 0.3, "start_round": 2})
        assert (attack.fraction, attack.start_round) == (0.3, 2)
