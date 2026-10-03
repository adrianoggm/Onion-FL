"""Tests for trainers, initialisation and the stub trainer (issue #85).

Following docs/RULES.md, the tests that run an optimiser use the real SWELL
extract and are skipped when it is absent. The rest check structure and
arithmetic with hand-picked numbers; nothing is trained on invented data.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from onion_fl.core.registry import PluginError
from onion_fl.learning.model import (
    DataShape,
    ModularMLP,
    ModularMLPConfig,
    load_arrays,
    state_arrays,
)
from onion_fl.learning.trainers import (
    TrainError,
    TrainResult,
    batches_of,
    inits,
    proximal_term,
    save_checkpoint,
    trainable,
    trainers,
)

SWELL = DataShape(dataset="swell", task="stress_binary", n_features=16, n_classes=2)
SWEET = DataShape(dataset="sweet", task="stress_binary", n_features=14, n_classes=2)


def build(*shapes: DataShape, seed: int = 0) -> ModularMLP:
    config = ModularMLPConfig(adapter_width=8, trunk_hidden=[4])
    return ModularMLP(config, list(shapes or [SWELL]), seed=seed)


# --- batching --------------------------------------------------------------------


def test_batches_cover_every_sample_once() -> None:
    batches = batches_of(10, 4, np.random.default_rng(0))

    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(np.concatenate(batches).tolist()) == list(range(10))


def test_batches_are_shuffled_by_the_rng() -> None:
    a = np.concatenate(batches_of(50, 50, np.random.default_rng(1)))
    b = np.concatenate(batches_of(50, 50, np.random.default_rng(1)))
    c = np.concatenate(batches_of(50, 50, np.random.default_rng(2)))

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, np.arange(50))


# --- frozen groups -----------------------------------------------------------------


def test_nothing_frozen_trains_every_parameter() -> None:
    model = build()

    assert trainable(model, []) == [name for name, _ in model.named_parameters()]


def test_frozen_groups_accept_names_and_patterns() -> None:
    model = build(SWELL, SWEET)

    names = trainable(model, ["trunk", "adapter.*"])

    assert names
    assert all(name.startswith("head.") for name in names)


def test_freezing_a_group_the_model_lacks_is_harmless() -> None:
    # The same trainer config reaches every edge; a SWELL edge has no SWEET adapter.
    model = build()

    assert trainable(model, ["adapter.sweet"]) == trainable(model, [])


# --- fedprox proximal term ------------------------------------------------------------


def test_the_proximal_term_is_zero_at_the_received_state() -> None:
    model = build()

    assert float(proximal_term(model, state_arrays(model), mu=0.5)) == 0.0


def test_the_proximal_term_is_half_mu_times_the_squared_distance() -> None:
    model = build()
    received = {k: v + 1.0 for k, v in state_arrays(model).items()}
    n_params = sum(p.numel() for p in model.parameters())

    value = float(proximal_term(model, received, mu=0.5))

    assert value == pytest.approx(0.25 * n_params)


def test_the_proximal_term_only_uses_received_trainable_keys() -> None:
    model = build()
    received = {
        k: v + 1.0 for k, v in state_arrays(model).items() if k.startswith("head.")
    }
    n_head = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith("head.")
    )

    assert float(proximal_term(model, received, mu=2.0)) == pytest.approx(n_head)
    assert float(proximal_term(model, received, mu=2.0, names=[])) == 0.0


def test_the_proximal_term_has_a_gradient() -> None:
    model = build()
    received = {k: v + 1.0 for k, v in state_arrays(model).items()}

    proximal_term(model, received, mu=1.0).backward()

    grad = model.head["stress_binary"].weight.grad
    assert grad is not None
    assert torch.allclose(grad, torch.full_like(grad, -1.0))


# --- trainer configs and errors (no optimiser step runs) ------------------------------


@pytest.mark.parametrize(
    "params",
    [{"lr": 0}, {"local_epochs": 0}, {"batch_size": 0}, {"optimizer": "rmsprop"}],
)
def test_standard_params_are_validated(params: dict) -> None:
    with pytest.raises(PluginError):
        trainers.create("standard", params)


def test_fedprox_mu_is_validated() -> None:
    with pytest.raises(PluginError, match="mu"):
        trainers.create("fedprox", {"mu": -1})


def test_training_without_samples_is_an_error() -> None:
    empty = SimpleNamespace(X=np.zeros((0, 16), np.float32), y=np.zeros(0, np.int64))

    with pytest.raises(TrainError, match="no samples"):
        trainers.create("standard").train(build(), empty)


def test_features_and_labels_must_have_the_same_length() -> None:
    data = SimpleNamespace(X=np.zeros((3, 16), np.float32), y=np.zeros(2, np.int64))

    with pytest.raises(TrainError, match="3 rows"):
        trainers.create("standard").train(build(), data)


def test_freezing_everything_is_an_error() -> None:
    trainer = trainers.create("standard", {"frozen": ["*"]})
    data = SimpleNamespace(X=np.zeros((1, 16), np.float32), y=np.zeros(1, np.int64))

    with pytest.raises(TrainError, match="frozen"):
        trainer.train(build(), data)


def test_fedprox_needs_the_received_state() -> None:
    data = SimpleNamespace(X=np.zeros((1, 16), np.float32), y=np.zeros(1, np.int64))

    with pytest.raises(TrainError, match="received"):
        trainers.create("fedprox").train(build(), data, received=None)


# --- stub trainer (protocol tests) ----------------------------------------------------


def test_the_stub_shifts_every_weight_deterministically() -> None:
    model = build()
    before = state_arrays(model)

    result = trainers.create("stub", {"shift": 0.5, "examples": 7}).train(model)

    after = state_arrays(model)
    for key, value in before.items():
        np.testing.assert_allclose(after[key], value + 0.5)
    assert result == TrainResult(loss=0.0, samples=7, examples=7, batches=1)


def test_the_stub_needs_no_data() -> None:
    result = trainers.create("stub").train(build(), None, received=None, ctx=None)

    assert result.examples > 0


def test_trainers_registry_lists_the_built_ins() -> None:
    assert trainers.names() == [
        "apfl",
        "ditto",
        "fedbabu",
        "feddyn",
        "fednova",
        "fedprox",
        "fedrep",
        "moon",
        "scaffold",
        "standard",
        "stub",
    ]
    assert inits.names() == ["checkpoint", "random"]


# --- initialisation ---------------------------------------------------------------------


def test_random_init_with_a_seed_matches_a_model_built_with_it() -> None:
    model = build(seed=0)

    inits.create("random", {"seed": 9}).init(model)

    expected = state_arrays(build(seed=9))
    for key, value in state_arrays(model).items():
        np.testing.assert_array_equal(value, expected[key])


def test_random_init_without_a_seed_keeps_the_experiment_seed() -> None:
    model = build(seed=4)

    inits.create("random").init(model)

    expected = state_arrays(build(seed=4))
    for key, value in state_arrays(model).items():
        np.testing.assert_array_equal(value, expected[key])


def test_checkpoint_loads_only_the_chosen_groups(tmp_path: Path) -> None:
    path = tmp_path / "pretrained.npz"
    save_checkpoint(build(seed=1), path)
    model = build(seed=2)
    before = state_arrays(model)

    loaded = inits.create("checkpoint", {"path": str(path), "groups": ["trunk"]}).init(
        model
    )

    pretrained, after = state_arrays(build(seed=1)), state_arrays(model)
    assert loaded and all(key.startswith("trunk.") for key in loaded)
    for key, value in after.items():
        expected = pretrained[key] if key.startswith("trunk.") else before[key]
        np.testing.assert_array_equal(value, expected)


def test_checkpoint_loads_everything_by_default(tmp_path: Path) -> None:
    path = tmp_path / "pretrained.npz"
    save_checkpoint(build(SWELL, SWEET, seed=1), path)  # a global model
    model = build(seed=2)  # a SWELL edge

    loaded = inits.create("checkpoint", {"path": str(path)}).init(model)

    assert set(loaded) == set(model.state_dict())


def test_checkpoint_rejects_a_group_it_does_not_hold(tmp_path: Path) -> None:
    path = tmp_path / "pretrained.npz"
    save_checkpoint(build(), path)

    with pytest.raises(TrainError, match="adapter.wesad"):
        inits.create(
            "checkpoint", {"path": str(path), "groups": ["adapter.wesad"]}
        ).init(build())


def test_checkpoint_must_share_keys_with_the_model(tmp_path: Path) -> None:
    path = tmp_path / "pretrained.npz"
    save_checkpoint(build(SWEET), path)

    with pytest.raises(TrainError, match="no key"):
        inits.create(
            "checkpoint", {"path": str(path), "groups": ["adapter.sweet"]}
        ).init(build(SWELL))


def test_checkpoint_never_unpickles(tmp_path: Path) -> None:
    path = tmp_path / "evil.npz"
    np.savez(path, **{"trunk.0.weight": np.array([object()], dtype=object)})

    with pytest.raises(ValueError, match="pickle"):
        inits.create("checkpoint", {"path": str(path)}).init(build())


# --- real data: these tests train, so they need the real extract ------------------------

SWELL_SAMPLE = Path("data/samples/swell_real_sample.pkl")
real = pytest.mark.skipif(
    not SWELL_SAMPLE.exists(), reason="swell_real_sample.pkl not available"
)


@pytest.fixture
def swell():
    from onion_fl.datasets.samples import load_swell_sample_features

    X, y = load_swell_sample_features(SWELL_SAMPLE)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return SimpleNamespace(X=X.astype(np.float32), y=y)


def ctx(seed: int = 0) -> SimpleNamespace:
    return SimpleNamespace(rng=np.random.default_rng(seed))


@real
def test_standard_training_lowers_the_training_loss(swell) -> None:
    model = build()
    trainer = trainers.create("standard", {"local_epochs": 1, "lr": 0.01})

    first = trainer.train(model, swell, ctx=ctx())
    for _ in range(4):
        last = trainer.train(model, swell, ctx=ctx())

    assert last.loss < first.loss
    assert first.examples == len(swell.y)
    assert first.samples == len(swell.y)
    assert first.batches == -(-len(swell.y) // 32)


@real
def test_frozen_groups_do_not_change(swell) -> None:
    model = build()
    before = state_arrays(model)

    trainers.create("standard", {"frozen": ["trunk", "adapter.*"]}).train(
        model, swell, ctx=ctx()
    )

    after = state_arrays(model)
    for key, value in before.items():
        changed = not np.array_equal(after[key], value)
        assert changed == key.startswith("head."), key
    assert all(p.requires_grad for p in model.parameters())


@real
def test_the_same_rng_gives_the_same_weights(swell) -> None:
    a, b = build(), build()
    trainer = trainers.create("standard", {"local_epochs": 2})

    trainer.train(a, swell, ctx=ctx(3))
    trainer.train(b, swell, ctx=ctx(3))

    for key, value in state_arrays(a).items():
        np.testing.assert_array_equal(value, state_arrays(b)[key])


@real
def test_fedprox_stays_closer_to_the_received_state(swell) -> None:
    received = state_arrays(build())
    plain, prox = build(), build()

    params = {"local_epochs": 3, "lr": 0.05}
    trainers.create("standard", params).train(plain, swell, received, ctx())
    trainers.create("fedprox", params | {"mu": 10.0}).train(
        prox, swell, received, ctx()
    )

    def distance(model: ModularMLP) -> float:
        return sum(
            float(((v - received[k]) ** 2).sum())
            for k, v in state_arrays(model).items()
        )

    assert distance(prox) < distance(plain)


@real
def test_ditto_keeps_a_personal_model_apart_from_the_global(swell) -> None:
    model = build()
    received = state_arrays(model)
    trainer = trainers.create("ditto", {"local_epochs": 2, "lr": 0.05, "lam": 0.1})

    result = trainer.train(model, swell, received, ctx())

    personal, trained = state_arrays(trainer.personal()), state_arrays(model)
    assert any(not np.array_equal(personal[k], trained[k]) for k in trained)
    assert any(not np.array_equal(personal[k], received[k]) for k in received)
    assert result.samples == 4 * len(swell.y)  # global and personal epochs
    assert result.examples == len(swell.y)


@real
def test_a_larger_lambda_keeps_the_personal_model_near_the_global(swell) -> None:
    received = state_arrays(build())

    def distance(lam: float) -> float:
        trainer = trainers.create("ditto", {"local_epochs": 3, "lr": 0.05, "lam": lam})
        trainer.train(build(), swell, received, ctx())
        return sum(
            float(((v - received[k]) ** 2).sum())
            for k, v in state_arrays(trainer.personal()).items()
        )

    assert distance(10.0) < distance(0.0)


@real
def test_ditto_trains_when_the_heads_stay_on_the_edge(swell) -> None:
    model = build()
    received = {
        k: v for k, v in state_arrays(model).items() if not k.startswith("head.")
    }
    trainer = trainers.create("ditto", {"lr": 0.05})

    trainer.train(model, swell, received, ctx())

    assert trainer.personal() is not None


@real
def test_apfl_with_alpha_zero_is_the_global_model(swell) -> None:
    model = build()
    trainer = trainers.create("apfl", {"lr": 0.05, "alpha": 0.0, "adapt_alpha": False})

    trainer.train(model, swell, state_arrays(model), ctx())

    for key, value in state_arrays(trainer.personal()).items():
        np.testing.assert_allclose(value, state_arrays(model)[key])


@real
def test_apfl_learns_its_alpha_within_bounds(swell) -> None:
    model = build()
    trainer = trainers.create(
        "apfl", {"local_epochs": 2, "lr": 0.05, "alpha": 0.5, "alpha_lr": 0.5}
    )

    result = trainer.train(model, swell, state_arrays(model), ctx())

    assert trainer.alpha != 0.5 and 0.0 <= trainer.alpha <= 1.0
    personal = state_arrays(trainer.personal())
    assert any(not np.allclose(personal[k], v) for k, v in state_arrays(model).items())
    assert result.samples == 2 * 2 * len(swell.y)  # two models per batch


@real
def test_fedrep_trains_the_head_then_the_body(swell) -> None:
    model = build()
    before = state_arrays(model)
    trainer = trainers.create(
        "fedrep", {"head_epochs": 2, "local_epochs": 1, "lr": 0.05}
    )

    result = trainer.train(model, swell, before, ctx())

    after = state_arrays(model)
    assert all(not np.array_equal(after[k], v) for k, v in before.items())
    assert result.samples == 3 * len(swell.y)
    assert trainer.local_groups == ["head*"]


@real
def test_fedbabu_leaves_the_head_as_it_was_initialised(swell) -> None:
    model = build()
    before = state_arrays(model)

    trainers.create("fedbabu", {"lr": 0.05}).train(model, swell, before, ctx())

    after = state_arrays(model)
    for key, value in before.items():
        assert np.array_equal(after[key], value) == key.startswith("head."), key


def rounds(trainer, swell, n: int = 2):
    """Train ``n`` rounds as an edge does: each starts from the current global."""
    model, context = build(), ctx(5)
    for _ in range(n):
        trainer.train(model, swell, state_arrays(model), context)
    return state_arrays(model)


@real
@pytest.mark.parametrize("name", ["ditto", "apfl"])
def test_the_global_model_trains_exactly_as_standard(swell, name: str) -> None:
    params = {"local_epochs": 1, "lr": 0.05}

    plain = rounds(trainers.create("standard", params), swell)
    personalised = rounds(trainers.create(name, params), swell)

    for key, value in plain.items():
        np.testing.assert_array_equal(value, personalised[key], err_msg=key)


@real
def test_ditto_keeps_its_personal_model_between_rounds(swell) -> None:
    trainer = trainers.create("ditto", {"lr": 0.05})
    model = build()
    trainer.train(model, swell, state_arrays(model), ctx())
    kept = trainer.personal()
    second = state_arrays(model)
    trainer.train(model, swell, second, ctx(1))

    fresh = trainers.create("ditto", {"lr": 0.05})
    fresh.train(build(), swell, second, ctx(1))

    assert trainer.personal() is kept
    mine, theirs = state_arrays(kept), state_arrays(fresh.personal())
    assert any(not np.array_equal(mine[k], theirs[k]) for k in mine)


@real
def test_apfl_starts_each_round_from_its_last_alpha(swell) -> None:
    trainer = trainers.create("apfl", {"lr": 0.05, "alpha_lr": 1e-9})
    model = build()
    trainer.train(model, swell, state_arrays(model), ctx())

    trainer.alpha = 0.9  # as if the first round had learnt it
    trainer.train(model, swell, state_arrays(model), ctx(1))

    assert trainer.alpha == pytest.approx(0.9, abs=1e-4)


@real
def test_apfl_keeps_its_personal_model_between_rounds(swell) -> None:
    params = {"lr": 0.05, "adapt_alpha": False}
    trainer = trainers.create("apfl", params)
    model = build()
    trainer.train(model, swell, state_arrays(model), ctx())
    second = state_arrays(model)
    trainer.train(model, swell, second, ctx(1))

    fresh, start = trainers.create("apfl", params), build()
    load_arrays(start, second)
    fresh.train(start, swell, second, ctx(1))

    mine, theirs = state_arrays(trainer.personal()), state_arrays(fresh.personal())
    assert any(not np.allclose(mine[k], theirs[k]) for k in mine)


@real
def test_scaffold_without_control_variates_is_sgd(swell) -> None:
    params = {"local_epochs": 1, "lr": 0.05, "optimizer": "sgd"}
    plain, corrected = build(), build()
    received = state_arrays(plain)

    trainers.create("standard", params).train(plain, swell, received, ctx())
    trainers.create("scaffold", params).train(corrected, swell, received, ctx())

    for key, value in state_arrays(plain).items():
        np.testing.assert_array_equal(value, state_arrays(corrected)[key])


@real
def test_scaffold_sends_its_new_control_variate(swell) -> None:
    model = build()
    received = state_arrays(model)
    trainer = trainers.create("scaffold", {"local_epochs": 1, "lr": 0.05})

    result = trainer.train(model, swell, received, ctx())

    after = state_arrays(model)
    for key, x in received.items():
        expected = (x - after[key]) / (result.batches * 0.05)  # c = c_i = 0
        np.testing.assert_allclose(
            result.aux[f"scaffold/{key}"], expected, rtol=1e-5, atol=1e-7
        )


@real
def test_scaffold_corrects_with_the_received_control_variate(swell) -> None:
    received = state_arrays(build())
    shifted = received | {
        f"scaffold/{k}": np.full_like(v, 0.1) for k, v in received.items()
    }
    plain, corrected = build(), build()
    params = {"local_epochs": 1, "lr": 0.05}

    trainers.create("scaffold", params).train(plain, swell, received, ctx())
    trainers.create("scaffold", params).train(corrected, swell, shifted, ctx())

    assert any(
        not np.array_equal(v, state_arrays(corrected)[k])
        for k, v in state_arrays(plain).items()
    )


@real
def test_scaffold_trains_when_the_heads_stay_on_the_edge(swell) -> None:
    model = build()
    received = {
        k: v for k, v in state_arrays(model).items() if not k.startswith("head.")
    }

    result = trainers.create("scaffold", {"lr": 0.05}).train(
        model, swell, received, ctx()
    )

    assert any(k.startswith("scaffold/head.") for k in result.aux)


@real
def test_fednova_sends_its_normalised_update(swell) -> None:
    model = build()
    received = state_arrays(model)

    result = trainers.create("fednova", {"lr": 0.05}).train(
        model, swell, received, ctx()
    )

    after = state_arrays(model)
    for key, x in received.items():
        np.testing.assert_allclose(
            result.aux[f"fednova/{key}"],
            (after[key] - x) / result.batches,
            rtol=1e-5,
            atol=1e-8,
        )


@real
def test_feddyn_without_history_is_proximal(swell) -> None:
    received = state_arrays(build())
    prox, dyn = build(), build()
    params = {"local_epochs": 1, "lr": 0.05}

    trainers.create("fedprox", params | {"mu": 0.3}).train(prox, swell, received, ctx())
    trainers.create("feddyn", params | {"alpha": 0.3}).train(
        dyn, swell, received, ctx()
    )

    for key, value in state_arrays(prox).items():
        np.testing.assert_allclose(value, state_arrays(dyn)[key], rtol=1e-6)


@real
def test_feddyn_remembers_its_gradient_term(swell) -> None:
    trainer = trainers.create("feddyn", {"lr": 0.05, "alpha": 0.3})
    model = build()
    trainer.train(model, swell, state_arrays(model), ctx())
    second = state_arrays(model)
    trainer.train(model, swell, second, ctx(1))

    fresh, start = trainers.create("feddyn", {"lr": 0.05, "alpha": 0.3}), build()
    load_arrays(start, second)
    fresh.train(start, swell, second, ctx(1))

    assert any(
        not np.allclose(v, state_arrays(start)[k])
        for k, v in state_arrays(model).items()
    )


@real
def test_moon_in_its_first_round_is_standard(swell) -> None:
    params = {"local_epochs": 1, "lr": 0.05}
    plain, moon = build(), build()
    received = state_arrays(plain)

    trainers.create("standard", params).train(plain, swell, received, ctx())
    trainers.create("moon", params).train(moon, swell, received, ctx())

    for key, value in state_arrays(plain).items():
        np.testing.assert_array_equal(value, state_arrays(moon)[key])


@real
def test_moon_pulls_towards_the_global_representation_from_round_two(swell) -> None:
    params = {"local_epochs": 1, "lr": 0.05}
    plain, moon = build(), build()
    standard = trainers.create("standard", params)
    contrastive = trainers.create("moon", params | {"mu": 5.0})
    for model, trainer in ((plain, standard), (moon, contrastive)):
        trainer.train(model, swell, state_arrays(model), ctx())

    standard.train(plain, swell, state_arrays(plain), ctx(1))
    contrastive.train(moon, swell, state_arrays(moon), ctx(1))

    assert any(
        not np.allclose(v, state_arrays(moon)[k])
        for k, v in state_arrays(plain).items()
    )


@real
def test_scaffold_alone_with_local_heads_is_sgd_for_two_rounds(swell) -> None:
    # One edge: c equals its own c_i on shared keys and never arrives for the
    # local head, so the correction must be zero everywhere, round after round.
    params = {"local_epochs": 1, "lr": 0.05, "optimizer": "sgd"}
    plain, corrected = build(), build()
    sgd, scaffold = (
        trainers.create("standard", params),
        trainers.create("scaffold", params),
    )
    c: dict = {}
    for round_ in range(2):
        shared = {
            k: v for k, v in state_arrays(plain).items() if not k.startswith("head.")
        }
        sgd.train(plain, swell, shared, ctx(round_))
        received = {
            k: v
            for k, v in state_arrays(corrected).items()
            if not k.startswith("head.")
        } | {k: v for k, v in c.items() if not k.startswith("scaffold/head.")}
        sent = dict(scaffold.train(corrected, swell, received, ctx(round_)).aux)
        c = {k: c.get(k, 0.0) + v for k, v in sent.items()}  # the server, alone

    for key, value in state_arrays(plain).items():
        np.testing.assert_allclose(
            value, state_arrays(corrected)[key], rtol=1e-5, atol=1e-6
        )


@real
def test_fednova_sends_its_steps_with_every_key(swell) -> None:
    model = build()

    result = trainers.create("fednova", {"lr": 0.05}).train(
        model, swell, state_arrays(model), ctx()
    )

    for key in state_arrays(model):
        np.testing.assert_array_equal(
            result.aux[f"fednova_steps/{key}"], [float(result.batches)]
        )


def test_fednova_defaults_to_sgd() -> None:
    assert trainers.create("fednova").params.optimizer == "sgd"


@pytest.mark.parametrize("name", ["scaffold", "fednova"])
def test_step_based_trainers_refuse_adam(name: str) -> None:
    with pytest.raises(ValueError, match="sgd"):
        trainers.create(name, {"optimizer": "adam"})


def test_moon_contrast_is_lower_near_the_global_than_near_the_previous() -> None:
    import copy

    trainer = trainers.create("moon", {"mu": 1.0, "temperature": 0.5})
    local, other = build(seed=0).eval(), build(seed=1).eval()
    x = torch.linspace(-1.0, 1.0, 4 * 16).reshape(4, 16)  # a shape fixture

    trainer._global, trainer._previous = copy.deepcopy(local), other
    near_global = float(trainer._batch_term(local, x))
    trainer._global, trainer._previous = other, copy.deepcopy(local)
    near_previous = float(trainer._batch_term(local, x))

    assert near_global < near_previous


def test_ditto_needs_the_received_state() -> None:
    data = SimpleNamespace(X=np.zeros((1, 16), np.float32), y=np.zeros(1, np.int64))

    with pytest.raises(TrainError, match="received"):
        trainers.create("ditto").train(build(), data, None, ctx())


def test_the_stub_can_add_seeded_noise_per_node() -> None:
    def shifted(seed: int) -> dict:
        model = build()
        trainers.create("stub", {"shift": 0.0, "noise": 0.1}).train(
            model, ctx=ctx(seed)
        )
        return state_arrays(model)

    a, b, c = shifted(1), shifted(1), shifted(2)

    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert any(not np.array_equal(a[k], c[k]) for k in a)
    assert any(not np.array_equal(a[k], state_arrays(build())[k]) for k in a)


@real
def test_scaffold_sends_the_change_in_its_control_variate(swell) -> None:
    model = build()
    trainer = trainers.create("scaffold", {"local_epochs": 1, "lr": 0.05})
    first = trainer.train(model, swell, state_arrays(model), ctx())
    received = state_arrays(model)  # no c arrives: c = c_i, no correction

    second = trainer.train(model, swell, received, ctx(1))

    after = state_arrays(model)
    for key, x in received.items():
        c_i = (x - after[key]) / (second.batches * 0.05)
        np.testing.assert_allclose(
            second.aux[f"scaffold/{key}"],
            c_i - first.aux[f"scaffold/{key}"],
            rtol=1e-4,
            atol=1e-6,
        )
