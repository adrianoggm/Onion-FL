"""Tests for the second model family, ``feature_transformer`` (issue #229).

Structure, keys and aggregation use the weights the model is initialised with;
nothing is trained on invented data. The tests that train use the real SWELL
extract and are skipped when it is absent (docs/RULES.md).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from onion_fl.core.registry import PluginError
from onion_fl.learning.aggregators import Contribution, aggregators
from onion_fl.learning.model import (
    NAMESPACES,
    DataShape,
    FeatureTransformer,
    FeatureTransformerConfig,
    group_of,
    load_arrays,
    models,
    param_groups,
    state_arrays,
)
from onion_fl.learning.sharing import SharingError, sharing
from onion_fl.learning.trainers import trainers

SWELL = DataShape(dataset="swell", task="stress_binary", n_features=16, n_classes=2)
WESAD = DataShape(dataset="wesad", task="stress_binary", n_features=9, n_classes=2)
WESAD3 = DataShape(dataset="wesad", task="affect", n_features=9, n_classes=3)


def build(*shapes: DataShape, seed: int = 0, **config) -> FeatureTransformer:
    return FeatureTransformer(
        FeatureTransformerConfig(**config), list(shapes or [SWELL]), seed=seed
    )


def groups(model: FeatureTransformer) -> set[str]:
    return set(param_groups(model.state_dict()))


# --- keys follow the namespace contract ------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"trunk": "per_dataset", "heads": "per_dataset"},
        {"blocks": 3, "attention_heads": 4},
    ],
)
def test_every_key_is_in_the_namespace(config: dict) -> None:
    model = build(SWELL, WESAD3, **config)

    for key in model.state_dict():
        assert key.split(".")[0] in NAMESPACES
        group_of(key)  # raises outside the namespace


def test_edge_model_holds_only_its_own_parts() -> None:
    assert groups(build(SWELL)) == {"adapter.swell", "trunk", "head.stress_binary"}


def test_global_model_holds_every_part_and_shares_what_it_should() -> None:
    assert groups(build(SWELL, WESAD)) == {
        "adapter.swell",
        "adapter.wesad",
        "trunk",
        "head.stress_binary",
    }
    assert groups(build(SWELL, WESAD3)) == {
        "adapter.swell",
        "adapter.wesad",
        "trunk",
        "head.stress_binary",
        "head.affect",
    }


def test_per_dataset_trunk_and_heads_are_namespaced_by_dataset() -> None:
    model = build(SWELL, WESAD, trunk="per_dataset", heads="per_dataset")

    assert groups(model) == {
        "adapter.swell",
        "adapter.wesad",
        "trunk.swell",
        "trunk.wesad",
        "head.swell",
        "head.wesad",
    }


def test_a_shared_adapter_is_one_group_and_needs_the_same_features() -> None:
    other = DataShape(dataset="other", task="stress_binary", n_features=16, n_classes=2)

    assert groups(build(SWELL, other, adapters="shared")) == {
        "adapter",
        "trunk",
        "head.stress_binary",
    }
    with pytest.raises(ValueError, match="same n_features"):
        build(SWELL, WESAD, adapters="shared")


def test_the_trunk_does_not_depend_on_the_number_of_features() -> None:
    # one token per feature: datasets of 16 and 9 features share the trunk as it is
    swell, wesad = build(SWELL).state_dict(), build(WESAD).state_dict()
    trunk = [k for k in swell if k.startswith("trunk.")]

    assert trunk == [k for k in wesad if k.startswith("trunk.")]
    assert all(swell[k].shape == wesad[k].shape for k in trunk)
    assert swell["adapter.swell.0.weight"].shape == (16, 16)
    assert wesad["adapter.wesad.0.weight"].shape == (9, 16)


def test_forward_routes_each_dataset_and_the_head_reads_the_features() -> None:
    model = build(SWELL, WESAD3, dropout=0.0).eval()
    x = torch.zeros((3, SWELL.n_features))  # a shape fixture

    assert model(torch.zeros((3, 9)), dataset="wesad").shape == (3, 3)
    torch.testing.assert_close(
        model(x, dataset="swell"),
        model.head["stress_binary"](model.features(x, dataset="swell")),
    )
    with pytest.raises(ValueError, match="pass dataset"):
        model(x)


def test_an_edge_loads_only_its_keys_from_the_global_state() -> None:
    edge = build(WESAD, seed=1)

    loaded = load_arrays(edge, state_arrays(build(SWELL, WESAD, seed=0)))

    assert sorted(loaded) == sorted(edge.state_dict())


# --- registry, config and sharing ---------------------------------------------------


def test_the_registry_builds_and_describes_it() -> None:
    family = models.create("feature_transformer", {"token_dim": 8, "blocks": 2})

    assert isinstance(family.build([SWELL]), FeatureTransformer)
    entry = next(e for e in models.describe() if e["name"] == "feature_transformer")
    assert entry["title"] == "Transformer sobre features"
    assert set(entry["params"]["properties"]) == {
        "token_dim",
        "blocks",
        "attention_heads",
        "ffn_factor",
        "adapters",
        "trunk",
        "heads",
        "dropout",
    }


@pytest.mark.parametrize(
    "params, field",
    [
        ({"token_dim": 10, "attention_heads": 4}, "multiple"),
        ({"blocks": 0}, "blocks"),
        ({"dropout": 1.0}, "dropout"),
        ({"adapter_width": 8}, "adapter_width"),  # an MLP option
    ],
)
def test_invalid_configs_are_rejected(params: dict, field: str) -> None:
    with pytest.raises(PluginError, match=field):
        models.create("feature_transformer", params)


def test_the_sharing_requirements_apply_to_it() -> None:
    config = FeatureTransformerConfig()

    sharing.create("independent").check_model(
        FeatureTransformerConfig(trunk="per_dataset", heads="per_dataset")
    )
    with pytest.raises(SharingError, match="trunk"):
        sharing.create("independent").check_model(config)
    with pytest.raises(SharingError, match="adapters"):
        sharing.create("harmonized").check_model(config)


def test_a_requirement_the_family_cannot_meet_is_a_sharing_error() -> None:
    from onion_fl.learning.sharing import SharingPolicy

    policy = SharingPolicy(name="odd", model_requirements={"adapter_width": "8"})

    with pytest.raises(SharingError, match="no such option"):
        policy.check_model(FeatureTransformerConfig())


# --- hierarchical aggregation equals flat aggregation ---------------------------------


def _contributions() -> list[Contribution]:
    # Five edges with this family's real keys and shapes: SWELL edges 0-2 and
    # WESAD edges 3-4, so the adapters are partial keys. Different seeds give
    # different weights; the samples per key are hand-picked.
    shapes = [SWELL, SWELL, SWELL, WESAD, WESAD]
    samples = [30, 12, 7, 21, 40]
    out = []
    for i, (shape, n) in enumerate(zip(shapes, samples, strict=True)):
        state = state_arrays(build(shape, seed=i))
        out.append(
            Contribution(
                source=f"edge_{i}", state=state, weights=dict.fromkeys(state, n)
            )
        )
    return out


@pytest.mark.parametrize("split", [2, 3])
def test_hierarchical_fedavg_equals_flat_fedavg(split: int) -> None:
    updates = _contributions()
    fedavg = aggregators.create("fedavg")
    fog_a = fedavg.aggregate(updates[:split], source="fog_a")
    fog_b = fedavg.aggregate(updates[split:], source="fog_b")

    hierarchical = fedavg.aggregate([fog_a, fog_b], source="cloud")
    flat = fedavg.aggregate(updates, source="cloud")

    assert set(flat.state) == set(build(SWELL, WESAD).state_dict())
    assert hierarchical.weights == flat.weights
    assert flat.weights["trunk.0.token"] == 110
    assert flat.weights["adapter.wesad.0.weight"] == 61
    for key, value in flat.state.items():
        np.testing.assert_allclose(hierarchical.state[key], value, rtol=1e-6, atol=1e-7)


# --- determinism -------------------------------------------------------------------


def test_the_same_seed_gives_the_same_weights_and_another_differs() -> None:
    a, b, c = (state_arrays(build(SWELL, WESAD, seed=s)) for s in (0, 0, 1))

    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["trunk.1.qkv.weight"], c["trunk.1.qkv.weight"])
    assert not np.array_equal(a["adapter.swell.0.weight"], c["adapter.swell.0.weight"])


def test_each_part_starts_the_same_in_edge_and_global_models() -> None:
    edge = state_arrays(build(WESAD, seed=3))
    global_ = state_arrays(build(SWELL, WESAD, seed=3))

    for key, value in edge.items():
        np.testing.assert_array_equal(value, global_[key])


def test_random_init_restores_every_parameter() -> None:
    model = build(SWELL, seed=5)
    before = state_arrays(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    model.init_parameters(5)

    after = state_arrays(model)
    assert all(np.array_equal(before[k], after[k]) for k in before)


# --- real data: these tests train ---------------------------------------------------

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


def _shape(swell) -> DataShape:
    return DataShape(
        dataset="swell",
        task="stress_binary",
        n_features=swell.X.shape[1],
        n_classes=int(swell.y.max()) + 1,
    )


TORCH_TRAINERS = [n for n in trainers.names() if n != "stub"]


@real
@pytest.mark.parametrize("name", TORCH_TRAINERS)
def test_every_trainer_trains_it(swell, name: str) -> None:
    params = {"local_epochs": 1, "lr": 0.01}
    if name == "fedrep":
        params["head_epochs"] = 1
    model = build(_shape(swell), token_dim=8)
    received = state_arrays(model)

    result = trainers.create(name, params).train(
        model, swell, received, SimpleNamespace(rng=np.random.default_rng(0))
    )

    assert np.isfinite(result.loss) and result.samples > 0
    changed = state_arrays(model)
    assert any(not np.array_equal(received[k], changed[k]) for k in received)


@real
def test_training_lowers_the_loss(swell) -> None:
    model = build(_shape(swell), token_dim=8)
    trainer = trainers.create("standard", {"local_epochs": 1, "lr": 0.01})
    ctx = SimpleNamespace(rng=np.random.default_rng(0))

    first = trainer.train(model, swell, ctx=ctx)
    for _ in range(4):
        last = trainer.train(model, swell, ctx=ctx)

    assert last.loss < first.loss


@real
def test_the_same_seed_trains_to_the_same_model(swell) -> None:
    def trained() -> dict[str, np.ndarray]:
        model = build(_shape(swell), seed=2, token_dim=8)
        trainers.create("standard", {"local_epochs": 2, "lr": 0.01}).train(
            model, swell, ctx=SimpleNamespace(rng=np.random.default_rng(7))
        )
        return state_arrays(model)

    a, b = trained(), trained()

    assert all(np.array_equal(a[k], b[k]) for k in a)
