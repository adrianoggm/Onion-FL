"""Tests for the modular model with namespaced parameter keys (issue #82).

These tests check structure and weights; they train nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from onion_fl.core.registry import PluginError
from onion_fl.learning.model import (
    DataShape,
    ModularMLP,
    ModularMLPConfig,
    group_of,
    is_aux,
    load_arrays,
    models,
    param_groups,
    state_arrays,
)

SWELL = DataShape(dataset="swell", task="stress_binary", n_features=16, n_classes=2)
SWEET = DataShape(dataset="sweet", task="stress_binary", n_features=14, n_classes=2)
SWEET3 = DataShape(dataset="sweet", task="stress_3class", n_features=14, n_classes=3)


def build(*shapes: DataShape, seed: int = 0, **config) -> ModularMLP:
    return ModularMLP(ModularMLPConfig(**config), list(shapes), seed=seed)


def groups(model: ModularMLP) -> set[str]:
    return set(param_groups(model.state_dict()))


# --- keys and structure ------------------------------------------------------


def test_edge_model_holds_only_its_own_parts() -> None:
    assert groups(build(SWELL)) == {"adapter.swell", "trunk", "head.stress_binary"}


def test_per_dataset_trunk_and_heads_are_namespaced_by_dataset() -> None:
    model = build(SWELL, trunk="per_dataset", heads="per_dataset")

    assert groups(model) == {"adapter.swell", "trunk.swell", "head.swell"}


def test_global_model_holds_every_part_and_shares_what_it_should() -> None:
    assert groups(build(SWELL, SWEET)) == {
        "adapter.swell",
        "adapter.sweet",
        "trunk",
        "head.stress_binary",
    }
    assert groups(build(SWELL, SWEET3)) == {
        "adapter.swell",
        "adapter.sweet",
        "trunk",
        "head.stress_binary",
        "head.stress_3class",
    }


def test_layers_are_sized_from_the_data() -> None:
    model = build(SWELL, SWEET3, adapter_width=8, trunk_hidden=[6, 4])
    state = model.state_dict()

    assert state["adapter.swell.0.weight"].shape == (8, 16)
    assert state["adapter.sweet.0.weight"].shape == (8, 14)
    assert state["head.stress_3class.weight"].shape == (3, 4)


def test_forward_routes_each_dataset_through_its_parts() -> None:
    model = build(SWELL, SWEET3).eval()

    assert model(torch.zeros(5, 16), dataset="swell").shape == (5, 2)
    assert model(torch.zeros(5, 14), dataset="sweet").shape == (5, 3)


def test_single_dataset_models_do_not_need_the_dataset_argument() -> None:
    assert build(SWELL).eval()(torch.zeros(3, 16)).shape == (3, 2)


def test_ambiguous_or_unknown_dataset_is_an_error() -> None:
    model = build(SWELL, SWEET)

    with pytest.raises(ValueError, match="dataset"):
        model(torch.zeros(1, 16))
    with pytest.raises(ValueError, match="wesad"):
        model(torch.zeros(1, 16), dataset="wesad")


def test_a_shared_head_needs_the_same_number_of_classes() -> None:
    clash = DataShape(dataset="sweet", task="stress_binary", n_features=14, n_classes=3)

    with pytest.raises(ValueError, match="stress_binary"):
        build(SWELL, clash)
    assert groups(build(SWELL, clash, heads="per_dataset")) >= {
        "head.swell",
        "head.sweet",
    }


@pytest.mark.parametrize(
    "fields",
    [
        {"dataset": "swell.v2"},
        {"dataset": "1swell"},
        {"task": "stress-binary"},
        {"n_features": 0},
        {"n_classes": 1},
    ],
)
def test_invalid_data_shapes_are_rejected(fields: dict) -> None:
    base = {
        "dataset": "swell",
        "task": "stress_binary",
        "n_features": 16,
        "n_classes": 2,
    }

    with pytest.raises(ValueError):
        DataShape(**(base | fields))


def test_a_model_needs_at_least_one_dataset() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build()


def test_datasets_must_be_unique() -> None:
    with pytest.raises(ValueError, match="swell"):
        build(SWELL, SWELL)


@pytest.mark.parametrize(
    "fields", [{"dropout": 1.0}, {"adapter_width": 0}, {"trunk_hidden": [0]}]
)
def test_invalid_configs_are_rejected(fields: dict) -> None:
    with pytest.raises(ValueError):
        ModularMLPConfig(**fields)


# --- shared adapter (harmonized features) -------------------------------------


SWELL_H = DataShape(dataset="swell", task="stress_binary", n_features=8, n_classes=2)
SWEET_H = DataShape(dataset="sweet", task="stress_binary", n_features=8, n_classes=2)


def test_a_shared_adapter_is_one_group_for_every_dataset() -> None:
    model = build(SWELL_H, SWEET_H, adapters="shared")

    assert groups(model) == {"adapter", "trunk", "head.stress_binary"}
    assert "adapter.0.weight" in model.state_dict()


def test_a_shared_adapter_serves_every_dataset() -> None:
    model = build(SWELL_H, SWEET_H, adapters="shared").eval()

    assert model(torch.zeros(2, 8), dataset="sweet").shape == (2, 2)


def test_a_shared_adapter_needs_the_same_features_everywhere() -> None:
    with pytest.raises(ValueError, match="n_features"):
        build(SWELL, SWEET, adapters="shared")


# --- deterministic initialisation --------------------------------------------


def test_the_same_seed_gives_the_same_weights() -> None:
    assert _equal_states(build(SWELL, seed=3), build(SWELL, seed=3))


def test_a_different_seed_gives_different_weights() -> None:
    assert not _equal_states(build(SWELL, seed=3), build(SWELL, seed=4))


def test_each_part_starts_the_same_in_edge_and_global_models() -> None:
    edge = build(SWELL, seed=5).state_dict()
    global_model = build(SWEET3, SWELL, seed=5).state_dict()

    for key, value in edge.items():
        assert torch.equal(value, global_model[key]), key


def _equal_states(a: ModularMLP, b: ModularMLP) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


# --- parameter groups --------------------------------------------------------


@pytest.mark.parametrize(
    "key, group",
    [
        ("adapter.swell.0.weight", "adapter.swell"),
        ("adapter.0.weight", "adapter"),
        ("trunk.0.weight", "trunk"),
        ("trunk.3.bias", "trunk"),
        ("trunk.swell.0.weight", "trunk.swell"),
        ("head.stress_binary.weight", "head.stress_binary"),
    ],
)
def test_group_of_classifies_every_key(key: str, group: str) -> None:
    assert group_of(key) == group


def test_group_of_rejects_keys_outside_the_namespace() -> None:
    with pytest.raises(ValueError, match="encoder"):
        group_of("encoder.0.weight")


# --- moving state in and out --------------------------------------------------


def test_state_arrays_are_numpy_copies() -> None:
    model = build(SWELL)
    arrays = state_arrays(model)

    assert all(
        isinstance(a, np.ndarray) and a.dtype == np.float32 for a in arrays.values()
    )
    arrays["trunk.0.weight"][:] = 99.0
    assert not torch.any(model.state_dict()["trunk.0.weight"] == 99.0)


def test_an_edge_loads_only_its_keys_from_the_global_state() -> None:
    global_state = state_arrays(build(SWELL, SWEET, seed=1))
    edge = build(SWELL, seed=2)

    loaded = load_arrays(edge, global_state)

    assert set(loaded) == set(edge.state_dict())
    assert "adapter.sweet.0.weight" not in loaded
    assert _equal_states(edge, _restricted(build(SWELL, SWEET, seed=1), edge))


def _restricted(source: ModularMLP, target: ModularMLP) -> ModularMLP:
    clone = build(SWELL, seed=0)
    clone.load_state_dict(
        {k: v for k, v in source.state_dict().items() if k in target.state_dict()}
    )
    return clone


def test_load_arrays_rejects_shape_mismatches() -> None:
    edge = build(SWELL)
    wrong = {"adapter.swell.0.weight": np.zeros((3, 3), dtype=np.float32)}

    with pytest.raises(ValueError, match="adapter.swell.0.weight"):
        load_arrays(edge, wrong)


# --- registry ----------------------------------------------------------------


def test_models_registry_builds_a_modular_mlp() -> None:
    family = models.create("modular_mlp", {"adapter_width": 8, "trunk_hidden": [4]})
    model = family.build([SWELL], seed=0)

    assert isinstance(model, ModularMLP)
    assert model.state_dict()["adapter.swell.0.weight"].shape == (8, 16)


def test_models_registry_validates_and_describes_the_config() -> None:
    with pytest.raises(PluginError, match="dropout"):
        models.create("modular_mlp", {"dropout": 2})

    (entry,) = models.describe()
    assert entry["name"] == "modular_mlp"
    assert set(entry["params"]["properties"]) >= {
        "adapter_width",
        "trunk_hidden",
        "trunk",
        "heads",
        "dropout",
    }


def test_auxiliary_keys_belong_to_their_parameter_group() -> None:
    assert is_aux("scaffold/trunk.0.weight") and not is_aux("trunk.0.weight")
    assert group_of("scaffold/trunk.0.weight") == "trunk"
    assert group_of("fednova/adapter.swell.0.weight") == "adapter.swell"


def test_the_head_reads_the_features() -> None:
    config = ModularMLPConfig(adapter_width=4, trunk_hidden=[3], dropout=0.0)
    model = ModularMLP(config, [SWELL], seed=0).eval()
    x = torch.zeros((2, SWELL.n_features))  # a shape fixture

    features = model.features(x)

    assert features.shape == (2, 3)
    torch.testing.assert_close(model(x), model.head[SWELL.task](features))
