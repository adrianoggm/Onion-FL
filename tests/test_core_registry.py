"""Tests for the generic plugin registry (issue #79)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from onion_fl.core.codec import JsonCodec, codecs, get_codec
from onion_fl.core.registry import PluginError, PluginSpec, Registry


class FedProxParams(BaseModel):
    mu: float = Field(0.01, ge=0, description="Peso del término proximal")


class FedProx:
    def __init__(self, mu: float) -> None:
        self.mu = mu


class Plain:
    """A plugin defined outside the core, without registry metadata."""


EXTERNAL_SPEC = PluginSpec(
    name="external",
    factory=FedProx,
    title="Externo",
    description="Plugin definido fuera del núcleo",
    params=FedProxParams,
)


@pytest.fixture
def trainers() -> Registry:
    registry = Registry("trainer")
    registry.register(
        "fedprox",
        title="FedProx",
        description="Entrenamiento local con término proximal",
        params=FedProxParams,
        explain="Penaliza alejarse del modelo global recibido.",
    )(FedProx)
    return registry


def test_registered_plugin_is_created_with_default_params(trainers: Registry) -> None:
    trainer = trainers.create("fedprox")

    assert isinstance(trainer, FedProx)
    assert trainer.mu == 0.01


def test_params_are_validated_and_passed_to_the_factory(trainers: Registry) -> None:
    assert trainers.create("fedprox", {"mu": 0.5}).mu == 0.5


def test_invalid_params_name_the_field(trainers: Registry) -> None:
    with pytest.raises(PluginError, match="mu"):
        trainers.create("fedprox", {"mu": -1})


def test_unknown_params_are_rejected_instead_of_ignored(trainers: Registry) -> None:
    with pytest.raises(PluginError, match="muu"):
        trainers.create("fedprox", {"muu": 0.5})


def test_params_for_a_plugin_without_params_are_rejected() -> None:
    registry = Registry("codec")
    registry.register("json", title="JSON", description="Texto")(JsonCodec)

    with pytest.raises(PluginError, match="takes no parameters"):
        registry.create("json", {"indent": 2})


def test_unknown_name_lists_the_available_plugins(trainers: Registry) -> None:
    with pytest.raises(PluginError, match="fedprox"):
        trainers.get("scaffold")


def test_duplicate_names_are_rejected(trainers: Registry) -> None:
    with pytest.raises(PluginError, match="already registered"):
        trainers.register("fedprox", title="Otro", description="Duplicado")(FedProx)


def test_dotted_path_resolves_an_external_plugin_spec(trainers: Registry) -> None:
    spec = trainers.get(f"{__name__}:EXTERNAL_SPEC")

    assert spec is EXTERNAL_SPEC
    assert trainers.create(f"{__name__}:EXTERNAL_SPEC", {"mu": 0.2}).mu == 0.2


def test_dotted_path_to_a_plain_factory_is_wrapped(trainers: Registry) -> None:
    assert isinstance(trainers.create(f"{__name__}:Plain"), Plain)


@pytest.mark.parametrize(
    "path", ["no_such_module_xyz:Thing", f"{__name__}:missing_attr", f"{__name__}:"]
)
def test_bad_dotted_paths_are_reported(trainers: Registry, path: str) -> None:
    with pytest.raises(PluginError):
        trainers.get(path)


NOT_A_PLUGIN = 42


def test_dotted_path_to_a_non_callable_is_rejected(trainers: Registry) -> None:
    with pytest.raises(PluginError, match="neither"):
        trainers.get(f"{__name__}:NOT_A_PLUGIN")


def test_describe_exposes_metadata_and_param_schema(trainers: Registry) -> None:
    (entry,) = trainers.describe()

    assert entry["name"] == "fedprox"
    assert entry["kind"] == "trainer"
    assert entry["title"] == "FedProx"
    assert entry["explain"] == "Penaliza alejarse del modelo global recibido."
    assert entry["params"]["properties"]["mu"]["minimum"] == 0


def test_codecs_live_in_the_registry() -> None:
    assert codecs.names() == ["json", "npz"]
    assert get_codec("npz").name == "npz"
    assert {entry["name"] for entry in codecs.describe()} == {"json", "npz"}
