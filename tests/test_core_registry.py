"""Tests for the generic plugin registry (issue #79)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from onion_fl.core.codec import JsonCodec, codecs, get_codec
from onion_fl.core.registry import TRUSTED_ENV, PluginError, PluginSpec, Registry


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


@pytest.fixture(autouse=True)
def trust_this_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator trusts the tests' package, as ONIONFL_TRUSTED_MODULES=tests."""
    monkeypatch.setenv(TRUSTED_ENV, __name__.split(".")[0])


BOOM = "onionfl_untrusted_plugin"


@pytest.fixture
def boom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / f"{BOOM}.py").write_text(
        "raise RuntimeError('boom: imported')", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield f"{BOOM}:Plugin"
    sys.modules.pop(BOOM, None)


@pytest.mark.parametrize("trusted", ["", "mylab", "onionfl_untrusted"])
def test_an_untrusted_path_is_rejected_without_importing_it(
    trainers: Registry, boom: str, monkeypatch: pytest.MonkeyPatch, trusted: str
) -> None:
    monkeypatch.setenv(TRUSTED_ENV, trusted)
    with pytest.raises(PluginError, match=f"not trusted.*{TRUSTED_ENV}"):
        trainers.get(boom)

    assert BOOM not in sys.modules


def test_a_trusted_path_is_imported(
    trainers: Registry, boom: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TRUSTED_ENV, f"mylab, {BOOM}")
    with pytest.raises(RuntimeError, match="boom: imported"):
        trainers.get(boom)


def test_a_framework_path_resolves_only_a_registered_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TRUSTED_ENV, raising=False)
    assert codecs.get("onion_fl.core.codec:JsonCodec") is codecs.get("json")
    for other in ["onion_fl.core.codec:get_codec", "onion_fl.core.registry:Registry"]:
        with pytest.raises(PluginError, match="not a registered codec"):
            codecs.get(other)


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
