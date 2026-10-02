from __future__ import annotations

"""Named plugins with metadata, one registry per experimental axis (spec §4, §11).

A config picks a plugin by name, or by a ``package.module:Name`` path for
plugins that live outside the framework. The metadata (title, description,
explanation and the pydantic model of the parameters) is what the front uses
to build forms and the tutorial without plugin-specific UI code.
"""

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

F = TypeVar("F", bound=Callable[..., Any])


class PluginError(ValueError):
    """A plugin cannot be found, registered or built."""


@dataclass(frozen=True)
class PluginSpec:
    """A plugin: its factory plus the metadata shown to the researcher."""

    name: str
    factory: Callable[..., Any]
    title: str
    description: str
    params: type[BaseModel] | None = None
    explain: str = ""

    def build(self, params: Mapping[str, Any] | None = None) -> Any:
        """Validate ``params`` and call the factory with them as keyword arguments."""
        given = dict(params or {})
        if self.params is None:
            if given:
                raise PluginError(
                    f"plugin {self.name!r} takes no parameters, got {sorted(given)}"
                )
            return self.factory()
        expected = set(self.params.model_fields)
        unknown = sorted(set(given) - expected)
        if unknown:
            raise PluginError(
                f"plugin {self.name!r}: unknown parameters {unknown}; "
                f"expected {sorted(expected)}"
            )
        try:
            validated = self.params.model_validate(given)
        except ValidationError as exc:
            raise PluginError(
                f"plugin {self.name!r}: invalid parameters: {exc}"
            ) from exc
        return self.factory(**dict(validated))

    def describe(self, kind: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "explain": self.explain,
            "params": None if self.params is None else self.params.model_json_schema(),
        }


class Registry:
    """Plugins of one kind (codec, trainer, placement, ...), looked up by name."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._specs: dict[str, PluginSpec] = {}

    def register(
        self,
        name: str,
        *,
        title: str,
        description: str,
        params: type[BaseModel] | None = None,
        explain: str = "",
    ) -> Callable[[F], F]:
        """Decorator that registers a factory (usually a class) under ``name``."""

        def decorator(factory: F) -> F:
            self.add(PluginSpec(name, factory, title, description, params, explain))
            return factory

        return decorator

    def add(self, spec: PluginSpec) -> None:
        if spec.name in self._specs:
            raise PluginError(f"{self.kind} plugin {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> PluginSpec:
        """Find a plugin by registered name or by ``package.module:Name``."""
        if ":" in name:
            return _import_spec(name)
        try:
            return self._specs[name]
        except KeyError:
            raise PluginError(
                f"unknown {self.kind} plugin {name!r}; available: {self.names()}"
            ) from None

    def create(self, name: str, params: Mapping[str, Any] | None = None) -> Any:
        return self.get(name).build(params)

    def describe(self) -> list[dict[str, Any]]:
        """Catalogue of every registered plugin, with the JSON Schema of its parameters."""
        return [self._specs[name].describe(self.kind) for name in self.names()]


def _import_spec(path: str) -> PluginSpec:
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise PluginError(
            f"plugin path must look like 'package.module:Name', got {path!r}"
        )
    try:
        obj = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise PluginError(f"cannot import plugin {path!r}: {exc}") from exc
    if isinstance(obj, PluginSpec):
        return obj
    if not callable(obj):
        raise PluginError(f"plugin {path!r} is neither a PluginSpec nor callable")
    return PluginSpec(
        name=path, factory=obj, title=attr, description=(obj.__doc__ or "").strip()
    )
