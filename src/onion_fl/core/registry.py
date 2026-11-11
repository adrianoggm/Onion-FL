from __future__ import annotations

"""Named plugins with metadata, one registry per experimental axis (spec §4, §11).

A config picks a plugin by name, or by a ``package.module:Name`` path for
plugins that live outside the framework. Importing a module runs its code, and
the Studio validates configs that arrive over HTTP, so a path is imported only
when its module lies under a prefix the operator lists in
``ONIONFL_TRUSTED_MODULES`` (comma separated, empty by default), or is the
framework's own and names a plugin already registered. The metadata (title, description,
explanation and the pydantic model of the parameters) is what the front uses
to build forms and the tutorial without plugin-specific UI code.
"""

import importlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

F = TypeVar("F", bound=Callable[..., Any])

TRUSTED_ENV = "ONIONFL_TRUSTED_MODULES"
# The framework's own modules are already installed and loaded: always trusted.
BUILTIN = "onion_fl"


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
        validated = self.validate(params)
        if validated is None:
            return self.factory()
        return self.factory(**dict(validated))

    def validate(self, params: Mapping[str, Any] | None = None) -> BaseModel | None:
        """``params`` checked against the plugin's model, without building it."""
        given = dict(params or {})
        if self.params is None:
            if given:
                raise PluginError(
                    f"plugin {self.name!r} takes no parameters, got {sorted(given)}"
                )
            return None
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
        return validated

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
            spec = _import_spec(name)
            if is_trusted(name.partition(":")[0], builtin=False):
                return spec
            # A framework path names a plugin this registry already holds (an old
            # module name, say), never any other callable of the package.
            for known in self._specs.values():
                if spec is known or spec.factory is known.factory:
                    return known
            raise PluginError(
                f"{name!r} is not a registered {self.kind} plugin; framework "
                f"plugins are chosen by name ({self.names()})"
            )
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


def trusted_prefixes() -> list[str]:
    """The module prefixes the operator trusts, from ``ONIONFL_TRUSTED_MODULES``."""
    return [p.strip() for p in os.environ.get(TRUSTED_ENV, "").split(",") if p.strip()]


def is_trusted(module_name: str, *, builtin: bool = True) -> bool:
    """Whether a module may be imported: under a trusted prefix, or the framework's
    own when ``builtin`` (its paths must then name a registered plugin)."""
    prefixes = [*trusted_prefixes(), *([BUILTIN] if builtin else [])]
    return any(module_name == p or module_name.startswith(p + ".") for p in prefixes)


def _import_spec(path: str) -> PluginSpec:
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise PluginError(
            f"plugin path must look like 'package.module:Name', got {path!r}"
        )
    if not is_trusted(module_name):
        raise PluginError(
            f"external plugin {path!r} is not trusted: the operator must list its "
            f"package in {TRUSTED_ENV} (comma separated) before it can be imported"
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
