from __future__ import annotations

"""The physical infrastructure: profiles, sites, hosts, network, services (I1 §5, §7, §8).

::

    name: edge_fog_cloud_lab
    profiles:
      cloud_vm:     {cpus: 8, memory: 16GB, train_rate: 20000, aggregate_rate: 5.0e8}
      raspberry_pi: {cpus: 4, memory: 4GB, train_rate: 900}
    sites:
      dc:    {hosts: {cloud-1: cloud_vm}}
      field: {pool: {raspberry_pi: 20}}    # hosts field-raspberry_pi-1 .. -20
    network:
      links:
        - {between: [field, dc], latency: 70ms, bandwidth: 10Mbps, loss: 0.01}
    services: {mqtt_broker: cloud-1}

A link's ``up`` direction goes from ``between[0]`` to ``between[1]``; inside a
site, hosts see each other through the site's ``lan``.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

HOST_ID = r"^[A-Za-z0-9_.-]+$"

_UNITS = {
    "duration": ({"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}, "10ms, 2s"),
    "bandwidth": (
        {"bps": 1.0, "kbps": 1e3, "mbps": 1e6, "gbps": 1e9},
        "5kbps, 100Mbps, 1Gbps",
    ),
    "memory": (
        {"b": 1.0, "kb": 2.0**10, "mb": 2.0**20, "gb": 2.0**30},
        "512MB, 16GB",
    ),
}


class InfrastructureError(ValueError):
    """An infrastructure is malformed. The message says where."""


def quantity(value: Any, kind: str) -> Any:
    """A number, or a string with a unit (``10ms``, ``100Mbps``, ``16GB``)."""
    if not isinstance(value, str):
        return value
    units, examples = _UNITS[kind]
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([A-Za-z]+)\s*", value)
    if match is None or match.group(2).lower() not in units:
        raise ValueError(f"{value!r}: use a number or a {kind} such as {examples}")
    return float(match.group(1)) * units[match.group(2).lower()]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Profile(Strict):
    """What a host can do. Rates left out cost nothing, as without infrastructure."""

    train_rate: float | None = Field(
        None, gt=0, description="Muestras entrenadas por segundo"
    )
    aggregate_rate: float | None = Field(
        None, gt=0, description="Bytes de actualización combinados por segundo"
    )
    cpus: int | None = Field(None, gt=0, description="Trabajos a la vez")
    memory: float | None = Field(None, gt=0, description="Bytes de memoria")

    _memory = field_validator("memory", mode="before")(lambda v: quantity(v, "memory"))


class LinkFields(Strict):
    """Behaviour of a network segment: what ``LinkProfile`` takes, with units."""

    latency: float = Field(0.0, ge=0, description="Latencia media de un sentido (s)")
    jitter: float = Field(0.0, ge=0, description="Desviación típica de la latencia")
    distribution: Literal["fixed", "normal", "lognormal"] = "fixed"
    bandwidth_up: float | None = Field(None, gt=0, description="bps; None: sin límite")
    bandwidth_down: float | None = Field(
        None, gt=0, description="bps; None: sin límite"
    )
    loss: float = Field(
        0.0, ge=0, le=1, description="Probabilidad de perder un mensaje"
    )

    @model_validator(mode="before")
    @classmethod
    def _bandwidth(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "bandwidth" in data:
            data = dict(data)
            both = data.pop("bandwidth")
            data.setdefault("bandwidth_up", both)
            data.setdefault("bandwidth_down", both)
        return data

    _durations = field_validator("latency", "jitter", mode="before")(
        lambda v: quantity(v, "duration")
    )
    _bandwidths = field_validator("bandwidth_up", "bandwidth_down", mode="before")(
        lambda v: quantity(v, "bandwidth")
    )


# ponytail: the same numbers as the runtime's "lan" preset, which this package
# cannot import (it depends on core only).
LAN = LinkFields(latency=0.001, jitter=0.0002, distribution="normal", bandwidth=1e9)


class Site(Strict):
    """Hosts in one place: named ones, and a pool of identical devices."""

    hosts: dict[str, str] = Field(default_factory=dict, description="host -> perfil")
    pool: dict[str, int] = Field(
        default_factory=dict, description="perfil -> cuántos dispositivos"
    )
    lan: LinkFields = Field(default=LAN, description="Red local entre sus hosts")

    @field_validator("hosts")
    @classmethod
    def _host_ids(cls, value: dict[str, str]) -> dict[str, str]:
        for host in value:
            if not re.fullmatch(HOST_ID, host):
                raise ValueError(f"host {host!r}: use letters, digits, '_', '.', '-'")
        return value

    @field_validator("pool")
    @classmethod
    def _counts(cls, value: dict[str, int]) -> dict[str, int]:
        for profile, count in value.items():
            if count < 1:
                raise ValueError(f"pool {profile!r}: at least one device, got {count}")
        return value


class Link(LinkFields):
    """A physical link between two sites."""

    between: tuple[str, str]


class Change(Strict):
    """A change of a physical link at a time of the experiment (I1 §8).

    ``set`` overrides some of its fields (``{latency: 50ms, loss: 0.1}``);
    ``down: true`` cuts it and ``down: false`` brings it back; ``trace`` is a
    CSV of measurements (a ``t`` column plus any of ``latency``, ``jitter``,
    ``bandwidth``, ``bandwidth_up``, ``bandwidth_down``, ``loss``), one change
    per row.
    """

    between: tuple[str, str]
    at: float = Field(0.0, ge=0, description="Instante del cambio (s)")
    set: dict[str, Any] | None = None
    down: bool | None = None
    trace: str | None = None

    _at = field_validator("at", mode="before")(lambda v: quantity(v, "duration"))

    @field_validator("set")
    @classmethod
    def _fields(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return None if value is None else patch(value)

    @model_validator(mode="after")
    def _one_kind(self) -> Change:
        given = [k for k in ("set", "down", "trace") if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError(
                f"a change takes exactly one of set, down or trace, got {given}"
            )
        return self


def patch(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Link fields with their units converted, only those given."""
    parsed = LinkFields.model_validate(dict(fields))
    return parsed.model_dump(include=parsed.model_fields_set)


class Network(Strict):
    links: list[Link] = Field(default_factory=list)
    bandwidth: Literal["shared", "dedicated"] = Field(
        "shared",
        description="shared: los enlaces lógicos que cruzan un enlace físico "
        "comparten su ancho de banda; dedicated: cada uno lo tiene entero",
    )
    schedule: list[Change] = Field(
        default_factory=list, description="Cambios de la red con el tiempo"
    )


class Infrastructure(Strict):
    """A validated infrastructure: every host has a known profile, every link known sites."""

    name: str = ""
    profiles: dict[str, Profile] = Field(min_length=1)
    sites: dict[str, Site] = Field(min_length=1)
    network: Network = Field(default_factory=Network)
    services: dict[str, str] = Field(
        default_factory=dict, description="servicio -> host (p. ej. mqtt_broker)"
    )

    @model_validator(mode="after")
    def _check(self) -> Infrastructure:
        seen: dict[str, str] = {}
        for site, spec in self.sites.items():
            used = list(spec.hosts.values()) + list(spec.pool)
            for profile in used:
                if profile not in self.profiles:
                    raise ValueError(
                        f"sites.{site}: unknown profile {profile!r}; "
                        f"profiles are {sorted(self.profiles)}"
                    )
            for host in self._site_hosts(site, spec):
                if host in seen:
                    raise ValueError(
                        f"host {host!r} is in sites {seen[host]!r} and {site!r}"
                    )
                seen[host] = site
        pairs = set()
        for i, link in enumerate(self.network.links):
            a, b = link.between
            for site in (a, b):
                if site not in self.sites:
                    raise ValueError(
                        f"network.links[{i}]: unknown site {site!r}; "
                        f"sites are {sorted(self.sites)}"
                    )
            if a == b:
                raise ValueError(f"network.links[{i}]: a link joins two sites")
            if frozenset((a, b)) in pairs:
                raise ValueError(f"network.links[{i}]: {a} and {b} are linked twice")
            pairs.add(frozenset((a, b)))
        for i, change in enumerate(self.network.schedule):
            if frozenset(change.between) not in pairs:
                raise ValueError(
                    f"network.schedule[{i}]: no link between {list(change.between)}"
                )
        for service, host in self.services.items():
            if host not in seen:
                raise ValueError(f"services.{service}: unknown host {host!r}")
        return self

    @staticmethod
    def _site_hosts(site: str, spec: Site) -> dict[str, str]:
        hosts = dict(spec.hosts)
        for profile, count in spec.pool.items():
            for i in range(1, count + 1):
                hosts[f"{site}-{profile}-{i}"] = profile
        return hosts

    def hosts(self) -> dict[str, tuple[str, str]]:
        """Every host, pools expanded: host -> (site, profile)."""
        return {
            host: (site, profile)
            for site, spec in self.sites.items()
            for host, profile in self._site_hosts(site, spec).items()
        }

    def profile_of(self, host: str) -> Profile:
        try:
            return self.profiles[self.hosts()[host][1]]
        except KeyError:
            raise InfrastructureError(f"unknown host {host!r}") from None

    @property
    def infrastructure_id(self) -> str:
        """SHA-256 of the canonical infrastructure; the name does not count."""
        canonical = self.model_dump(mode="json", exclude={"name"})
        canonical["network"]["links"] = sorted(
            canonical["network"]["links"], key=lambda link: sorted(link["between"])
        )
        text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_infrastructure(raw: Mapping[str, Any]) -> Infrastructure:
    if not isinstance(raw, Mapping):
        raise InfrastructureError("an infrastructure must be a mapping")
    try:
        return Infrastructure.model_validate(dict(raw))
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'infrastructure'}: "
            f"{err['msg'].removeprefix('Value error, ')}"
            for err in exc.errors()
        )
        raise InfrastructureError(details) from exc


def load_infrastructure(path: str | Path) -> Infrastructure:
    import yaml

    path = Path(path)
    if not path.exists():
        raise InfrastructureError(f"infrastructure file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise InfrastructureError(f"{path.name} is not valid YAML: {exc}") from None
    if isinstance(raw, Mapping):  # traces are read next to the file
        for change in (raw.get("network") or {}).get("schedule") or []:
            if isinstance(change, dict) and change.get("trace"):
                change["trace"] = str(path.parent / change["trace"])
    return parse_infrastructure(raw)
