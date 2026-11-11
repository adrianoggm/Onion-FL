from __future__ import annotations

"""The inventory of real machines for the ``distributed`` backend (I1 §10, I14).

An infrastructure names hosts; an inventory says which real machine plays each
one and how the launcher reaches it over SSH::

    # inventories/lab.yaml
    machines:
      cloud-1: {address: 192.168.1.10, user: onion, python: .venv/bin/python}
      fog-1:   {address: 192.168.1.11, user: onion, key: ~/.ssh/id_onion}
      field-pi-1:
        address: 192.168.1.20
        user: pi
        key: ${ONIONFL_PI_KEY}        # a path, from the environment
        root: onion-fl                # relative to the user's home
    clock: {reference: ntp, server: pool.ntp.org, max_offset: 0.05}

The inventory holds no secrets: ``key`` is the path of a private key (``~``
and ``$VARS`` are expanded when the launcher connects), never the key itself,
and there is no field for a password. Hosts that the assignment does not use
may be left out; a host that runs nodes and has no machine is refused.
"""

import ipaddress
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from onion_fl.infrastructure.model import HOST_ID, InfrastructureError

SECRET = re.compile(r"-----BEGIN|PRIVATE KEY|ssh-(rsa|ed25519) AAAA")
# Every value below reaches a command line (ssh's destination or options, the
# remote shell line, a probe's request): nothing may read as an option or
# break a line. Labels may hold "_" for container names on a Docker network.
LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9])?"
HOSTNAME = re.compile(rf"{LABEL}(?:\.{LABEL})*")
USER = re.compile(r"[A-Za-z_][A-Za-z0-9._-]{0,31}")
CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def check_address(value: str) -> str:
    """A hostname, an IPv4 address or a bracketed IPv6 one; nothing else."""
    if value.startswith("[") and value.endswith("]"):
        try:
            ipaddress.IPv6Address(value[1:-1])
        except ValueError:
            raise ValueError(f"{value!r} is not a bracketed IPv6 address") from None
        return value
    try:
        ipaddress.IPv4Address(value)
        return value
    except ValueError:
        pass
    if len(value) > 253 or not HOSTNAME.fullmatch(value):
        raise ValueError(
            f"{value!r} is not a hostname, an IPv4 address or a bracketed IPv6 one"
        )
    return value


def check_text(value: str, what: str, *, option_like: bool = True) -> str:
    """No control characters, and (``option_like``) no leading ``-``."""
    if CONTROL.search(value):
        raise ValueError(f"{what}: control characters are not allowed")
    if option_like and value.startswith("-"):
        raise ValueError(f"{what} {value!r} may not start with '-'")
    return value


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Machine(Strict):
    """A real machine and how to reach it."""

    address: str = Field(min_length=1, description="Dirección SSH desde el lanzador")
    port: int = Field(22, gt=0, le=65535, description="Puerto SSH")
    user: str | None = Field(None, description="Usuario SSH; por defecto, el de ssh")
    key: str | None = Field(
        None, description="Ruta de la clave privada (admite ~ y $VARIABLES)"
    )
    python: str = Field("python3", description="Intérprete con onion_fl instalado")
    root: str = Field(
        "onionfl", min_length=1, description="Carpeta de trabajo, relativa al home"
    )
    advertise: str | None = Field(
        None, description="Dirección con la que la ven las demás máquinas"
    )
    grpc_port: int = Field(50051, gt=0, le=65535, description="Puerto del gRPC")
    listen: str = Field("0.0.0.0", description="Interfaz en la que escucha su gRPC")
    env: dict[str, str] = Field(
        default_factory=dict, description="Variables de entorno de sus procesos"
    )

    @field_validator("address", "advertise")
    @classmethod
    def _address(cls, value: str | None) -> str | None:
        return None if value is None else check_address(value)

    @field_validator("listen")
    @classmethod
    def _listen(cls, value: str) -> str:
        return check_address(value)

    @field_validator("user")
    @classmethod
    def _user(cls, value: str | None) -> str | None:
        if value is not None and not USER.fullmatch(value):
            raise ValueError(f"{value!r} is not a POSIX user name")
        return value

    @field_validator("key")
    @classmethod
    def _key_is_a_path(cls, value: str | None) -> str | None:
        if value is not None and (SECRET.search(value) or "\n" in value):
            raise ValueError("give the path of the private key, never the key itself")
        return None if value is None else check_text(value, "key")

    @field_validator("python", "root")
    @classmethod
    def _command_parts(cls, value: str, info: Any) -> str:
        return check_text(value, info.field_name)

    @field_validator("env")
    @classmethod
    def _env_names(cls, value: dict[str, str]) -> dict[str, str]:
        for name, text in value.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"{name!r} is not an environment variable name")
            check_text(text, f"env.{name}", option_like=False)
        return value

    @property
    def peer_address(self) -> str:
        """The address the other machines use: ``advertise``, or ``address``."""
        return self.advertise or self.address

    def key_path(self) -> str | None:
        """``key`` with ``~`` and ``$VARS`` expanded; refused when a var is unset."""
        if self.key is None:
            return None
        expanded = os.path.expanduser(os.path.expandvars(self.key))
        if "$" in expanded:
            raise InfrastructureError(
                f"key {self.key!r}: an environment variable in it is not set"
            )
        try:  # what the variables brought in is checked too
            return check_text(expanded, "key")
        except ValueError as exc:
            raise InfrastructureError(str(exc)) from None


class Clock(Strict):
    """How the clocks are checked before a run."""

    reference: Literal["launcher", "ntp"] = Field(
        "launcher",
        description="Contra qué se mide el desfase: el reloj del lanzador o NTP",
    )
    server: str = Field("pool.ntp.org", description="Servidor NTP")

    @field_validator("server")
    @classmethod
    def _server(cls, value: str) -> str:
        host, colon, port = value.rpartition(":")
        if colon and port.isdigit() and not value.endswith("]"):
            check_address(host)
        else:
            check_address(value)
        return value

    max_offset: float = Field(
        0.05, gt=0, description="Desfase máximo admitido de cada reloj (s)"
    )


class Inventory(Strict):
    machines: dict[str, Machine] = Field(min_length=1)
    clock: Clock = Field(default_factory=Clock)
    known_hosts: str | None = Field(
        None, description="Fichero known_hosts de SSH; por defecto, el del usuario"
    )
    host_key_check: Literal["yes", "accept-new", "no"] = Field(
        "accept-new", description="StrictHostKeyChecking de SSH"
    )
    connect_timeout: float = Field(10.0, gt=0, description="Plazo de conexión (s)")

    @field_validator("known_hosts")
    @classmethod
    def _known_hosts(cls, value: str | None) -> str | None:
        if value is not None and '"' in value:
            raise ValueError("known_hosts may not hold '\"'")
        return None if value is None else check_text(value, "known_hosts")

    @field_validator("machines")
    @classmethod
    def _host_ids(cls, value: dict[str, Machine]) -> dict[str, Machine]:
        for host in value:
            if not re.fullmatch(HOST_ID, host):
                raise ValueError(f"{host!r} is not a host id ({HOST_ID})")
        return value

    def machine(self, host: str) -> Machine:
        try:
            return self.machines[host]
        except KeyError:
            raise InfrastructureError(
                f"inventory: no machine for host {host!r}; it has "
                f"{sorted(self.machines)}"
            ) from None


def parse_inventory(raw: Mapping[str, Any]) -> Inventory:
    if not isinstance(raw, Mapping):
        raise InfrastructureError("an inventory must be a mapping")
    try:
        return Inventory.model_validate(dict(raw))
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'inventory'}: "
            f"{err['msg'].removeprefix('Value error, ')}"
            for err in exc.errors()
        )
        raise InfrastructureError(f"inventory: {details}") from exc


def load_inventory(ref: str | Path | Mapping[str, Any]) -> Inventory:
    """An inventory from a YAML file, or from a mapping given inline."""
    if isinstance(ref, Mapping):
        return parse_inventory(ref)
    path = Path(os.path.expanduser(os.path.expandvars(str(ref))))
    if not path.is_file():
        raise InfrastructureError(f"inventory file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise InfrastructureError(f"{path.name} is not valid YAML: {exc}") from None
    return parse_inventory(raw)
