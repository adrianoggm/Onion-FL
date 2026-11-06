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

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from onion_fl.infrastructure.model import HOST_ID, InfrastructureError

SECRET = re.compile(r"-----BEGIN|PRIVATE KEY|ssh-(rsa|ed25519) AAAA")


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

    @field_validator("key")
    @classmethod
    def _key_is_a_path(cls, value: str | None) -> str | None:
        if value is not None and (SECRET.search(value) or "\n" in value):
            raise ValueError("give the path of the private key, never the key itself")
        return value

    @field_validator("env")
    @classmethod
    def _env_names(cls, value: dict[str, str]) -> dict[str, str]:
        for name in value:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"{name!r} is not an environment variable name")
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
        return expanded


class Clock(Strict):
    """How the clocks are checked before a run."""

    reference: Literal["launcher", "ntp"] = Field(
        "launcher",
        description="Contra qué se mide el desfase: el reloj del lanzador o NTP",
    )
    server: str = Field("pool.ntp.org", description="Servidor NTP")
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
