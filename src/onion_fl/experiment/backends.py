from __future__ import annotations

"""The ``sim`` and ``local`` backends (I1 §10).

Both follow the contract of ``onion_fl.infrastructure.backends`` and build the
same federation with ``build_federation``:

- ``sim``: the virtual clock in one process, with the resources and network
  the infrastructure models;
- ``local``: the wall clock, one process per group on this machine over MQTT,
  with its own resources and no emulated network. ``runtime.mode: real`` is
  its alias.

An experiment picks one with ``backend``; without it, ``runtime.mode`` does.
"""

import sys
from dataclasses import dataclass
from ipaddress import IPv4Network
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from onion_fl.infrastructure.backends import backends

ALIASES = {"real": "local"}  # run.json of runs before the backends said "real"


def backend_name(ref: str | dict[str, Any]) -> str:
    return ref if isinstance(ref, str) else ref["name"]


@dataclass
class SimPlan:
    scenario: Any
    evaluate: Any = None
    path: Path | None = None


@backends.register(
    "sim",
    title="Simulación",
    description="Reloj virtual en un proceso; recursos y red modelados",
    explain=(
        "Ejecuta la federación entera en un proceso con un reloj virtual. "
        "Reproduce la infraestructura declarada: cómputo, contención, red "
        "física y fallos. Es el backend de los barridos rápidos."
    ),
)
class SimBackend:
    def supports(self, infrastructure: Any) -> list[str]:
        return []

    def prepare(
        self,
        scenario: Any,
        infrastructure: Any = None,
        assignment: Any = None,
        evaluate: Any = None,
    ) -> SimPlan:
        return SimPlan(scenario, evaluate)

    def run(self, plan: SimPlan) -> None:
        from onion_fl.experiment.runner import simulate

        plan.path = simulate(plan.scenario, plan.evaluate)

    def collect(self, plan: SimPlan) -> Path:
        if plan.path is None:
            raise ValueError("sim: collect before run")
        return plan.path


@backends.register(
    "local",
    title="Local",
    description="Reloj de pared, un proceso por grupo en esta máquina, sobre MQTT",
    explain=(
        "Arranca un proceso `onion_fl node` por coordinador o agregador, con "
        "sus edges, y los comunica por MQTT. Usa los recursos de la máquina, "
        "sin límites ni red emulada: es la referencia de la fidelidad "
        "funcional. `runtime.mode: real` es su alias."
    ),
)
class LocalBackend:
    def __init__(self, python: str = sys.executable) -> None:
        self.python = python

    def supports(self, infrastructure: Any) -> list[str]:
        if infrastructure is None:
            return []
        return [
            "infrastructure: real runs take the machines they run on; an "
            "infrastructure is simulated only for now (I10, I14)"
        ]

    def prepare(
        self,
        scenario: Any,
        infrastructure: Any = None,
        assignment: Any = None,
        evaluate: Any = None,
    ) -> Any:
        from onion_fl.experiment.real import prepare_local

        if evaluate is not None:
            raise ValueError("a custom evaluate cannot be sent to the node processes")
        return prepare_local(scenario, python=self.python)

    def run(self, plan: Any) -> None:
        from onion_fl.experiment.real import launch

        launch(plan)

    def collect(self, plan: Any) -> Path:
        from onion_fl.experiment.real import merge

        return merge(plan)


class PinSpec(BaseModel):
    """Fixed CPUs for the containers (``cpuset``), the same in every session or run.

    ``cpus`` is either a list that the hosts take in name order, each as many
    as its profile's ``cpus``, or a mapping from host to its CPUs, so that a
    run on fewer hosts keeps each host's CPUs (those of a calibration's
    session); ``others`` is the cpuset of the routers and the broker. The CPUs
    are those Docker's VM shows, not physical cores: on Docker Desktop the
    hypervisor still moves them over the machine's cores.
    """

    model_config = ConfigDict(extra="forbid")

    cpus: list[int] | dict[str, list[int]] = Field(
        min_length=1, description="CPUs de los hosts, por orden o por host"
    )
    others: str = Field(description="cpuset de los routers y del broker, p. ej. 20-23")


class EmulatedParams(BaseModel):
    image: str = Field(
        "onionfl-emulated:latest",
        description="Imagen de los hosts; se construye si no existe",
    )
    broker_image: str = Field("eclipse-mosquitto:2", description="Imagen del broker")
    project: str = Field(
        "onionfl",
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        description="Prefijo del proyecto Compose, sus contenedores y sus redes",
    )
    subnet: str = Field(
        "10.195.0.0/16", description="Red de la que sale una /24 por LAN y por enlace"
    )
    max_containers: int = Field(
        64, gt=0, description="Tope de contenedores de una ejecución"
    )
    ready_timeout: float = Field(
        120.0, gt=0, description="Espera máxima a que los contenedores estén listos (s)"
    )
    pin: PinSpec | None = Field(
        None,
        description="CPUs fijas de cada contenedor, como en la sesión de calibración",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )

    @field_validator("subnet")
    @classmethod
    def _subnet(cls, value: str) -> str:
        net = IPv4Network(value)
        if net.prefixlen > 23:
            raise ValueError(f"{value}: give a /23 or wider; each network takes a /24")
        return value


@backends.register(
    "emulated",
    title="Emulado",
    description="Un contenedor por host con su CPU y memoria, y red emulada con netem",
    explain=(
        "Genera un proyecto de Docker Compose desde la infraestructura: un "
        "contenedor por host con los límites de CPU y memoria de su perfil, un "
        "router por sitio con netem en cada enlace físico (latencia, ancho de "
        "banda y pérdidas) y el broker MQTT en su host. Dentro arranca un "
        "proceso `onion_fl node` por host, como `local`. Necesita Docker; mide "
        "la fidelidad de rendimiento sin hardware."
    ),
    params=EmulatedParams,
)
class EmulatedBackend:
    def __init__(self, **params: Any) -> None:
        self.params = EmulatedParams(**params)

    def supports(self, infrastructure: Any) -> list[str]:
        if infrastructure is None:
            return [
                "backend emulated: give the experiment an infrastructure; the one "
                "implicit in the topology is not emulated"
            ]
        return []

    def inspect(self, resolved: Any) -> dict[str, Any]:
        """The containers ``onion_fl plan`` shows; refuses what cannot be emulated."""
        from onion_fl.experiment.emulated import inspect

        return inspect(self.params, resolved)

    def prepare(
        self,
        scenario: Any,
        infrastructure: Any = None,
        assignment: Any = None,
        evaluate: Any = None,
    ) -> Any:
        from onion_fl.experiment.emulated import prepare_emulated

        if evaluate is not None:
            raise ValueError("a custom evaluate cannot be sent to the containers")
        return prepare_emulated(scenario, self.params)

    def run(self, plan: Any) -> None:
        from onion_fl.experiment.emulated import launch_emulated

        launch_emulated(plan)

    def collect(self, plan: Any) -> Path:
        from onion_fl.experiment.real import merge

        return merge(plan)
