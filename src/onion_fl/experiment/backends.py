from __future__ import annotations

"""The ``sim`` and ``local`` backends (I1 §10).

Both follow the contract of ``onion_fl.infrastructure.backends`` and build the
same federation with ``build_federation``:

- ``sim``: the virtual clock in one process, with the resources and network
  the infrastructure models;
- ``local``: the wall clock, one process per group on this machine over MQTT,
  gRPC or Flower, with its own resources and no emulated network.
  ``runtime.mode: real`` is its alias.

An experiment picks one with ``backend``; without it, ``runtime.mode`` does.
"""

import sys
from dataclasses import dataclass
from ipaddress import IPv4Network
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from onion_fl.infrastructure.backends import backends

ALIASES = {"real": "local"}  # run.json of runs before the backends said "real"


def backend_name(ref: str | dict[str, Any]) -> str:
    return ref if isinstance(ref, str) else ref["name"]


@dataclass
class SimPlan:
    scenario: Any
    evaluate: Any = None
    paused: bool = False  # for the operators to resume (continual.control)
    path: Path | None = None
    faults: Any = None


class Restart(BaseModel):
    host: str = Field(description="Host cuyo proceso se mata (kill -9)")
    at: float = Field(gt=0, description="Segundo del reloj de la ejecución en que cae")
    down: float = Field(ge=0, description="Segundos caído antes de volver con --resume")


class FaultParams(BaseModel):
    """When a run on hosts starts and what fails in it, the same keys in
    ``sim`` and ``distributed`` so that one scenario runs on both (C9)."""

    start_in: float = Field(
        0.0,
        ge=0,
        description=(
            "Segundos entre el lanzamiento y el cero del reloj de la ejecución, "
            "para que un stream no empiece antes de que arranquen los nodos"
        ),
    )
    checkpoints: bool = Field(
        False,
        description=(
            "Cada edge guarda su estado (su parte del bundle) en su máquina "
            "tras cada evento, para reanudar si su proceso cae"
        ),
    )
    restarts: list[Restart] = Field(
        default_factory=list,
        description="Fallos reales: el proceso de un host cae y vuelve con su estado",
    )
    joins: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "host -> segundo del reloj de la ejecución en que arranca: sus edges "
            "se incorporan a mitad"
        ),
    )

    @model_validator(mode="after")
    def _faults(self) -> FaultParams:
        if self.restarts and not self.checkpoints:
            raise ValueError(
                "restarts: a host comes back with its state only if its edges "
                "save it; set checkpoints: true"
            )
        both = {r.host for r in self.restarts} & set(self.joins)
        if both:
            raise ValueError(f"joins: {sorted(both)} also restart; pick one")
        return self


class SimParams(FaultParams):
    pass


@backends.register(
    "sim",
    title="Simulación",
    description="Reloj virtual en un proceso; recursos y red modelados",
    explain=(
        "Ejecuta la federación entera en un proceso con un reloj virtual. "
        "Reproduce la infraestructura declarada: cómputo, contención, red "
        "física y fallos. Es el backend de los barridos rápidos. Con "
        "`start_in`, `restarts` y `joins` reproduce el arranque y los fallos "
        "de `distributed`: los nodos arrancan antes del cero del reloj, y el "
        "proceso de un host cae y vuelve con el estado guardado de sus edges, "
        "o arranca tarde."
    ),
    params=SimParams,
)
class SimBackend:
    def __init__(self, **params: Any) -> None:
        self.params = SimParams(**params)

    def supports(self, infrastructure: Any) -> list[str]:
        return []

    def prepare(
        self,
        scenario: Any,
        infrastructure: Any = None,
        assignment: Any = None,
        evaluate: Any = None,
        paused: bool = False,
    ) -> SimPlan:
        params = self.params
        faults = params if params.restarts or params.joins or params.start_in else None
        if faults is not None and infrastructure is None:
            raise ValueError(
                "backend sim: start_in, restarts and joins are for hosts; give "
                "the experiment an infrastructure"
            )
        return SimPlan(scenario, evaluate, paused, faults=faults)

    def run(self, plan: SimPlan) -> None:
        from onion_fl.experiment.runner import simulate

        plan.path = simulate(
            plan.scenario, plan.evaluate, plan.paused, faults=plan.faults
        )

    def collect(self, plan: SimPlan) -> Path:
        if plan.path is None:
            raise ValueError("sim: collect before run")
        return plan.path


@backends.register(
    "local",
    title="Local",
    description=(
        "Reloj de pared, un proceso por grupo en esta máquina, sobre MQTT, gRPC "
        "o Flower"
    ),
    explain=(
        "Arranca un proceso `onion_fl node` por coordinador o agregador, con "
        "sus edges, y los comunica por MQTT, por gRPC (sin broker) o por "
        "Flower (mensajes de Flower sobre gRPC). Usa los recursos de la "
        "máquina, sin límites ni red emulada: es la referencia de la fidelidad "
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
            "infrastructure: local runs on this machine as it is; an "
            "infrastructure runs on the emulated or distributed backend"
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


class DistributedParams(FaultParams):
    inventory: str | dict[str, Any] = Field(
        description=(
            "Inventario de máquinas: ruta de un YAML o el mapa en línea; "
            "sin secretos, solo rutas de claves"
        ),
    )
    broker: str | None = Field(
        None,
        pattern=r"^[^:\s]+:\d+$",
        description=(
            "host:puerto del broker MQTT visto desde las máquinas; por defecto, "
            "el host de services.mqtt_broker"
        ),
    )
    mqtt_port: int = Field(1883, gt=0, le=65535, description="Puerto del broker")
    push_data: bool = Field(
        True, description="Enviar la caché de datos a las máquinas que no la tienen"
    )
    ssh: str = Field("ssh", description="Cliente OpenSSH del lanzador")
    clock_samples: int = Field(
        5, ge=1, description="Muestras para medir el desfase de cada reloj"
    )
    probe_timeout: float = Field(
        60.0, gt=0, description="Plazo de la comprobación previa de cada máquina (s)"
    )
    grace: float = Field(
        60.0, ge=0, description="Margen de arranque de los procesos (s)"
    )


@backends.register(
    "distributed",
    title="Distribuido",
    description="Máquinas reales por SSH: un proceso por host, con relojes comprobados",
    explain=(
        "Ejecuta la federación en las máquinas reales de un inventario: un "
        "proceso `onion_fl node` por host, lanzado por SSH. Antes comprueba "
        "cada máquina (Python, el mismo código de onion_fl, los datos, el "
        "desfase del reloj frente al lanzador o NTP, el broker y los puertos "
        "gRPC de las demás), les envía la configuración y la caché de datos y, "
        "al acabar, recoge sus eventos y firma la ejecución como `local`. Es la "
        "referencia física."
    ),
    params=DistributedParams,
)
class DistributedBackend:
    def __init__(self, **params: Any) -> None:
        self.params = DistributedParams(**params)

    def supports(self, infrastructure: Any) -> list[str]:
        if infrastructure is None:
            return [
                "backend distributed: give the experiment an infrastructure; its "
                "hosts are the machines of the inventory"
            ]
        return []

    def inspect(self, resolved: Any) -> dict[str, Any]:
        """The machines ``onion_fl plan`` shows; contacts none of them."""
        from onion_fl.experiment.distributed import inspect

        return inspect(self.params, resolved)

    def prepare(
        self,
        scenario: Any,
        infrastructure: Any = None,
        assignment: Any = None,
        evaluate: Any = None,
    ) -> Any:
        from onion_fl.experiment.distributed import prepare_distributed

        if evaluate is not None:
            raise ValueError("a custom evaluate cannot be sent to the machines")
        return prepare_distributed(scenario, self.params)

    def run(self, plan: Any) -> None:
        from onion_fl.experiment.distributed import launch_distributed

        launch_distributed(plan)

    def collect(self, plan: Any) -> Path:
        from onion_fl.experiment.real import merge

        return merge(plan)
