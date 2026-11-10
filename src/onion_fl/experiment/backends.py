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
from pathlib import Path
from typing import Any

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
        paused: bool = False,
    ) -> SimPlan:
        return SimPlan(scenario, evaluate, paused)

    def run(self, plan: SimPlan) -> None:
        from onion_fl.experiment.runner import simulate

        plan.path = simulate(plan.scenario, plan.evaluate, plan.paused)

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
