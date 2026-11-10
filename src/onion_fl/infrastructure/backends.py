from __future__ import annotations

"""The backend contract and its registry (I1 §10).

A backend runs one scenario of an experiment on some infrastructure and
returns the signed run. It gives clock, transport, resources and failures,
never its own roles, trainers or aggregators: every backend builds the
federation with ``build_federation``. The contract has four steps:

- ``supports(infrastructure)``: the reasons it cannot reproduce what the
  infrastructure asks for, empty when it can. ``None`` is the infrastructure
  implicit in the topology. The config refuses a backend with any reason.
- ``prepare(scenario, infrastructure, assignment)``: an execution plan (the
  run folder, what to start and where), checked before anything runs.
- ``run(plan)``: execute it.
- ``collect(plan)``: the folder of the signed run, with its events merged.

This module only holds the contract and the ``backends`` registry: the
backends themselves, which build federations, live with the experiment
(``onion_fl.experiment.backends``), so the core keeps out of the FL engine.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from onion_fl.core.registry import Registry

backends = Registry("backend")


@runtime_checkable
class Backend(Protocol):
    def supports(self, infrastructure: Any) -> Sequence[str]: ...

    def prepare(
        self, scenario: Any, infrastructure: Any = None, assignment: Any = None
    ) -> Any: ...

    def run(self, plan: Any) -> None: ...

    def collect(self, plan: Any) -> Path: ...


def execute(
    backend: Backend,
    scenario: Any,
    infrastructure: Any = None,
    assignment: Any = None,
    **options: Any,
) -> Path:
    """``prepare``, ``run`` and ``collect`` one scenario; the signed run folder.

    ``options`` go to ``prepare`` (hooks such as a custom scorer, which only an
    in-process backend can take).
    """
    refused = list(backend.supports(infrastructure))
    if refused:
        raise ValueError("; ".join(refused))
    plan = backend.prepare(scenario, infrastructure, assignment, **options)
    backend.run(plan)
    return backend.collect(plan)
