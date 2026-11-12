from __future__ import annotations

"""The ``emulated`` backend: the real launcher in containers (I1 §10, §11).

``prepare_emulated`` resolves the scenario's infrastructure and assignment,
creates the run and writes, in ``runs/<run_id>/emulation/``, the Compose
project that ``onion_fl.infrastructure.emulators`` generates: one container
per host that runs logical nodes (with its CPU and memory limits), a router
per site, netem on every link and the MQTT broker on its host. It also writes
``scenario.json`` with the paths as the containers see them, the logical
nodes of each host (``groups``) and the broker's address.

``launch_emulated`` brings the project up, waits until every container has
shaped its interfaces and the broker answers, and then starts one ``onion_fl
node`` per host with ``docker exec``, as the ``local`` backend starts one
process per group; the run is merged and signed by the same ``merge``. The
project is always taken down, with its logs kept in ``compose.log``. The
infrastructure's link outages (``network.schedule``, #289) are applied on the
routers at their instants of the run's clock, each recorded as the
simulation records it: ``link.down`` or ``link.up`` of the node ``network``.

Inside the containers the roles, trainers and aggregators are those of every
backend (``build_scenario``); only the clock, the transport's path and the
resources differ.
"""

import json
import os
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from onion_fl.experiment.backends import EmulatedParams
from onion_fl.experiment.config import ConfigError
from onion_fl.experiment.sweep import Scenario
from onion_fl.infrastructure.assignment import Resolved
from onion_fl.infrastructure.emulators import (
    MQTT_PORT,
    READY,
    Emulation,
    Layout,
    compose_project,
    layout,
    outages,
    refusals,
    slug,
)
from onion_fl.infrastructure.model import InfrastructureError
from onion_fl.observability.events import JsonlSink
from onion_fl.observability.run import Run, enrich
from onion_fl.runtime.real import TRACE

ROOT = "/onionfl"  # where the containers see the source and the run's paths
DOCKERFILE = "emulated.Dockerfile"  # in onion_fl.infrastructure
NUMERIC = ("torch", "numpy", "scipy", "pandas", "scikit-learn")  # pinned in the image
LABEL = "org.onion-fl.pins"  # the image label that records them


@dataclass
class EmulatedPlan:
    """What the ``emulated`` backend starts: a Compose project and one node per host."""

    run: Run
    scenario: Scenario
    groups: list[str]  # the hosts that run logical nodes
    emulation: Emulation
    folder: Path  # runs/<run_id>/emulation
    params: EmulatedParams
    codes: list[int] = field(default_factory=list)
    grace: float = 30.0  # seconds a process may take to start, on top of the timeout

    def command(self, group: str) -> list[str]:
        """The node process of host ``group``, inside its container.

        It sees the launcher's ``ONIONFL_TRACE``, which ``docker exec`` would
        not pass on by itself.
        """
        trace = os.environ.get(TRACE)
        return [
            "docker",
            "exec",
            *([] if trace is None else ["-e", f"{TRACE}={trace}"]),
            self.emulation.container(group),
            "python",
            "-m",
            "onion_fl",
            "node",
            "--id",
            group,
            "--run",
            self.run.run_id,
            "--config",
            f"{ROOT}/paths/runs/{self.run.run_id}/scenario.json",
        ]

    @property
    def compose(self) -> list[str]:
        return [
            "docker",
            "compose",
            "-p",
            self.emulation.project,
            "-f",
            str(self.folder / "compose.yaml"),
        ]


def check(params: EmulatedParams, resolved: Resolved) -> Layout:
    """The layout of an emulation, or a ``ConfigError`` with what it cannot do."""
    refused = refusals(resolved.infrastructure)
    broker = resolved.services.get("mqtt_broker")
    if broker is None:
        refused.append("services: name the host of the mqtt_broker")
    if refused:
        raise ConfigError("backend emulated: " + "; ".join(refused))
    try:
        lay = layout(
            resolved.infrastructure,
            sorted(set(resolved.hosts.values())),
            broker,
            params.subnet,
        )
    except InfrastructureError as exc:
        raise ConfigError(f"backend emulated: {exc}") from None
    if lay.containers > params.max_containers:
        raise ConfigError(
            f"backend emulated: {lay.containers} containers ({len(lay.hosts)} hosts, "
            f"{len(lay.routers)} routers and the broker) exceed max_containers "
            f"{params.max_containers}; colocate edges, or raise it"
        )
    return lay


def inspect(params: EmulatedParams, resolved: Resolved) -> dict[str, Any]:
    """What ``onion_fl plan`` shows of an emulation; refuses it as ``check`` does."""
    lay = check(params, resolved)
    return {
        "name": "emulated",
        **lay.summary(),
        "max_containers": params.max_containers,
    }


def host_load(
    resolved: Resolved, groups: Mapping[str, list[str]]
) -> dict[str, list[str]]:
    """The logical nodes of each host; an evaluator runs on its parent's host.

    ``groups`` is ``real.edge_groups``: the edges and evaluators under each
    leaf, and the root's evaluators.
    """
    load: dict[str, list[str]] = {}
    for node, host in resolved.hosts.items():
        load.setdefault(host, []).append(node)
    for parent, members in groups.items():
        for node in members:
            if node not in resolved.hosts:
                load[resolved.hosts[parent]].append(node)
    return {host: sorted(nodes) for host, nodes in sorted(load.items())}


def container_config(config: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The mounts of a run and its config with the paths the containers see.

    Every folder of ``paths`` that exists, the folder of each dataset
    descriptor and of a topology or infrastructure given as a file, and the
    ``onion_fl`` source go under ``/onionfl``; only the runs are writable. The
    prepared data is in the cache, so the raw data is not mounted.
    """
    import onion_fl

    raw = config.dump()
    volumes: list[dict[str, Any]] = []

    def bind(source: Path, target: str, writable: bool = False) -> None:
        volumes.append(
            {
                "type": "bind",
                "source": source.resolve().as_posix(),
                "target": target,
                "read_only": not writable,
            }
        )

    bind(Path(onion_fl.__file__).parent, f"{ROOT}/src/onion_fl")
    paths = {}
    for key in type(config.paths).model_fields:
        source, target = Path(getattr(config.paths, key)), f"{ROOT}/paths/{key}"
        if source.is_dir():
            bind(source, target, writable=key == "runs")
        paths[key] = target
    raw["paths"] = paths

    def file(ref: str, target: str) -> str:
        path = Path(ref)
        bind(path.parent, target)
        return f"{target}/{path.name}"

    for name, use in config.data.datasets.items():
        if use.descriptor:
            descriptor = file(use.descriptor, f"{ROOT}/descriptors/{slug(name)}")
            raw["data"]["datasets"][name]["descriptor"] = descriptor
    if isinstance(config.topology, str) and Path(config.topology).suffix in (
        ".yaml",
        ".yml",
    ):
        raw["topology"] = file(config.topology, f"{ROOT}/topology")
    infra = config.infrastructure
    if isinstance(infra.name, str) and Path(infra.name).suffix in (".yaml", ".yml"):
        raw["infrastructure"]["name"] = file(infra.name, f"{ROOT}/infrastructure")
    return volumes, raw


class _Plain(yaml.SafeDumper):
    """YAML without anchors: every service written out in full."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def prepare_emulated(scenario: Scenario, params: EmulatedParams) -> EmulatedPlan:
    """Resolve the hosts, create the run and write its Compose project."""
    from onion_fl.experiment.real import (
        _check_links,
        edge_groups,
        open_run,
        write_scenario,
    )
    from onion_fl.experiment.runner import (
        _initial_state,
        _scenario_data,
        _shapes,
        check_memory,
        physical,
    )

    config = scenario.config
    topology, split, placement, digests = _scenario_data(scenario)
    _check_links(topology)
    _, state = _initial_state(config, _shapes(split), scenario.seed)
    resolved = physical(scenario, topology, placement, state)
    if resolved is None:
        raise ConfigError("backend emulated: give the experiment an infrastructure")
    check_memory(resolved, placement, state)
    lay = check(params, resolved)
    load = host_load(resolved, edge_groups(config, topology, placement))
    run = open_run(
        scenario,
        topology,
        split,
        placement,
        digests,
        backend="emulated",
        infrastructure_id=resolved.infrastructure_id,
    )
    try:
        folder = run.path / "emulation"
        folder.mkdir()
        volumes, raw = container_config(config)
        emulation = compose_project(
            lay,
            project=f"{params.project}-{run.run_id.rsplit('-', 1)[-1]}",
            image=params.image,
            broker_image=params.broker_image,
            volumes=volumes,
            setup=folder.resolve().as_posix(),
        )
        if params.nodelay:  # Nagle off on the broker too, as on the clients
            from onion_fl.experiment.calibrate import broker_nodelay

            broker_nodelay(emulation, folder)
        cpusets = {}
        if params.pin is not None:  # the CPUs the calibration's session pinned
            from onion_fl.experiment.calibrate import pin

            cpusets = pin(emulation, params.pin)
        # The containers' torch draws other initial bits: they take these.
        np.savez(folder / "initial.npz", **state)
        write_scenario(
            run,
            scenario,
            raw,
            groups=load,
            broker=emulation.broker,
            nodelay=params.nodelay,
            initial="emulation/initial.npz",
        )
        (folder / "compose.yaml").write_text(
            yaml.dump(emulation.compose, Dumper=_Plain, sort_keys=False),
            encoding="utf-8",
        )
        for service, script in emulation.scripts.items():
            with open(
                folder / f"{service}.sh", "w", encoding="utf-8", newline="\n"
            ) as f:
                f.write(script)
        run.meta["emulation"] = {
            "project": emulation.project,
            "image": params.image,
            "broker_image": params.broker_image,
            "nodelay": params.nodelay,
            **lay.summary(),
            "shaping": emulation.shaping,
        } | ({"cpusets": cpusets} if cpusets else {})
        steps = outages(lay)
        if steps:  # only then, so other runs record what they did
            run.meta["emulation"]["outages"] = [
                {"at": at, "name": name, **tags, "commands": commands}
                for at, name, tags, commands in steps
            ]
    except BaseException:  # never leave a run that looks still running
        run.finish(status="failed")
        raise
    # Python and torch load slowly in CPU-limited containers.
    grace = params.ready_timeout
    return EmulatedPlan(
        run, scenario, sorted(load), emulation, folder, params, grace=grace
    )


def _docker(*args: str, **kw: Any) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, **kw)


def check_docker(plan: EmulatedPlan) -> None:
    """Docker answers, has the CPUs and memory the hosts ask for, and the image."""
    info = _docker("info", "--format", "{{.NCPU}} {{.MemTotal}}")
    if info.returncode != 0:
        raise RuntimeError(
            f"backend emulated: Docker is not available: {info.stderr.strip()}"
        )
    cpus, memory = (int(x) for x in info.stdout.split())
    infra = plan.emulation.layout.infra
    for host in plan.emulation.layout.hosts:
        profile = infra.profile_of(host)
        if (profile.cpus or 0) > cpus or (profile.memory or 0) > memory:
            raise ConfigError(
                f"backend emulated: host {host} asks for {profile.cpus} CPUs and "
                f"{profile.memory} bytes; Docker has {cpus} and {memory}"
            )
    ensure_image(plan.params.image)


def pins() -> dict[str, str]:
    """The versions of this environment that decide the bits of a run."""
    return {name: version(name).split("+")[0] for name in NUMERIC}


def ensure_image(image: str) -> None:
    """Build ``image`` unless it exists with this environment's ``pins``.

    The same torch, numpy, pandas and scikit-learn as the launcher: another
    torch initialises the model with other bits, and the equivalence with the
    simulation is bit for bit.
    """
    wanted = " ".join(f"{name}=={v}" for name, v in pins().items())
    found = _docker(
        "image", "inspect", "-f", '{{index .Config.Labels "' + LABEL + '"}}', image
    )
    if found.returncode == 0 and found.stdout.strip() == wanted:
        return
    print(f"backend emulated: building {image} with {wanted} (minutes)", flush=True)
    dockerfile = resources.files("onion_fl.infrastructure") / DOCKERFILE
    torch, *numeric = wanted.split()
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            image,
            "--label",
            f"{LABEL}={wanted}",
            "--build-arg",
            f"TORCH={torch}",
            "--build-arg",
            f"NUMERIC={' '.join(numeric)}",
            "-",
        ],
        input=dockerfile.read_bytes(),
        check=True,
    )


def wait_ready(plan: EmulatedPlan) -> None:
    """Every container has shaped its interfaces, and the broker answers."""
    deadline = time.monotonic() + plan.params.ready_timeout
    emulation = plan.emulation
    broker = f"{emulation.project}-broker"
    pending = [c for c in emulation.containers if c != broker]
    probe = (
        "import socket; "
        f"socket.create_connection(('127.0.0.1', {MQTT_PORT}), 1).close()"
    )
    checks = [(c, ["test", "-f", READY]) for c in pending]
    if emulation.layout.broker is not None:  # through its host's namespace
        owner = emulation.container(emulation.layout.broker)
        checks.append((owner, ["python", "-c", probe]))
    for container, command in checks:
        while _docker("exec", container, *command).returncode != 0:
            state = _docker("inspect", "-f", "{{.State.Running}}", container)
            if state.stdout.strip() != "true":
                raise RuntimeError(
                    f"backend emulated: {container} stopped while setting up; "
                    f"see {plan.folder / 'compose.log'}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"backend emulated: {container} not ready after "
                    f"{plan.params.ready_timeout:g} s"
                )
            time.sleep(0.2)


def partition(plan: EmulatedPlan, epoch: float) -> Any:
    """Apply the link outages at their instants of the run's clock, in a thread.

    Each change runs its command on the router at each end of the link and
    goes in the run as an event of the node ``network`` (with ``failed``, the
    routers whose command failed, if any). Returns a function that stops the
    thread and waits for it; None without outages.
    """
    from onion_fl.experiment.real import PART

    steps = outages(plan.emulation.layout)
    if not steps:
        return None
    run, stop = plan.run, threading.Event()
    sink = JsonlSink(run.path / PART.format(group="network"), flush=True)

    def apply() -> None:
        try:
            for at, name, tags, commands in steps:
                if stop.wait(max(0.0, epoch + at - time.time())):
                    return
                failed = [
                    service
                    for service, command in commands.items()
                    if _docker(
                        "exec",
                        f"{plan.emulation.project}-{service}",
                        "sh",
                        "-c",
                        command,
                    ).returncode
                ]
                raw = {
                    "t": time.time() - epoch,
                    "node": "network",
                    "name": name,
                    "value": None,
                    "tags": tags | ({"failed": failed} if failed else {}),
                }
                sink.write(
                    enrich(
                        raw,
                        run_id=run.run_id,
                        topology_id=run.topology.topology_id,
                        scenario=run.scenario,
                        seed=run.seed,
                        nodes={},
                    )
                )
        finally:
            sink.close()

    thread = threading.Thread(target=apply, name="onionfl-partition", daemon=True)
    thread.start()

    def halt() -> None:
        stop.set()
        thread.join()

    return halt


def take_down(plan: EmulatedPlan) -> None:
    """Keep the containers' logs and remove the project, whatever happened."""
    logs = subprocess.run(
        [*plan.compose, "logs", "--no-color", "--timestamps"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (plan.folder / "compose.log").write_text(
        logs.stdout + logs.stderr, encoding="utf-8"
    )
    subprocess.run(
        [*plan.compose, "down", "--volumes", "--remove-orphans", "--timeout", "1"],
        capture_output=True,
    )


def launch_emulated(plan: EmulatedPlan) -> None:
    """Bring the project up, run one node per host in it, and take it down."""
    from onion_fl.experiment.real import launch

    try:
        check_docker(plan)
        start = time.monotonic()
        up = subprocess.run(
            [*plan.compose, "up", "--detach", "--quiet-pull"],
            capture_output=True,
            text=True,
        )
        if up.returncode != 0:
            raise RuntimeError(f"backend emulated: compose up failed: {up.stderr}")
        wait_ready(plan)
        plan.run.meta["emulation"]["setup_s"] = round(time.monotonic() - start, 3)
        # The run's clock starts now, not when the project was written.
        path = plan.run.path / "scenario.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["epoch"] = time.time()
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        halt = partition(plan, document["epoch"])
    except BaseException:  # never leave a run that looks still running
        take_down(plan)
        plan.run.finish(status="failed")
        raise
    try:
        launch(plan)
    finally:
        if halt is not None:
            halt()
        take_down(plan)
