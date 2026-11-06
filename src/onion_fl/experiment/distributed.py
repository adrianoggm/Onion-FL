from __future__ import annotations

"""The ``distributed`` backend: the real launcher on real machines (I1 §10, I14).

An inventory (``onion_fl.infrastructure.inventory``) says which machine plays
each host of the infrastructure and how to reach it over SSH. The launcher
uses the system's OpenSSH client and needs nothing on a machine but ``sh``,
``tar`` and a Python with ``onion_fl`` installed.

``prepare_distributed`` resolves the assignment (the logical nodes of each
host, as ``emulated`` does) and runs the **preflight** on every machine at
once (``onion_fl.infrastructure.probe``): Python 3.11 or newer, the same
``onion_fl`` source as the launcher, the numeric libraries (a different
version is a warning: the bits may differ from ``sim``), the data cache, the
clock offset (against the launcher's clock, measured over the SSH session, or
against NTP) within ``clock.max_offset``, the broker, and every other
machine's gRPC port. Any failure refuses the run, with every reason at once,
before a run folder exists.

``run`` sends each machine, under its ``root``, the config files, the
descriptors, the data cache it lacks, ``scenario.json`` (paths relative to
``root``) and the launcher's initial model; starts one ``onion_fl node`` per
host over SSH, as ``local`` starts one per group; and brings every
``events.<host>.jsonl`` and the root's ``model.npz`` back. ``collect`` merges
and signs the run with the same ``merge`` as ``local``.

gRPC links find their peers through an explicit map (``grpc`` in
``scenario.json``: each host's ``advertise:grpc_port``), not the local
discovery folder; MQTT links all go to one broker, the host named by the
infrastructure's ``services.mqtt_broker`` or the backend's ``broker``.
"""

import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from inspect import getsource
from pathlib import Path
from typing import Any

import numpy as np

from onion_fl.experiment.config import ConfigError
from onion_fl.experiment.sweep import Scenario
from onion_fl.infrastructure import probe
from onion_fl.infrastructure.assignment import Resolved
from onion_fl.infrastructure.emulators import slug
from onion_fl.infrastructure.inventory import (
    CONTROL,
    Inventory,
    Machine,
    load_inventory,
)
from onion_fl.infrastructure.model import InfrastructureError
from onion_fl.observability.run import Run

NUMERIC = ("torch", "numpy", "scipy", "pandas", "scikit-learn")
MIN_PYTHON = (3, 11)
MODEL = "model.npz"
MAX_MEMBER = 1 << 30  # bytes of one file a machine sends back
MAX_ARCHIVE = 2 << 30  # bytes of all it sends back
PATHS = ("topologies", "infrastructures", "datasets", "runs", "cache")


class PreflightError(RuntimeError):
    """A machine cannot take part in the run. The message lists every reason."""


# --- SSH -------------------------------------------------------------------------------


def ssh_command(
    ssh: str,
    inventory: Inventory,
    machine: Machine,
    line: str,
    *,
    no_stdin: bool = False,
) -> list[str]:
    """``ssh`` options for ``machine`` and the shell ``line`` it runs there."""
    command = [
        ssh,
        "-o",
        "BatchMode=yes",
        "-o",
        f"StrictHostKeyChecking={inventory.host_key_check}",
        "-o",
        f"ConnectTimeout={max(1, round(inventory.connect_timeout))}",
        "-o",
        "ServerAliveInterval=15",
        "-p",
        str(machine.port),
    ]
    if inventory.known_hosts is not None:
        known = known_hosts_path(inventory.known_hosts)
        # ssh splits this option's value on whitespace, so a path with a space
        # is quoted; any other is passed bare, since Python's quoting of '"'
        # for Windows reaches an MSYS ssh (Git's) as "invalid quotes".
        if any(c.isspace() for c in known):
            known = f'"{known}"'
        command += ["-o", f"UserKnownHostsFile={known}"]
    key = machine.key_path()
    if key is not None:
        command += ["-i", key, "-o", "IdentitiesOnly=yes"]
    if no_stdin:
        command.append("-n")
    target = (
        machine.address if machine.user is None else f"{machine.user}@{machine.address}"
    )
    # "--": whatever the destination says, ssh never reads it as an option.
    return [*command, "--", target, line]


def known_hosts_path(path: str) -> str:
    """The known_hosts file, expanded; quoted by the caller if it has spaces."""
    known = expand(path)
    if '"' in known or CONTROL.search(known) or known.startswith("-"):
        raise ConfigError(f"known_hosts {path!r}: not a usable path for ssh")
    return known


def expand(path: str) -> str:
    """``~`` and ``$VARS`` expanded, as for an SSH key."""
    return os.path.expanduser(os.path.expandvars(path))


def root_of(machine: Machine) -> str:
    """``root`` as the remote shell takes it: relative paths start at the home."""
    return machine.root.removeprefix("~/") or "."


def remote_line(
    machine: Machine, argv: Sequence[str], *, pidfile: str | None = None
) -> str:
    """A POSIX shell line: into ``root``, with the machine's env, run ``argv``."""
    root = shlex.quote(root_of(machine))
    env = [f"{k}={shlex.quote(v)}" for k, v in sorted(machine.env.items())]
    run = " ".join(shlex.quote(a) for a in argv)
    if env:
        run = f"env {' '.join(env)} {run}"
    if pidfile is not None:  # so that a process the launcher gave up on can be stopped
        run = f"echo $$ > {shlex.quote(pidfile)} && exec {run}"
    return f"mkdir -p {root} && cd {root} && {run}"


# --- the preflight ------------------------------------------------------------------------


@dataclass
class Probe:
    """One machine's probe session: its facts, then clock samples and reach."""

    host: str
    process: subprocess.Popen
    facts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def ask(self, line: str) -> str:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        answer = self.process.stdout.readline()
        if not answer:
            raise ConnectionError("the probe stopped answering")
        return answer.strip()

    def close(self) -> str:
        """End the session; what the probe wrote to stderr."""
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except OSError:
            pass
        try:
            _, err = self.process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            _, err = self.process.communicate()
        return (err or "").strip()


def clock_offset(samples: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
    """``(offset, round_trip)`` from ``(sent, remote, received)`` samples.

    As NTP: the remote time against the middle of the round trip, from the
    sample with the shortest one; the offset is within half of it.
    """
    sent, remote, received = min(samples, key=lambda s: s[2] - s[0])
    return remote - (sent + received) / 2, received - sent


def open_probe(
    ssh: str,
    inventory: Inventory,
    host: str,
    request: Mapping[str, Any],
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> Probe:
    machine = inventory.machine(host)
    source = getsource(probe)
    line = remote_line(machine, [machine.python, "-c", source, json.dumps(request)])
    process = popen(
        ssh_command(ssh, inventory, machine, line),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    return Probe(host, process)


def run_probes(
    ssh: str,
    inventory: Inventory,
    requests: Mapping[str, Mapping[str, Any]],
    reach: Mapping[str, Sequence[str]],
    *,
    samples: int = 5,
    timeout: float = 60.0,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> dict[str, dict[str, Any]]:
    """The facts of every machine, probed at once.

    Every probe first answers with its facts and holds its gRPC port open;
    then the launcher takes ``samples`` clock samples from each, and only
    then asks each what it ``reach``es, so the other machines' ports are up.
    A machine that cannot be probed gets an ``error``.
    """
    probes = {
        host: open_probe(ssh, inventory, host, request, popen)
        for host, request in requests.items()
    }
    timers = [threading.Timer(timeout, p.process.kill) for p in probes.values()]
    for timer in timers:
        timer.daemon = True
        timer.start()
    try:
        for p in probes.values():
            try:
                assert p.process.stdout is not None
                first = p.process.stdout.readline()
                p.facts = json.loads(first)
            except (ValueError, OSError):
                p.error = "no answer"
        for p in probes.values():
            if p.error is not None:
                continue
            try:
                taken = []
                for _ in range(samples):
                    sent = time.time()
                    remote = float(p.ask("time"))
                    taken.append((sent, remote, time.time()))
                p.facts["offset"], p.facts["rtt"] = clock_offset(taken)
            except (ValueError, OSError) as exc:
                p.error = f"clock samples: {exc}"
        for p in probes.values():
            if p.error is not None:
                continue
            try:
                p.facts["reach"] = json.loads(
                    p.ask("reach " + json.dumps(list(reach.get(p.host, []))))
                )
            except (ValueError, OSError) as exc:
                p.error = f"reach: {exc}"
    finally:
        for timer in timers:
            timer.cancel()
        out = {}
        for host, p in probes.items():
            stderr = p.close()
            if p.error is not None:
                out[host] = {"error": f"{p.error}; ssh said: {stderr or 'nothing'}"}
            else:
                out[host] = p.facts
    return out


@dataclass
class Expected:
    """What every machine must match: the launcher's side of the preflight."""

    source: str
    pins: dict[str, str | None]
    files: dict[str, str]  # remote path -> SHA-256 the launcher has
    max_offset: float
    reference: str = "launcher"
    bind: bool = False


@dataclass
class Verdict:
    host: str
    errors: list[str]
    warnings: list[str]
    stale: list[str]  # the files of ``Expected.files`` it lacks or has otherwise


def judge(host: str, facts: Mapping[str, Any], expected: Expected) -> Verdict:
    """Whether a machine can take part, from what its probe said."""
    errors: list[str] = []
    warnings: list[str] = []
    if "error" in facts and facts.get("python") is None:
        return Verdict(host, [f"cannot probe it: {facts['error']}"], [], [])
    python = tuple(int(x) for x in str(facts.get("python", "0")).split(".")[:2])
    if python < MIN_PYTHON:
        errors.append(f"Python {facts.get('python')}; onion_fl needs 3.11 or newer")
    if facts.get("source") is None:
        errors.append(f"onion_fl is not installed: {facts.get('error', 'no import')}")
    elif facts["source"] != expected.source:
        errors.append(
            "it runs other onion_fl code than the launcher (the source digest "
            "differs): install the launcher's commit"
        )
    for name, wanted in expected.pins.items():
        found = facts.get("pins", {}).get(name)
        if wanted is not None and found != wanted:
            warnings.append(
                f"{name} {found or 'missing'} here, {wanted} on the launcher: the "
                "bits may differ from sim"
            )
    offset = facts.get("offset")
    if offset is not None and abs(offset) > expected.max_offset:
        errors.append(
            f"its clock is {offset:+.3f} s off the launcher's (max_offset "
            f"{expected.max_offset:g} s): synchronise it with NTP"
        )
    if expected.reference == "ntp":
        ntp = facts.get("ntp")
        if ntp is None:
            errors.append(
                f"no NTP answer ({facts.get('ntp_error', 'not asked')}); the clock "
                "cannot be checked"
            )
        elif abs(ntp) > expected.max_offset:
            errors.append(
                f"its clock is {ntp:+.3f} s off NTP (max_offset "
                f"{expected.max_offset:g} s): synchronise it"
            )
    if expected.bind and facts.get("bind") is False:
        errors.append("its gRPC port is in use")
    for target, ok in facts.get("reach", {}).items():
        if not ok:
            errors.append(f"cannot reach {target}")
    files = facts.get("files", {})
    stale = [f for f, digest in expected.files.items() if files.get(f) != digest]
    return Verdict(host, errors, warnings, stale)


# --- what each machine gets ------------------------------------------------------------------


def remote_config(config: Any) -> tuple[dict[str, Path], dict[str, Path], dict]:
    """The files a machine needs, the data caches, and the config it reads.

    Paths are relative to the machine's ``root``: ``topologies``,
    ``infrastructures``, ``datasets``, ``runs`` and ``cache``, and
    ``descriptors/<dataset>/``, ``topology/`` and ``infrastructure/`` for the
    files the config names. The raw data is not sent: the cache is.
    """
    from onion_fl.data.cache import cache_key
    from onion_fl.data.ingest import load_spec

    raw = config.dump()
    files: dict[str, Path] = {}
    caches: dict[str, Path] = {}
    for key in ("topologies", "infrastructures", "datasets"):
        folder = Path(getattr(config.paths, key))
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix in (".yaml", ".yml"):
                    files[f"{key}/{path.name}"] = path
    raw["paths"] = {key: key for key in PATHS}

    def send(ref: str, folder: str) -> str:
        path = Path(ref)
        files[f"{folder}/{path.name}"] = path
        return f"{folder}/{path.name}"

    for name, use in sorted(config.data.datasets.items()):
        descriptor = use.descriptor or Path(config.paths.datasets) / f"{name}.yaml"
        if use.descriptor:
            remote = send(use.descriptor, f"descriptors/{slug(name)}")
            raw["data"]["datasets"][name]["descriptor"] = remote
        spec = load_spec(descriptor)
        local = Path(config.paths.cache) / spec.name / cache_key(spec, use.options)
        caches[f"cache/{spec.name}/{local.name}"] = local
    if isinstance(config.topology, str) and Path(config.topology).suffix in (
        ".yaml",
        ".yml",
    ):
        raw["topology"] = send(config.topology, "topology")
    infra = config.infrastructure
    if isinstance(infra.name, str) and Path(infra.name).suffix in (".yaml", ".yml"):
        raw["infrastructure"]["name"] = send(infra.name, "infrastructure")
    return files, caches, raw


def push(
    ssh: str,
    inventory: Inventory,
    machine: Machine,
    entries: Mapping[str, Path],
    run: Callable[..., Any] = subprocess.Popen,
) -> None:
    """Send ``entries`` (remote path -> local file or folder) as one tar stream."""
    root = shlex.quote(root_of(machine))
    line = f"mkdir -p {root} && tar xf - -C {root}"
    process = run(
        ssh_command(ssh, inventory, machine, line),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    err = b""
    try:
        with tarfile.open(
            fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT
        ) as tar:
            for arcname, path in sorted(entries.items()):
                tar.add(path, arcname=arcname)
        process.stdin.close()
        _, err = process.communicate(timeout=600)
    except BaseException:
        process.kill()
        raise
    if process.returncode != 0:
        raise RuntimeError(
            f"cannot send the run to {machine.address}: {err.decode(errors='replace')}"
        )


def expected_files(host: str, root: bool) -> list[str]:
    """What ``host`` may send back: its own event file, and the model if it
    hosts the root."""
    return [f"events.{host}.jsonl", *([MODEL] if root else [])]


def pull_line(run_id: str, host: str, root: bool) -> str:
    """Archive, to stdout, the files of ``expected_files`` that exist."""
    folder = shlex.quote(f"runs/{run_id}")
    names = " ".join(shlex.quote(n) for n in expected_files(host, root))
    return (
        f"cd {folder} 2>/dev/null || exit 0; for f in {names}; "
        'do [ -f "$f" ] && echo "$f"; done | tar cf - -T -'
    )


class SpoofError(RuntimeError):
    """A machine sent back something it may not: the run is not signed."""


def unpack(
    data: bytes,
    into: Path,
    host: str,
    *,
    root: bool,
    max_member: int = MAX_MEMBER,
    max_archive: int = MAX_ARCHIVE,
) -> list[str]:
    """Write what ``host`` sent back (a tar in ``data``) into ``into``.

    Only ``events.<host>.jsonl`` and, from the root's host, ``model.npz``
    are taken, each once and never over a file already there. Anything else
    (another host's events, a link, a folder, a path, an oversized member or
    archive) refuses the whole archive before anything is written: what a
    machine sends is data, never where to write it or whose it is.
    """
    if not data:
        return []
    if len(data) > max_archive:
        raise SpoofError(f"{host}: sent {len(data)} bytes, over {max_archive}")
    allowed = set(expected_files(host, root))
    taken: dict[str, bytes] = {}
    try:
        tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
    except tarfile.TarError as exc:
        raise SpoofError(f"{host}: not a tar archive: {exc}") from None
    with tar:
        for member in tar.getmembers():
            name = member.name
            if not member.isreg():
                raise SpoofError(f"{host}: {name!r} is not a regular file")
            if name not in allowed:  # also any "/", "..", or absolute name
                raise SpoofError(
                    f"{host}: may send back only {sorted(allowed)}, not {name!r}"
                )
            if name in taken or (into / name).exists():
                raise SpoofError(f"{host}: {name!r} sent twice")
            if member.size > max_member:
                raise SpoofError(
                    f"{host}: {name!r} has {member.size} bytes, over {max_member}"
                )
            extracted = tar.extractfile(member)
            if extracted is None:
                raise SpoofError(f"{host}: {name!r} cannot be read")
            taken[name] = extracted.read(max_member + 1)
    for name, content in taken.items():
        with open(into / name, "xb") as out:  # never overwrite
            out.write(content)
    return sorted(taken)


# --- the plan -------------------------------------------------------------------------------


@dataclass
class DistributedPlan:
    """What the ``distributed`` backend starts: one node per host, over SSH."""

    run: Run
    scenario: Scenario
    groups: list[str]  # the hosts that run logical nodes
    inventory: Inventory
    params: Any  # DistributedParams
    files: dict[str, Path]  # config files and descriptors, the same for every host
    caches: dict[str, Path]
    stale: dict[str, list[str]]  # host -> caches to send
    codes: list[int] = field(default_factory=list)
    grace: float = 60.0  # seconds a process may take to start, on top of the timeout
    root_host: str | None = None  # the host of the coordinator: it sends the model

    def machine(self, host: str) -> Machine:
        return self.inventory.machine(host)

    def command(self, group: str) -> list[str]:
        """The node process of host ``group``, on its machine."""
        machine = self.machine(group)
        rid = self.run.run_id
        argv = [
            machine.python,
            "-m",
            "onion_fl",
            "node",
            "--id",
            group,
            "--run",
            rid,
            "--config",
            f"runs/{rid}/scenario.json",
        ]
        line = remote_line(machine, argv, pidfile=f"runs/{rid}/node.{group}.pid")
        return ssh_command(
            self.params.ssh, self.inventory, machine, line, no_stdin=True
        )

    def stop_command(self, group: str) -> list[str]:
        """Stop the node of ``group`` if it is still running."""
        machine = self.machine(group)
        pid = shlex.quote(f"runs/{self.run.run_id}/node.{group}.pid")
        root = shlex.quote(root_of(machine))
        line = f"cd {root} && [ -f {pid} ] && kill $(cat {pid}) 2>/dev/null; true"
        return ssh_command(
            self.params.ssh, self.inventory, machine, line, no_stdin=True
        )


def _links(topology: Any) -> list[Any]:
    links = [n.link_up for n in topology.nodes if n.link_up is not None]
    return [*links, topology.edge.link_up]


def _transport_name(spec: Any) -> str:
    return spec if isinstance(spec, str) else spec.get("name", "mqtt")


def check_grpc(topology: Any) -> None:
    """One gRPC server per machine: every gRPC link must say the same."""
    specs = {
        json.dumps(link.transport, sort_keys=True)
        for link in _links(topology)
        if _transport_name(link.transport) == "grpc"
    }
    if len(specs) > 1:
        raise ConfigError(
            "backend distributed: every grpc link must have the same transport "
            "settings; each machine serves its inboxes on one gRPC port"
        )


def machines_of(inventory: Inventory, hosts: Sequence[str]) -> dict[str, Machine]:
    missing = [h for h in hosts if h not in inventory.machines]
    if missing:
        raise ConfigError(
            f"backend distributed: the inventory has no machine for {missing}; "
            f"it has {sorted(inventory.machines)}"
        )
    return {h: inventory.machines[h] for h in hosts}


def refused(resolved: Resolved) -> list[str]:
    """What real machines cannot reproduce of an infrastructure."""
    out = []
    if resolved.infrastructure.network.schedule:
        out.append(
            "network.schedule: a real network cannot be changed on schedule; "
            "use the emulated backend"
        )
    return out


def wiring(
    params: Any,
    resolved: Resolved,
    machines: Mapping[str, Machine],
    load: Mapping[str, Sequence[str]],
    names: set[str],
    inventory: Inventory,
) -> tuple[str | None, dict[str, Any] | None]:
    """The broker every MQTT link takes, and the gRPC address map."""
    broker = None
    if "mqtt" in names:
        broker = params.broker
        if broker is None:
            host = resolved.services.get("mqtt_broker")
            if host is None:
                raise ConfigError(
                    "backend distributed: name the host of the mqtt_broker in "
                    "services, or give the backend a broker"
                )
            try:
                machine = inventory.machine(host)
            except InfrastructureError as exc:
                raise ConfigError(f"backend distributed: {exc}") from None
            broker = f"{machine.peer_address}:{params.mqtt_port}"
    grpc = None
    if "grpc" in names:
        grpc = {
            "listen": {h: f"{m.listen}:{m.grpc_port}" for h, m in machines.items()},
            "peers": {
                node: f"{machines[h].peer_address}:{machines[h].grpc_port}"
                for h, nodes in sorted(load.items())
                for node in nodes
            },
        }
    return broker, grpc


def inspect(params: Any, resolved: Resolved) -> dict[str, Any]:
    """What ``onion_fl plan`` shows of a distributed run; no machine is contacted."""
    problems = refused(resolved)
    if problems:
        raise ConfigError("backend distributed: " + "; ".join(problems))
    inventory = _inventory(params)
    hosts = sorted(set(resolved.hosts.values()))
    machines = machines_of(inventory, hosts)
    return {
        "name": "distributed",
        "machines": {
            h: {"address": m.address, "port": m.port, "advertise": m.peer_address}
            for h, m in machines.items()
        },
        "clock": inventory.clock.model_dump(),
    }


def _inventory(params: Any) -> Inventory:
    try:
        return load_inventory(params.inventory)
    except InfrastructureError as exc:
        raise ConfigError(f"backend distributed: {exc}") from None


def preflight(
    params: Any,
    inventory: Inventory,
    machines: Mapping[str, Machine],
    expected: Expected,
    broker: str | None,
    grpc: Mapping[str, Any] | None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> dict[str, tuple[dict[str, Any], Verdict]]:
    """Probe every machine and judge it; a ``PreflightError`` with all reasons."""
    clock = inventory.clock
    requests, reach = {}, {}
    for host, machine in machines.items():
        requests[host] = {
            "pins": list(NUMERIC),
            "files": sorted(expected.files),
            "bind": None if grpc is None else [machine.listen, machine.grpc_port],
            "ntp": clock.server if clock.reference == "ntp" else None,
        }
        targets = [] if broker is None else [broker]
        if grpc is not None:
            targets += [
                f"{m.peer_address}:{m.grpc_port}"
                for other, m in machines.items()
                if other != host
            ]
        reach[host] = targets
    facts = run_probes(
        params.ssh,
        inventory,
        requests,
        reach,
        samples=params.clock_samples,
        timeout=params.probe_timeout,
        popen=popen,
    )
    out = {host: (facts[host], judge(host, facts[host], expected)) for host in machines}
    errors = [
        f"{host}: {reason}" for host, (_, v) in out.items() for reason in v.errors
    ]
    if not params.push_data:
        errors += [
            f"{host}: its data cache {f} is missing or differs, and push_data is off"
            for host, (_, v) in out.items()
            for f in v.stale
        ]
    if errors:
        raise PreflightError(
            "backend distributed: preflight failed:\n- " + "\n- ".join(errors)
        )
    return out


def prepare_distributed(
    scenario: Scenario,
    params: Any,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> DistributedPlan:
    """Resolve the hosts, check every machine, and create the run."""
    from onion_fl.experiment.emulated import host_load
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
    check_grpc(topology)
    _, state = _initial_state(config, _shapes(split), scenario.seed)
    resolved = physical(scenario, topology, placement, state)
    if resolved is None:
        raise ConfigError("backend distributed: give the experiment an infrastructure")
    check_memory(resolved, placement, state)
    problems = refused(resolved)
    if problems:
        raise ConfigError("backend distributed: " + "; ".join(problems))
    inventory = _inventory(params)
    load = host_load(resolved, edge_groups(config, topology, placement))
    machines = machines_of(inventory, sorted(load))
    names = {_transport_name(link.transport) for link in _links(topology)}
    broker, grpc = wiring(params, resolved, machines, load, names, inventory)
    files, caches, raw = remote_config(config)
    expected = Expected(
        source=probe.source_digest(Path(probe.__file__).parents[1]),
        pins=numeric_pins(),
        files={
            f"{rel}/meta.json": _sha256(local / "meta.json")
            for rel, local in caches.items()
        },
        max_offset=inventory.clock.max_offset,
        reference=inventory.clock.reference,
        bind=grpc is not None,
    )
    checked = preflight(params, inventory, machines, expected, broker, grpc, popen)
    stale = {
        host: sorted(rel for rel in caches if f"{rel}/meta.json" in verdict.stale)
        for host, (_, verdict) in checked.items()
    }
    run = open_run(
        scenario,
        topology,
        split,
        placement,
        digests,
        backend="distributed",
        infrastructure_id=resolved.infrastructure_id,
    )
    try:
        np.savez(run.path / "initial.npz", **state)
        launcher: dict[str, Any] = {"groups": load, "initial": "initial.npz"}
        if broker is not None:
            launcher["broker"] = broker
        if grpc is not None:
            launcher["grpc"] = grpc
        write_scenario(run, scenario, raw, **launcher)
        warnings = [
            f"{host}: {w}" for host, (_, v) in checked.items() for w in v.warnings
        ]
        for warning in warnings:
            print(f"backend distributed: {warning}", file=sys.stderr, flush=True)
        run.meta["distributed"] = {
            "machines": {
                host: {
                    "address": machines[host].address,
                    "advertise": machines[host].peer_address,
                    "root": machines[host].root,
                    **{
                        k: facts.get(k)
                        for k in (
                            "python",
                            "machine",
                            "system",
                            "onion_fl",
                            "pins",
                            "offset",
                            "rtt",
                            "ntp",
                        )
                        if k in facts
                    },
                    "nodes": len(load[host]),
                }
                for host, (facts, _) in checked.items()
            },
            "clock": inventory.clock.model_dump(),
            "broker": broker,
            "transports": sorted(names),
            "warnings": warnings,
            "sent_data": stale,
        }
    except BaseException:  # never leave a run that looks still running
        run.finish(status="failed")
        raise
    return DistributedPlan(
        run,
        scenario,
        sorted(load),
        inventory,
        params,
        files,
        caches,
        stale,
        grace=params.grace,
        root_host=resolved.hosts[topology.root.id],
    )


def numeric_pins() -> dict[str, str | None]:
    """The launcher's versions of the libraries that decide the bits."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in NUMERIC:
        try:
            out[name] = version(name).split("+")[0]
        except PackageNotFoundError:
            out[name] = None
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launch_distributed(
    plan: DistributedPlan, popen: Callable[..., Any] = subprocess.Popen
) -> None:
    """Send each machine its files, run one node per host, bring the events back."""
    from onion_fl.experiment.real import launch

    rid = plan.run.run_id
    try:
        for host in plan.groups:  # the bulk first: the clock starts after it
            entries = dict(plan.files)
            entries |= {rel: plan.caches[rel] for rel in plan.stale.get(host, [])}
            push(plan.params.ssh, plan.inventory, plan.machine(host), entries, popen)
        path = plan.run.path / "scenario.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["epoch"] = time.time()  # the run's clock starts now
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        own = {
            f"runs/{rid}/scenario.json": path,
            f"runs/{rid}/initial.npz": plan.run.path / "initial.npz",
        }
        for host in plan.groups:
            push(plan.params.ssh, plan.inventory, plan.machine(host), own, popen)
    except BaseException:  # never leave a run that looks still running
        plan.run.finish(status="failed")
        raise
    try:
        launch(plan)
    finally:
        for host in plan.groups:
            subprocess.run(plan.stop_command(host), capture_output=True, timeout=60)
        pulled, refused = {}, []
        root_host = plan.root_host
        for host in plan.groups:
            machine = plan.machine(host)
            line = pull_line(rid, host, host == root_host)
            fetched = subprocess.run(
                ssh_command(
                    plan.params.ssh,
                    plan.inventory,
                    machine,
                    f"cd {shlex.quote(root_of(machine))} && {line}",
                    no_stdin=True,
                ),
                capture_output=True,
                timeout=600,
            )
            try:
                pulled[host] = unpack(
                    fetched.stdout, plan.run.path, host, root=host == root_host
                )
            except SpoofError as exc:
                refused.append(str(exc))
        plan.run.meta["distributed"]["pulled"] = pulled
        if refused:
            plan.run.meta["distributed"]["refused"] = refused
            plan.run.finish(status="failed")
            raise SpoofError("; ".join(refused))
