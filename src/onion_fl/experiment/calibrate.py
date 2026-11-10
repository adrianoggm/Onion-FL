from __future__ import annotations

"""Calibrating the profiles and links of an infrastructure on a backend (I1 §13, I12).

Three steps, each its own command, so that a calibration is never judged on
the measurements it was fitted on:

1. ``onion_fl calibrate measure <spec.yaml> --out <session>`` brings up the
   emulation of the hosts the spec names and runs the microbenchmarks
   (``onion_fl.experiment.microbench``) in their containers:

   - **train**, per workload: every train host at once, so that they cohabit
     as in an experiment (the I9 spike lost 12–16 % to cohabiting hosts);
   - **aggregate**: every aggregate host at once;
   - **links**: each pair of hosts, in both directions, one at a time and
     with the other hosts idle, through the MQTT broker of the
     infrastructure, as the nodes talk.

2. ``onion_fl calibrate fit <session>...`` turns one or more sessions into a
   signed calibration ``calibrations/<calibration_id>/``, and writes the
   infrastructure that cites it.
3. ``onion_fl calibrate check <session> --infrastructure <calibrated.yaml>``
   predicts, with the simulator's own code, what another session measured,
   and says whether that session's values fall inside the calibration's
   intervals (refusing a session the calibration was fitted on).

How each value is fitted:

- **Training rate** of a (profile, workload): the samples charged over the
  wall time, summed over the trainings of every host of the profile. A
  host's first ``train.warmup`` trainings warm it up and are left out, and so
  is any training outside the window in which every host of the phase was
  training.
- **Aggregation rate** of a profile: bytes combined over wall time, the same
  way (the first combination of each size is the warm-up).
- **Path** between two hosts: a least-squares line ``delay = a + c·size``
  over its one-way delays up to ``links.fit_up_to`` bytes; ``a`` is its
  latency and ``c`` its seconds per byte. The first message of each pair is
  a warm-up and is left out. Past the first TCP window (about 14.6 kB) slow
  start after an idle connection adds round trips that no line describes, so
  the path model holds up to ``fit_up_to`` only (``valid_up_to``).
- **Link latency**: least squares of the paths' ``a`` over the links they
  cross (the broker makes a path cross some links twice), so the links absorb
  the transport's own overhead.
- **Link bandwidth**: through the broker, a message is sent once per leg
  (``paths.legs``), each leg at the rate of its narrowest segment, so
  ``c = Σ 8·xₕ / Bₕ`` over the legs, ``Bₕ`` the nominal rate of the leg's
  bottleneck ``h`` and ``xₕ`` the nominal over the goodput. The ``xₕ`` are
  solved by least squares over the paths; a link direction that bottlenecks
  no leg takes their mean (listed in ``derived``).
- **Link loss**: the lowest loss of the paths that cross it (a link loses at
  most what its least lossy path does).

Within a session every interval is a 95 % percentile bootstrap over its
measurements. Fitted on k ≥ 2 sessions, each value is the geometric mean of
the sessions' values and its interval the 95 % prediction interval of the
value of one more session, ``exp(m ± t₀.₉₇₅,ₖ₋₁ · s · √(1 + 1/k))`` with ``m``
and ``s`` the mean and standard deviation of the logs (``between``): the
variation between sessions, not the noise within one, decides whether a
calibration holds on another day. A link's loss pools the sessions' messages.
"""

import json
import math
import pickle
import platform
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from onion_fl.core.ids import code_version
from onion_fl.experiment.backends import EmulatedParams, PinSpec
from onion_fl.experiment.config import ConfigError, DataConfig, PathsConfig
from onion_fl.infrastructure.calibration import (
    Calibration,
    Estimate,
    LinkCalibration,
    ProfileCalibration,
    Workload,
    aggregate_rate,
    cited,
    link_key,
    load_calibration,
    measurements_digest,
    train_rate,
    workload_key,
    write_calibration,
)
from onion_fl.infrastructure.emulators import SETUP, compose_project, layout
from onion_fl.infrastructure.model import (
    Infrastructure,
    InfrastructureError,
    load_infrastructure,
)
from onion_fl.infrastructure.paths import effective, legs, route, segment

ROOT = "/onionfl"
SESSION = f"{ROOT}/session"  # where the containers see the session folder
BOOTSTRAP = 2000
MEASUREMENTS = "measurements.jsonl"
META = "session.json"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkloadSpec(Strict):
    model: Any = Field(description="Modelo, como en learning.model")
    trainer: Any = Field(description="Entrenador, como en learning.trainer")


class TrainSpec(Strict):
    hosts: list[str] = Field(min_length=1, description="Hosts que entrenan a la vez")
    per_host: int = Field(
        4, gt=0, description="Clientes por host, entrenados por turno"
    )
    jobs: int = Field(12, gt=1, description="Entrenamientos por host y carga")
    warmup: int = Field(
        1, ge=1, description="Entrenamientos iniciales de cada host que no cuentan"
    )


class AggregateSpec(Strict):
    hosts: list[str] = Field(min_length=1, description="Hosts que agregan a la vez")
    workload: int = Field(
        0, ge=0, description="Índice de la carga cuyo modelo se agrega"
    )
    updates: list[int] = Field([10, 20], min_length=1)
    seconds: float = Field(5.0, gt=0, description="Duración por número de updates (s)")
    batch: int = Field(100, gt=0, description="Combinaciones por medida")


class LinksSpec(Strict):
    pairs: list[tuple[str, str]] = Field(min_length=1)
    sizes: list[int] = Field([100, 1_000, 5_000, 10_000], min_length=2)
    fit_up_to: int | None = Field(
        None, gt=0, description="Tamaño máximo de los mensajes del ajuste (bytes)"
    )
    repeats: int = Field(15, gt=1)
    qos: int = Field(1, ge=0, le=2)
    gap: float = Field(0.5, ge=0, description="Pausa entre mensajes (s)")
    wait: float = Field(30.0, gt=0, description="Espera máxima de un acuse (s)")
    nodelay: bool = Field(
        False,
        description="TCP_NODELAY en los clientes y en el broker (sin Nagle); "
        "por defecto no, como en las sesiones de b55104b3 (las ejecuciones FL "
        "lo activan desde #198)",
    )


class CalibrationSpec(Strict):
    """What to measure; ``lead`` is how long the hosts of a phase have to
    start their processes before they all begin at once."""

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = ""
    infrastructure: str
    data: DataConfig
    workloads: list[WorkloadSpec] = Field(min_length=1)
    train: TrainSpec | None = None
    aggregate: AggregateSpec | None = None
    links: LinksSpec | None = None
    seed: int = 0
    lead: float = Field(20.0, gt=0, description="Margen para arrancar una fase (s)")
    pin: PinSpec | None = None
    paths: PathsConfig = Field(default_factory=PathsConfig)
    emulation: EmulatedParams = Field(default_factory=EmulatedParams)


def load_spec(path: str | Path) -> CalibrationSpec:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return CalibrationSpec.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"calibration spec {path}: {exc}") from None


def _infrastructure(spec: CalibrationSpec) -> Infrastructure:
    path = Path(spec.infrastructure)
    if path.suffix not in (".yaml", ".yml"):
        path = Path(spec.paths.infrastructures) / f"{spec.infrastructure}.yaml"
    try:
        return load_infrastructure(path)
    except InfrastructureError as exc:
        raise ConfigError(f"infrastructure: {exc}") from None


# --- measure -------------------------------------------------------------------------


@dataclass
class _Session:
    """What ``emulated.wait_ready`` and ``take_down`` need of a plan."""

    emulation: Any
    params: EmulatedParams
    folder: Path

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


def _clients(spec: CalibrationSpec) -> tuple[list[tuple[str, Any]], Any]:
    """The training clients of the spec's real data, and their data shape."""
    from onion_fl.data.roles import split_subjects
    from onion_fl.experiment.runner import load_data

    subjects, _ = load_data(SimpleNamespace(data=spec.data, paths=spec.paths))
    split = split_subjects(subjects, spec.data.roles)
    clients = [(c.id, c.train) for c in split.clients]
    if not clients:
        raise ConfigError("data: no training subjects")
    return clients, clients[0][1].shape


def _workloads(spec: CalibrationSpec, shape: Any) -> dict[str, dict[str, Any]]:
    from onion_fl.learning.model import models, state_arrays
    from onion_fl.learning.trainers import trainers
    from onion_fl.roles.policies import create

    out = {}
    for w in spec.workloads:
        model = create(models, w.model).build([shape], seed=spec.seed)
        shapes = {k: list(np.shape(v)) for k, v in state_arrays(model).items()}
        params = getattr(create(trainers, w.trainer), "params", None)
        batch = getattr(params, "batch_size", None)
        wid, flops = workload_key(shapes, batch)
        if wid in out:
            raise ConfigError(f"workloads: {w.model} and another are the same workload")
        out[wid] = Workload(
            model=w.model,
            trainer=w.trainer,
            batch_size=batch,
            shapes=shapes,
            flops_per_sample=flops,
        ).model_dump(mode="json")
    return out


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def environment(image: str) -> dict[str, Any]:
    """Where the measurements were taken."""
    from onion_fl.experiment.emulated import pins

    info = _docker(
        "info",
        "--format",
        "{{json .ServerVersion}} {{json .NCPU}} {{json .MemTotal}} "
        "{{json .KernelVersion}} {{json .OperatingSystem}}",
    )
    fields = ["docker", "ncpu", "memory", "kernel", "os"]
    values = info.stdout.split(" ", 4) if info.returncode == 0 else []
    image_id = _docker("image", "inspect", "-f", "{{.Id}}", image).stdout.strip()
    return {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "docker": {k: json.loads(v) for k, v in zip(fields, values, strict=False)},
        "image": {"name": image, "id": image_id, "pins": pins()},
    }


def measure(
    spec_path: str | Path, out: str | Path, params: Mapping[str, Any] | None = None
) -> Path:
    """Run the microbenchmarks of a spec in a fresh emulation; ``out`` is the
    session folder. ``params`` overrides the spec's ``emulation``."""
    from onion_fl.experiment.emulated import ensure_image, take_down, wait_ready

    spec = load_spec(spec_path)
    if params:
        spec = spec.model_copy(
            update={"emulation": spec.emulation.model_copy(update=dict(params))}
        )
    infra = _infrastructure(spec)
    known = infra.hosts()
    folder = Path(out)
    if folder.exists() and any(folder.iterdir()):
        raise ConfigError(f"{folder} is not empty; give a new session folder")
    for d in ("emulation", "jobs", "out"):
        (folder / d).mkdir(parents=True, exist_ok=True)
    clients, shape = _clients(spec)
    workloads = _workloads(spec, shape)
    hosts = set()
    data: dict[str, Any] = {"shape": shape}
    if spec.train is not None:
        need = len(spec.train.hosts) * spec.train.per_host
        if need > len(clients):
            raise ConfigError(f"train: {need} clients asked, {len(clients)} exist")
        for i, host in enumerate(spec.train.hosts):
            per = spec.train.per_host
            data[host] = clients[i * per : (i + 1) * per]
        hosts |= set(spec.train.hosts)
    if spec.aggregate is not None:
        hosts |= set(spec.aggregate.hosts)
    if spec.links is not None:
        hosts |= {h for pair in spec.links.pairs for h in pair}
    broker = infra.services.get("mqtt_broker")
    unknown = sorted(
        h for h in hosts | ({broker} if broker else set()) if h not in known
    )
    if unknown:
        raise ConfigError(f"unknown hosts {unknown}")
    with open(folder / "data.pkl", "wb") as f:
        pickle.dump(data, f)
    import onion_fl

    volumes = [
        {
            "type": "bind",
            "source": Path(onion_fl.__file__).parent.resolve().as_posix(),
            "target": f"{ROOT}/src/onion_fl",
            "read_only": True,
        },
        {
            "type": "bind",
            "source": folder.resolve().as_posix(),
            "target": SESSION,
            "read_only": False,
        },
    ]
    p = spec.emulation
    lay = layout(infra, sorted(hosts), broker, p.subnet)
    stamp = datetime.now(UTC).strftime("%Y%m%dt%H%M%S")
    emulation = compose_project(
        lay,
        project=f"{p.project}-cal-{stamp}",
        image=p.image,
        broker_image=p.broker_image,
        volumes=volumes,
        setup=(folder / "emulation").resolve().as_posix(),
    )
    session = _Session(emulation, p, folder / "emulation")
    cpusets = pin(emulation, spec.pin) if spec.pin else {}
    if spec.links is not None and spec.links.nodelay:
        broker_nodelay(emulation, session.folder)
    from onion_fl.experiment.emulated import _Plain

    (session.folder / "compose.yaml").write_text(
        yaml.dump(emulation.compose, Dumper=_Plain, sort_keys=False), encoding="utf-8"
    )
    for service, script in emulation.scripts.items():
        with open(
            session.folder / f"{service}.sh", "w", encoding="utf-8", newline="\n"
        ) as f:
            f.write(script)
    meta = {
        "spec": spec.model_dump(mode="json"),
        "infrastructure_id": infra.infrastructure_id,
        "hosts": {h: known[h][1] for h in sorted(hosts)},
        "broker": broker,
        "workloads": workloads,
        "clients": {
            h: [c for c, _ in data[h]] for h in (spec.train.hosts if spec.train else [])
        },
        "code": code_version(),
        "project": emulation.project,
        "shaping": emulation.shaping,
        "cpusets": cpusets,
    }
    ensure_image(p.image)
    meta["environment"] = environment(p.image)
    loads: list[dict[str, Any]] = []
    meta["load"] = loads
    started = time.time()
    try:
        up = subprocess.run(
            [*session.compose, "up", "--detach", "--quiet-pull"],
            capture_output=True,
            text=True,
        )
        if up.returncode != 0:
            raise RuntimeError(f"calibrate: compose up failed: {up.stderr}")
        wait_ready(session)
        _phases(spec, emulation, folder, workloads, known, broker, loads)
    finally:
        take_down(session)
    meta["started_at"] = datetime.fromtimestamp(started, UTC).isoformat()
    meta["duration_s"] = round(time.time() - started, 3)
    rows = []
    for name in sorted((folder / "out").glob("*.jsonl")):
        rows += [json.loads(x) for x in name.read_text(encoding="utf-8").splitlines()]
    with open(folder / MEASUREMENTS, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (folder / META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def pin(emulation: Any, spec: PinSpec) -> dict[str, str]:
    """Give every container of ``emulation`` its fixed CPUs (``PinSpec``);
    the cpuset of each service."""
    lay = emulation.layout
    by_host = spec.cpus if isinstance(spec.cpus, dict) else None
    free, sets = ([] if by_host is not None else list(spec.cpus)), {}
    for host in sorted(lay.hosts):
        need = lay.infra.profile_of(host).cpus or 1
        if by_host is not None:
            mine = by_host.get(host, [])
            if len(mine) != need:
                raise ConfigError(
                    f"pin.cpus: host {host} has {need} cpus in its profile, but "
                    f"pin gives it {len(mine)}"
                )
        elif len(free) < need:
            raise ConfigError(
                f"pin.cpus: {len(spec.cpus)} CPUs do not cover the hosts' cpus"
            )
        else:
            mine, free = free[:need], free[need:]
        sets[lay.services[host]] = ",".join(str(c) for c in mine)
    for site in lay.routers:
        sets[lay.services[site]] = spec.others
    if lay.broker is not None:
        sets["broker"] = spec.others
    for name, cpuset in sets.items():
        emulation.compose["services"][name]["cpuset"] = cpuset
    return sets


def broker_nodelay(emulation: Any, setup: Path) -> None:
    """Run the broker with ``set_tcp_nodelay true`` (its config in ``setup``)."""
    with open(setup / "mosquitto.conf", "w", encoding="utf-8", newline="\n") as f:
        f.write("listener 1883\nallow_anonymous true\nset_tcp_nodelay true\n")
    broker = emulation.compose["services"]["broker"]
    broker["volumes"] = [
        {
            "type": "bind",
            "source": setup.resolve().as_posix(),
            "target": SETUP,
            "read_only": True,
        }
    ]
    broker["command"] = ["mosquitto", "-c", f"{SETUP}/mosquitto.conf"]


WINDOWS_LOAD = (  # the whole machine: busy %, performance % (turbo) and MHz
    "$p = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation "
    "-Filter \"Name='_Total'\"; "
    '"$($p.PercentProcessorTime) $($p.PercentProcessorPerformance) '
    '$($p.ProcessorFrequency)"'
)


def host_load(emulation: Any, phase: str) -> dict[str, Any]:
    """How busy the machine and Docker's VM are, and how fast the CPU runs.

    Taken at the edges of each phase, not during it, so as not to load it;
    on Windows, the processor counters of the whole machine.
    """
    out: dict[str, Any] = {"phase": phase, "at": time.time()}
    if platform.system() == "Windows":
        try:
            ps = subprocess.run(
                ["powershell", "-NoProfile", "-Command", WINDOWS_LOAD],
                capture_output=True,
                text=True,
                timeout=60,
            )
            busy, performance, mhz = (float(x) for x in ps.stdout.split())
            out["machine"] = {
                "busy_percent": busy,
                "performance_percent": performance,
                "frequency_mhz": mhz,
            }
        except (OSError, ValueError, subprocess.SubprocessError):
            out["machine"] = None
    host = sorted(emulation.layout.hosts)[0]
    vm = _docker("exec", emulation.container(host), "cat", "/proc/loadavg")
    if vm.returncode == 0:
        out["vm_loadavg"] = [float(x) for x in vm.stdout.split()[:3]]
    return out


def _job(folder: Path, name: str, **job: Any) -> tuple[str, str]:
    """Write a job; its path and its output as the containers see them."""
    out = f"{SESSION}/out/{name}.jsonl"
    payload = {"out": out, "data": f"{SESSION}/data.pkl", **job}
    (folder / "jobs" / f"{name}.json").write_text(json.dumps(payload), "utf-8")
    return f"{SESSION}/jobs/{name}.json", out


def _exec(emulation: Any, host: str, job: str) -> subprocess.Popen:
    command = [
        "docker",
        "exec",
        emulation.container(host),
        "python",
        "-m",
        "onion_fl.experiment.microbench",
        job,
    ]
    return subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def _wait(procs: Mapping[str, subprocess.Popen], what: str) -> None:
    failed = {}
    for name, proc in procs.items():
        output, _ = proc.communicate()
        if proc.returncode != 0:
            failed[name] = output[-2000:]
    if failed:
        raise RuntimeError(f"calibrate: {what} failed: {failed}")


def _phases(
    spec: CalibrationSpec,
    emulation: Any,
    folder: Path,
    workloads: Mapping[str, Any],
    known: Mapping[str, tuple[str, str]],
    broker: str | None,
    loads: list[dict[str, Any]] | None = None,
) -> None:
    def load(phase: str) -> None:
        if loads is not None:
            loads.append(host_load(emulation, phase))

    order = list(workloads)
    if spec.train is not None:
        for k, wid in enumerate(order):
            w = spec.workloads[k]
            load(f"train {wid} before")
            procs, start_at = {}, time.time() + spec.lead
            for host in spec.train.hosts:
                job, _ = _job(
                    folder,
                    f"train-{k}-{host}",
                    kind="train",
                    host=host,
                    profile=known[host][1],
                    workload=wid,
                    model=w.model,
                    trainer=w.trainer,
                    jobs=spec.train.jobs,
                    seed=spec.seed,
                    start_at=start_at,
                )
                procs[host] = _exec(emulation, host, job)
            _wait(procs, f"training workload {wid}")
            load(f"train {wid} after")
    if spec.aggregate is not None:
        a = spec.aggregate
        wid = order[a.workload]
        load("aggregate before")
        procs, start_at = {}, time.time() + spec.lead
        for host in a.hosts:
            job, _ = _job(
                folder,
                f"aggregate-{host}",
                kind="aggregate",
                host=host,
                profile=known[host][1],
                workload=wid,
                model=spec.workloads[a.workload].model,
                updates=a.updates,
                seconds=a.seconds,
                batch=a.batch,
                seed=spec.seed,
                start_at=start_at,
            )
            procs[host] = _exec(emulation, host, job)
        _wait(procs, "aggregation")
        load("aggregate after")
    if spec.links is not None:
        if emulation.broker is None:
            raise ConfigError("links: the infrastructure has no mqtt_broker")
        x = spec.links
        load("links before")
        for i, (a, b) in enumerate(x.pairs):
            for j, (src, dst) in enumerate(((a, b), (b, a))):
                if j and src == dst:
                    continue
                name = f"link-{i}-{j}"
                topic = f"onionfl-calibration/{name}"
                common = {
                    "broker": emulation.broker,
                    "qos": x.qos,
                    "inbox": f"{topic}/inbox",
                    "ack": f"{topic}/ack",
                    "nodelay": x.nodelay,
                }
                ready = folder / "out" / f"{name}.ready"
                timeout = len(x.sizes) * x.repeats * (x.wait + x.gap) + 60
                echo, _ = _job(
                    folder,
                    f"{name}-echo",
                    kind="echo",
                    host=dst,
                    peer=src,
                    ready=f"{SESSION}/out/{name}.ready",
                    timeout=timeout,
                    **common,
                )
                listening = _exec(emulation, dst, echo)
                deadline = time.monotonic() + 120
                while not ready.exists():
                    if listening.poll() is not None or time.monotonic() > deadline:
                        _wait({dst: listening}, f"echo on {dst}")
                        raise RuntimeError(f"calibrate: echo on {dst} never ready")
                    time.sleep(0.2)
                sender, _ = _job(
                    folder,
                    f"{name}-send",
                    kind="send",
                    host=src,
                    peer=dst,
                    sizes=x.sizes,
                    repeats=x.repeats,
                    gap=x.gap,
                    wait=x.wait,
                    **common,
                )
                _wait({src: _exec(emulation, src, sender)}, f"sending {src} -> {dst}")
                _wait({dst: listening}, f"echo on {dst}")
        load("links after")


# --- fit -----------------------------------------------------------------------------


def read_session(folder: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folder = Path(folder)
    meta = json.loads((folder / META).read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (folder / MEASUREMENTS).read_text(encoding="utf-8").splitlines()
    ]
    return meta, rows


def concurrent(
    rows: Sequence[Mapping[str, Any]], warm: str, skip: int = 1
) -> list[Mapping[str, Any]]:
    """The rows taken while every host of the phase was busy, warm-ups left out.

    ``warm`` names the field whose first ``skip`` values per host (and per
    ``updates`` for an aggregation) are the warm-up.
    """
    hosts: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:  # a host warming up is busy too
        hosts.setdefault(row["host"], []).append(row)
    if not hosts:
        return []
    lo = max(min(r["t0"] for r in rs) for rs in hosts.values())
    hi = min(max(r["t1"] for r in rs) for rs in hosts.values())
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["host"], row.get("updates")), []).append(row)
    kept = []
    for group in groups.values():
        warmup = sorted({r[warm] for r in group})[:skip]
        kept += [r for r in group if r[warm] not in warmup]
    return [r for r in kept if r["t0"] >= lo and r["t1"] <= hi]


def _percentiles(draws: Iterable[float]) -> tuple[float, float]:
    values = np.asarray([d for d in draws if np.isfinite(d)], dtype=float)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def ratio(num: Sequence[float], den: Sequence[float], seed: int = 0) -> Estimate:
    """Σnum / Σden with a bootstrap interval over the rows."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(BOOTSTRAP, len(num)))
    low, high = _percentiles(num[idx].sum(1) / den[idx].sum(1))
    return Estimate(value=float(num.sum() / den.sum()), low=low, high=high, n=len(num))


def mean(values: Sequence[float], seed: int = 0) -> Estimate:
    return ratio(values, [1.0] * len(values), seed)


def _line(sizes: np.ndarray, delays: np.ndarray) -> tuple[float, float]:
    """``delay = a + c·size`` by least squares: (a, c)."""
    c, a = np.polyfit(sizes, delays, 1)
    return float(a), float(c)


def link_rows(
    rows: Sequence[Mapping[str, Any]], up_to: int | None = None
) -> dict[tuple[str, str], dict]:
    """Per directed pair: the delays by size (``up_to`` bytes at most) and how
    many messages were lost. The first message of a pair, the first on its
    connections, is a warm-up and is left out."""
    out: dict[tuple[str, str], dict] = {}
    first = {}
    for r in rows:
        if r["kind"] == "sent":
            key = (r["src"], r["dst"])
            first[key] = min(first.get(key, r["seq"]), r["seq"])
    arrived = {
        (r["src"], r["dst"], r["seq"]): r["t_recv"] - r["t_send"]
        for r in rows
        if r["kind"] == "link"
    }
    for r in rows:
        if r["kind"] != "sent" or r["seq"] == first[(r["src"], r["dst"])]:
            continue
        if up_to is not None and r["size"] > up_to:
            continue
        pair = out.setdefault((r["src"], r["dst"]), {"sizes": [], "delays": [], "n": 0})
        pair["n"] += 1
        delay = arrived.get((r["src"], r["dst"], r["seq"]))
        if delay is not None:
            pair["sizes"].append(r["size"])
            pair["delays"].append(delay)
    return out


def _wilson(lost: int, n: int) -> Estimate:
    z = 1.959964
    p = lost / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return Estimate(
        value=p, low=max(0.0, centre - half), high=min(1.0, centre + half), n=n
    )


def _links(
    infra: Infrastructure,
    broker: str | None,
    pairs: Mapping[tuple[str, str], dict],
    seed: int,
) -> tuple[dict[str, LinkCalibration], dict[str, Estimate], list[str]]:
    """Per physical link its fitted latency, bandwidths and loss (module doc)."""
    notes: list[str] = []
    transport: dict[str, Estimate] = {}
    crossings: dict[tuple[str, str], list[tuple[str, bool]]] = {}
    for (src, dst), data in sorted(pairs.items()):
        if src == dst:  # only the transport: a process to the broker and back
            latency, goodput = _boot_line(data, seed)
            transport["loopback_latency_s"] = latency
            transport["loopback_goodput_bps"] = goodput
            notes.append(
                f"{src} to itself through the broker: {latency.value * 1e3:.2f} ms, "
                "the transport's own one-way latency, which the links absorb"
            )
            continue
        crossings[(src, dst)] = route(infra, src, dst, broker)[1]
    keys = sorted(
        {k for hops in crossings.values() for k, _ in hops if k[:5] == "link:"}
    )
    if not keys:
        return {}, transport, notes
    order = sorted(crossings)
    design = np.array(
        [[sum(1 for k, _ in crossings[p] if k == key) for key in keys] for p in order],
        float,
    )
    if np.linalg.matrix_rank(design) < len(keys):
        raise ConfigError(
            f"links: the pairs measured cannot tell apart the latencies of {keys}; "
            "measure pairs that cross them in different combinations"
        )
    fixed = np.array(  # the LAN segments a path crosses keep their latency
        [
            sum(segment(infra, k).latency for k, _ in crossings[p] if k[:4] == "lan:")
            for p in order
        ]
    )
    # Each leg is sent at its narrowest segment: (hop, nominal rate) per leg.
    narrowest: dict[tuple[str, str], list[tuple[tuple[str, bool], float]]] = {}
    for p in order:
        narrowest[p] = []
        for leg in legs(crossings[p]):
            nominal = {}
            for k, forward in leg:
                s = segment(infra, k)  # nominal: nothing cites a calibration yet
                rate = s.bandwidth_up if forward else s.bandwidth_down
                if rate is not None:
                    nominal[(k, forward)] = rate
            if nominal:
                hop = min(nominal, key=lambda h, n=nominal: (n[h], h))
                narrowest[p].append((hop, nominal[hop]))
    unknown = sorted(
        {hop for p in order for hop, _ in narrowest[p] if hop[0][:5] == "link:"}
    )
    slope_design = np.array(
        [
            [sum(8 / r for h, r in narrowest[p] if h == u) for u in unknown]
            for p in order
        ]
    ).reshape(len(order), len(unknown))
    slope_fixed = np.array(  # a LAN bottleneck keeps its nominal rate
        [sum(8 / r for h, r in narrowest[p] if h not in unknown) for p in order]
    )
    if unknown and np.linalg.matrix_rank(slope_design) < len(unknown):
        raise ConfigError(
            f"links: the pairs measured cannot tell apart the bandwidths of "
            f"{unknown}; measure pairs whose legs they bottleneck in different "
            "combinations"
        )

    def fit(lines: Mapping[tuple[str, str], tuple[float, float]]) -> dict[str, Any]:
        a = np.array([lines[p][0] for p in order]) - fixed
        latency = np.linalg.lstsq(design, a, rcond=None)[0]
        x: dict[tuple[str, bool], float] = {}
        if unknown:
            c = np.array([lines[p][1] for p in order]) - slope_fixed
            solved = np.linalg.lstsq(slope_design, c, rcond=None)[0]
            x = {h: float(v) for h, v in zip(unknown, solved, strict=True) if v > 0}
        return {
            "latency": dict(zip(keys, latency, strict=True)),
            "x": x,
            "overall": float(np.mean(list(x.values()))) if x else 1.0,
        }

    rng = np.random.default_rng(seed)
    point = fit(
        {
            p: _line(
                np.array(pairs[p]["sizes"], float), np.array(pairs[p]["delays"], float)
            )
            for p in order
        }
    )
    for hop in unknown:
        if hop not in point["x"]:
            notes.append(f"{hop}: no positive fit of its bandwidth; it takes the mean")
    draws = []
    for _ in range(BOOTSTRAP // 4):
        lines = {}
        for p in order:
            sizes = np.array(pairs[p]["sizes"], float)
            delays = np.array(pairs[p]["delays"], float)
            idx = rng.integers(0, len(sizes), len(sizes))
            if len(set(sizes[idx])) < 2:
                idx = np.arange(len(sizes))
            lines[p] = _line(sizes[idx], delays[idx])
        draws.append(fit(lines))
    out = {}
    for key in keys:
        link = segment(infra, key)  # nominal: nothing cites a calibration yet
        n = sum(
            len(pairs[p]["delays"])
            for p in order
            if any(k == key for k, _ in crossings[p])
        )
        value = float(point["latency"][key])
        if value < 0:  # the paths disagree with the sum of their links
            notes.append(f"{key}: fitted latency {value * 1e3:.2f} ms, taken as 0")
        latency = Estimate(
            value=max(value, 0.0),
            **dict(
                zip(
                    ("low", "high"),
                    _percentiles(d["latency"][key] for d in draws),
                    strict=True,
                )
            ),
            n=n,
        )
        fitted: dict[str, Any] = {"latency": latency}
        derived = []
        for side, forward in (("bandwidth_up", True), ("bandwidth_down", False)):
            nominal = getattr(link, side)
            if nominal is None:
                continue
            hop = (key, forward)
            if hop not in point["x"]:
                derived.append(side)

            def x(d: Mapping[str, Any], hop: tuple[str, bool] = hop) -> float:
                return d["x"].get(hop, d["overall"])

            low, high = _percentiles(nominal / x(d) for d in draws)
            fitted[side] = Estimate(value=nominal / x(point), low=low, high=high, n=n)
        losses = []
        for p in order:
            if any(k == key for k, _ in crossings[p]):
                lost = pairs[p]["n"] - len(pairs[p]["delays"])
                losses.append((lost / pairs[p]["n"], lost, pairs[p]["n"]))
        _, lost, total = min(losses)
        fitted["loss"] = _wilson(lost, total)
        paths = [
            f"{s}>{d}" for s, d in order if any(k == key for k, _ in crossings[(s, d)])
        ]
        out[link_key(link.between)] = LinkCalibration(
            **fitted, pairs=paths, derived=derived
        )
    return out, transport, notes


def _boot_line(data: Mapping[str, Any], seed: int) -> tuple[Estimate, Estimate]:
    """A path's latency (s) and goodput (bit/s) with their intervals."""
    sizes, delays = np.array(data["sizes"], float), np.array(data["delays"], float)
    a, c = _line(sizes, delays)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(sizes), len(sizes))
        if len(set(sizes[idx])) >= 2:
            draws.append(_line(sizes[idx], delays[idx]))
    low, high = _percentiles(d[0] for d in draws)
    latency = Estimate(value=a, low=low, high=high, n=len(sizes))
    low, high = _percentiles(8.0 / d[1] for d in draws if d[1] > 0)
    goodput = Estimate(
        value=8.0 / c if c > 0 else np.inf, low=low, high=high, n=len(sizes)
    )
    return latency, goodput


def session_values(
    meta: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    infra: Infrastructure,
    spec: CalibrationSpec,
) -> dict[str, Any]:
    """What one session measured of every calibrated quantity, each with its
    interval within the session; ``infra`` as the session measured it."""
    out: dict[str, Any] = {"train": {}, "aggregate": {}, "paths": {}, "loopback": {}}
    warmup = spec.train.warmup if spec.train else 1
    train = [r for r in rows if r["kind"] == "train"]
    for wid in meta["workloads"]:
        mine = concurrent([r for r in train if r["workload"] == wid], "job", warmup)
        for name in sorted({r["profile"] for r in mine}):
            these = [r for r in mine if r["profile"] == name]
            samples = [r["samples"] for r in these]
            walls = [r["wall_s"] for r in these]
            out["train"][(name, wid)] = {
                "rate": ratio(samples, walls, spec.seed),
                "host": these[0]["host"],
                "samples": samples,
                "walls": walls,
            }
    combined = concurrent([r for r in rows if r["kind"] == "aggregate"], "repeat")
    for name in sorted({r["profile"] for r in combined}):
        these = [r for r in combined if r["profile"] == name]
        sizes = [r["bytes"] for r in these]
        walls = [r["wall_s"] for r in these]
        out["aggregate"][name] = {
            "rate": ratio(sizes, walls, spec.seed),
            "host": these[0]["host"],
            "samples": sizes,
            "walls": walls,
        }
    up_to = spec.links.fit_up_to if spec.links else None
    out["links"], out["transport"], out["notes"] = _links(
        infra, meta["broker"], link_rows(rows, up_to), spec.seed
    )
    for (src, dst), data in sorted(link_rows(rows).items()):
        sizes, delays = np.array(data["sizes"], float), np.array(data["delays"], float)
        where = "loopback" if src == dst else "paths"
        for size in sorted(set(sizes.tolist())):
            out[where][(src, dst, int(size))] = mean(
                delays[sizes == size].tolist(), spec.seed
            )
    return out


def between(estimates: Sequence[Estimate]) -> Estimate:
    """One value from the values of k sessions: their geometric mean and the
    95 % prediction interval of one more session's value (module doc); a
    single session keeps its own interval. Values that are not all positive
    are taken on their own scale."""
    if len(estimates) == 1:
        return estimates[0]
    from scipy import stats

    values = np.array([e.value for e in estimates], float)
    k = len(values)
    logs = bool(np.all(values > 0))
    v = np.log(values) if logs else values
    m, sd = float(v.mean()), float(v.std(ddof=1))
    half = float(stats.t.ppf(0.975, k - 1)) * sd * math.sqrt(1 + 1 / k)

    def back(x: float) -> float:
        return float(np.exp(x)) if logs else float(x)

    return Estimate(
        value=back(m),
        low=back(m - half),
        high=back(m + half),
        n=sum(e.n for e in estimates),
        sessions=values.tolist(),
    )


def path_key(src: str, dst: str, size: int) -> str:
    return f"{src}>{dst}@{size}"


def fit(
    sessions: str | Path | Sequence[str | Path], out: str | Path = "calibrations"
) -> Path:
    """The signed calibration of one or more sessions of one spec (module doc)."""
    if isinstance(sessions, str | Path):
        sessions = [sessions]
    folders = [Path(f) for f in sessions]
    read = [read_session(f) for f in folders]
    first = read[0][0]
    for folder, (meta, _) in zip(folders, read, strict=True):
        for field in ("spec", "infrastructure_id", "hosts", "workloads", "broker"):
            if meta[field] != first[field]:
                raise ConfigError(
                    f"{folder}: its {field} differs from that of {folders[0]}; a "
                    "calibration fits sessions of one spec"
                )
    spec = CalibrationSpec.model_validate(first["spec"])
    infra = _infrastructure(spec)
    if infra.infrastructure_id != first["infrastructure_id"]:
        raise ConfigError(
            f"{spec.infrastructure} changed since the session measured it"
        )
    raws = [(f / MEASUREMENTS).read_bytes() for f in folders]
    digests = [measurements_digest(raw) for raw in raws]
    if len(set(digests)) < len(digests):
        raise ConfigError("fit: the same session is given twice")
    values = [session_values(meta, rows, infra, spec) for meta, rows in read]

    def every(kind: str, key: Any) -> list[Any]:
        found = [v[kind].get(key) for v in values]
        if any(x is None for x in found):
            raise ConfigError(f"fit: {kind} {key} was not measured in every session")
        return found

    profiles: dict[str, dict[str, Any]] = {}
    for name in first["hosts"].values():
        p = infra.profiles[name]
        profiles.setdefault(name, {"cpus": p.cpus, "memory": p.memory, "train": {}})
    for name, wid in values[0]["train"]:
        rates = [x["rate"] for x in every("train", (name, wid))]
        profiles[name]["train"][wid] = between(rates)
    for name in values[0]["aggregate"]:
        rates = [x["rate"] for x in every("aggregate", name)]
        profiles[name]["aggregate"] = between(rates)
    links = {}
    for key, link in values[0]["links"].items():
        mine = every("links", key)
        fitted: dict[str, Any] = {
            "latency": between([x.latency for x in mine]),
            "pairs": link.pairs,
            "derived": link.derived,
        }
        for side in ("bandwidth_up", "bandwidth_down"):
            if getattr(link, side) is not None:
                fitted[side] = between([getattr(x, side) for x in mine])
        if len(mine) == 1:
            fitted["loss"] = link.loss
        else:  # the sessions' messages pooled
            lost = sum(round(x.loss.value * x.loss.n) for x in mine)
            fitted["loss"] = _wilson(lost, sum(x.loss.n for x in mine))
        links[key] = LinkCalibration(**fitted)
    transport = {
        key: between(every("transport", key)) for key in values[0]["transport"]
    }
    up_to = spec.links.fit_up_to if spec.links else None
    paths = {
        path_key(*key): between(every("paths", key))
        for key in values[0]["paths"]
        if up_to is None or key[2] <= up_to
    }
    notes = [
        note if len(values) == 1 else f"session {i}: {note}"
        for i, v in enumerate(values)
        for note in v["notes"]
    ]
    if len(raws) == 1:
        raw = raws[0]
    else:  # every row says which session it comes from
        raw = "".join(
            json.dumps({**row, "session": i}, sort_keys=True) + "\n"
            for i, (_, rows) in enumerate(read)
            for row in rows
        ).encode("utf-8")
    calibration = Calibration(
        backend="emulated",
        infrastructure_id=first["infrastructure_id"],
        created_at=first["started_at"],
        code=first["code"],
        environment=first["environment"],
        spec=first["spec"] | {"hosts": first["hosts"], "clients": first["clients"]},
        measurements_sha256=measurements_digest(raw),
        workloads=first["workloads"],
        profiles={
            name: ProfileCalibration(**p)
            for name, p in profiles.items()
            if p["train"] or p.get("aggregate")
        },
        links=links,
        transport=transport,
        notes=notes,
        sessions=[
            {
                "started_at": meta["started_at"],
                "duration_s": meta.get("duration_s"),
                "code": meta["code"],
                "measurements_sha256": digest,
                "cpusets": meta.get("cpusets", {}),
                "load": meta.get("load", []),
            }
            for (meta, _), digest in zip(read, digests, strict=True)
        ],
        paths=paths,
        valid_up_to=up_to,
    )
    return write_calibration(out, calibration, raw)


def calibrated_infrastructure(
    infra_path: str | Path, calibration: Calibration
) -> dict[str, Any]:
    """The infrastructure as written, citing ``calibration`` for every profile
    and link it measured (their rates removed: they come from it)."""
    raw = yaml.safe_load(Path(infra_path).read_text(encoding="utf-8"))
    cid = calibration.calibration_id
    for name, profile in raw["profiles"].items():
        if name in calibration.profiles:
            profile.pop("train_rate", None)
            profile.pop("aggregate_rate", None)
            profile["calibration"] = cid
    for link in raw.get("network", {}).get("links", []):
        if link_key(link["between"]) in calibration.links:
            link["calibration"] = cid
    return raw


# --- check ---------------------------------------------------------------------------


def _profile_estimate(
    infra: Infrastructure, host: str, kind: str, workload: str | None = None
) -> Estimate | None:
    """The calibrated estimate of a host's rate, with its interval."""
    profile = infra.profile_of(host)
    if not profile.calibration:
        return None
    name = infra.hosts()[host][1]
    measured = infra._calibrations[profile.calibration].profiles.get(name)
    if measured is None:
        return None
    return measured.aggregate if kind == "aggregate" else measured.train.get(workload)


def _tally(entries: Sequence[Mapping[str, Any]], *fields: str) -> dict[str, int]:
    return {"quantities": len(entries)} | {
        f: sum(1 for e in entries if e[f]) for f in fields
    }


def check(
    session: str | Path, infrastructure: str | Path, calibrations: str | Path
) -> dict[str, Any]:
    """What the simulator predicts for the measurements of ``session`` with the
    calibrated ``infrastructure``, and whether the session's values fall inside
    the calibration's intervals.

    A quantity passes when the value this session measured is inside the
    calibrated interval (``inside``); a path also needs the simulator's
    prediction inside it (``predicted_inside``), since its prediction is not
    a calibrated value but the sum of the links'. ``inside_session_ci`` says
    whether the prediction is inside this session's own interval, the
    criterion of a calibration fitted on one session.
    """
    from onion_fl.runtime.devices import compute_models
    from onion_fl.runtime.network import LinkProfile

    meta, rows = read_session(session)
    spec = CalibrationSpec.model_validate(meta["spec"])
    nominal = _infrastructure(spec)
    if nominal.infrastructure_id != meta["infrastructure_id"]:
        raise ConfigError(
            f"{spec.infrastructure} changed since the session measured it"
        )
    infra = load_infrastructure(infrastructure, calibrations)
    cids = cited(infra)
    loaded = {cid: load_calibration(calibrations, cid) for cid in cids}
    digest = measurements_digest((Path(session) / MEASUREMENTS).read_bytes())
    for cid, calibration in loaded.items():  # never judge on what it was fitted on
        if digest in calibration.digests():
            raise ConfigError(
                f"{session} is a session calibration {cid[:12]} was fitted on; "
                "check it on another session"
            )
    mine = session_values(meta, rows, nominal, spec)
    out: dict[str, Any] = {
        "session": {
            "started_at": meta["started_at"],
            "code": meta["code"],
            "infrastructure_id": meta["infrastructure_id"],
            "measurements_sha256": digest,
            "load": meta.get("load", []),
        },
        "calibrations": cids,
        "infrastructure_id": infra.infrastructure_id,
        "train": [],
        "aggregate": [],
        "links": [],
        "paths": [],
        "loopback": [],
        "flops_estimates": [],
    }

    def compute(kind: str, name: str, wid: str | None, x: Mapping[str, Any]) -> dict:
        host = x["host"]
        if kind == "train":
            flops = meta["workloads"][wid]["flops_per_sample"]
            rate, note = train_rate(infra, host, (wid, flops))
        else:
            rate, note = aggregate_rate(infra, host), None
        model = compute_models.create(
            "samples_per_second", {"samples_per_second": rate}
        )
        predicted = sum(model.duration(n, 0.0) for n in x["samples"])
        measured = x["rate"]
        calibrated = _profile_estimate(infra, host, kind, wid)
        return {
            "profile": name,
            **({"workload": wid} if wid else {}),
            "measured_n": len(x["walls"]),
            "predicted_rate": rate,
            "estimated": note is not None,
            "calibrated_rate": None if calibrated is None else calibrated.model_dump(),
            "measured_rate": measured.model_dump(),
            "inside": calibrated is not None and calibrated.covers(measured.value),
            "inside_session_ci": measured.covers(rate),
            "relative_error": predicted / sum(x["walls"]) - 1,
        }

    for (name, wid), x in mine["train"].items():
        out["train"].append(compute("train", name, wid, x))
    for name, x in mine["aggregate"].items():
        out["aggregate"].append(compute("aggregate", name, None, x))
    by_profile: dict[str, dict[str, Any]] = {}
    for (name, wid), x in mine["train"].items():
        by_profile.setdefault(name, {})[wid] = x["rate"]
    for name, measured in by_profile.items():
        for wid, m in measured.items():
            flops = meta["workloads"][wid]["flops_per_sample"]
            for other, m2 in measured.items():
                if other == wid:
                    continue
                flops2 = meta["workloads"][other]["flops_per_sample"]
                estimate = m2.value * flops2 / flops
                out["flops_estimates"].append(
                    {
                        "profile": name,
                        "workload": wid,
                        "from_workload": other,
                        "estimated_rate": estimate,
                        "measured_rate": m.model_dump(),
                        "inside": m.covers(estimate),
                        "relative_error": estimate / m.value - 1,
                    }
                )
    cited_links = {
        link_key(x.between): loaded[x.calibration].links.get(link_key(x.between))
        for x in infra.network.links
        if x.calibration
    }
    for key, measured in sorted(mine["links"].items()):
        calibrated = cited_links.get(key)
        if calibrated is None:
            continue
        for field in ("latency", "bandwidth_up", "bandwidth_down", "loss"):
            c, m = getattr(calibrated, field), getattr(measured, field)
            if c is None or m is None or field in calibrated.derived:
                continue
            out["links"].append(
                {
                    "link": key,
                    "field": field,
                    "calibrated": c.model_dump(),
                    "measured": m.model_dump(),
                    "inside": c.covers(m.value),
                }
            )
    valid = min(
        (c.valid_up_to for c in loaded.values() if c.valid_up_to is not None),
        default=spec.links.fit_up_to if spec.links else None,
    )
    for (src, dst, size), measured in sorted(mine["paths"].items()):
        _, profile = effective(infra, src, dst, meta["broker"])
        link = LinkProfile(**profile)
        predicted = link.latency_s + link.transmission_s(size, "up")
        calibrated = next(
            (
                c.paths[path_key(src, dst, size)]
                for c in loaded.values()
                if c.paths and path_key(src, dst, size) in c.paths
            ),
            None,
        )
        out["paths"].append(
            {
                "src": src,
                "dst": dst,
                "size": size,
                "fitted_range": valid is None or size <= valid,
                "predicted_s": predicted,
                "calibrated_s": None if calibrated is None else calibrated.model_dump(),
                "measured_s": measured.model_dump(),
                "inside": calibrated is not None and calibrated.covers(measured.value),
                "predicted_inside": calibrated is not None
                and calibrated.covers(predicted),
                "inside_session_ci": measured.covers(predicted),
                "relative_error": predicted / measured.value - 1,
            }
        )
    for (src, _, size), measured in sorted(mine["loopback"].items()):
        out["loopback"].append(
            {"host": src, "size": size, "measured_s": measured.model_dump()}
        )
    in_range = [p for p in out["paths"] if p["fitted_range"]]
    for p in in_range:
        p["passed"] = p["inside"] and p["predicted_inside"]
    out["acceptance"] = {
        "train": _tally(out["train"], "inside", "inside_session_ci"),
        "aggregate": _tally(out["aggregate"], "inside", "inside_session_ci"),
        "links": _tally(out["links"], "inside"),
        "paths": _tally(
            in_range, "inside", "predicted_inside", "passed", "inside_session_ci"
        ),
    }
    out["acceptance"]["passed"] = all(
        e["inside"] for kind in ("train", "aggregate", "links") for e in out[kind]
    ) and all(p["passed"] for p in in_range)
    return out
