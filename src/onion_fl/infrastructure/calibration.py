from __future__ import annotations

"""Calibrations: measured rates and links that an infrastructure cites (I1 §7, §13).

A calibration is a signed folder ``calibrations/<calibration_id>/``:

- ``calibration.json``: what was fitted, with a 95 % interval for each value,
  the environment, the code version, the specification of the
  microbenchmarks and the SHA-256 of the raw measurements;
- ``measurements.jsonl``: every measurement, one per line (with the
  ``session`` it comes from, when it was fitted on several).

``calibration_id`` is the SHA-256 of the canonical ``calibration.json``
without the id itself, so it signs the measurements too. The JSON holds the
fields that were given, so that a calibration written before a field existed
keeps its id. A profile or a link cites one by id::

    profiles:
      edge_device: {cpus: 1, memory: 2GB, calibration: 3f2a…}   # rates measured
    network:
      links:
        - {between: [field, fog_a], latency: 10ms, bandwidth: 50Mbps,
           calibration: 3f2a…}

- **Profiles.** The training rate is measured per pair (profile, workload): a
  workload is the shapes of the model's parameters and the trainer's batch
  size (``workload_key``). A workload the calibration did not measure on that
  profile is estimated by scaling the rate of the measured one nearest in
  FLOPs per sample by the ratio of their FLOPs, and the run says so
  (``estimated``). The ``cpus`` and ``memory`` of the profile stay given, and
  must be those the calibration was measured with.
- **Links.** The nominal fields of a link are what the emulator shapes; the
  calibrated ones, measured with the transport over that link, are what the
  simulation uses (``link_fields``).

The package only reads calibrations; ``onion_fl calibrate`` writes them.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from onion_fl.infrastructure.model import (
    Infrastructure,
    InfrastructureError,
    Link,
    LinkFields,
    Strict,
)

CALIBRATION = "calibration.json"
MEASUREMENTS = "measurements.jsonl"


class Estimate(Strict):
    """A fitted value and its 95 % interval, from ``n`` measurements.

    With ``sessions``, the value each session it was fitted on gave, the
    interval is the 95 % prediction interval of what one more session
    measures (``between sessions``, see ``onion_fl.experiment.calibrate``);
    without, it is a bootstrap within one session.
    """

    value: float
    low: float
    high: float
    n: int = Field(ge=1)
    sessions: list[float] | None = None

    def covers(self, x: float) -> bool:
        return self.low <= x <= self.high


class Workload(Strict):
    """What a training rate was measured on."""

    model: Any  # the model reference as configured
    trainer: Any  # the trainer reference as configured
    batch_size: int | None
    shapes: dict[str, list[int]]
    flops_per_sample: float = Field(gt=0)


class ProfileCalibration(Strict):
    cpus: int | None = None
    memory: float | None = None
    train: dict[str, Estimate] = Field(
        default_factory=dict, description="workload -> muestras por segundo"
    )
    aggregate: Estimate | None = Field(
        None, description="Bytes de actualización combinados por segundo"
    )


class LinkCalibration(Strict):
    latency: Estimate  # one way, s
    bandwidth_up: Estimate | None = None  # bit/s, between[0] -> between[1]
    bandwidth_down: Estimate | None = None
    loss: Estimate
    pairs: list[str] = Field(default_factory=list)  # the measured paths it comes from
    derived: list[str] = Field(default_factory=list)  # fields not measured on it


class Calibration(Strict):
    backend: str
    infrastructure_id: str  # the infrastructure measured, as it was written
    created_at: str
    code: dict[str, Any]
    environment: dict[str, Any]
    spec: dict[str, Any]
    measurements_sha256: str
    workloads: dict[str, Workload] = Field(default_factory=dict)
    profiles: dict[str, ProfileCalibration] = Field(default_factory=dict)
    links: dict[str, LinkCalibration] = Field(default_factory=dict)  # "a|b"
    transport: dict[str, Estimate] = Field(default_factory=dict)  # reported only
    notes: list[str] = Field(default_factory=list)
    # fitted on several sessions: each one's start, code, load and digest
    sessions: list[dict[str, Any]] | None = None
    # "src>dst@size": the one-way delay of a measured path, as the fit
    # sessions measured it; the path model holds up to ``valid_up_to`` bytes
    paths: dict[str, Estimate] | None = None
    valid_up_to: int | None = None

    def document(self) -> dict[str, Any]:
        """The JSON of the calibration: the fields that were given."""
        return self.model_dump(mode="json", exclude_unset=True)

    @property
    def calibration_id(self) -> str:
        return calibration_id(self.document())

    def digests(self) -> set[str]:
        """The SHA-256 of every session's measurements it was fitted on."""
        out = {self.measurements_sha256}
        return out | {x["measurements_sha256"] for x in self.sessions or []}


def calibration_id(document: Mapping[str, Any]) -> str:
    body = {k: v for k, v in document.items() if k != "calibration_id"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def measurements_digest(raw: bytes) -> str:
    """SHA-256 of the measurements with LF line ends, so that a checkout that
    turns them into CRLF keeps the signature."""
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def link_key(between: Sequence[str]) -> str:
    return f"{between[0]}|{between[1]}"


def workload_key(
    shapes: Mapping[str, Sequence[int]], batch_size: int | None
) -> tuple[str, float]:
    """A workload's id and its FLOPs per training sample.

    ponytail: 6 FLOPs per parameter and sample (2 forward, 4 backward), right
    for dense layers; only ratios are used, so the constant cancels.
    """
    canonical = {
        "shapes": {k: [int(d) for d in v] for k, v in sorted(shapes.items())},
        "batch_size": batch_size,
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    flops = 6.0 * sum(math.prod(v) for v in canonical["shapes"].values())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], flops


def write_calibration(
    folder: str | Path, calibration: Calibration, measurements: bytes
) -> Path:
    """Write ``<folder>/<calibration_id>/`` and return it."""
    digest = measurements_digest(measurements)
    if digest != calibration.measurements_sha256:
        raise InfrastructureError("calibration: the measurements are not those signed")
    document = calibration.document()
    cid = calibration_id(document)
    out = Path(folder) / cid
    out.mkdir(parents=True, exist_ok=True)
    (out / MEASUREMENTS).write_bytes(measurements)
    text = json.dumps({"calibration_id": cid, **document}, indent=2, sort_keys=True)
    with open(out / CALIBRATION, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return out


def load_calibration(folder: str | Path, cid: str) -> Calibration:
    """The calibration ``cid`` of ``folder``, after checking its signature."""
    path = Path(folder) / cid / CALIBRATION
    if not path.exists():
        raise InfrastructureError(f"calibration {cid} not found in {Path(folder)}")
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        calibration = Calibration.model_validate(
            {k: v for k, v in document.items() if k != "calibration_id"}
        )
    except ValidationError as exc:
        raise InfrastructureError(f"calibration {cid}: {exc}") from None
    if calibration.calibration_id != cid or document.get("calibration_id") != cid:
        raise InfrastructureError(f"calibration {cid}: its content does not match")
    raw = (path.parent / MEASUREMENTS).read_bytes()
    if measurements_digest(raw) != calibration.measurements_sha256:
        raise InfrastructureError(f"calibration {cid}: its measurements changed")
    return calibration


def cited(infra: Infrastructure) -> list[str]:
    """The calibrations an infrastructure cites."""
    ids = {p.calibration for p in infra.profiles.values()}
    ids |= {x.calibration for x in infra.network.links}
    return sorted(i for i in ids if i)


def bind(infra: Infrastructure, folder: str | Path) -> Infrastructure:
    """Load the calibrations ``infra`` cites and check they fit it."""
    loaded = {cid: load_calibration(folder, cid) for cid in cited(infra)}
    for name, profile in infra.profiles.items():
        if not profile.calibration:
            continue
        calibration = loaded[profile.calibration]
        mine = calibration.profiles.get(name)
        if mine is None:
            raise InfrastructureError(
                f"profiles.{name}: calibration {profile.calibration[:12]} did not "
                f"measure it; it measured {sorted(calibration.profiles)}"
            )
        for limit in ("cpus", "memory"):
            given, measured = getattr(profile, limit), getattr(mine, limit)
            if given != measured:
                raise InfrastructureError(
                    f"profiles.{name}: {limit} {given} but calibration "
                    f"{profile.calibration[:12]} measured it with {measured}"
                )
    for i, link in enumerate(infra.network.links):
        if link.calibration and link_key(link.between) not in (
            loaded[link.calibration].links
        ):
            raise InfrastructureError(
                f"network.links[{i}]: calibration {link.calibration[:12]} did not "
                f"measure {list(link.between)}"
            )
    infra._calibrations = loaded
    return infra


def _loaded(infra: Infrastructure, cid: str, where: str) -> Calibration:
    try:
        return infra._calibrations[cid]
    except KeyError:
        raise InfrastructureError(
            f"{where} cites calibration {cid[:12]}, which is not loaded; "
            "load the infrastructure with its calibrations folder"
        ) from None


def train_rate(
    infra: Infrastructure, host: str, workload: tuple[str, float] | None
) -> tuple[float | None, dict[str, Any] | None]:
    """Samples per second of ``workload`` on ``host``, and a note when it is
    estimated by FLOPs rather than measured."""
    profile = infra.profile_of(host)
    if not profile.calibration:
        return profile.train_rate, None
    name = infra.hosts()[host][1]
    measured = _loaded(infra, profile.calibration, f"profiles.{name}")
    rates = measured.profiles[name].train
    if not rates:
        return None, None
    if workload is None:
        raise InfrastructureError(f"profiles.{name}: a calibrated rate needs a model")
    wid, flops = workload
    if wid in rates:
        return rates[wid].value, None
    nearest = min(
        rates,
        key=lambda w: (
            abs(math.log(measured.workloads[w].flops_per_sample / flops)),
            w,
        ),
    )
    rate = rates[nearest].value * measured.workloads[nearest].flops_per_sample / flops
    note = {
        "host": host,
        "profile": name,
        "calibration": profile.calibration,
        "workload": wid,
        "from_workload": nearest,
        "train_rate": rate,
    }
    return rate, note


def aggregate_rate(infra: Infrastructure, host: str) -> float | None:
    profile = infra.profile_of(host)
    if not profile.calibration:
        return profile.aggregate_rate
    name = infra.hosts()[host][1]
    measured = _loaded(infra, profile.calibration, f"profiles.{name}")
    estimate = measured.profiles[name].aggregate
    return None if estimate is None else estimate.value


def speed(infra: Infrastructure, host: str) -> float:
    """How fast a host trains, to order devices: its ``train_rate``, or the
    FLOPs per second of its fastest calibrated workload; infinite if free."""
    profile = infra.profile_of(host)
    if not profile.calibration:
        return profile.train_rate or math.inf
    name = infra.hosts()[host][1]
    measured = _loaded(infra, profile.calibration, f"profiles.{name}")
    rates = measured.profiles[name].train
    return max(
        (r.value * measured.workloads[w].flops_per_sample for w, r in rates.items()),
        default=math.inf,
    )


def link_fields(infra: Infrastructure, link: Link) -> LinkFields:
    """What the simulation uses of a physical link: its calibrated latency,
    bandwidth and loss when it cites a calibration, its own fields otherwise."""
    if not link.calibration:
        return link
    measured = _loaded(infra, link.calibration, f"link {list(link.between)}")
    fitted = measured.links[link_key(link.between)]
    update: dict[str, Any] = {"latency": fitted.latency.value}
    update["loss"] = fitted.loss.value
    for side in ("bandwidth_up", "bandwidth_down"):
        estimate = getattr(fitted, side)
        if estimate is not None:
            update[side] = estimate.value
    return link.model_copy(update=update)
