from __future__ import annotations

"""FedScale's client device traces as Onion-FL profiles (G9, issue #253).

FedScale (Apache-2.0) publishes two traces in
``benchmark/dataset/data/device_info/`` of its repository:

- ``client_device_capacity``: client id -> ``{computation, communication}``,
  ``computation`` the inference latency of a model in **ms per sample** and
  ``communication`` the client's bandwidth in **kbit/s** (FedScale's
  ``ClientMetadata.get_completion_time``).
- ``client_behave_trace``: client id -> ``{active, inactive, finish_time, ...}``,
  in **seconds**: the client is up inside ``[active[i], inactive[i]]`` of a
  trace that repeats every ``finish_time`` (FedScale's ``is_active``).

Both are pickles, and unpickling runs code. :func:`import_fedscale` reads them
once with an unpickler that builds only dicts, lists, numbers, strings and
numpy float scalars (decoded from their bytes), and writes plain files that the
runs read: ``capacity.csv``, ``behaviour.jsonl`` and a ``manifest.json`` with
the SHA-256 of the pickles and of what was written. Nothing else ever opens the
pickles.

The mapping (FedScale's own): training a sample costs ``3 ×`` its inference
latency (the backward pass is about twice the forward one), so
``train_rate = 1000 / (3 · computation)`` samples/s; ``bandwidth = 1000 ·
communication`` bit/s, the same both ways. These rates come from a trace of
other devices and models, not from a calibration (I1 §13): their profiles say
``origin: trace``, and a fidelity result cannot rest on them.
"""

import csv
import hashlib
import io
import json
import pickle
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from onion_fl.core.context import node_rng
from onion_fl.infrastructure.model import InfrastructureError

SOURCE = "https://github.com/SymbioticLab/FedScale"
TRACE_DIR = "benchmark/dataset/data/device_info"
COMMIT = "7ec441c2afa99510535adebe155d89fa8bb2c637"  # what the import was tested on
LICENSE = "Apache-2.0"
CAPACITY, BEHAVIOUR = "client_device_capacity", "client_behave_trace"
TRAIN_FACTOR = 3.0  # FedScale's augmentation_factor: forward + backward ≈ 3 × forward
UNITS = {
    "computation": "ms per sample (inference)",
    "communication": "kbit/s",
    "active, inactive, finish_time": "s",
    "train_rate": "samples/s = 1000 / (3 * computation)",
    "bandwidth": "bit/s = 1000 * communication",
}


# --- reading the pickles ---------------------------------------------------------


class _Dtype:
    """A stand-in for ``numpy.dtype`` that only names a float or int type."""

    ALLOWED = {"f4", "f8", "i4", "i8"}

    def __init__(self, name: Any, align: Any = False, copy: Any = False) -> None:
        if name not in self.ALLOWED:
            raise pickle.UnpicklingError(f"numpy dtype {name!r} is not allowed")
        self.name, self.order = name, "<"

    def __setstate__(self, state: Any) -> None:
        if not (isinstance(state, tuple) and len(state) > 1 and state[1] in "<>=|"):
            raise pickle.UnpicklingError("unexpected numpy dtype state")
        self.order = "<" if state[1] in "=|" else state[1]


def _scalar(dtype: Any, data: Any) -> float:
    if not isinstance(dtype, _Dtype) or not isinstance(data, bytes):
        raise pickle.UnpicklingError("unexpected numpy scalar")
    value = np.frombuffer(data, dtype=np.dtype(dtype.order + dtype.name))
    if value.size != 1:
        raise pickle.UnpicklingError("a numpy scalar holds one value")
    return value.item()


class _Restricted(pickle.Unpickler):
    """Only the classes FedScale's traces use, and none of them runs code."""

    SAFE = {
        ("collections", "OrderedDict"): OrderedDict,
        ("numpy", "dtype"): _Dtype,
        ("numpy.core.multiarray", "scalar"): _scalar,
        ("numpy._core.multiarray", "scalar"): _scalar,
    }

    def find_class(self, module: str, name: str) -> Any:
        try:
            return self.SAFE[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"{module}.{name} is not allowed") from None

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("persistent ids are not allowed")


def safe_load(data: bytes) -> Any:
    """Unpickle ``data`` allowing only plain values (see :class:`_Restricted`)."""
    return _Restricted(io.BytesIO(data)).load()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_fedscale(folder: str | Path) -> dict[str, Any]:
    """Convert the two pickles in ``folder`` to ``capacity.csv``,
    ``behaviour.jsonl`` and ``manifest.json`` next to them; return the manifest.

    Clients whose capacity is not positive, or whose windows are not ordered
    pairs inside their period, are left out and counted in the manifest.
    """
    folder = Path(folder)
    sources = {name: folder / name for name in (CAPACITY, BEHAVIOUR)}
    for name, path in sources.items():
        if not path.exists():
            raise InfrastructureError(
                f"{path} not found: download {name} from {SOURCE}/tree/{COMMIT}/"
                f"{TRACE_DIR}"
            )
    try:
        capacity = safe_load(sources[CAPACITY].read_bytes())
        behaviour = safe_load(sources[BEHAVIOUR].read_bytes())
    except (pickle.UnpicklingError, EOFError) as exc:
        raise InfrastructureError(f"not a FedScale trace: {exc}") from None
    skipped = {CAPACITY: 0, BEHAVIOUR: 0}
    with (folder / "capacity.csv").open("w", encoding="utf-8", newline="\n") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["client", "computation_ms", "communication_kbps"])
        for client in sorted(capacity):
            row = capacity[client]
            comp, comm = float(row["computation"]), float(row["communication"])
            if not (comp > 0 and comm > 0 and np.isfinite(comp + comm)):
                skipped[CAPACITY] += 1
                continue
            out.writerow([int(client), repr(comp), repr(comm)])
    with (folder / "behaviour.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for client in sorted(behaviour):
            row = behaviour[client]
            windows = _windows(row)
            if windows is None:
                skipped[BEHAVIOUR] += 1
                continue
            line = {"client": int(client), "finish_time": float(row["finish_time"])}
            line |= {"active": windows[0], "inactive": windows[1]}
            line |= {"model": str(row.get("model", ""))}
            f.write(json.dumps(line, separators=(",", ":")) + "\n")
    manifest = {
        "source": SOURCE,
        "commit": COMMIT,
        "path": TRACE_DIR,
        "license": LICENSE,
        "sources": {name: sha256(path) for name, path in sources.items()},
        "files": {
            name: sha256(folder / name) for name in ("capacity.csv", "behaviour.jsonl")
        },
        "clients": {CAPACITY: len(capacity), BEHAVIOUR: len(behaviour)},
        "skipped": skipped,
        "units": UNITS,
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def _windows(row: Mapping[str, Any]) -> tuple[list[float], list[float]] | None:
    try:
        period = float(row["finish_time"])
        active = [float(a) for a in row["active"]]
        inactive = [float(b) for b in row["inactive"]]
    except (KeyError, TypeError, ValueError):
        return None
    if not (period > 0 and active and len(active) == len(inactive)):
        return None
    edges = [x for pair in zip(active, inactive, strict=True) for x in pair]
    if any(b < a for a, b in zip(edges, edges[1:], strict=False)) or edges[0] < 0:
        return None
    return active, inactive


# --- sampling profiles -------------------------------------------------------------


def expand(raw: Any, root: str | Path) -> Any:
    """``raw`` with each site's ``traces`` sampled into profiles and a pool.

    Each of the ``devices`` gets a capacity client and, with ``availability``,
    an availability client, both drawn without replacement from the stream
    ``node_rng(seed, "fedscale/<site>/...")``: the same seed gives the same
    devices. ``device_traces.<site>`` records the files' hashes and which
    clients were drawn, so the ``infrastructure_id`` covers them.
    """
    if not isinstance(raw, Mapping):
        return raw
    sites = raw.get("sites")
    pools = {
        site: spec["traces"]
        for site, spec in (sites.items() if isinstance(sites, Mapping) else ())
        if isinstance(spec, Mapping) and spec.get("traces") is not None
    }
    if not pools:
        return raw
    from onion_fl.infrastructure.model import TracePool

    raw = dict(raw)
    raw["sites"] = {k: dict(v) for k, v in raw["sites"].items()}
    raw["profiles"] = dict(raw.get("profiles") or {})
    raw["device_traces"] = dict(raw.get("device_traces") or {})
    traces = _traces(Path(root) / "FedScale")
    for site, given in sorted(pools.items()):
        try:
            pool = TracePool.model_validate(dict(given))
        except Exception as exc:  # pydantic's message names the field
            raise InfrastructureError(f"sites.{site}.traces: {exc}") from None
        names, record = traces.sample(site, pool, raw["profiles"])
        site_pool = dict(raw["sites"][site].get("pool") or {})
        site_pool |= dict.fromkeys(names, 1)
        raw["sites"][site]["pool"] = site_pool
        raw["device_traces"][site] = record
    return raw


_LOADED: dict[tuple[str, tuple[int, ...]], _Traces] = {}


def _traces(folder: Path) -> _Traces:
    """The traces in ``folder``, kept while their files do not change (a run
    resolves its infrastructure more than once)."""
    names = ("manifest.json", "capacity.csv", "behaviour.jsonl")
    stamp = tuple(
        (folder / n).stat().st_mtime_ns if (folder / n).exists() else 0 for n in names
    )
    key = (str(folder.resolve()), stamp)
    if key not in _LOADED:
        _LOADED.clear()
        _LOADED[key] = _Traces(folder)
    return _LOADED[key]


class _Traces:
    """The imported FedScale traces in ``folder``, read once when first needed."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._capacity: list[tuple[int, float, float]] | None = None
        self._behaviour: list[str] | None = None
        self._manifest: dict[str, Any] | None = None

    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            path = self.folder / "manifest.json"
            if not path.exists():
                raise InfrastructureError(
                    f"{path} not found: download {CAPACITY} and {BEHAVIOUR} from "
                    f"{SOURCE}/tree/{COMMIT}/{TRACE_DIR} into {self.folder} and run "
                    "`onion_fl traces import fedscale` (README §2)"
                )
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for name, digest in manifest["files"].items():
                if sha256(self.folder / name) != digest:
                    raise InfrastructureError(
                        f"{self.folder / name} does not match its manifest; "
                        "import the traces again"
                    )
            self._manifest = manifest
        return self._manifest

    def capacity(self) -> list[tuple[int, float, float]]:
        if self._capacity is None:
            self.manifest()
            with (self.folder / "capacity.csv").open(encoding="utf-8") as f:
                rows = csv.reader(f)
                next(rows)
                self._capacity = [(int(c), float(a), float(b)) for c, a, b in rows]
        return self._capacity

    def behaviour(self) -> list[str]:
        if self._behaviour is None:
            self.manifest()
            text = (self.folder / "behaviour.jsonl").read_text(encoding="utf-8")
            self._behaviour = text.splitlines()
        return self._behaviour

    def sample(
        self, site: str, pool: Any, profiles: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any]]:
        """Add a profile per sampled device to ``profiles``; return their names
        and what to record."""
        capacity = self.capacity()
        behaviour = self.behaviour() if pool.availability else []
        for kind, rows in (("capacity", capacity), ("availability", behaviour)):
            if rows and pool.devices > len(rows):
                raise InfrastructureError(
                    f"sites.{site}.traces: {pool.devices} devices, but the "
                    f"{kind} trace has {len(rows)} clients"
                )
        picks = node_rng(pool.seed, f"fedscale/{site}/capacity").choice(
            len(capacity), size=pool.devices, replace=False
        )
        windows = (
            node_rng(pool.seed, f"fedscale/{site}/availability").choice(
                len(behaviour), size=pool.devices, replace=False
            )
            if pool.availability
            else [None] * pool.devices
        )
        names, clients = [], []
        for i, (pick, window) in enumerate(zip(picks, windows, strict=True), 1):
            client, computation_ms, communication_kbps = capacity[int(pick)]
            name = f"fedscale_{site}_{i}"
            if name in profiles:
                raise InfrastructureError(
                    f"sites.{site}.traces: profile {name!r} already exists"
                )
            profile: dict[str, Any] = {
                "train_rate": 1000.0 / (TRAIN_FACTOR * computation_ms),
                "bandwidth": 1000.0 * communication_kbps,
                "origin": "trace",
            }
            profile |= {
                k: v for k, v in (("cpus", pool.cpus), ("memory", pool.memory)) if v
            }
            entry = {"profile": name, "capacity_client": client}
            if window is not None:
                row = json.loads(behaviour[int(window)])
                profile["availability"] = {
                    "name": "windows",
                    "active": row["active"],
                    "inactive": row["inactive"],
                    "period": row["finish_time"],
                }
                entry |= {"availability_client": row["client"], "model": row["model"]}
            profiles[name] = profile
            names.append(name)
            clients.append(entry)
        manifest = self.manifest()
        record = {
            "source": manifest["source"],
            "commit": manifest["commit"],
            "license": manifest["license"],
            "sources": manifest["sources"],
            "files": manifest["files"],
            "units": manifest["units"],
            "rates": "trace",  # derived from a trace: not calibrated (I1 §13)
            "seed": pool.seed,
            "devices": pool.devices,
            "availability": pool.availability,
            "clients": clients,
        }
        return names, record
