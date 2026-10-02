from __future__ import annotations

"""Prepared-data cache and dataset card (spec §8.3).

``data/cache/<dataset>/<key>/subject_<id>.npz`` plus ``meta.json``. The key
hashes the description and the resolved options, not the root: the same data
on another machine gets the same key. ``meta.json`` keeps a digest per subject,
which later feeds the ``data_id`` of a run.
"""

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from onion_fl.data.contract import DataError, SubjectData
from onion_fl.data.ingest import DatasetSpec, ingest, resolve_options

CORRELATION_WARNING = 0.95


def cache_key(spec: DatasetSpec, options: Mapping[str, Any] | None = None) -> str:
    payload = {
        "spec": spec.model_dump(mode="json", exclude={"root"}),
        "options": resolve_options(spec, options),
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _digest(data: SubjectData) -> str:
    return hashlib.sha256(data.X.tobytes() + data.y.tobytes()).hexdigest()


def prepare(
    spec: DatasetSpec,
    options: Mapping[str, Any] | None = None,
    *,
    cache_dir: str | Path = "data/cache",
    root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Ingest once and store the result; later calls reuse it unless ``force``."""
    key = cache_key(spec, options)
    out = Path(cache_dir) / spec.name / key
    if (out / "meta.json").exists() and not force:
        return out
    subjects = ingest(spec, options, root=root)  # fails before touching the cache
    if not subjects:
        raise DataError(f"{spec.name}: no labelled rows")
    # A private folder per writer, renamed into place: parallel runs may race
    # for the same cache, and nobody ever reads a half-written one.
    tmp = out.with_name(f"{key}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    tmp.mkdir(parents=True)
    first = subjects[0]
    meta = {
        "dataset": spec.name,
        "key": key,
        "task": first.task,
        "n_classes": first.n_classes,
        "feature_names": first.feature_names,
        "options": resolve_options(spec, options),
        "spec": spec.model_dump(mode="json", exclude={"root"}),
        "subjects": {},
    }
    for data in subjects:
        np.savez(tmp / f"subject_{data.subject}.npz", X=data.X, y=data.y)
        meta["subjects"][data.subject] = {
            "n_samples": data.n_samples,
            "class_counts": data.class_counts,
            "sha256": _digest(data),
        }
    (tmp / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    if force:
        shutil.rmtree(out, ignore_errors=True)
    try:
        tmp.rename(out)
    except OSError:  # another writer finished first: keep its cache, drop ours
        shutil.rmtree(tmp, ignore_errors=True)
        if not (out / "meta.json").exists():
            raise
    return out


def load_prepared(path: str | Path) -> list[SubjectData]:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    subjects = []
    for subject, info in meta["subjects"].items():
        with np.load(path / f"subject_{subject}.npz", allow_pickle=False) as archive:
            data = SubjectData(
                X=archive["X"],
                y=archive["y"],
                dataset=meta["dataset"],
                subject=subject,
                task=meta["task"],
                n_classes=meta["n_classes"],
                feature_names=meta["feature_names"],
            )
        if _digest(data) != info["sha256"]:
            raise DataError(f"{path}: subject {subject} does not match its digest")
        subjects.append(data)
    return subjects


def inspect(subjects: Sequence[SubjectData]) -> dict[str, Any]:
    """Dataset card: sizes, class balance, missing values and suspicious features."""
    X = np.concatenate([s.X for s in subjects])
    y = np.concatenate([s.y for s in subjects]).astype(np.float64)
    names = subjects[0].feature_names
    missing = np.isnan(X).mean(axis=0)
    warnings = []
    for j, name in enumerate(names):
        ok = ~np.isnan(X[:, j])
        x, t = X[ok, j], y[ok]
        if len(x) < 2 or x.std() == 0 or t.std() == 0:
            continue
        r = float(np.corrcoef(x, t)[0, 1])
        if abs(r) > CORRELATION_WARNING:
            warnings.append(
                f"feature {name!r} has |corr| = {abs(r):.3f} with the label: "
                "is it a meta column?"
            )
    return {
        "dataset": subjects[0].dataset,
        "task": subjects[0].task,
        "n_subjects": len(subjects),
        "n_samples": len(y),
        "n_features": len(names),
        "class_counts": np.sum([s.class_counts for s in subjects], axis=0).tolist(),
        "samples_per_subject": {s.subject: s.n_samples for s in subjects},
        "missing": {n: float(m) for n, m in zip(names, missing, strict=True) if m > 0},
        "warnings": warnings,
    }
