from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from sklearn.preprocessing import StandardScaler

from .swell import SWELLDatasetError, load_swell_all_samples

ScalerMode = Literal["global", "none"]
SplitStrategy = Literal["global", "per_subject"]


@dataclass
class FederatedConfig:
    data_dir: str = "data/SWELL"
    modalities: list[str] | None = None
    subject_ids: list[int] | None = None
    seed: int = 42
    split_train: float = 0.5
    split_val: float = 0.2
    split_test: float = 0.3
    scaler: ScalerMode = "global"
    split_strategy: SplitStrategy = (
        "per_subject"  # "global" = subjects in one split, "per_subject" = each subject has own splits
    )
    mode: Literal["manual", "auto"] = "manual"
    num_fog_nodes: int = 1
    manual_assignments: dict[str, list[int]] | None = None
    per_node_percentages: list[float] | None = None
    output_dir: str = "federated_runs/swell"
    run_name: str | None = None
    ensure_min_train_per_node: bool = True
    test_assignments: dict[str, list[int]] | None = None


def _read_config(config_path: str | os.PathLike) -> FederatedConfig:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    text = p.read_text(encoding="utf-8")
    data: dict
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed; use JSON or install pyyaml")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    # Flatten with defaults
    dataset = data.get("dataset", {})
    split = data.get("split", {})
    federation = data.get("federation", {})

    cfg = FederatedConfig(
        data_dir=dataset.get("data_dir", "data/SWELL"),
        modalities=dataset.get("modalities"),
        subject_ids=dataset.get("subjects"),
        seed=split.get("seed", 42),
        split_train=float(split.get("train", 0.5)),
        split_val=float(split.get("val", 0.2)),
        split_test=float(split.get("test", 0.3)),
        scaler=split.get("scaler", "global"),
        split_strategy=split.get(
            "strategy", "per_subject"
        ),  # Default to per_subject for FL
        mode=federation.get("mode", "manual"),
        num_fog_nodes=int(federation.get("num_fog_nodes", 1)),
        manual_assignments=federation.get("manual_assignments"),
        per_node_percentages=federation.get("per_node_percentages"),
        output_dir=federation.get("output_dir", "federated_runs/swell"),
        run_name=federation.get("run_name"),
        ensure_min_train_per_node=bool(
            federation.get("ensure_min_train_per_node", True)
        ),
        test_assignments=federation.get("test_assignments"),
    )

    total = cfg.split_train + cfg.split_val + cfg.split_test
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split percentages must sum to 1.0")

    return cfg


def _stable_subject_seed(base_seed: int, subject: str) -> int:
    """Build a stable per-subject seed (no dependence on PYTHONHASHSEED)."""
    crc = zlib.crc32(subject.encode("utf-8", errors="ignore")) & 0xFFFFFFFF
    return int((base_seed + crc) % (2**32 - 1))


def _split_subjects(
    subjects: list[str], train: float, val: float, test: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    uniq = np.array(sorted(set(subjects)))
    if uniq.size < 2:
        raise SWELLDatasetError("Not enough subjects to split")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = max(1, int(round(train * n)))
    n_val = max(0, int(round(val * n)))
    # ensure exact coverage
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_val > 0:
            n_val -= 1
        else:
            n_train -= 1

    train_ids = uniq[:n_train].tolist()
    val_ids = uniq[n_train : n_train + n_val].tolist()
    test_ids = uniq[n_train + n_val :].tolist()
    return train_ids, val_ids, test_ids


def _split_subjects_with_test(
    subjects: list[str],
    train: float,
    val: float,
    seed: int,
    test_subjects: list[str],
) -> tuple[list[str], list[str], list[str]]:
    test_set = set(test_subjects)
    remaining = [s for s in subjects if s not in test_set]
    if not remaining:
        raise SWELLDatasetError("No remaining subjects after test holdout")

    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)

    denom = train + val
    train_ratio = train / denom if denom > 0 else 1.0
    n = len(remaining)
    if val > 0 and n > 1:
        n_train = max(1, int(round(train_ratio * n)))
        if n_train >= n:
            n_train = n - 1
    else:
        n_train = n
    n_val = n - n_train

    train_ids = remaining[:n_train]
    val_ids = remaining[n_train:]
    return train_ids, val_ids, sorted(test_set)


def _auto_assign_nodes(
    all_subjects: list[str],
    num_nodes: int,
    percentages: list[float] | None,
    seed: int,
) -> dict[str, list[str]]:
    if num_nodes < 1:
        raise ValueError("num_fog_nodes must be >= 1")

    subs = np.array(sorted(all_subjects))
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)

    if not percentages:
        # Balanced split by count
        counts = [len(subs) // num_nodes] * num_nodes
        for i in range(len(subs) % num_nodes):
            counts[i] += 1
        idx = 0
        mapping: dict[str, list[str]] = {}
        for n, c in enumerate(counts):
            mapping[f"fog_{n}"] = subs[idx : idx + c].tolist()
            idx += c
        return mapping

    if abs(sum(percentages) - 1.0) > 1e-6:
        raise ValueError("per_node_percentages must sum to 1.0")
    counts = [int(round(p * len(subs))) for p in percentages]
    # fix rounding
    diff = len(subs) - sum(counts)
    for i in range(abs(diff)):
        counts[i % len(counts)] += 1 if diff > 0 else -1

    idx = 0
    mapping = {}
    for n, c in enumerate(counts):
        mapping[f"fog_{n}"] = subs[idx : idx + c].tolist()
        idx += c
    return mapping


def plan_and_materialize_swell_federated(config_path: str) -> dict:
    """Create federated SWELL subject splits from JSON/YAML config and save npz artifacts.

    Supports two split strategies:
    - "global": Each subject belongs to exactly one split (train OR val OR test)
    - "per_subject": Each subject has its own internal train/val/test split

    Output structure (under output_dir/run_name):
      - manifest.json: full config + subject splits + nodes mapping
      - scaler_global.json (if scaler==global)
      - fog_<id>/train.npz, val.npz, test.npz (aggregated arrays)
      - fog_<id>/subject_<N>/train.npz, val.npz, test.npz (per-client arrays)
    """
    cfg = _read_config(config_path)

    X_all, y_all, subjects_all, meta = load_swell_all_samples(
        data_dir=cfg.data_dir,
        modalities=cfg.modalities,
        subjects=cfg.subject_ids,
        normalize_features=False,
    )
    subjects_all = subjects_all.astype(str)
    uniq_subjects = sorted(set(subjects_all.tolist()))

    # Node subject mapping (which subjects go to which fog)
    if cfg.mode == "manual":
        if not cfg.manual_assignments:
            raise ValueError("manual mode requires 'manual_assignments'")
        node_map = {
            node: [str(s) for s in subs]
            for node, subs in cfg.manual_assignments.items()
        }
    else:
        node_map = _auto_assign_nodes(
            uniq_subjects, cfg.num_fog_nodes, cfg.per_node_percentages, cfg.seed
        )

    # Prepare output dir
    base = Path(cfg.output_dir)
    run = cfg.run_name or f"run_{cfg.seed}"
    out_dir = base / run
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strategy-specific processing
    if cfg.split_strategy == "per_subject":
        return _materialize_per_subject_strategy(
            cfg, X_all, y_all, subjects_all, uniq_subjects, node_map, out_dir, meta
        )
    else:  # "global" strategy
        return _materialize_global_strategy(
            cfg, X_all, y_all, subjects_all, uniq_subjects, node_map, out_dir, meta
        )


def _materialize_per_subject_strategy(
    cfg: FederatedConfig,
    X_all: np.ndarray,
    y_all: np.ndarray,
    subjects_all: np.ndarray,
    uniq_subjects: list[str],
    node_map: dict[str, list[str]],
    out_dir: Path,
    meta: dict,
) -> dict:
    """Each subject has its own internal train/val/test split.

    This is the recommended strategy for federated learning where:
    - Each client (subject) can train AND validate AND test locally
    - Global test is aggregated from all subjects' test splits
    """
    rng = np.random.default_rng(cfg.seed)

    # First pass: compute global scaler on ALL training data (from all subjects)
    # We need to know which samples will be train to fit scaler
    scaler_payload = None
    if cfg.scaler == "global":
        # Collect indices of train samples from all subjects
        train_indices = []
        for subj in uniq_subjects:
            subj_str = str(subj)
            subj_mask = subjects_all == subj_str
            subj_indices = np.where(subj_mask)[0]
            n_subj = len(subj_indices)
            # Same split logic as below, but just for indices
            n_train = max(1, int(round(cfg.split_train * n_subj)))
            # Use same seed+subject for reproducibility
            subj_rng = np.random.default_rng(_stable_subject_seed(cfg.seed, subj_str))
            perm = subj_rng.permutation(n_subj)
            train_indices.extend(subj_indices[perm[:n_train]])

        if train_indices:
            scaler = StandardScaler()
            scaler.fit(X_all[train_indices])
            scaler_payload = {
                "mean": scaler.mean_.astype(float).tolist(),
                "scale": scaler.scale_.astype(float).tolist(),
            }

    def _apply_scale(X: np.ndarray) -> np.ndarray:
        if scaler_payload is None:
            return X.astype(np.float32)
        mean = np.array(scaler_payload["mean"], dtype=np.float64)
        scale = np.array(scaler_payload["scale"], dtype=np.float64)
        scale = np.where(scale == 0.0, 1.0, scale)
        return ((X - mean) / scale).astype(np.float32)

    # Track which subjects have train data (all should in per_subject mode)
    subjects_with_train = set()
    clients_map = {}  # fog_id -> {client_id: subject_id}

    # Per-subject split storage for aggregation
    fog_splits = {node: {"train": [], "val": [], "test": []} for node in node_map}

    for node, node_subjects in node_map.items():
        node_dir = out_dir / node
        node_dir.mkdir(parents=True, exist_ok=True)
        clients_map[node] = {}

        for subj in node_subjects:
            subj_str = str(subj)
            client_id = f"{node}_client_{subj_str}"
            clients_map[node][client_id] = subj_str

            subj_dir = node_dir / f"subject_{subj_str}"
            subj_dir.mkdir(parents=True, exist_ok=True)

            # Get all data for this subject
            subj_mask = subjects_all == subj_str
            X_subj = X_all[subj_mask]
            y_subj = y_all[subj_mask]

            if len(X_subj) == 0:
                # Empty subject - create empty files
                for split_name in ("train", "val", "test"):
                    np.savez(
                        subj_dir / f"{split_name}.npz",
                        X=np.empty((0, X_all.shape[1]), dtype=np.float32),
                        y=np.empty((0,), dtype=np.int64),
                        subjects=np.empty((0,), dtype=object),
                    )
                continue

            # Split this subject's data internally
            n = len(X_subj)
            n_train = max(1, int(round(cfg.split_train * n)))
            n_val = max(0, int(round(cfg.split_val * n)))
            n_test = n - n_train - n_val

            # Ensure at least 1 test sample if possible
            if n_test < 1 and n > n_train:
                n_test = 1
                if n_val > 0:
                    n_val -= 1
                else:
                    n_train -= 1

            # Shuffle indices for this subject (reproducible per subject)
            subj_rng = np.random.default_rng(_stable_subject_seed(cfg.seed, subj_str))
            perm = subj_rng.permutation(n)

            idx_train = perm[:n_train]
            idx_val = perm[n_train : n_train + n_val]
            idx_test = perm[n_train + n_val :]

            # Mark subject as having train data
            if len(idx_train) > 0:
                subjects_with_train.add(subj_str)

            # Save splits for this subject
            for split_name, idx in [
                ("train", idx_train),
                ("val", idx_val),
                ("test", idx_test),
            ]:
                if len(idx) > 0:
                    X_split = _apply_scale(X_subj[idx])
                    y_split = y_subj[idx]
                    subj_arr = np.full(len(idx), subj_str, dtype=object)

                    # Store for fog aggregation
                    fog_splits[node][split_name].append((X_split, y_split, subj_arr))
                else:
                    X_split = np.empty((0, X_all.shape[1]), dtype=np.float32)
                    y_split = np.empty((0,), dtype=np.int64)
                    subj_arr = np.empty((0,), dtype=object)

                np.savez(
                    subj_dir / f"{split_name}.npz",
                    X=X_split,
                    y=y_split,
                    subjects=subj_arr,
                )

        # Save aggregated fog-level splits
        for split_name in ("train", "val", "test"):
            parts = fog_splits[node][split_name]
            if parts:
                X_agg = np.concatenate([p[0] for p in parts], axis=0)
                y_agg = np.concatenate([p[1] for p in parts], axis=0)
                s_agg = np.concatenate([p[2] for p in parts], axis=0)
            else:
                X_agg = np.empty((0, X_all.shape[1]), dtype=np.float32)
                y_agg = np.empty((0,), dtype=np.int64)
                s_agg = np.empty((0,), dtype=object)

            np.savez(node_dir / f"{split_name}.npz", X=X_agg, y=y_agg, subjects=s_agg)

    # Build manifest
    manifest = {
        "config": {
            "data_dir": cfg.data_dir,
            "modalities": cfg.modalities,
            "seed": cfg.seed,
            "split": {
                "train": cfg.split_train,
                "val": cfg.split_val,
                "test": cfg.split_test,
                "strategy": "per_subject",
            },
            "scaler": cfg.scaler,
            "mode": cfg.mode,
            "num_fog_nodes": cfg.num_fog_nodes,
            "per_node_percentages": cfg.per_node_percentages,
        },
        "global_subjects": {
            "train": sorted(subjects_with_train),  # All subjects with train data
            "val": sorted(uniq_subjects),  # All subjects have val (internal)
            "test": sorted(uniq_subjects),  # All subjects have test (internal)
            "all": sorted(uniq_subjects),
        },
        "nodes": node_map,
        "clients": clients_map,
        "meta": {
            k: v
            for k, v in meta.items()
            if k in ("modalities", "n_samples", "n_features", "n_subjects")
        },
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if scaler_payload is not None:
        (out_dir / "scaler_global.json").write_text(
            json.dumps(scaler_payload, indent=2), encoding="utf-8"
        )

    print(f"[MATERIALIZE] Strategy: per_subject")
    print(
        f"[MATERIALIZE] All {len(uniq_subjects)} subjects have internal train/val/test splits"
    )
    print(
        f"[MATERIALIZE] Split ratios: train={cfg.split_train:.0%}, val={cfg.split_val:.0%}, test={cfg.split_test:.0%}"
    )

    return {"output_dir": str(out_dir), "manifest": manifest}


def _materialize_global_strategy(
    cfg: FederatedConfig,
    X_all: np.ndarray,
    y_all: np.ndarray,
    subjects_all: np.ndarray,
    uniq_subjects: list[str],
    node_map: dict[str, list[str]],
    out_dir: Path,
    meta: dict,
) -> dict:
    """Original global subject-based split: each subject belongs to exactly one split.

    This means a subject in 'train' has ALL its data in train, none in val/test.
    """
    test_subjects: list[str] | None = None
    if cfg.test_assignments:
        test_subjects = []
        for node, subs in cfg.test_assignments.items():
            if subs is None:
                continue
            test_subjects.extend([str(s) for s in subs])
        test_subjects = sorted(set(test_subjects))
        missing = [s for s in test_subjects if s not in uniq_subjects]
        if missing:
            raise SWELLDatasetError(f"Test subjects not found in dataset: {missing}")
        # Ensure test subjects are assigned to some node
        assigned = set()
        for subs in node_map.values():
            assigned.update([str(s) for s in subs])
        unassigned = [s for s in test_subjects if s not in assigned]
        if unassigned:
            raise SWELLDatasetError(
                f"Test subjects not present in any node assignment: {unassigned}"
            )

    # Global subject-based split
    if test_subjects:
        tr_subj, va_subj, te_subj = _split_subjects_with_test(
            uniq_subjects, cfg.split_train, cfg.split_val, cfg.seed, test_subjects
        )
    else:
        tr_subj, va_subj, te_subj = _split_subjects(
            uniq_subjects, cfg.split_train, cfg.split_val, cfg.split_test, cfg.seed
        )

    # Ensure each node has at least one train subject (legacy behavior)
    if cfg.ensure_min_train_per_node:
        tr_set = set(tr_subj)
        node_tr_counts = {
            n: len(set(subs).intersection(tr_set)) for n, subs in node_map.items()
        }
        lacking_nodes = [n for n, c in node_tr_counts.items() if c == 0]
        donor_nodes = [n for n, c in node_tr_counts.items() if c > 1]
        for ln in lacking_nodes:
            moved = False
            for dn in donor_nodes:
                dn_tr_subjects = [s for s in node_map[dn] if s in tr_set]
                for s in dn_tr_subjects:
                    if node_tr_counts[dn] <= 1:
                        continue
                    if s not in node_map[ln]:
                        node_map[dn].remove(s)
                        node_map[ln].append(s)
                        node_tr_counts[dn] -= 1
                        node_tr_counts[ln] += 1
                        moved = True
                        break
                if moved:
                    break

    # Compute global scaler on train subjects only
    scaler_payload = None
    if cfg.scaler == "global":
        train_mask = np.isin(subjects_all, np.array(tr_subj))
        scaler = StandardScaler()
        scaler.fit(X_all[train_mask])
        scaler_payload = {
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
        }

    def _apply_scale(X: np.ndarray) -> np.ndarray:
        if scaler_payload is None:
            return X.astype(np.float32)
        mean = np.array(scaler_payload["mean"], dtype=np.float64)
        scale = np.array(scaler_payload["scale"], dtype=np.float64)
        scale = np.where(scale == 0.0, 1.0, scale)
        return ((X - mean) / scale).astype(np.float32)

    clients_map = {}

    for node, node_subjects in node_map.items():
        node_dir = out_dir / node
        node_dir.mkdir(parents=True, exist_ok=True)
        clients_map[node] = {}

        # Save aggregated fog-level splits
        for split_name, split_subjects in (
            ("train", tr_subj),
            ("val", va_subj),
            ("test", te_subj),
        ):
            ss = np.array(sorted(set(node_subjects).intersection(set(split_subjects))))
            if ss.size == 0:
                np.savez(
                    node_dir / f"{split_name}.npz",
                    X=np.empty((0, X_all.shape[1]), dtype=np.float32),
                    y=np.empty((0,), dtype=np.int64),
                    subjects=np.empty((0,), dtype=object),
                )
                continue

            mask = np.isin(subjects_all, ss)
            X_split = _apply_scale(X_all[mask])
            y_split = y_all[mask]
            sub_split = subjects_all[mask]
            np.savez(
                node_dir / f"{split_name}.npz", X=X_split, y=y_split, subjects=sub_split
            )

        # Save per-subject splits
        for subj in node_subjects:
            subj_str = str(subj)
            client_id = f"{node}_client_{subj_str}"
            clients_map[node][client_id] = subj_str

            subj_dir = node_dir / f"subject_{subj_str}"
            subj_dir.mkdir(parents=True, exist_ok=True)

            for split_name, split_subjects in (
                ("train", tr_subj),
                ("val", va_subj),
                ("test", te_subj),
            ):
                if subj_str not in split_subjects:
                    np.savez(
                        subj_dir / f"{split_name}.npz",
                        X=np.empty((0, X_all.shape[1]), dtype=np.float32),
                        y=np.empty((0,), dtype=np.int64),
                        subjects=np.empty((0,), dtype=object),
                    )
                    continue

                mask = subjects_all == subj_str
                X_split = _apply_scale(X_all[mask])
                y_split = y_all[mask]
                sub_split = subjects_all[mask]
                np.savez(
                    subj_dir / f"{split_name}.npz",
                    X=X_split,
                    y=y_split,
                    subjects=sub_split,
                )

    manifest = {
        "config": {
            "data_dir": cfg.data_dir,
            "modalities": cfg.modalities,
            "seed": cfg.seed,
            "split": {
                "train": cfg.split_train,
                "val": cfg.split_val,
                "test": cfg.split_test,
                "strategy": "global",
            },
            "scaler": cfg.scaler,
            "mode": cfg.mode,
            "num_fog_nodes": cfg.num_fog_nodes,
            "per_node_percentages": cfg.per_node_percentages,
            "test_assignments": cfg.test_assignments,
        },
        "global_subjects": {
            "train": tr_subj,
            "val": va_subj,
            "test": te_subj,
            "all": uniq_subjects,
        },
        "nodes": node_map,
        "clients": clients_map,
        "meta": {
            k: v
            for k, v in meta.items()
            if k in ("modalities", "n_samples", "n_features", "n_subjects")
        },
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if scaler_payload is not None:
        (out_dir / "scaler_global.json").write_text(
            json.dumps(scaler_payload, indent=2), encoding="utf-8"
        )

    print(f"[MATERIALIZE] Strategy: global (subject-level split)")
    print(
        f"[MATERIALIZE] Train subjects: {len(tr_subj)}, Val subjects: {len(va_subj)}, Test subjects: {len(te_subj)}"
    )

    return {"output_dir": str(out_dir), "manifest": manifest}


def load_node_split(
    split_file: str | os.PathLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a saved npz split (X, y, subjects)."""
    arr = np.load(split_file, allow_pickle=True)
    return arr["X"], arr["y"], arr["subjects"]
