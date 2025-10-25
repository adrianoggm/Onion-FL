from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from .swell import SWELLDatasetError, load_swell_all_samples
from sklearn.preprocessing import StandardScaler


SplitMode = Literal["global", "none"]


@dataclass
class FederatedConfig:
    data_dir: str = "data/SWELL"
    modalities: Optional[List[str]] = None
    subject_ids: Optional[List[int]] = None
    seed: int = 42
    split_train: float = 0.5
    split_val: float = 0.2
    split_test: float = 0.3
    scaler: SplitMode = "global"
    mode: Literal["manual", "auto"] = "manual"
    num_fog_nodes: int = 1
    manual_assignments: Optional[Dict[str, List[int]]] = None
    per_node_percentages: Optional[List[float]] = None
    output_dir: str = "federated_runs/swell"
    run_name: Optional[str] = None
    ensure_min_train_per_node: bool = True


def _read_config(config_path: str | os.PathLike) -> FederatedConfig:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    text = p.read_text(encoding="utf-8")
    data: Dict
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
        mode=federation.get("mode", "manual"),
        num_fog_nodes=int(federation.get("num_fog_nodes", 1)),
        manual_assignments=federation.get("manual_assignments"),
        per_node_percentages=federation.get("per_node_percentages"),
        output_dir=federation.get("output_dir", "federated_runs/swell"),
        run_name=federation.get("run_name"),
        ensure_min_train_per_node=bool(federation.get("ensure_min_train_per_node", True)),
    )

    total = cfg.split_train + cfg.split_val + cfg.split_test
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split percentages must sum to 1.0")

    return cfg


def _split_subjects(
    subjects: List[str], train: float, val: float, test: float, seed: int
) -> Tuple[List[str], List[str], List[str]]:
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


def _auto_assign_nodes(
    all_subjects: List[str], num_nodes: int, percentages: Optional[List[float]], seed: int
) -> Dict[str, List[str]]:
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
        mapping: Dict[str, List[str]] = {}
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


def plan_and_materialize_swell_federated(config_path: str) -> Dict:
    """Create federated SWELL subject splits from JSON/YAML config and save npz artifacts.

    Output structure (under output_dir/run_name):
      - manifest.json: full config + subject splits + nodes mapping
      - scaler_global.json (if scaler==global)
      - node_<id>/train.npz, val.npz, test.npz (arrays X, y, subjects)
    """
    cfg = _read_config(config_path)

    X_all, y_all, subjects_all, meta = load_swell_all_samples(
        data_dir=cfg.data_dir, modalities=cfg.modalities, subjects=cfg.subject_ids, normalize_features=False
    )
    subjects_all = subjects_all.astype(str)
    uniq_subjects = sorted(set(subjects_all.tolist()))

    # Global subject-based split (parity with centralized training)
    tr_subj, va_subj, te_subj = _split_subjects(
        uniq_subjects, cfg.split_train, cfg.split_val, cfg.split_test, cfg.seed
    )

    # Node subject mapping
    if cfg.mode == "manual":
        if not cfg.manual_assignments:
            raise ValueError("manual mode requires 'manual_assignments'")
        node_map = {node: [str(s) for s in subs] for node, subs in cfg.manual_assignments.items()}
    else:
        node_map = _auto_assign_nodes(uniq_subjects, cfg.num_fog_nodes, cfg.per_node_percentages, cfg.seed)

    # Ensure each node has at least one train subject assigned (to avoid empty train splits)
    if cfg.ensure_min_train_per_node:
        tr_set = set(tr_subj)
        # Build counts per node
        node_tr_counts = {n: len(set(subs).intersection(tr_set)) for n, subs in node_map.items()}
        lacking_nodes = [n for n, c in node_tr_counts.items() if c == 0]
        donor_nodes = [n for n, c in node_tr_counts.items() if c > 1]
        for ln in lacking_nodes:
            moved = False
            for dn in donor_nodes:
                # find a donor subject that is in train and not already in ln
                dn_tr_subjects = [s for s in node_map[dn] if s in tr_set]
                for s in dn_tr_subjects:
                    # avoid emptying donor to zero
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
            # If still not moved, leave as is (will result in empty train for that node)

    # Prepare output dir
    base = Path(cfg.output_dir)
    run = cfg.run_name or f"run_{cfg.seed}"
    out_dir = base / run
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally compute a GLOBAL scaler on global train samples only
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
            return X
        mean = np.array(scaler_payload["mean"], dtype=np.float64)
        scale = np.array(scaler_payload["scale"], dtype=np.float64)
        # avoid division by zero
        scale = np.where(scale == 0.0, 1.0, scale)
        return ((X - mean) / scale).astype(np.float32)

    # Save per-node splits
    for node, node_subjects in node_map.items():
        node_dir = out_dir / f"{node}"
        node_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_subjects in (
            ("train", tr_subj),
            ("val", va_subj),
            ("test", te_subj),
        ):
            # subjects for this node and split (intersection)
            ss = np.array(sorted(set(node_subjects).intersection(set(split_subjects))))
            if ss.size == 0:
                # create empty npz (consistent shape handling downstream)
                np.savez(node_dir / f"{split_name}.npz", X=np.empty((0, X_all.shape[1]), dtype=np.float32), y=np.empty((0,), dtype=np.int64), subjects=np.empty((0,), dtype=object))
                continue

            mask = np.isin(subjects_all, ss)
            X_split = _apply_scale(X_all[mask])
            y_split = y_all[mask]
            sub_split = subjects_all[mask]
            np.savez(node_dir / f"{split_name}.npz", X=X_split, y=y_split, subjects=sub_split)

    manifest = {
        "config": {
            "data_dir": cfg.data_dir,
            "modalities": cfg.modalities,
            "seed": cfg.seed,
            "split": {"train": cfg.split_train, "val": cfg.split_val, "test": cfg.split_test},
            "scaler": cfg.scaler,
            "mode": cfg.mode,
            "num_fog_nodes": cfg.num_fog_nodes,
            "per_node_percentages": cfg.per_node_percentages,
        },
        "global_subjects": {
            "train": tr_subj,
            "val": va_subj,
            "test": te_subj,
            "all": uniq_subjects,
        },
        "nodes": node_map,
        "meta": {k: v for k, v in meta.items() if k in ("modalities", "n_samples", "n_features", "n_subjects")},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if scaler_payload is not None:
        (out_dir / "scaler_global.json").write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")

    return {"output_dir": str(out_dir), "manifest": manifest}


def load_node_split(split_file: str | os.PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a saved npz split (X, y, subjects)."""
    arr = np.load(split_file, allow_pickle=True)
    return arr["X"], arr["y"], arr["subjects"]
