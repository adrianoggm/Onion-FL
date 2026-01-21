from __future__ import annotations

"""SWEET federated data preparation utilities.

Reuses the same infrastructure as SWELL but adapted for SWEET dataset structure.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from sklearn.preprocessing import StandardScaler

from .sweet_samples import load_sweet_sample_dataset, SWEETSampleLoaderError

ScalerMode = Literal["global", "none"]
SplitStrategy = Literal["global", "per_subject"]


@dataclass
class FederatedConfigSWEET:
    """Configuration for SWEET federated splits with transfer learning support."""

    # Pre-trained model (from selection1)
    pretrained_model_path: str | None = None
    pretrained_scaler_path: str | None = None

    # Federated training data (selection2)
    data_dir: str = "data/SWEET/selection2/users"
    label_strategy: str = "ordinal_3class"
    elevated_threshold: float = 2.0
    min_samples_per_subject: int = 5

    # Split configuration
    seed: int = 42
    split_train: float = 0.6
    split_val: float = 0.2
    split_test: float = 0.2
    scaler: ScalerMode = "global"
    split_strategy: SplitStrategy = (
        "global"  # "global" = each subject in ONE split only (STRICT)
    )

    # Federation configuration
    mode: Literal["manual", "auto"] = "auto"
    num_fog_nodes: int = 3
    manual_assignments: dict[str, list[str]] | None = None
    per_node_percentages: list[float] | None = None
    output_dir: str = "federated_runs/sweet"
    run_name: str | None = None
    ensure_min_train_per_node: bool = True

    # Transfer learning settings
    freeze_initial_weights: bool = False
    fine_tune_lr_multiplier: float = 0.1


def _read_config(config_path: str | os.PathLike) -> FederatedConfigSWEET:
    """Read SWEET federated config from JSON/YAML."""
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

    dataset = data.get("dataset", {})
    split = data.get("split", {})
    federation = data.get("federation", {})
    transfer_learning = data.get("transfer_learning", {})

    cfg = FederatedConfigSWEET(
        pretrained_model_path=transfer_learning.get("pretrained_model_path"),
        pretrained_scaler_path=transfer_learning.get("pretrained_scaler_path"),
        data_dir=dataset.get("data_dir", "data/SWEET/selection2/users"),
        label_strategy=dataset.get("label_strategy", "ordinal_3class"),
        elevated_threshold=float(dataset.get("elevated_threshold", 2.0)),
        min_samples_per_subject=int(dataset.get("min_samples_per_subject", 5)),
        seed=split.get("seed", 42),
        split_train=float(split.get("train", 0.6)),
        split_val=float(split.get("val", 0.2)),
        split_test=float(split.get("test", 0.2)),
        scaler=split.get("scaler", "global"),
        split_strategy=split.get("strategy", "per_subject"),
        mode=federation.get("mode", "auto"),
        num_fog_nodes=int(federation.get("num_fog_nodes", 3)),
        manual_assignments=federation.get("manual_assignments"),
        per_node_percentages=federation.get("per_node_percentages"),
        output_dir=federation.get("output_dir", "federated_runs/sweet"),
        run_name=federation.get("run_name"),
        ensure_min_train_per_node=bool(
            federation.get("ensure_min_train_per_node", True)
        ),
        freeze_initial_weights=bool(
            transfer_learning.get("freeze_initial_weights", False)
        ),
        fine_tune_lr_multiplier=float(
            transfer_learning.get("fine_tune_lr_multiplier", 0.1)
        ),
    )

    total = cfg.split_train + cfg.split_val + cfg.split_test
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split percentages must sum to 1.0")

    return cfg


def _auto_assign_nodes(
    all_subjects: list[str],
    num_nodes: int,
    percentages: list[float] | None,
    seed: int,
) -> dict[str, list[str]]:
    """Auto-assign subjects to fog nodes."""
    if num_nodes < 1:
        raise ValueError("num_fog_nodes must be >= 1")

    subs = np.array(sorted(all_subjects))
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)

    if not percentages:
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
    diff = len(subs) - sum(counts)
    for i in range(abs(diff)):
        counts[i % len(counts)] += 1 if diff > 0 else -1

    idx = 0
    mapping = {}
    for n, c in enumerate(counts):
        mapping[f"fog_{n}"] = subs[idx : idx + c].tolist()
        idx += c
    return mapping


def plan_and_materialize_sweet_federated(config_path: str) -> dict:
    """Create federated SWEET subject splits from JSON/YAML config.

    Supports two strategies:
    - "global": Each subject belongs to exactly ONE split (train OR val OR test)
      → Only training subjects become clients (~84 clients with 70/15/15 split)
    - "per_subject": Each subject has internal train/val/test splits
      → All subjects become clients (~120 clients), each with local validation

    Output structure (under output_dir/run_name):
      - manifest.json: full config + subject splits + nodes mapping + clients mapping
      - scaler_global.json (if scaler==global)
      - pretrained_model.json, pretrained_scaler.json (if transfer learning)
      - fog_<id>/train.npz, val.npz, test.npz (aggregated per fog node)
      - fog_<id>/subject_<N>/train.npz, val.npz, test.npz (per client - only train subjects in global mode)

    Returns:
        Manifest dictionary with splits and paths
    """
    cfg = _read_config(config_path)

    # Load full SWEET dataset (selection2 for federated fine-tuning)
    from .sweet_samples import load_sweet_sample_full
    from sklearn.model_selection import train_test_split

    X, y, subject_ids, feature_names = load_sweet_sample_full(
        data_dir=cfg.data_dir,
        label_strategy=cfg.label_strategy,
        elevated_threshold=cfg.elevated_threshold,
        min_samples_per_subject=cfg.min_samples_per_subject,
    )

    # Get unique subjects
    all_subjects = sorted(list(set(subject_ids.tolist())))

    # Check which strategy to use
    if cfg.split_strategy == "global":
        # Global strategy: each subject in ONE split only
        return _materialize_global_strategy_sweet(
            cfg, X, y, subject_ids, all_subjects, feature_names
        )
    else:
        # Per-subject strategy: each subject has internal splits
        return _materialize_per_subject_strategy_sweet(
            cfg, X, y, subject_ids, all_subjects, feature_names
        )


def _materialize_global_strategy_sweet(
    cfg: FederatedConfigSWEET,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    all_subjects: list[str],
    feature_names: list[str],
) -> dict:
    """Global strategy: each subject belongs to ONE split (train OR val OR test).

    Only training subjects become federated clients.
    """
    from sklearn.model_selection import train_test_split

    # Subject-level train/val/test split
    subjects_train, subjects_temp = train_test_split(
        all_subjects,
        train_size=cfg.split_train,
        random_state=cfg.seed,
        shuffle=True,
    )
    val_ratio = cfg.split_val / (cfg.split_val + cfg.split_test)
    subjects_val, subjects_test = train_test_split(
        subjects_temp,
        train_size=val_ratio,
        random_state=cfg.seed,
        shuffle=True,
    )

    print(
        f"[SPLIT] Global strategy: {len(subjects_train)} train, {len(subjects_val)} val, {len(subjects_test)} test subjects"
    )

    # Node subject mapping - distribute ONLY training subjects
    if cfg.mode == "manual":
        if not cfg.manual_assignments:
            raise ValueError("manual mode requires 'manual_assignments'")
        node_map = {
            node: [str(s) for s in subs if s in subjects_train]
            for node, subs in cfg.manual_assignments.items()
        }
    else:
        # Only distribute training subjects across fog nodes
        node_map = _auto_assign_nodes(
            subjects_train, cfg.num_fog_nodes, cfg.per_node_percentages, cfg.seed
        )

    # Prepare output directory
    run_id = cfg.run_name or f"run_{cfg.seed}"
    run_dir = Path(cfg.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy pre-trained model and scaler if provided
    if cfg.pretrained_model_path:
        pretrained_model_src = Path(cfg.pretrained_model_path)
        if pretrained_model_src.exists():
            pretrained_model_dst = run_dir / "pretrained_model.json"
            pretrained_model_dst.write_text(pretrained_model_src.read_text())
            print(f"✓ Copied pre-trained model to {pretrained_model_dst}")

    if cfg.pretrained_scaler_path:
        pretrained_scaler_src = Path(cfg.pretrained_scaler_path)
        if pretrained_scaler_src.exists():
            pretrained_scaler_dst = run_dir / "pretrained_scaler.json"
            pretrained_scaler_dst.write_text(pretrained_scaler_src.read_text())
            print(f"✓ Copied pre-trained scaler to {pretrained_scaler_dst}")

    # Global scaler - fit on training subjects only
    scaler_payload = None
    if cfg.scaler == "global":
        train_mask = np.isin(subject_ids, subjects_train)
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(X[train_mask])
        scaler_payload = {
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
            "var": scaler.var_.astype(float).tolist(),
        }
        scaler_path = run_dir / "scaler_global.json"
        scaler_path.write_text(json.dumps(scaler_payload, indent=2))

    def _apply_scale(X_arr: np.ndarray) -> np.ndarray:
        if scaler_payload is None:
            return X_arr.astype(np.float32)
        mean = np.array(scaler_payload["mean"], dtype=np.float64)
        scale = np.array(scaler_payload["scale"], dtype=np.float64)
        scale = np.where(scale == 0.0, 1.0, scale)
        return ((X_arr - mean) / scale).astype(np.float32)

    # Create clients mapping (only training subjects)
    clients_map = {}

    # Create per-node splits
    for node_id, node_subjects in node_map.items():
        node_dir = run_dir / node_id
        node_dir.mkdir(exist_ok=True)
        clients_map[node_id] = {}

        # Save aggregated fog-level splits
        for split_name, split_subjects in [
            ("train", subjects_train),
            ("val", subjects_val),
            ("test", subjects_test),
        ]:
            node_split_subjects = [s for s in node_subjects if s in split_subjects]

            if not node_split_subjects:
                # Empty split
                np.savez(
                    node_dir / f"{split_name}.npz",
                    X=np.empty((0, X.shape[1]), dtype=np.float32),
                    y=np.empty((0,), dtype=np.int64),
                    subjects=np.empty((0,), dtype=object),
                )
                continue

            mask = np.isin(subject_ids, node_split_subjects)
            X_split = _apply_scale(X[mask])
            y_split = y[mask]
            s_split = subject_ids[mask]
            np.savez(
                node_dir / f"{split_name}.npz",
                X=X_split,
                y=y_split,
                subjects=s_split,
            )

        # Create per-subject directories (ONLY for training subjects)
        for subj in node_subjects:
            if subj not in subjects_train:
                continue  # Skip non-training subjects

            subj_str = str(subj)
            client_id = f"{node_id}_client_{subj_str}"
            clients_map[node_id][client_id] = subj_str

            subj_dir = node_dir / f"subject_{subj_str}"
            subj_dir.mkdir(exist_ok=True)

            # Only create train.npz for training subjects
            mask = subject_ids == subj_str
            X_subj = _apply_scale(X[mask])
            y_subj = y[mask]
            s_subj = subject_ids[mask]

            np.savez(
                subj_dir / "train.npz",
                X=X_subj,
                y=y_subj,
                subjects=s_subj,
            )

            # Create empty val and test files
            for split_name in ["val", "test"]:
                np.savez(
                    subj_dir / f"{split_name}.npz",
                    X=np.empty((0, X.shape[1]), dtype=np.float32),
                    y=np.empty((0,), dtype=np.int64),
                    subjects=np.empty((0,), dtype=object),
                )

    # Save manifest
    total_clients = sum(len(clients) for clients in clients_map.values())
    manifest = {
        "config": {
            "data_dir": cfg.data_dir,
            "label_strategy": cfg.label_strategy,
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
            "pretrained_model_path": cfg.pretrained_model_path,
            "pretrained_scaler_path": cfg.pretrained_scaler_path,
            "freeze_initial_weights": cfg.freeze_initial_weights,
            "fine_tune_lr_multiplier": cfg.fine_tune_lr_multiplier,
        },
        "global_subjects": {
            "train": subjects_train,
            "val": subjects_val,
            "test": subjects_test,
            "all": all_subjects,
        },
        "nodes": node_map,
        "clients": clients_map,
        "meta": {
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "n_subjects": len(all_subjects),
            "n_train_subjects": len(subjects_train),
            "n_val_subjects": len(subjects_val),
            "n_test_subjects": len(subjects_test),
        },
        "output_dir": str(run_dir),
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n✓ SWEET federated splits materialized at: {run_dir}")
    print(f"  Strategy: global (each subject in ONE split only)")
    print(f"  Nodes: {len(node_map)}")
    print(f"  Features: {len(feature_names)}")
    print(
        f"  Total subjects: {len(all_subjects)} (train: {len(subjects_train)}, val: {len(subjects_val)}, test: {len(subjects_test)})"
    )
    print(f"  Total clients: {total_clients} (only training subjects)")
    if cfg.pretrained_model_path:
        print(
            f"  Transfer learning: Pre-trained model from {cfg.pretrained_model_path}"
        )
    for node_id, node_subjects in node_map.items():
        print(
            f"  {node_id}: {len(node_subjects)} subjects ({len(clients_map[node_id])} clients)"
        )

    return manifest


def _materialize_per_subject_strategy_sweet(
    cfg: FederatedConfigSWEET,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    all_subjects: list[str],
    feature_names: list[str],
) -> dict:
    """Per-subject strategy: each subject has internal train/val/test splits.

    All subjects become federated clients with local validation capability.
    """

    # Get unique subjects
    all_subjects = sorted(list(set(subject_ids.tolist())))

    # Node subject mapping (distribute ALL subjects across fog nodes)
    if cfg.mode == "manual":
        if not cfg.manual_assignments:
            raise ValueError("manual mode requires 'manual_assignments'")
        node_map = {
            node: [str(s) for s in subs]
            for node, subs in cfg.manual_assignments.items()
        }
    else:
        node_map = _auto_assign_nodes(
            all_subjects, cfg.num_fog_nodes, cfg.per_node_percentages, cfg.seed
        )

    # Prepare output directory
    run_id = cfg.run_name or f"run_{cfg.seed}"
    run_dir = Path(cfg.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy pre-trained model and scaler if provided
    if cfg.pretrained_model_path:
        pretrained_model_src = Path(cfg.pretrained_model_path)
        if pretrained_model_src.exists():
            pretrained_model_dst = run_dir / "pretrained_model.json"
            pretrained_model_dst.write_text(pretrained_model_src.read_text())
            print(f"✓ Copied pre-trained model to {pretrained_model_dst}")

    if cfg.pretrained_scaler_path:
        pretrained_scaler_src = Path(cfg.pretrained_scaler_path)
        if pretrained_scaler_src.exists():
            pretrained_scaler_dst = run_dir / "pretrained_scaler.json"
            pretrained_scaler_dst.write_text(pretrained_scaler_src.read_text())
            print(f"✓ Copied pre-trained scaler to {pretrained_scaler_dst}")

    # Global scaler - fit on ALL training data (from all subjects' training portions)
    scaler_payload = None
    if cfg.scaler == "global":
        # Collect training indices from all subjects
        train_indices = []
        for subj in all_subjects:
            subj_mask = subject_ids == subj
            subj_indices = np.where(subj_mask)[0]
            n_subj = len(subj_indices)
            if n_subj == 0:
                continue
            # Internal split for this subject
            n_train = max(1, int(round(cfg.split_train * n_subj)))
            # Reproducible shuffle per subject
            subj_rng = np.random.default_rng(cfg.seed + hash(subj) % (2**31))
            perm = subj_rng.permutation(n_subj)
            train_indices.extend(subj_indices[perm[:n_train]])

        if train_indices:
            scaler = StandardScaler()
            scaler.fit(X[train_indices])
            scaler_payload = {
                "mean": scaler.mean_.astype(float).tolist(),
                "scale": scaler.scale_.astype(float).tolist(),
                "var": scaler.var_.astype(float).tolist(),
            }
            scaler_path = run_dir / "scaler_global.json"
            scaler_path.write_text(json.dumps(scaler_payload, indent=2))

    def _apply_scale(X_arr: np.ndarray) -> np.ndarray:
        if scaler_payload is None:
            return X_arr.astype(np.float32)
        mean = np.array(scaler_payload["mean"], dtype=np.float64)
        scale = np.array(scaler_payload["scale"], dtype=np.float64)
        scale = np.where(scale == 0.0, 1.0, scale)
        return ((X_arr - mean) / scale).astype(np.float32)

    # Track subjects with train data and client mapping
    subjects_with_train = set()
    clients_map = {}
    fog_splits = {node: {"train": [], "val": [], "test": []} for node in node_map}

    # Create per-subject splits (per_subject strategy)
    for node_id, node_subjects in node_map.items():
        node_dir = run_dir / node_id
        node_dir.mkdir(exist_ok=True)
        clients_map[node_id] = {}

        for subj in node_subjects:
            subj_str = str(subj)
            client_id = f"{node_id}_client_{subj_str}"
            clients_map[node_id][client_id] = subj_str

            subj_dir = node_dir / f"subject_{subj_str}"
            subj_dir.mkdir(exist_ok=True)

            # Get all data for this subject
            subj_mask = subject_ids == subj_str
            X_subj = X[subj_mask]
            y_subj = y[subj_mask]

            if len(X_subj) == 0:
                # Empty subject - create empty files
                for split_name in ("train", "val", "test"):
                    np.savez(
                        subj_dir / f"{split_name}.npz",
                        X=np.empty((0, X.shape[1]), dtype=np.float32),
                        y=np.empty((0,), dtype=np.int64),
                        subjects=np.empty((0,), dtype=object),
                    )
                continue

            # Internal split for this subject
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

            # Reproducible shuffle per subject
            subj_rng = np.random.default_rng(cfg.seed + hash(subj_str) % (2**31))
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
                    fog_splits[node_id][split_name].append((X_split, y_split, subj_arr))
                else:
                    X_split = np.empty((0, X.shape[1]), dtype=np.float32)
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
            parts = fog_splits[node_id][split_name]
            if parts:
                X_agg = np.concatenate([p[0] for p in parts], axis=0)
                y_agg = np.concatenate([p[1] for p in parts], axis=0)
                s_agg = np.concatenate([p[2] for p in parts], axis=0)
            else:
                X_agg = np.empty((0, X.shape[1]), dtype=np.float32)
                y_agg = np.empty((0,), dtype=np.int64)
                s_agg = np.empty((0,), dtype=object)

            np.savez(node_dir / f"{split_name}.npz", X=X_agg, y=y_agg, subjects=s_agg)

    # Save manifest
    manifest = {
        "config": {
            "data_dir": cfg.data_dir,
            "label_strategy": cfg.label_strategy,
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
            "pretrained_model_path": cfg.pretrained_model_path,
            "pretrained_scaler_path": cfg.pretrained_scaler_path,
            "freeze_initial_weights": cfg.freeze_initial_weights,
            "fine_tune_lr_multiplier": cfg.fine_tune_lr_multiplier,
        },
        "global_subjects": {
            "train": sorted(subjects_with_train),
            "val": sorted(all_subjects),  # All subjects have val (internal)
            "test": sorted(all_subjects),  # All subjects have test (internal)
            "all": sorted(all_subjects),
        },
        "nodes": node_map,
        "clients": clients_map,
        "meta": {
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "n_subjects": len(all_subjects),
        },
        "output_dir": str(run_dir),
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n✓ SWEET federated splits materialized at: {run_dir}")
    print(f"  Strategy: per_subject (each subject has internal train/val/test)")
    print(f"  Nodes: {len(node_map)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Total subjects: {len(all_subjects)}")
    print(f"  Total clients: {sum(len(clients) for clients in clients_map.values())}")
    print(
        f"  Split ratios: train={cfg.split_train:.0%}, val={cfg.split_val:.0%}, test={cfg.split_test:.0%}"
    )
    if cfg.pretrained_model_path:
        print(
            f"  Transfer learning: Pre-trained model from {cfg.pretrained_model_path}"
        )
    for node_id, node_subjects in node_map.items():
        print(
            f"  {node_id}: {len(node_subjects)} subjects ({len(clients_map[node_id])} clients)"
        )

    return manifest


def load_node_split(
    split_file: str | os.PathLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a saved npz split (X, y, subjects).

    Args:
        split_file: Path to the .npz file (e.g., fog_0/train.npz or fog_0/subject_123/train.npz)

    Returns:
        Tuple of (X, y, subjects) arrays
    """
    arr = np.load(split_file, allow_pickle=True)
    return arr["X"], arr["y"], arr["subjects"]


def load_subject_split(
    run_dir: str | Path,
    node_id: str,
    subject_id: str,
    split: Literal["train", "val", "test"] = "train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load split for a specific subject within a node.

    Args:
        run_dir: Path to federated run directory
        node_id: Node identifier (e.g., 'fog_0')
        subject_id: Subject identifier
        split: Which split to load

    Returns:
        Tuple of (X, y, subjects) arrays
    """
    path = Path(run_dir) / node_id / f"subject_{subject_id}" / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Subject split not found: {path}")

    return load_node_split(path)
