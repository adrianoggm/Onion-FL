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
    split_strategy: SplitStrategy = "global"  # "global" = each subject in ONE split only (STRICT)
    
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
        freeze_initial_weights=bool(transfer_learning.get("freeze_initial_weights", False)),
        fine_tune_lr_multiplier=float(transfer_learning.get("fine_tune_lr_multiplier", 0.1)),
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
    
    Output structure (under output_dir/run_name):
      - manifest.json: full config + subject splits + nodes mapping
      - scaler_global.json (if scaler==global)
      - pretrained_model.json, pretrained_scaler.json (if transfer learning)
      - fog_<id>/train.npz, val.npz, test.npz
      - fog_<id>/subject_<N>/train.npz, val.npz, test.npz
    
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
    all_subjects = list(np.unique(subject_ids))
    
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
    
    # Node subject mapping
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
    
    # Global scaler (if needed)
    scaler = None
    if cfg.scaler == "global":
        scaler = StandardScaler()
        # Fit only on training subjects
        train_mask = np.isin(subject_ids, subjects_train)
        scaler.fit(X[train_mask])
        
        scaler_path = run_dir / "scaler_global.json"
        scaler_data = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
        }
        scaler_path.write_text(json.dumps(scaler_data, indent=2))
    
    # Create per-node splits
    node_stats = {}
    for node_id, node_subjects in node_map.items():
        node_dir = run_dir / node_id
        node_dir.mkdir(exist_ok=True)
        
        # Aggregate node data
        train_X_list, train_y_list = [], []
        val_X_list, val_y_list = [], []
        test_X_list, test_y_list = [], []
        
        for subj in node_subjects:
            # Get subject data from full dataset
            subj_mask = subject_ids == subj
            X_subj = X[subj_mask]
            y_subj = y[subj_mask]
            
            # Determine split for this subject
            if subj in subjects_train:
                train_X_list.append(X_subj)
                train_y_list.append(y_subj)
            elif subj in subjects_val:
                val_X_list.append(X_subj)
                val_y_list.append(y_subj)
            elif subj in subjects_test:
                test_X_list.append(X_subj)
                test_y_list.append(y_subj)
            
            # Per-subject splits
            subj_dir = node_dir / f"subject_{subj}"
            subj_dir.mkdir(exist_ok=True)
            
            if subj in subjects_train:
                X_s = X_subj.copy()
                if scaler:
                    X_s = scaler.transform(X_s)
                np.savez(
                    subj_dir / "train.npz",
                    X=X_s,
                    y=y_subj,
                    subject_id=np.array([subj] * len(y_subj)),
                )
            
            if subj in subjects_val:
                X_s = X_subj.copy()
                if scaler:
                    X_s = scaler.transform(X_s)
                np.savez(
                    subj_dir / "val.npz",
                    X=X_s,
                    y=y_subj,
                    subject_id=np.array([subj] * len(y_subj)),
                )
            
            if subj in subjects_test:
                X_s = X_subj.copy()
                if scaler:
                    X_s = scaler.transform(X_s)
                np.savez(
                    subj_dir / "test.npz",
                    X=X_s,
                    y=y_subj,
                    subject_id=np.array([subj] * len(y_subj)),
                )
        
        # Aggregated node splits
        if train_X_list:
            train_X_agg = np.vstack(train_X_list)
            train_y_agg = np.concatenate(train_y_list)
            if scaler:
                train_X_agg = scaler.transform(train_X_agg)
            np.savez(node_dir / "train.npz", X=train_X_agg, y=train_y_agg)
        
        if val_X_list:
            val_X_agg = np.vstack(val_X_list)
            val_y_agg = np.concatenate(val_y_list)
            if scaler:
                val_X_agg = scaler.transform(val_X_agg)
            np.savez(node_dir / "val.npz", X=val_X_agg, y=val_y_agg)
        
        if test_X_list:
            test_X_agg = np.vstack(test_X_list)
            test_y_agg = np.concatenate(test_y_list)
            if scaler:
                test_X_agg = scaler.transform(test_X_agg)
            np.savez(node_dir / "test.npz", X=test_X_agg, y=test_y_agg)
        
        node_stats[node_id] = {
            "subjects": node_subjects,
            "train_samples": sum(len(y) for y in train_y_list) if train_y_list else 0,
            "val_samples": sum(len(y) for y in val_y_list) if val_y_list else 0,
            "test_samples": sum(len(y) for y in test_y_list) if test_y_list else 0,
        }
    
    # Save manifest
    manifest = {
        "config": {
            "data_dir": cfg.data_dir,
            "label_strategy": cfg.label_strategy,
            "seed": cfg.seed,
            "scaler": cfg.scaler,
            "split_strategy": cfg.split_strategy,
            "pretrained_model_path": cfg.pretrained_model_path,
            "pretrained_scaler_path": cfg.pretrained_scaler_path,
            "freeze_initial_weights": cfg.freeze_initial_weights,
            "fine_tune_lr_multiplier": cfg.fine_tune_lr_multiplier,
        },
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "nodes": node_stats,
        "node_mapping": node_map,
        "output_dir": str(run_dir),
        "subjects_train": subjects_train,
        "subjects_val": subjects_val,
        "subjects_test": subjects_test,
    }
    
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    print(f"✓ SWEET federated splits materialized at: {run_dir}")
    print(f"  Nodes: {len(node_map)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Total subjects: {len(all_subjects)} (train: {len(subjects_train)}, val: {len(subjects_val)}, test: {len(subjects_test)})")
    if cfg.pretrained_model_path:
        print(f"  Transfer learning: Pre-trained model from {cfg.pretrained_model_path}")
    for node_id, stats in node_stats.items():
        print(
            f"  {node_id}: {len(stats['subjects'])} subjects, "
            f"{stats['train_samples']} train, {stats['val_samples']} val, "
            f"{stats['test_samples']} test samples"
        )
    
    return manifest


def load_node_split(
    run_dir: str | Path,
    node_id: str,
    split: Literal["train", "val", "test"] = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Load aggregated split for a fog node.
    
    Args:
        run_dir: Path to federated run directory
        node_id: Node identifier (e.g., 'fog_0')
        split: Which split to load
        
    Returns:
        Tuple of (X, y) arrays
    """
    path = Path(run_dir) / node_id / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    
    data = np.load(path)
    return data["X"], data["y"]


def load_subject_split(
    run_dir: str | Path,
    node_id: str,
    subject_id: str,
    split: Literal["train", "val", "test"] = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Load split for a specific subject within a node.
    
    Args:
        run_dir: Path to federated run directory
        node_id: Node identifier (e.g., 'fog_0')
        subject_id: Subject identifier
        split: Which split to load
        
    Returns:
        Tuple of (X, y) arrays
    """
    path = Path(run_dir) / node_id / f"subject_{subject_id}" / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Subject split not found: {path}")
    
    data = np.load(path)
    return data["X"], data["y"]
