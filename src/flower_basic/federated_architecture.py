from __future__ import annotations

"""Config-driven orchestration utilities for fog–cloud federated learning.

This module ingests a JSON/YAML architecture description (orchestrator, fog
nodes, clients, datasets/workflows) and produces:
  - A validated, strongly typed in-memory representation
  - MQTT payloads to distribute per-fog configuration
  - A runtime plan (commands + environment) to launch the existing components
    (server, fog bridge, broker, dataset-specific clients)
"""

import json
import os
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

SUPPORTED_WORKFLOWS = {"swell", "wesad", "sweet"}


@dataclass
class MQTTTopics:
    updates: str = "fl/updates"
    partial: str = "fl/partial"
    global_model: str = "fl/global_model"


@dataclass
class MQTTConfig:
    broker: str = "localhost"
    port: int = 1883
    topics: MQTTTopics = field(default_factory=MQTTTopics)


@dataclass
class ModelConfig:
    """Global model settings."""

    type: str = "ecg_cnn"
    input_dim: int | None = None


@dataclass
class DatasetConfig:
    """Dataset/split settings so the orchestrator can materialize shards."""

    name: str = "swell"
    data_dir: str = "data/SWELL"
    modalities: list[str] | None = None
    subjects: list[int] | None = None
    seed: int = 42
    split_train: float = 0.5
    split_val: float = 0.2
    split_test: float = 0.3
    split_strategy: str = "per_subject"
    scaler: str = "global"
    output_dir: str = "federated_runs/swell"
    run_name: str | None = None
    mode: str = "auto"
    manual_assignments: dict[str, list[int]] | None = None
    per_node_percentages: list[float] | None = None
    ensure_min_train_per_node: bool = True
    test_assignments: dict[str, list[int]] | None = None


@dataclass
class ClientSpec:
    id: str
    dataset: str
    rounds: int = 3
    data_dir: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    workflow: str | None = None


@dataclass
class FogNodeSpec:
    id: str
    address: str | None = None
    k: int = 1
    clients: list[ClientSpec] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorSpec:
    address: str = "0.0.0.0:8080"
    protocol: str = "MQTT"
    rounds: int = 3
    stale_update_policy: str = "accept"
    mqtt: MQTTConfig = field(default_factory=MQTTConfig)


@dataclass
class FederatedArchitecture:
    orchestrator: OrchestratorSpec
    fog_nodes: list[FogNodeSpec]
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig | None = None
    workflow: str | None = None
    client_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeCommand:
    """Command + environment to launch a federated component."""

    role: str
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass(frozen=True)
class ManifestSubjectStatus:
    """Filesystem-backed readiness check for one manifest subject."""

    subject_dir: str
    has_train_data: bool
    reason: str | None = None


@dataclass(frozen=True)
class ManifestApplicationResult:
    """Pure result of applying a manifest to an architecture."""

    architecture: FederatedArchitecture
    messages: list[str]


@dataclass(frozen=True)
class SwellMaterializationPlan:
    """Pure plan describing how SWELL partitions should be materialized."""

    config: dict[str, Any]
    config_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SwellMaterializationResult:
    """Result of materializing SWELL partitions and rehydrating the architecture."""

    architecture: FederatedArchitecture
    config_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class RuntimePlan:
    """Resolved architecture plus runnable commands."""

    architecture: FederatedArchitecture
    commands: list[RuntimeCommand]


def _normalize_workflow(name: str | None) -> str | None:
    if name is None:
        return None
    slug = str(name).strip().lower()
    if slug not in SUPPORTED_WORKFLOWS:
        return slug
    return slug


def _read_architecture_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML no está instalado; usa JSON o instala pyyaml")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("La configuración debe ser un objeto JSON/YAML")
    return data


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _resolve_path(path_like: str | os.PathLike[str], repo_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return repo_root / path


def parse_architecture_config(raw: Mapping[str, Any]) -> FederatedArchitecture:
    """Parse a config mapping into a validated architecture without I/O."""
    root = _as_mapping(raw.get("federated_architecture", raw))
    if not root:
        raise ValueError("La configuración debe incluir un objeto federated_architecture")

    orchestrator_raw = _as_mapping(root.get("orchestrator"))
    mqtt_raw = _as_mapping(orchestrator_raw.get("mqtt"))
    topics_raw = _as_mapping(mqtt_raw.get("topics"))

    orchestrator = OrchestratorSpec(
        address=str(orchestrator_raw.get("address", "0.0.0.0:8080")),
        protocol=str(orchestrator_raw.get("protocol", "MQTT")),
        rounds=int(orchestrator_raw.get("rounds", 3)),
        stale_update_policy=str(
            orchestrator_raw.get("stale_update_policy", "accept")
        ).lower(),
        mqtt=MQTTConfig(
            broker=str(mqtt_raw.get("broker", "localhost")),
            port=int(mqtt_raw.get("port", 1883)),
            topics=MQTTTopics(
                updates=str(topics_raw.get("updates", "fl/updates")),
                partial=str(topics_raw.get("partial", "fl/partial")),
                global_model=str(topics_raw.get("global_model", "fl/global_model")),
            ),
        ),
    )

    model_raw = _as_mapping(root.get("model"))
    model = ModelConfig(
        type=str(model_raw.get("type", "ecg_cnn")),
        input_dim=model_raw.get("input_dim"),
    )

    dataset_raw = _as_mapping(root.get("dataset"))
    split_raw = _as_mapping(dataset_raw.get("split"))
    dataset = None
    if dataset_raw:
        dataset = DatasetConfig(
            name=str(
                dataset_raw.get("name", dataset_raw.get("dataset", "swell"))
            ).lower(),
            data_dir=str(dataset_raw.get("data_dir", "data/SWELL")),
            modalities=deepcopy(dataset_raw.get("modalities")),
            subjects=deepcopy(dataset_raw.get("subjects")),
            seed=int(dataset_raw.get("seed", split_raw.get("seed", 42))),
            split_train=float(split_raw.get("train", 0.5)),
            split_val=float(split_raw.get("val", 0.2)),
            split_test=float(split_raw.get("test", 0.3)),
            split_strategy=str(split_raw.get("strategy", "per_subject")),
            scaler=str(split_raw.get("scaler", "global")),
            output_dir=str(dataset_raw.get("output_dir", "federated_runs/swell")),
            run_name=dataset_raw.get("run_name"),
            mode=str(dataset_raw.get("mode", "auto")),
            manual_assignments=deepcopy(dataset_raw.get("manual_assignments")),
            per_node_percentages=deepcopy(dataset_raw.get("per_node_percentages")),
            ensure_min_train_per_node=bool(
                dataset_raw.get("ensure_min_train_per_node", True)
            ),
            test_assignments=deepcopy(dataset_raw.get("test_assignments")),
        )

    fog_nodes: list[FogNodeSpec] = []
    for node_raw_any in _as_list(root.get("fog_nodes")):
        node_raw = _as_mapping(node_raw_any)
        clients: list[ClientSpec] = []
        for client_raw_any in _as_list(node_raw.get("clients")):
            client_raw = _as_mapping(client_raw_any)
            clients.append(
                ClientSpec(
                    id=str(client_raw.get("id")),
                    dataset=str(client_raw.get("dataset")),
                    rounds=int(client_raw.get("rounds", 3)),
                    data_dir=client_raw.get("data_dir"),
                    params=deepcopy(_as_mapping(client_raw.get("params"))),
                    workflow=_normalize_workflow(
                        client_raw.get("workflow") or client_raw.get("dataset")
                    ),
                )
            )
        fog_nodes.append(
            FogNodeSpec(
                id=str(node_raw.get("id")),
                address=node_raw.get("address"),
                k=int(node_raw.get("k", node_raw.get("K", 1))),
                clients=clients,
                params=deepcopy(_as_mapping(node_raw.get("params"))),
            )
        )

    arch = FederatedArchitecture(
        orchestrator=orchestrator,
        fog_nodes=fog_nodes,
        model=model,
        dataset=dataset,
        workflow=_normalize_workflow(root.get("workflow")),
        client_params=deepcopy(_as_mapping(root.get("client_params"))),
    )
    _validate_architecture(arch)
    return arch


def load_architecture_config(config_path: str | os.PathLike) -> FederatedArchitecture:
    """Load and validate a federated architecture config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")

    return parse_architecture_config(_read_architecture_file(path))


def _validate_architecture(arch: FederatedArchitecture) -> None:
    if not arch.fog_nodes:
        raise ValueError("La configuración debe definir al menos un nodo fog")
    for fog in arch.fog_nodes:
        if not fog.clients:
            raise ValueError(f"El nodo fog {fog.id} no tiene clientes configurados")
        for client in fog.clients:
            if not client.dataset:
                raise ValueError(f"Cliente sin dataset en nodo {fog.id}")
    if arch.dataset is not None:
        total = (
            arch.dataset.split_train + arch.dataset.split_val + arch.dataset.split_test
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Las proporciones de split (train/val/test) deben sumar 1.0"
            )
    if arch.orchestrator.stale_update_policy not in {"accept", "strict"}:
        raise ValueError(
            "orchestrator.stale_update_policy debe ser 'accept' o 'strict'"
        )


def infer_primary_workflow(arch: FederatedArchitecture) -> str:
    """Infer the main workflow (SWELL/SWEET/WESAD) for the run."""
    if arch.workflow:
        return arch.workflow
    for fog in arch.fog_nodes:
        for client in fog.clients:
            wf = _normalize_workflow(client.workflow or client.dataset)
            if wf:
                return wf
    return "wesad"


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _infer_input_dim_from_clients(
    arch: FederatedArchitecture, repo_root: Path
) -> int | None:
    """Try to infer input_dim from the first available client split (train.npz)."""
    for fog in arch.fog_nodes:
        for client in fog.clients:
            if not client.data_dir:
                continue
            p = Path(client.data_dir)
            if not p.is_absolute():
                p = repo_root / p
            train_npz = p / "train.npz"
            if not train_npz.exists():
                continue
            try:
                import numpy as np  # local import to avoid hard dep at module import time

                arr = np.load(train_npz, allow_pickle=True)
                X = arr["X"]
                return int(X.shape[1])
            except Exception:
                continue
    return None


def _run_swell_materialization(config_path: str) -> dict[str, Any]:
    from .datasets.swell_federated import plan_and_materialize_swell_federated

    return plan_and_materialize_swell_federated(config_path)


def materialize_swell_partitions(
    arch: FederatedArchitecture, repo_root: Path | None = None
) -> SwellMaterializationResult:
    """Materialize SWELL federated splits and return a new architecture."""
    if arch.dataset is None:
        raise ValueError(
            "La configuración debe incluir sección 'dataset' para preparar particiones SWELL"
        )

    repo_root = repo_root or _default_repo_root()
    ds = arch.dataset
    run_name = ds.run_name or f"arch_{int(time.time())}"
    plan = plan_swell_materialization(arch, repo_root=repo_root, run_name=run_name)

    plan.config_path.parent.mkdir(parents=True, exist_ok=True)
    plan.config_path.write_text(json.dumps(plan.config, indent=2), encoding="utf-8")

    result = _run_swell_materialization(str(plan.config_path))
    manifest_path = Path(result["output_dir"]) / "manifest.json"
    updated_arch = apply_manifest_paths(arch, manifest_path)
    return SwellMaterializationResult(
        architecture=updated_arch,
        config_path=plan.config_path,
        manifest_path=manifest_path,
    )


def plan_swell_materialization(
    arch: FederatedArchitecture, repo_root: Path, run_name: str | None = None
) -> SwellMaterializationPlan:
    """Build the SWELL split-materialization config without writing files."""
    if arch.dataset is None:
        raise ValueError(
            "La configuración debe incluir sección 'dataset' para preparar particiones SWELL"
        )

    ds = arch.dataset
    resolved_run_name = run_name or ds.run_name
    if resolved_run_name is None:
        raise ValueError(
            "dataset.run_name es obligatorio para planificar la materialización SWELL"
        )

    output_dir = _resolve_path(ds.output_dir, repo_root)
    data_dir = _resolve_path(ds.data_dir, repo_root)
    config = {
        "dataset": {
            "data_dir": str(data_dir),
            "modalities": deepcopy(ds.modalities),
            "subjects": deepcopy(ds.subjects),
        },
        "split": {
            "train": ds.split_train,
            "val": ds.split_val,
            "test": ds.split_test,
            "seed": ds.seed,
            "scaler": ds.scaler,
            "strategy": ds.split_strategy,
        },
        "federation": {
            "mode": ds.mode,
            "num_fog_nodes": len(arch.fog_nodes),
            "manual_assignments": deepcopy(ds.manual_assignments),
            "per_node_percentages": deepcopy(ds.per_node_percentages),
            "output_dir": str(output_dir),
            "run_name": resolved_run_name,
            "ensure_min_train_per_node": ds.ensure_min_train_per_node,
            "test_assignments": deepcopy(ds.test_assignments),
        },
    }
    run_dir = output_dir / resolved_run_name
    return SwellMaterializationPlan(
        config=config,
        config_path=run_dir / "_arch_auto_config.json",
        manifest_path=run_dir / "manifest.json",
    )


def _inspect_manifest_subjects(
    manifest: Mapping[str, Any], manifest_base: Path
) -> tuple[dict[str, dict[str, ManifestSubjectStatus]], int | None]:
    """Read per-subject train files so the pure planner can stay side-effect free."""
    subject_statuses: dict[str, dict[str, ManifestSubjectStatus]] = {}
    inferred_n_features: int | None = None
    nodes = _as_mapping(manifest.get("nodes"))

    for fog_id, subjects in nodes.items():
        fog_statuses: dict[str, ManifestSubjectStatus] = {}
        for subject_id in _as_list(subjects):
            subject_str = str(subject_id)
            subject_dir = manifest_base / str(fog_id) / f"subject_{subject_str}"
            train_file = subject_dir / "train.npz"
            if not train_file.exists():
                fog_statuses[subject_str] = ManifestSubjectStatus(
                    subject_dir=str(subject_dir),
                    has_train_data=False,
                    reason="train.npz not found",
                )
                continue

            import numpy as np

            try:
                arr = np.load(train_file, allow_pickle=True)
                x_data = arr["X"]
            except Exception as exc:
                fog_statuses[subject_str] = ManifestSubjectStatus(
                    subject_dir=str(subject_dir),
                    has_train_data=False,
                    reason=f"error reading train.npz: {exc}",
                )
                continue

            if inferred_n_features is None and getattr(x_data, "ndim", 0) >= 2:
                inferred_n_features = int(x_data.shape[1])
            if x_data.shape[0] == 0:
                fog_statuses[subject_str] = ManifestSubjectStatus(
                    subject_dir=str(subject_dir),
                    has_train_data=False,
                    reason="train.npz is empty",
                )
                continue

            fog_statuses[subject_str] = ManifestSubjectStatus(
                subject_dir=str(subject_dir),
                has_train_data=True,
            )

        subject_statuses[str(fog_id)] = fog_statuses

    return subject_statuses, inferred_n_features


def plan_manifest_application(
    arch: FederatedArchitecture,
    manifest: Mapping[str, Any],
    subject_statuses: Mapping[str, Mapping[str, ManifestSubjectStatus]],
    *,
    n_features: int | None = None,
) -> ManifestApplicationResult:
    """Apply a manifest to an architecture without mutating the input."""
    nodes = _as_mapping(manifest.get("nodes"))
    clients_map = _as_mapping(manifest.get("clients"))
    global_subjects = _as_mapping(manifest.get("global_subjects"))
    config = _as_mapping(manifest.get("config"))
    split_config = _as_mapping(config.get("split"))
    split_strategy = str(split_config.get("strategy", "global"))

    messages: list[str] = []
    if split_strategy == "per_subject":
        train_subjects = {str(subject_id) for subject_id in _as_list(global_subjects.get("all"))}
        messages.append(
            f"[MANIFEST] Strategy: per_subject - all {len(train_subjects)} subjects have train data"
        )
    else:
        train_subjects = {
            str(subject_id) for subject_id in _as_list(global_subjects.get("train"))
        }
        messages.append(
            f"[MANIFEST] Strategy: global - only {len(train_subjects)} subjects in train split"
        )

    resolved_arch = deepcopy(arch)
    if n_features is not None:
        resolved_arch.model.input_dim = int(n_features)

    for fog in resolved_arch.fog_nodes:
        if fog.id not in nodes:
            continue

        fog_clients = _as_mapping(clients_map.get(fog.id))
        if not fog_clients:
            continue

        statuses_for_fog = subject_statuses.get(fog.id, {})
        new_clients: list[ClientSpec] = []
        for client_id, subject_id in fog_clients.items():
            subject_str = str(subject_id)
            if subject_str not in train_subjects:
                messages.append(
                    f"[MANIFEST] Skipping {client_id} (subject {subject_str} has no train data)"
                )
                continue

            status = statuses_for_fog.get(subject_str)
            if status is None or not status.has_train_data:
                reason = (
                    status.reason if status is not None and status.reason else "train.npz not found"
                )
                messages.append(f"[MANIFEST] Skipping {client_id} ({reason})")
                continue

            new_clients.append(
                ClientSpec(
                    id=str(client_id),
                    dataset="swell",
                    workflow="swell",
                    rounds=arch.orchestrator.rounds,
                    data_dir=status.subject_dir,
                )
            )

        available_clients = len(new_clients)
        new_k = fog.k
        if available_clients == 0:
            messages.append(f"[MANIFEST] {fog.id}: 0 clientes con training data")
        else:
            if new_k <= 0:
                new_k = available_clients
            elif new_k > available_clients:
                messages.append(
                    f"[MANIFEST] Adjusting {fog.id} K from {fog.k} "
                    f"to {available_clients} to match spawned clients"
                )
                new_k = available_clients
            messages.append(
                f"[MANIFEST] {fog.id}: {available_clients} clientes con training data (K={new_k})"
            )

        fog.clients = new_clients
        fog.k = new_k

    return ManifestApplicationResult(architecture=resolved_arch, messages=messages)


def apply_manifest_paths(
    arch: FederatedArchitecture,
    manifest_path: Path | str,
    *,
    emit: Callable[[str], None] | None = print,
) -> FederatedArchitecture:
    """Rehydrate SWELL clients from a generated manifest and return a new arch."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta = _as_mapping(manifest.get("meta"))
    subject_statuses, inferred_n_features = _inspect_manifest_subjects(
        manifest, manifest_path.parent
    )
    resolved_n_features = meta.get("n_features", inferred_n_features)
    result = plan_manifest_application(
        arch,
        manifest,
        subject_statuses,
        n_features=int(resolved_n_features) if resolved_n_features is not None else None,
    )
    if emit is not None:
        for message in result.messages:
            emit(message)
    return result.architecture


def resolve_runtime_architecture(
    arch: FederatedArchitecture, *, inferred_input_dim: int | None = None
) -> FederatedArchitecture:
    """Resolve derived runtime fields without mutating the input architecture."""
    resolved_arch = deepcopy(arch)
    primary = infer_primary_workflow(arch)
    if primary != "swell":
        return resolved_arch

    input_dim = resolved_arch.model.input_dim
    if input_dim is None:
        input_dim = inferred_input_dim
    if input_dim is None:
        raise ValueError("model.input_dim es obligatorio para ejecutar el flujo SWELL")
    resolved_arch.model.input_dim = int(input_dim)
    return resolved_arch


def plan_runtime_commands(
    arch: FederatedArchitecture,
    repo_root: Path,
    *,
    python_exec: str,
    python_path: str | None = None,
    manifest_path: Path | str | None = None,
) -> list[RuntimeCommand]:
    """Translate an already-resolved architecture into runnable commands."""
    mqtt = arch.orchestrator.mqtt
    topics = mqtt.topics
    primary = infer_primary_workflow(arch)
    py_path = str(repo_root / "src")
    if python_path:
        py_path = py_path + os.pathsep + python_path
    env_path = {"PYTHONPATH": py_path}

    def _merge_env(extra: dict[str, str] | None = None) -> dict[str, str]:
        env = env_path.copy()
        if extra:
            env.update(extra)
        return env

    for fog in arch.fog_nodes:
        for client in fog.clients:
            wf = _normalize_workflow(client.workflow or client.dataset)
            if wf and wf != primary:
                raise ValueError(
                    f"Workflow mixto no soportado en un mismo run: primario={primary}, cliente={client.id} usa {wf}"
                )

    active_fogs = [fog for fog in arch.fog_nodes if fog.clients]
    if not active_fogs:
        raise ValueError("No hay nodos fog activos con clientes para lanzar")

    k_map = {fog.id: fog.k for fog in active_fogs}
    broker_env = {
        "MQTT_BROKER": mqtt.broker,
        "MQTT_PORT": str(mqtt.port),
        "MQTT_TOPIC_UPDATES": topics.updates,
        "MQTT_TOPIC_PARTIAL": topics.partial,
        "MQTT_TOPIC_GLOBAL": topics.global_model,
    }
    broker_k_arg = None
    if len(set(k_map.values())) > 1:
        broker_env["FOG_K_MAP"] = json.dumps(k_map)
    else:
        broker_k_arg = next(iter(k_map.values())) if k_map else 1

    commands: list[RuntimeCommand] = []
    if primary == "swell":
        if arch.model.input_dim is None:
            raise ValueError("model.input_dim es obligatorio para ejecutar el flujo SWELL")
        server_cmd = [
            python_exec,
            "-m",
            "flower_basic.servers.swell",
            "--input_dim",
            str(int(arch.model.input_dim)),
            "--rounds",
            str(arch.orchestrator.rounds),
            "--mqtt-broker",
            mqtt.broker,
            "--mqtt-port",
            str(mqtt.port),
            "--topic-global",
            topics.global_model,
            "--min-fit-clients",
            str(len(active_fogs)),
            "--min-available-clients",
            str(len(active_fogs)),
        ]
        if manifest_path is not None:
            server_cmd.extend(["--manifest", str(manifest_path)])
    else:
        server_cmd = [
            python_exec,
            "-m",
            "flower_basic.server",
            "--server_addr",
            arch.orchestrator.address,
            "--rounds",
            str(arch.orchestrator.rounds),
            "--mqtt-broker",
            mqtt.broker,
            "--mqtt-port",
            str(mqtt.port),
            "--topic-global",
            topics.global_model,
        ]
    commands.append(
        RuntimeCommand(
            role="server",
            cmd=server_cmd,
            cwd=str(repo_root),
            env=_merge_env(),
        )
    )

    if primary == "swell":
        if arch.model.input_dim is None:
            raise ValueError("model.input_dim es obligatorio para ejecutar el flujo SWELL")
        for fog in active_fogs:
            bridge_cmd = [
                python_exec,
                "-m",
                "flower_basic.clients.fog_bridge_swell",
                "--input_dim",
                str(int(arch.model.input_dim)),
                "--server",
                arch.orchestrator.address,
                "--region",
                fog.id,
                "--mqtt-broker",
                mqtt.broker,
                "--mqtt-port",
                str(mqtt.port),
                "--topic-partial",
                topics.partial,
            ]
            commands.append(
                RuntimeCommand(
                    role=f"fog_bridge_{fog.id}",
                    cmd=bridge_cmd,
                    cwd=str(repo_root),
                    env=_merge_env(),
                )
            )
    else:
        bridge_script = repo_root / "src" / "flower_basic" / "fog_flower_client.py"
        bridge_cmd = [
            python_exec,
            str(bridge_script),
            "--server",
            arch.orchestrator.address,
            "--mqtt-broker",
            mqtt.broker,
            "--mqtt-port",
            str(mqtt.port),
            "--topic-partial",
            topics.partial,
        ]
        commands.append(
            RuntimeCommand(
                role="fog_bridge",
                cmd=bridge_cmd,
                cwd=str(repo_root),
                env=_merge_env(),
            )
        )

    broker_cmd = [
        python_exec,
        "-m",
        "flower_basic.brokers.fog",
        "--mqtt-broker",
        mqtt.broker,
        "--mqtt-port",
        str(mqtt.port),
        "--topic-updates",
        topics.updates,
        "--topic-partial",
        topics.partial,
        "--topic-global",
        topics.global_model,
        "--stale-update-policy",
        arch.orchestrator.stale_update_policy,
    ]
    if broker_k_arg is not None:
        broker_cmd.extend(["--k", str(int(broker_k_arg))])
    commands.append(
        RuntimeCommand(
            role="broker",
            cmd=broker_cmd,
            env=_merge_env(broker_env),
            cwd=str(repo_root),
        )
    )

    client_index = 0
    for fog in active_fogs:
        for client in fog.clients:
            workflow = _normalize_workflow(client.workflow or client.dataset) or primary
            merged_params: dict[str, Any] = {}
            merged_params.update(arch.client_params or {})
            merged_params.update(fog.params or {})
            merged_params.update(client.params or {})
            env_client = {
                "MQTT_BROKER": mqtt.broker,
                "MQTT_PORT": str(mqtt.port),
                "MQTT_TOPIC_UPDATES": topics.updates,
                "MQTT_TOPIC_PARTIAL": topics.partial,
                "MQTT_TOPIC_GLOBAL": topics.global_model,
                "MQTT_REGION": fog.id,
            }
            if workflow == "swell":
                if client.data_dir is None:
                    raise ValueError(f"Cliente {client.id} requiere data_dir para SWELL")
                cmd = [
                    python_exec,
                    "-m",
                    "flower_basic.clients.swell",
                    "--node_dir",
                    str(client.data_dir),
                    "--region",
                    fog.id,
                    "--rounds",
                    str(client.rounds),
                    "--mqtt-broker",
                    mqtt.broker,
                    "--mqtt-port",
                    str(mqtt.port),
                    "--topic-updates",
                    topics.updates,
                    "--topic-global",
                    topics.global_model,
                    "--client-index",
                    str(client_index),
                    "--client-id",
                    client.id,
                ]
                if "lr" in merged_params and merged_params["lr"] is not None:
                    cmd.extend(["--lr", str(merged_params["lr"])])
                if (
                    "batch_size" in merged_params
                    and merged_params["batch_size"] is not None
                ):
                    cmd.extend(["--batch_size", str(merged_params["batch_size"])])
                if "seed" in merged_params and merged_params["seed"] is not None:
                    cmd.extend(["--seed", str(merged_params["seed"])])
                local_epochs = merged_params.get(
                    "local_epochs", merged_params.get("local-epochs")
                )
                if local_epochs is not None:
                    cmd.extend(["--local-epochs", str(local_epochs)])
                commands.append(
                    RuntimeCommand(
                        role=f"client_{client.id}",
                        cmd=cmd,
                        env=_merge_env(env_client),
                        cwd=str(repo_root),
                    )
                )
                client_index += 1
            elif workflow == "wesad":
                client_script = repo_root / "src" / "flower_basic" / "client.py"
                commands.append(
                    RuntimeCommand(
                        role=f"client_{client.id}",
                        cmd=[
                            python_exec,
                            str(client_script),
                            "--rounds",
                            str(client.rounds),
                            "--region",
                            fog.id,
                        ],
                        env=_merge_env(env_client),
                        cwd=str(repo_root),
                    )
                )
            else:
                raise ValueError(f"Workflow no soportado aún: {workflow}")

    return commands


def build_runtime_plan(
    arch: FederatedArchitecture,
    repo_root: Path | None = None,
    manifest_path: Path | str | None = None,
) -> RuntimePlan:
    """Resolve the architecture and build commands without mutating the input."""
    root = repo_root or _default_repo_root()
    inferred = None
    if infer_primary_workflow(arch) == "swell" and arch.model.input_dim is None:
        inferred = _infer_input_dim_from_clients(arch, root)

    resolved_arch = resolve_runtime_architecture(arch, inferred_input_dim=inferred)
    commands = plan_runtime_commands(
        resolved_arch,
        root,
        python_exec=os.getenv("PYTHON", sys.executable),
        python_path=os.getenv("PYTHONPATH"),
        manifest_path=manifest_path,
    )
    return RuntimePlan(architecture=resolved_arch, commands=commands)


def build_distribution_payloads(arch: FederatedArchitecture) -> list[dict[str, Any]]:
    """Build per-fog payloads without publishing them."""
    payloads: list[dict[str, Any]] = []
    topics = arch.orchestrator.mqtt.topics
    for fog in arch.fog_nodes:
        payloads.append(
            {
                "fog_id": fog.id,
                "k": fog.k,
                "clients": [
                    {
                        "id": c.id,
                        "dataset": c.dataset,
                        "workflow": _normalize_workflow(c.workflow or c.dataset),
                        "rounds": c.rounds,
                        "data_dir": c.data_dir,
                        "params": deepcopy(c.params),
                    }
                    for c in fog.clients
                ],
                "topics": {
                    "updates": topics.updates,
                    "partial": topics.partial,
                    "global_model": topics.global_model,
                },
                "orchestrator": {
                    "address": arch.orchestrator.address,
                    "rounds": arch.orchestrator.rounds,
                    "protocol": arch.orchestrator.protocol,
                },
            }
        )
    return payloads


def distribute_architecture(
    arch: FederatedArchitecture,
    mqtt_client: Any = None,
    base_topic: str = "fl/ctrl/plan",
) -> list[dict[str, Any]]:
    """Prepare and optionally publish per-fog configuration over MQTT."""
    payloads = build_distribution_payloads(arch)
    if mqtt_client is None:
        return payloads

    for payload in payloads:
        try:
            topic = f"{base_topic}/{payload['fog_id']}"
            mqtt_client.publish(topic, json.dumps(payload))
        except Exception:
            # Fall back silently; publishing is best-effort in tests/offline modes.
            pass
    return payloads
