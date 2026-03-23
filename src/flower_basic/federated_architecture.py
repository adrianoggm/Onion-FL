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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def load_architecture_config(config_path: str | os.PathLike) -> FederatedArchitecture:
    """Load and validate a federated architecture config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")

    raw = _read_architecture_file(path)
    root = raw.get("federated_architecture", raw)

    orchestrator_raw = root.get("orchestrator") or {}
    mqtt_raw = orchestrator_raw.get("mqtt") or {}
    topics_raw = mqtt_raw.get("topics") or {}

    orchestrator = OrchestratorSpec(
        address=str(orchestrator_raw.get("address", "0.0.0.0:8080")),
        protocol=str(orchestrator_raw.get("protocol", "MQTT")),
        rounds=int(orchestrator_raw.get("rounds", 3)),
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

    model_raw = root.get("model") or {}
    model = ModelConfig(
        type=str(model_raw.get("type", "ecg_cnn")),
        input_dim=model_raw.get("input_dim"),
    )

    dataset_raw = root.get("dataset") or {}
    split_raw = dataset_raw.get("split", {})
    dataset = None
    if dataset_raw:
        dataset = DatasetConfig(
            name=str(
                dataset_raw.get("name", dataset_raw.get("dataset", "swell"))
            ).lower(),
            data_dir=str(dataset_raw.get("data_dir", "data/SWELL")),
            modalities=dataset_raw.get("modalities"),
            subjects=dataset_raw.get("subjects"),
            seed=int(dataset_raw.get("seed", split_raw.get("seed", 42))),
            split_train=float(split_raw.get("train", 0.5)),
            split_val=float(split_raw.get("val", 0.2)),
            split_test=float(split_raw.get("test", 0.3)),
            split_strategy=str(split_raw.get("strategy", "per_subject")),
            scaler=str(split_raw.get("scaler", "global")),
            output_dir=str(dataset_raw.get("output_dir", "federated_runs/swell")),
            run_name=dataset_raw.get("run_name"),
            mode=str(dataset_raw.get("mode", "auto")),
            manual_assignments=dataset_raw.get("manual_assignments"),
            per_node_percentages=dataset_raw.get("per_node_percentages"),
            ensure_min_train_per_node=bool(
                dataset_raw.get("ensure_min_train_per_node", True)
            ),
            test_assignments=dataset_raw.get("test_assignments"),
        )

    fog_nodes: list[FogNodeSpec] = []
    for node_raw in root.get("fog_nodes", []):
        clients: list[ClientSpec] = []
        for c in node_raw.get("clients", []):
            clients.append(
                ClientSpec(
                    id=str(c.get("id")),
                    dataset=str(c.get("dataset")),
                    rounds=int(c.get("rounds", 3)),
                    data_dir=c.get("data_dir"),
                    params=c.get("params", {}) or {},
                    workflow=_normalize_workflow(c.get("workflow") or c.get("dataset")),
                )
            )
        fog_nodes.append(
            FogNodeSpec(
                id=str(node_raw.get("id")),
                address=node_raw.get("address"),
                k=int(node_raw.get("k", node_raw.get("K", 1))),
                clients=clients,
                params=node_raw.get("params", {}) or {},
            )
        )

    workflow = _normalize_workflow(root.get("workflow"))
    client_params = root.get("client_params", {}) or {}
    arch = FederatedArchitecture(
        orchestrator=orchestrator,
        fog_nodes=fog_nodes,
        model=model,
        dataset=dataset,
        workflow=workflow,
        client_params=client_params,
    )
    _validate_architecture(arch)
    return arch


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


def materialize_swell_partitions(
    arch: FederatedArchitecture, repo_root: Path | None = None
) -> Path:
    """Materialize SWELL federated splits based on the architecture config.

    Returns the path to the generated manifest.json and mutates `arch` to fill client.data_dir.
    """
    if arch.dataset is None:
        raise ValueError(
            "La configuración debe incluir sección 'dataset' para preparar particiones SWELL"
        )

    from .datasets.swell_federated import plan_and_materialize_swell_federated

    repo_root = repo_root or _default_repo_root()
    ds = arch.dataset
    run_name = ds.run_name or f"arch_{int(time.time())}"
    output_dir = Path(ds.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    data_dir = Path(ds.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    cfg = {
        "dataset": {
            "data_dir": str(data_dir),
            "modalities": ds.modalities,
            "subjects": ds.subjects,
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
            "manual_assignments": ds.manual_assignments,
            "per_node_percentages": ds.per_node_percentages,
            "output_dir": str(output_dir),
            "run_name": run_name,
            "ensure_min_train_per_node": ds.ensure_min_train_per_node,
            "test_assignments": ds.test_assignments,
        },
    }

    config_path = output_dir / run_name / "_arch_auto_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    result = plan_and_materialize_swell_federated(str(config_path))
    manifest_path = Path(result["output_dir"]) / "manifest.json"
    apply_manifest_paths(arch, manifest_path)

    return manifest_path


def apply_manifest_paths(
    arch: FederatedArchitecture, manifest_path: Path | str
) -> None:
    """Rehydrate SWELL clients from a generated manifest.

    The manifest is the source of truth for:
    - stable client ids generated per subject
    - the exact per-subject train directories to launch
    - the effective number of trainable clients per fog node
    """

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = manifest.get("nodes", {})
    clients_map = manifest.get("clients", {})
    global_subjects = manifest.get("global_subjects", {})
    config = manifest.get("config", {})
    split_config = config.get("split", {})
    split_strategy = split_config.get("strategy", "global")

    if split_strategy == "per_subject":
        train_subjects = {
            str(subject_id) for subject_id in global_subjects.get("all", [])
        }
        print(
            f"[MANIFEST] Strategy: per_subject - all {len(train_subjects)} subjects have train data"
        )
    else:
        train_subjects = {
            str(subject_id) for subject_id in global_subjects.get("train", [])
        }
        print(
            f"[MANIFEST] Strategy: global - only {len(train_subjects)} subjects in train split"
        )

    base = manifest_path.parent
    meta = manifest.get("meta", {})
    n_features = meta.get("n_features")
    if n_features is not None:
        arch.model.input_dim = int(n_features)

    for fog in arch.fog_nodes:
        if fog.id not in nodes:
            continue

        fog_clients = clients_map.get(fog.id, {})
        if not fog_clients:
            continue

        new_clients: list[ClientSpec] = []
        for client_id, subject_id in fog_clients.items():
            subject_str = str(subject_id)
            if subject_str not in train_subjects:
                print(
                    f"[MANIFEST] Skipping {client_id} (subject {subject_str} has no train data)"
                )
                continue

            subject_dir = base / fog.id / f"subject_{subject_str}"
            train_file = subject_dir / "train.npz"
            if train_file.exists():
                import numpy as np

                try:
                    arr = np.load(train_file, allow_pickle=True)
                    if arr["X"].shape[0] == 0:
                        print(f"[MANIFEST] Skipping {client_id} (train.npz is empty)")
                        continue
                except Exception as exc:
                    print(
                        f"[MANIFEST] Skipping {client_id} (error reading train.npz: {exc})"
                    )
                    continue
            else:
                print(f"[MANIFEST] Skipping {client_id} (train.npz not found)")
                continue

            new_clients.append(
                ClientSpec(
                    id=client_id,
                    dataset="swell",
                    workflow="swell",
                    rounds=arch.orchestrator.rounds,
                    data_dir=str(subject_dir),
                )
            )

        fog.clients = new_clients
        available_clients = len(new_clients)
        if available_clients == 0:
            print(f"[MANIFEST] {fog.id}: 0 clientes con training data")
            continue

        if fog.k <= 0:
            fog.k = available_clients
        elif fog.k > available_clients:
            print(
                f"[MANIFEST] Adjusting {fog.id} K from {fog.k} "
                f"to {available_clients} to match spawned clients"
            )
            fog.k = available_clients

        print(
            f"[MANIFEST] {fog.id}: {available_clients} clientes con training data (K={fog.k})"
        )


def build_runtime_plan(
    arch: FederatedArchitecture,
    repo_root: Path | None = None,
    manifest_path: Path | str | None = None,
) -> list[RuntimeCommand]:
    """Translate architecture into runnable commands (without starting them)."""
    root = repo_root or _default_repo_root()
    python_exec = os.getenv("PYTHON", sys.executable)
    mqtt = arch.orchestrator.mqtt
    topics = mqtt.topics
    primary = infer_primary_workflow(arch)
    py_path = str(root / "src")
    if os.getenv("PYTHONPATH"):
        py_path = py_path + os.pathsep + os.getenv("PYTHONPATH")
    env_path = {"PYTHONPATH": py_path}

    def _merge_env(extra: dict[str, str] | None = None) -> dict[str, str]:
        env = env_path.copy()
        if extra:
            env.update(extra)
        return env

    if primary == "swell":
        inferred = _infer_input_dim_from_clients(arch, root)
        if arch.model.input_dim is None and inferred is not None:
            arch.model.input_dim = inferred
        if arch.model.input_dim is None:
            raise ValueError(
                "model.input_dim es obligatorio para ejecutar el flujo SWELL"
            )

    # Validar que todos los clientes usen el mismo workflow primario (una sola jerarquía por run)
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
    # Solo usar K_MAP si hay valores distintos
    if len(set(k_map.values())) > 1:
        broker_env["FOG_K_MAP"] = json.dumps(k_map)
        broker_k_arg = None
    else:
        broker_k_arg = next(iter(k_map.values())) if k_map else 1

    commands: list[RuntimeCommand] = []

    # Central server
    if primary == "swell":
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
        RuntimeCommand(role="server", cmd=server_cmd, cwd=str(root), env=_merge_env())
    )

    # Fog bridge client(s)
    if primary == "swell":
        for fog in active_fogs:
            bridge_cmd = [
                python_exec,
                "-m",
                "flower_basic.clients.fog_bridge_swell",
                "--input_dim",
                str(int(arch.model.input_dim)),
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
                    cwd=str(root),
                    env=_merge_env(),
                )
            )
    else:
        bridge_script = root / "src" / "flower_basic" / "fog_flower_client.py"
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
                role="fog_bridge", cmd=bridge_cmd, cwd=str(root), env=_merge_env()
            )
        )

    # Fog broker
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
    ]
    if broker_k_arg is not None:
        broker_cmd.extend(["--k", str(int(broker_k_arg))])
    commands.append(
        RuntimeCommand(
            role="broker", cmd=broker_cmd, env=_merge_env(broker_env), cwd=str(root)
        )
    )

    # Clients per fog node (with unique index for metrics port)
    client_index = 0  # Global client index for deterministic metrics ports
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
                    raise ValueError(
                        f"Cliente {client.id} requiere data_dir para SWELL"
                    )
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
                        cwd=str(root),
                    )
                )
                client_index += 1
            elif workflow == "wesad":
                client_script = root / "src" / "flower_basic" / "client.py"
                cmd = [
                    python_exec,
                    str(client_script),
                    "--rounds",
                    str(client.rounds),
                    "--region",
                    fog.id,
                ]
                commands.append(
                    RuntimeCommand(
                        role=f"client_{client.id}",
                        cmd=cmd,
                        env=_merge_env(env_client),
                        cwd=str(root),
                    )
                )
            else:
                raise ValueError(f"Workflow no soportado aún: {workflow}")

    return commands


def distribute_architecture(
    arch: FederatedArchitecture,
    mqtt_client: Any = None,
    base_topic: str = "fl/ctrl/plan",
) -> list[dict[str, Any]]:
    """Prepare and optionally publish per-fog configuration over MQTT.

    Returns the payloads so they can be inspected or unit-tested without a broker.
    """
    payloads: list[dict[str, Any]] = []
    topics = arch.orchestrator.mqtt.topics
    for fog in arch.fog_nodes:
        payload = {
            "fog_id": fog.id,
            "k": fog.k,
            "clients": [
                {
                    "id": c.id,
                    "dataset": c.dataset,
                    "workflow": _normalize_workflow(c.workflow or c.dataset),
                    "rounds": c.rounds,
                    "data_dir": c.data_dir,
                    "params": c.params,
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
        payloads.append(payload)

        if mqtt_client is not None:
            try:
                topic = f"{base_topic}/{fog.id}"
                mqtt_client.publish(topic, json.dumps(payload))
            except Exception:
                # Fall back silently; publishing is best-effort in tests/offline modes.
                pass
    return payloads
