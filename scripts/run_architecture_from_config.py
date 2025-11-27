#!/usr/bin/env python3
from __future__ import annotations

"""Config-driven launcher for the fog–cloud federated stack.

Usage:
  python scripts/run_architecture_from_config.py --config configs/federated_architecture.example.yaml --plan-only
  python scripts/run_architecture_from_config.py --config ... --manifest federated_runs/swell/example_manual/manifest.json --launch
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flower_basic.federated_architecture import (  # noqa: E402
    FederatedArchitecture,
    build_runtime_plan,
    distribute_architecture,
    infer_primary_workflow,
    load_architecture_config,
    materialize_swell_partitions,
)


def _apply_manifest_paths(arch: FederatedArchitecture, manifest_path: Path) -> None:
    """Fill missing SWELL client data_dir paths using a manifest (nodes -> dirs)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = manifest.get("nodes", {})
    base = manifest_path.parent
    meta = manifest.get("meta", {})
    n_features = meta.get("n_features")
    if n_features is not None:
        arch.model.input_dim = int(n_features)
    for fog in arch.fog_nodes:
        node_dir = base / fog.id
        if fog.id not in nodes:
            continue
        for client in fog.clients:
            wf = (client.workflow or client.dataset or "").lower()
            if wf == "swell" and client.data_dir is None:
                client.data_dir = str(node_dir)


def _print_plan(commands) -> None:
    print("\n=== Plan de ejecución (orden sugerido) ===")
    for cmd in commands:
        env_hint = f" env={list(cmd.env.keys())}" if cmd.env else ""
        print(f"- {cmd.role}: {' '.join(cmd.cmd)}{env_hint}")


def _launch(commands, delay: float = 1.0) -> None:
    procs: List[subprocess.Popen] = []
    try:
        for cmd in commands:
            env = os.environ.copy()
            env.update(cmd.env)
            print(f"[LAUNCH] {cmd.role}: {' '.join(cmd.cmd)}")
            proc = subprocess.Popen(cmd.cmd, cwd=cmd.cwd, env=env)
            procs.append(proc)
            time.sleep(delay)

        print("\n[RUNNING] Todos los procesos lanzados. Ctrl+C para detener.")
        # Detectar fin del servidor para cortar clientes y broker
        server_proc = procs[0] if procs else None
        while True:
            time.sleep(0.5)
            if server_proc and server_proc.poll() is not None:
                print("[INFO] Server terminó; deteniendo resto de procesos...")
                break
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Deteniendo procesos...")
    finally:
        for proc in procs:
            try:
                if proc.poll() is None:
                    if os.name == "nt":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                        proc.terminate()
                    else:
                        proc.terminate()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Lanza la arquitectura fog-cloud desde un config JSON/YAML")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON/YAML con federated_architecture")
    parser.add_argument("--manifest", help="Manifest SWELL para rellenar data_dir por nodo (opcional)")
    parser.add_argument("--plan-only", action="store_true", help="Solo mostrar plan, no lanzar procesos (por defecto)")
    parser.add_argument("--launch", action="store_true", help="Lanzar procesos siguiendo el plan")
    parser.add_argument("--dispatch-config", action="store_true", help="Publicar plan por MQTT en fl/ctrl/plan/<fog_id>")
    parser.add_argument(
        "--prepare-splits",
        action="store_true",
        help="Materializa splits SWELL automáticamente desde el YAML si no hay manifest",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Delay entre lanzamientos (segundos)")
    args = parser.parse_args()

    arch = load_architecture_config(args.config)

    primary = infer_primary_workflow(arch)
    print(f"[INFO] Flujo principal inferido: {primary}")

    # Materializar particiones SWELL si es necesario
    manifest_path = Path(args.manifest) if args.manifest else None
    if primary == "swell":
        if manifest_path:
            _apply_manifest_paths(arch, manifest_path)
        else:
            needs_materialization = args.prepare_splits or any(
                (c.data_dir is None or not Path(str(c.data_dir)).exists())
                for fog in arch.fog_nodes
                for c in fog.clients
                if (c.workflow or c.dataset or "").lower() == "swell"
            )
            if needs_materialization:
                manifest_path = materialize_swell_partitions(arch, repo_root=REPO_ROOT)
                print(f"[INFO] Particiones SWELL preparadas automáticamente: {manifest_path}")

    if args.dispatch_config:
        try:
            import paho.mqtt.client as mqtt  # type: ignore

            mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            mqttc.connect(arch.orchestrator.mqtt.broker, arch.orchestrator.mqtt.port)
            mqttc.loop_start()
        except Exception:
            mqttc = None
        payloads = distribute_architecture(arch, mqtt_client=mqttc)
        print(f"[INFO] Plan enviado por MQTT a {len(payloads)} nodos fog")
        if mqttc is not None:
            mqttc.loop_stop()

    commands = build_runtime_plan(arch, repo_root=REPO_ROOT, manifest_path=manifest_path)
    _print_plan(commands)

    should_launch = args.launch
    if should_launch:
        _launch(commands, delay=args.delay)


if __name__ == "__main__":
    main()
