#!/usr/bin/env python3
from __future__ import annotations

"""Config-driven launcher for the fog–cloud federated stack.

Usage:
  python scripts/run_architecture_from_config.py --config configs/federated_architecture.example.yaml --plan-only
  python scripts/run_architecture_from_config.py --config ... --manifest federated_runs/swell/example_manual/manifest.json --launch
"""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flower_basic.federated_architecture import (  # noqa: E402
    FederatedArchitecture,
    apply_manifest_paths,
    build_runtime_plan,
    distribute_architecture,
    infer_primary_workflow,
    load_architecture_config,
    materialize_swell_partitions,
)


def _apply_manifest_paths(
    arch: FederatedArchitecture, manifest_path: Path
) -> FederatedArchitecture:
    """Backwards-compatible wrapper used by tests and the CLI script."""
    return apply_manifest_paths(arch, manifest_path)


def _print_plan(commands) -> None:
    print("\n=== Plan de ejecución (orden sugerido) ===")
    for cmd in commands:
        env_hint = f" env={list(cmd.env.keys())}" if cmd.env else ""
        print(f"- {cmd.role}: {' '.join(cmd.cmd)}{env_hint}")


def _extract_arg(cmd: list[str], flag: str) -> str | None:
    try:
        idx = cmd.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(cmd):
        return None
    return cmd[idx + 1]


def _normalize_connect_host(host: str) -> str:
    stripped = host.strip().strip("[]")
    if stripped in {"", "0.0.0.0", "::", "*"}:
        return "127.0.0.1"
    return stripped


def _wait_for_server_ready(server_addr: str, timeout_s: float = 20.0) -> bool:
    host, sep, port_str = server_addr.rpartition(":")
    if not sep:
        return False

    connect_host = _normalize_connect_host(host)
    try:
        port = int(port_str)
    except ValueError:
        return False

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((connect_host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _launch(commands, delay: float = 1.0) -> None:
    procs: list[subprocess.Popen] = []

    # Load .env file to ensure OTEL variables are available
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        env_file = REPO_ROOT / "docker" / ".env"

    otel_env = {}
    if env_file.exists():
        print(f"[LAUNCH] Loading environment from {env_file}")
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Only load OTEL and MQTT related vars
                if key.startswith(("OTEL_", "MQTT_")):
                    otel_env[key] = value
        print(f"[LAUNCH] Loaded OTEL vars: {list(otel_env.keys())}")

    try:
        for cmd in commands:
            env = os.environ.copy()
            env.update(otel_env)  # Add OTEL vars from .env
            env.update(cmd.env)  # Add command-specific vars
            print(f"[LAUNCH] {cmd.role}: {' '.join(cmd.cmd)}")
            proc = subprocess.Popen(cmd.cmd, cwd=cmd.cwd, env=env)
            procs.append(proc)
            if cmd.role == "server":
                server_addr = _extract_arg(cmd.cmd, "--server_addr")
                if server_addr:
                    print(f"[LAUNCH] Waiting for server readiness at {server_addr}...")
                    if _wait_for_server_ready(server_addr):
                        print(
                            f"[LAUNCH] Server is accepting connections at {server_addr}"
                        )
                    else:
                        print(
                            f"[LAUNCH] Server did not become ready within timeout: {server_addr}"
                        )
            time.sleep(delay)

        print("\n[RUNNING] Todos los procesos lanzados. Ctrl+C para detener.")
        # Detectar fin del servidor para cortar clientes y broker
        server_proc = procs[0] if procs else None
        while True:
            time.sleep(0.5)
            if server_proc and server_proc.poll() is not None:
                print(
                    f"[INFO] Server terminó (exit code: {server_proc.returncode}); deteniendo resto de procesos..."
                )
                break
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Deteniendo procesos (Ctrl+C)...")
    finally:
        # Terminate all remaining processes
        for i, proc in enumerate(procs):
            try:
                if proc.poll() is None:
                    print(f"[SHUTDOWN] Terminando proceso {i}...")
                    if os.name == "nt":
                        proc.terminate()
                    else:
                        proc.terminate()
            except Exception as e:
                print(f"[SHUTDOWN] Error terminando proceso {i}: {e}")

        # Wait for processes to actually terminate (with timeout)
        print("[SHUTDOWN] Esperando que los procesos terminen...")
        for i, proc in enumerate(procs):
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                print(f"[SHUTDOWN] Proceso {i} no respondió, forzando kill...")
                proc.kill()
            except Exception:
                pass

        print("[DONE] Todos los procesos terminados.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lanza la arquitectura fog-cloud desde un config JSON/YAML"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Ruta al archivo JSON/YAML con federated_architecture",
    )
    parser.add_argument(
        "--manifest", help="Manifest SWELL para rellenar data_dir por nodo (opcional)"
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Solo mostrar plan, no lanzar procesos (por defecto)",
    )
    parser.add_argument(
        "--launch", action="store_true", help="Lanzar procesos siguiendo el plan"
    )
    parser.add_argument(
        "--dispatch-config",
        action="store_true",
        help="Publicar plan por MQTT en fl/ctrl/plan/<fog_id>",
    )
    parser.add_argument(
        "--prepare-splits",
        action="store_true",
        help="Materializa splits SWELL automáticamente desde el YAML si no hay manifest",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay entre lanzamientos (segundos)"
    )
    args = parser.parse_args()

    arch = load_architecture_config(args.config)

    primary = infer_primary_workflow(arch)
    print(f"[INFO] Flujo principal inferido: {primary}")

    # Materializar particiones SWELL si es necesario
    manifest_path = Path(args.manifest) if args.manifest else None
    if primary == "swell":
        if manifest_path:
            arch = _apply_manifest_paths(arch, manifest_path)
        else:
            needs_materialization = args.prepare_splits or any(
                (c.data_dir is None or not Path(str(c.data_dir)).exists())
                for fog in arch.fog_nodes
                for c in fog.clients
                if (c.workflow or c.dataset or "").lower() == "swell"
            )
            if needs_materialization:
                materialized = materialize_swell_partitions(arch, repo_root=REPO_ROOT)
                arch = materialized.architecture
                manifest_path = materialized.manifest_path
                print(
                    f"[INFO] Particiones SWELL preparadas automáticamente: {manifest_path}"
                )

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

    runtime_plan = build_runtime_plan(
        arch, repo_root=REPO_ROOT, manifest_path=manifest_path
    )
    _print_plan(runtime_plan.commands)

    should_launch = args.launch
    if should_launch:
        _launch(runtime_plan.commands, delay=args.delay)


if __name__ == "__main__":
    main()
