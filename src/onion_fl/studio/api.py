from __future__ import annotations

"""The Studio API over the repository's files (spec of #103, §3).

Everything reads and writes the same files as the command line: ``topologies/``,
``experiments/``, ``runs/`` and ``datasets/`` under ``root``. Names are checked
against ``^[A-Za-z0-9_.-]+$`` so no request leaves its folder.
"""

import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from onion_fl.core.topology import TopologyError, load_topology, parse_topology
from onion_fl.data.contract import DataError
from onion_fl.experiment.config import (
    REGISTRIES,
    ConfigError,
    ExperimentConfig,
    experiment_schema,
    load_experiment,
    parse_experiment,
)
from onion_fl.experiment.sweep import scenarios
from onion_fl.observability.analysis import RunRecord, Runs, load_runs
from onion_fl.observability.events import read_events
from onion_fl.studio.previews import link_preview, placement_preview, sharing_preview

SAFE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
STATIC = Path(__file__).parent / "static"

# The tutorial: what each axis decides, and which dry-run preview it has.
AXES: list[tuple[str, str, str, str | None]] = [
    (
        "sharing",
        "Compartición",
        "Qué grupos de parámetros viajan y hasta qué nivel: global, por nivel o local.",
        "sharing",
    ),
    (
        "placement",
        "Reparto",
        "Qué sujetos van bajo cada fog: segregados por dataset, mezclados o sesgados.",
        "placement",
    ),
    (
        "link_profile",
        "Perfil de red",
        "Latencia, jitter, ancho de banda y pérdidas de cada enlace en simulación.",
        "link",
    ),
    (
        "transport",
        "Transporte",
        "Cómo viajan los mensajes en una ejecución real.",
        None,
    ),
    (
        "codec",
        "Codec",
        "Cómo se codifica un mensaje; su tamaño cuenta para la red y las métricas.",
        None,
    ),
    ("model", "Modelo", "Qué partes tiene el modelo de cada edge y del global.", None),
    (
        "aggregator",
        "Agregador",
        "Cómo combina cada nodo las actualizaciones de sus hijos, clave a clave.",
        None,
    ),
    (
        "server_optimizer",
        "Optimizador de servidor",
        "Cómo pasa el coordinador del agregado al nuevo modelo global.",
        None,
    ),
    ("trainer", "Entrenador", "Cómo entrena cada edge con sus datos.", None),
    (
        "attack",
        "Ataque",
        "Qué fracción de edges de cada dataset es maliciosa y cómo envenena "
        "sus datos o su actualización.",
        None,
    ),
    (
        "privacy",
        "Privacidad",
        "Ruido de privacidad diferencial que cada edge añade antes de enviar.",
        None,
    ),
    (
        "memory",
        "Memoria",
        "Qué filas ya entrenadas guarda cada edge de un stream para volver a "
        "entrenar con ellas (replay).",
        None,
    ),
    ("init", "Inicialización", "De dónde sale el modelo inicial.", None),
    ("participation", "Participación", "Qué hijos participan en cada ronda.", None),
    (
        "staleness",
        "Actualizaciones tardías",
        "Qué se hace con las que llegan después del cierre de ronda.",
        None,
    ),
    (
        "stale_weighting",
        "Peso por antigüedad",
        "Cuánto pesa una actualización tardía.",
        None,
    ),
    (
        "metric",
        "Métricas",
        "Qué se mide al evaluar en el edge, en cada zona y en el global.",
        None,
    ),
    (
        "diagnostic",
        "Diagnósticos",
        "Qué calcula cada agregador al cerrar una ronda.",
        None,
    ),
    (
        "compute_model",
        "Cómputo",
        "Cuánto tarda en simulación el entrenamiento de un edge.",
        None,
    ),
    (
        "availability_model",
        "Disponibilidad",
        "Cuándo está un nodo encendido en simulación.",
        None,
    ),
    ("reader", "Lectores", "Cómo se leen los ficheros de un dataset.", None),
    (
        "step",
        "Pasos de ingesta",
        "Cómo se transforma la tabla de un dataset hasta SubjectData.",
        None,
    ),
    (
        "baseline",
        "Baselines",
        "Modelos clásicos sobre los mismos sujetos de test.",
        None,
    ),
]


def rooted(config: ExperimentConfig, root: Path) -> ExperimentConfig:
    """The config with its relative paths resolved against the Studio root."""
    raw = config.dump()
    for key, value in raw["paths"].items():
        if not Path(value).is_absolute():
            raw["paths"][key] = str(root / value)
    for use in raw["data"]["datasets"].values():
        if use.get("descriptor") and not Path(use["descriptor"]).is_absolute():
            use["descriptor"] = str(root / use["descriptor"])
    if isinstance(raw["topology"], str) and raw["topology"].endswith((".yaml", ".yml")):
        if not Path(raw["topology"]).is_absolute():
            raw["topology"] = str(root / raw["topology"])
    return parse_experiment(raw)


def _clean(value: Any) -> Any:
    """JSON-ready: NaN and infinities become null, tuples lists, numpy scalars numbers."""
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        return _clean(value.item())
    return value


def _json(value: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(
        _clean(json.loads(json.dumps(value, default=str))), status_code=status
    )


def _errors(message: str, status: int = 422) -> JSONResponse:
    return JSONResponse(
        {"errors": [line for line in str(message).splitlines() if line.strip()]},
        status_code=status,
    )


def _safe(name: str) -> str:
    if not SAFE.fullmatch(name) or ".." in name:
        raise HTTPException(400, f"invalid name {name!r}")
    return name


def create_app(root: str | Path = ".") -> FastAPI:
    root = Path(root).resolve()
    folders = {
        name: root / name for name in ("topologies", "experiments", "runs", "datasets")
    }
    app = FastAPI(title="Onion-FL Studio")

    @app.middleware("http")
    async def revalidate(request, call_next):
        # Files change under a running Studio (a run, an edit): always revalidate.
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    def existing(kind: str, name: str, suffix: str = ".yaml") -> Path:
        path = folders[kind] / f"{_safe(name)}{suffix}"
        if not path.is_file():
            raise HTTPException(404, f"{kind[:-1]} {name!r} not found")
        return path

    # --- schema and tutorial ----------------------------------------------------------

    @app.get("/api/schema")
    def schema() -> JSONResponse:
        return _json(experiment_schema())

    @app.get("/api/tutorial")
    def tutorial() -> JSONResponse:
        return _json(
            [
                {
                    "kind": kind,
                    "title": title,
                    "explain": explain,
                    "preview": preview,
                    "plugins": REGISTRIES[kind].describe(),
                }
                for kind, title, explain, preview in AXES
            ]
        )

    # --- topologies -------------------------------------------------------------------

    @app.get("/api/topologies")
    def topologies() -> JSONResponse:
        out = []
        for path in sorted(folders["topologies"].glob("*.yaml")):
            try:
                topology = load_topology(path)
            except TopologyError:
                continue
            out.append(
                {
                    "name": path.stem,
                    "title": topology.name,
                    "topology_id": topology.topology_id,
                    "levels": list(topology.levels),
                    "nodes": len(topology.nodes),
                    "leaves": len(topology.leaves()),
                }
            )
        return _json(out)

    @app.get("/api/topologies/{name}")
    def topology(name: str) -> JSONResponse:
        path = existing("topologies", name)
        try:
            graph = load_topology(path).to_graph()
        except TopologyError as exc:
            return _errors(str(exc))
        return _json(
            {"name": name, "yaml": path.read_text(encoding="utf-8"), "graph": graph}
        )

    @app.post("/api/topologies/validate")
    def validate_topology(body: dict = Body(...)) -> JSONResponse:
        try:
            return _json({"graph": parse_topology(body).to_graph()})
        except TopologyError as exc:
            return _errors(str(exc).replace("; ", "\n"))

    @app.put("/api/topologies/{name}")
    def save_topology(name: str, body: dict = Body(...)) -> JSONResponse:
        target = folders["topologies"] / f"{_safe(name)}.yaml"
        try:
            graph = parse_topology(body).to_graph()
        except TopologyError as exc:
            return _errors(str(exc).replace("; ", "\n"))
        folders["topologies"].mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return _json({"graph": graph})

    # --- experiments ------------------------------------------------------------------

    def experiment(name: str) -> ExperimentConfig:
        return load_experiment(existing("experiments", name))

    def scenario_rows(config: ExperimentConfig) -> list[dict[str, Any]]:
        return [
            {"name": s.name, "seed": s.seed, "config_id": s.config_id}
            for s in scenarios(config)
        ]

    @app.get("/api/experiments")
    def experiments() -> JSONResponse:
        out = []
        for path in sorted(folders["experiments"].glob("*.yaml")):
            try:
                config = load_experiment(path)
            except (ConfigError, ValueError):
                continue
            combos = (
                math.prod(len(v) for v in config.sweep.values()) if config.sweep else 1
            )
            topology = (
                config.topology
                if isinstance(config.topology, str)
                else config.topology.get("name", "inline")
            )
            out.append(
                {
                    "name": path.stem,
                    "description": config.description,
                    "topology": topology,
                    "scenarios": combos,
                    "seeds": len(config.seeds),
                }
            )
        return _json(out)

    @app.get("/api/experiments/{name}")
    def experiment_detail(name: str) -> JSONResponse:
        path = existing("experiments", name)
        try:
            config = load_experiment(path)
            rows = scenario_rows(config)  # a sweep can break what loads
        except ConfigError as exc:
            return _errors(str(exc))
        return _json(
            {
                "name": name,
                "yaml": path.read_text(encoding="utf-8"),
                "config": config.dump(),
                "scenarios": rows,
            }
        )

    @app.post("/api/experiments/validate")
    def validate_experiment(body: dict = Body(...)) -> JSONResponse:
        try:
            return _json({"scenarios": scenario_rows(parse_experiment(body))})
        except ConfigError as exc:
            return _errors(str(exc))

    @app.post("/api/experiments/{name}/plan")
    def plan_experiment(name: str) -> JSONResponse:
        from onion_fl.experiment.runner import plan

        try:
            config = experiment(name)
            previews = plan(rooted(config, root))
        except OSError as exc:  # the file it misses, not where it lives
            missing = Path(exc.filename).name if exc.filename else ""
            return _errors(f"{exc.strerror or type(exc).__name__}: {missing}")
        except (ConfigError, DataError, TopologyError, ValueError) as exc:
            return _errors(str(exc))
        # Runs start from the root with the paths as written: their config_id.
        for preview, scenario in zip(previews, scenarios(config), strict=True):
            preview["config_id"] = scenario.config_id
        return _json(previews)

    @app.post("/api/experiments/{name}/run")
    def run_experiment(name: str, body: dict = Body(default={})) -> JSONResponse:
        path = existing("experiments", name)
        mode = body.get("mode", "sim")
        if mode not in ("sim", "real"):
            raise HTTPException(400, f"mode must be sim or real, got {mode!r}")
        args = [
            sys.executable,
            "-m",
            "onion_fl",
            "run",
            str(path),
            "--mode",
            mode,
            "--workers",
            str(int(body.get("workers", 1))),
        ]
        if body.get("scenario"):
            args += ["--scenario", str(body["scenario"])]
        process = subprocess.Popen(args, cwd=str(root))
        return _json({"pid": process.pid})

    # --- runs -------------------------------------------------------------------------

    def run_record(run_id: str) -> RunRecord:
        path = folders["runs"] / _safe(run_id)
        if not (path / "run.json").is_file():
            raise HTTPException(404, f"run {run_id!r} not found")
        return RunRecord(
            path, json.loads((path / "run.json").read_text(encoding="utf-8"))
        )

    def levels_of(record: RunRecord) -> list[str]:
        """The topology's levels, root first, from the run's config."""
        from onion_fl.experiment.runner import resolve_topology

        try:
            return list(
                resolve_topology(
                    rooted(parse_experiment(record.meta["config"]), root)
                ).levels
            )
        except (ConfigError, TopologyError, KeyError, ValueError):
            return ["global"]

    def summary_of(path: Path) -> dict[str, Any]:
        file = path / "summary.json"
        return json.loads(file.read_text(encoding="utf-8")) if file.is_file() else {}

    @app.get("/api/runs")
    def runs(experiment: str | None = None, status: str | None = None) -> JSONResponse:
        out = []
        for record in load_runs(folders["runs"], experiment=experiment):
            meta, summary = record.meta, summary_of(record.path)
            if status and meta.get("status") != status:
                continue
            out.append(
                {
                    "run_id": meta.get("run_id"),
                    "experiment": record.experiment,
                    "scenario": meta.get("scenario"),
                    "seed": meta.get("seed"),
                    "status": meta.get("status"),
                    "started_at": meta.get("started_at"),
                    "finished_at": meta.get("finished_at"),
                    "topology_id": meta.get("topology_id"),
                    "config_id": meta.get("config_id"),
                    "rounds": summary.get("rounds"),
                    "final": summary.get("final", {}),
                }
            )
        return _json(sorted(out, key=lambda r: r["run_id"] or "", reverse=True))

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> JSONResponse:
        record = run_record(run_id)
        roles, composition = {}, {}
        events = record.path / "events.jsonl"
        for event in read_events(events) if events.is_file() else []:
            if event["name"] == "data.roles":
                roles = event["tags"].get("roles", {})
            elif event["name"] == "data.composition":
                tags = dict(event["tags"])
                composition[tags.pop("leaf")] = tags
        return _json(
            {
                "meta": record.meta,
                "summary": summary_of(record.path),
                "roles": roles,
                "composition": composition,
                "levels": levels_of(record),
            }
        )

    @app.get("/api/runs/{run_id}/events")
    def run_events(
        run_id: str, after: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=5000)
    ) -> JSONResponse:
        record = run_record(run_id)
        events = record.path / "events.jsonl"
        lines = (
            events.read_text(encoding="utf-8").splitlines() if events.is_file() else []
        )
        # ponytail: reads the whole file per poll; an offset index if runs get huge
        page = [
            json.loads(line) for line in lines[after : after + limit] if line.strip()
        ]
        return _json({"events": page, "next": after + len(page), "total": len(lines)})

    @app.get("/api/runs/{run_id}/series")
    def run_series(
        run_id: str,
        level: str,
        name: str,
        model: str | None = None,
        dataset: str | None = None,
        source: str | None = None,
    ) -> JSONResponse:
        tags = {k: v for k, v in {"model": model, "source": source}.items() if v}
        table = Runs([run_record(run_id)]).metrics(level=level, name=name, **tags)
        if dataset and not table.empty:
            if dataset == "*":
                table = table[table["dataset"].isna()] if "dataset" in table else table
            else:
                table = table[table.get("dataset") == dataset]
        if table.empty:
            return _json([])
        grouped = (
            table.groupby("round")["value"]
            .agg(["mean", "min", "max", "count"])
            .reset_index()
        )
        return _json(grouped.to_dict(orient="records"))

    @app.get("/api/compare")
    def compare(
        experiment: str | None = None,
        level: str = "global",
        metric: str = "accuracy",
        by: str = "topology_id",
    ) -> JSONResponse:
        keys = [k for k in by.split(",") if k]
        table = load_runs(folders["runs"], experiment=experiment).compare(
            level=level, metric=metric, by=keys
        )
        split = table.attrs.get("by", keys)  # what the series were split by
        return _json([row | {"_by": split} for row in table.to_dict(orient="records")])

    # --- previews ---------------------------------------------------------------------

    @app.post("/api/preview/sharing")
    def preview_sharing(body: dict = Body(...)) -> JSONResponse:
        try:
            return _json(
                sharing_preview(
                    body["topology"],
                    body.get("sharing", "fedavg"),
                    body.get("datasets", ["a"]),
                    body.get("model", "modular_mlp"),
                )
            )
        except (TopologyError, ValueError, KeyError) as exc:
            return _errors(str(exc))

    @app.post("/api/preview/placement")
    def preview_placement(body: dict = Body(...)) -> JSONResponse:
        try:
            return _json(
                placement_preview(
                    body["topology"],
                    body.get("placement", "mixing"),
                    body["datasets"],
                    int(body.get("seed", 0)),
                )
            )
        except (TopologyError, ValueError, KeyError) as exc:
            return _errors(str(exc))

    @app.post("/api/preview/link")
    def preview_link(body: dict = Body(...)) -> JSONResponse:
        try:
            return _json(
                link_preview(
                    body.get("profile", "lan"), int(body.get("bytes", 100_000))
                )
            )
        except (ValueError, KeyError) as exc:
            return _errors(str(exc))

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="app")
    return app
