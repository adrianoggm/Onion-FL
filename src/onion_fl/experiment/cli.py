from __future__ import annotations

"""The ``onion_fl`` command line (spec §12.4).

::

    onion_fl data prepare|inspect <dataset> [--option key=value]...
    onion_fl topology show <file> [--graph | --mermaid]
    onion_fl plan <experiment.yaml>
    onion_fl run <experiment.yaml> [--scenario NAME] [--backend sim|local]
                 [--mode sim|real] [--workers N]
    onion_fl node --id <node> --run <run_id> --config <experiment.yaml>
    onion_fl ceiling <experiment.yaml> [--replicas R] [--metric NAME] [--series KEY]
                     [--resample] [--paired] [--workers N]
    onion_fl fidelity <runs A> <runs B> [--metric NAME]... [--target METRIC=VALUE]
                      [--ceiling FILE] [--calibration ID] [--csv FILE] [--html FILE]
    onion_fl calibrate measure <spec.yaml> --out <session> [--image IMAGE]
                       [--project NAME] [--subnet CIDR]
    onion_fl calibrate fit <session>... [--out calibrations] [--infrastructure-out FILE]
    onion_fl calibrate check <session> --infrastructure FILE [--calibrations DIR]
                       [--out FILE]
    onion_fl report <experiment.yaml | runs_dir>... [--out FILE] [--metric NAME]... [--by TAG]...
    onion_fl baseline <experiment.yaml> [--models lr rf xgboost] [--cv K]
    onion_fl schema [--out FILE]
    onion_fl serve [--root .] [--host 127.0.0.1] [--port 8765]

Results go to stdout as JSON (or a path per line); errors to stderr with
exit code 2.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from onion_fl.core.topology import Topology, TopologyError, load_topology


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _options(pairs: Sequence[str]) -> dict[str, Any]:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--option {pair!r}: use key=value")
        out[key] = yaml.safe_load(value)
    return out


def _dataset(args: argparse.Namespace) -> tuple[Any, Path]:
    from onion_fl.data.cache import prepare
    from onion_fl.data.ingest import load_spec

    spec = load_spec(Path(args.datasets_dir) / f"{args.dataset}.yaml")
    return spec, prepare(
        spec, _options(args.option), cache_dir=args.cache_dir, force=args.force
    )


def cmd_data(args: argparse.Namespace) -> int:
    from onion_fl.data.cache import inspect, load_prepared

    _, path = _dataset(args)
    if args.action == "prepare":
        print(path)
    else:
        _print_json(inspect(load_prepared(path)))
    return 0


def _tree(topology: Topology) -> str:
    lines = [f"{topology.name or '(unnamed)'}  topology_id={topology.topology_id}"]

    def walk(node_id: str, prefix: str) -> None:
        children = topology.children(node_id)
        for i, child in enumerate(children):
            last = i == len(children) - 1
            link = child.link_up
            profile = link.profile if isinstance(link.profile, str) else "custom"
            lines.append(
                f"{prefix}{'└── ' if last else '├── '}{child.id} ({child.level}, "
                f"{topology.role(child.id)}) {link.transport}/{link.codec}/{profile}"
            )
            walk(child.id, prefix + ("    " if last else "│   "))

    root = topology.root
    lines.append(f"{root.id} ({root.level}, coordinator)")
    walk(root.id, "")
    edge = topology.edge.link_up
    lines.append(
        f"edges ({topology.levels[-1]}): {edge.transport}/{edge.codec}/{edge.profile}"
    )
    return "\n".join(lines)


def _link_label(link: Any) -> str:
    transport = link.transport
    if not isinstance(transport, str):
        transport = transport.get("name", "custom")
    profile = link.profile if isinstance(link.profile, str) else "custom"
    return f"{transport}/{link.codec}/{profile}"


def _mermaid(topology: Topology) -> str:
    """Mermaid flowchart of the tree; the edges of each leaf drawn as one box."""
    lines = ["flowchart TD"]
    for node in topology.nodes:
        role = topology.role(node.id)
        lines.append(f'    {node.id}["{node.id}<br/>{node.level} · {role}"]')
    for node in topology.nodes:
        if node.parent is not None:
            lines.append(
                f"    {node.id} -->|{_link_label(node.link_up)}| {node.parent}"
            )
    edge = _link_label(topology.edge.link_up)
    for leaf in topology.leaves():
        box = f'edges_{leaf.id}(["edges ({topology.levels[-1]})"])'
        lines.append(f"    {box} -->|{edge}| {leaf.id}")
    return "\n".join(lines)


def cmd_topology(args: argparse.Namespace) -> int:
    topology = load_topology(args.file)
    if args.graph:
        _print_json(topology.to_graph())
    elif args.mermaid:
        print(_mermaid(topology))
    else:
        print(_tree(topology))
    return 0


def _experiment(path: str) -> Any:
    from onion_fl.experiment.config import load_experiment

    if not Path(path).exists():
        raise FileNotFoundError(f"experiment not found: {path}")
    return load_experiment(path)


def cmd_plan(args: argparse.Namespace) -> int:
    from onion_fl.experiment.runner import plan

    _print_json(plan(_experiment(args.experiment)))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from onion_fl.experiment.config import parse_experiment
    from onion_fl.experiment.runner import run_experiment

    config = _experiment(args.experiment)
    if args.mode:
        config = parse_experiment(
            config.dump() | {"runtime": config.dump()["runtime"] | {"mode": args.mode}}
        )
    if args.backend:
        config = parse_experiment(config.dump() | {"backend": args.backend})
    for path in run_experiment(config, workers=args.workers, only=args.scenario):
        print(path)
    return 0


def cmd_node(args: argparse.Namespace) -> int:
    from onion_fl.experiment.real import run_node

    if not Path(args.config).exists():
        raise FileNotFoundError(f"config not found: {args.config}")
    return run_node(
        args.id, args.run, args.config, scenario=args.scenario, seed=args.seed
    )


def cmd_ceiling(args: argparse.Namespace) -> int:
    from onion_fl.experiment.ceiling import run_ceiling

    _print_json(
        run_ceiling(
            _experiment(args.experiment),
            n=args.replicas,
            metric=args.metric,
            series=args.series,
            resample=args.resample,
            paired=args.paired,
            minimize=args.minimize,
            workers=args.workers,
        )
    )
    return 0


def _target(text: str | None) -> tuple[str, float] | None:
    if text is None:
        return None
    metric, sep, value = text.partition("=")
    if not sep:
        raise ValueError(f"--target {text!r}: use metric=value, as accuracy=0.8")
    return metric, float(value)


def cmd_fidelity(args: argparse.Namespace) -> int:
    from onion_fl.experiment.fidelity import fidelity, write_csv, write_html

    noise = None
    if args.ceiling:
        noise = json.loads(Path(args.ceiling).read_text(encoding="utf-8"))
    result = fidelity(
        args.a,
        args.b,
        metrics=args.metric or ["accuracy", "time"],
        series=args.series,
        target=_target(args.target),
        experiment=args.experiment,
        noise_ceiling=noise,
        minimize=args.minimize or [],
        resamples=args.bootstrap,
        seed=args.seed,
        calibration=args.calibration,
    )
    if args.csv:
        write_csv(result, args.csv)
    if args.html:
        write_html(result, args.html)
    _print_json(result)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from onion_fl.experiment import calibrate

    if args.step == "measure":
        given = {"image": args.image, "project": args.project, "subnet": args.subnet}
        params = {k: v for k, v in given.items() if v is not None}
        print(calibrate.measure(args.spec, args.out, params))
        return 0
    if args.step == "fit":
        from onion_fl.infrastructure.calibration import load_calibration

        folder = calibrate.fit(args.sessions, args.out)
        if args.infrastructure_out:
            meta, _ = calibrate.read_session(args.sessions[0])
            spec = calibrate.CalibrationSpec.model_validate(meta["spec"])
            source = Path(spec.infrastructure)
            if source.suffix not in (".yaml", ".yml"):
                source = Path(spec.paths.infrastructures) / f"{source}.yaml"
            raw = calibrate.calibrated_infrastructure(
                source, load_calibration(args.out, folder.name)
            )
            head = (
                f"# {source.name} with the profiles and links that calibration\n"
                f"# {folder.name} measured citing it (onion_fl calibrate fit).\n"
            )
            Path(args.infrastructure_out).write_text(
                head + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
            )
        print(folder)
        return 0
    result = calibrate.check(args.session, args.infrastructure, args.calibrations)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_json(result)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from onion_fl.observability.analysis import Runs, load_runs

    records = []
    for target in args.targets:
        if target.endswith((".yaml", ".yml")):
            config = _experiment(target)
            records += load_runs(config.paths.runs, experiment=config.name).records
        else:
            records += load_runs(target).records
    path = Runs(records).report(
        args.out,
        metrics=args.metric or ["accuracy", "loss"],
        by=args.by or ["topology_id"],
    )
    print(path)
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    from onion_fl.baselines import cross_validate, evaluate
    from onion_fl.data.roles import split_subjects
    from onion_fl.experiment.runner import load_data

    config = _experiment(args.experiment)
    subjects, _ = load_data(config)
    if args.cv:
        _print_json(
            {
                m: cross_validate(subjects, m, k=args.cv, roles=config.data.roles)
                for m in args.models
            }
        )
    else:
        _print_json(
            evaluate(split_subjects(subjects, config.data.roles), models=args.models)
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from onion_fl.studio.api import create_app
    except ImportError as exc:
        raise ValueError(
            f"Studio needs its extra: pip install 'onion-fl[studio]' ({exc})"
        ) from None
    print(
        f"Onion-FL Studio on http://{args.host}:{args.port}  (root {Path(args.root).resolve()})"
    )
    uvicorn.run(create_app(args.root), host=args.host, port=args.port)
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    from onion_fl.experiment.config import experiment_schema

    text = json.dumps(experiment_schema(), indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="onion_fl", description="Hierarchical federated learning experiments."
    )
    sub = p.add_subparsers(dest="command", required=True)

    data = sub.add_parser(
        "data", help="prepare the cache of a dataset or show its card"
    )
    data.add_argument("action", choices=["prepare", "inspect"])
    data.add_argument("dataset")
    data.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    data.add_argument("--datasets-dir", default="datasets")
    data.add_argument("--cache-dir", default="data/cache")
    data.add_argument("--force", action="store_true")
    data.set_defaults(func=cmd_data)

    topology = sub.add_parser(
        "topology", help="draw a topology and give its topology_id"
    )
    topology.add_argument("action", choices=["show"])
    topology.add_argument("file")
    topology.add_argument(
        "--mermaid", action="store_true", help="print a Mermaid flowchart instead"
    )
    topology.add_argument(
        "--graph", action="store_true", help="print the JSON graph instead"
    )
    topology.set_defaults(func=cmd_topology)

    plan = sub.add_parser(
        "plan", help="dry run: composition per leaf and groups per link"
    )
    plan.add_argument("experiment")
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="run an experiment")
    run.add_argument("experiment")
    run.add_argument("--scenario", help="only the scenario with this name")
    run.add_argument("--backend", help="sim, local...; overrides the experiment's")
    run.add_argument("--mode", choices=["sim", "real"], help="real: alias of local")
    run.add_argument("--workers", type=int, default=1)
    run.set_defaults(func=cmd_run)

    node = sub.add_parser("node", help="start one node of a real run")
    node.add_argument("--id", required=True)
    node.add_argument("--run", required=True)
    node.add_argument(
        "--config",
        required=True,
        help="scenario.json of the run, or an experiment YAML",
    )
    node.add_argument("--scenario", help="with an experiment YAML: which scenario")
    node.add_argument("--seed", type=int, help="with an experiment YAML: which seed")
    node.set_defaults(func=cmd_node)

    ceiling = sub.add_parser(
        "ceiling", help="noise ceiling: top-1 and regret between replicas"
    )
    ceiling.add_argument("experiment")
    ceiling.add_argument(
        "--replicas", type=int, help="replicas per configuration; default len(seeds)"
    )
    ceiling.add_argument("--metric", default="accuracy")
    ceiling.add_argument(
        "--series", help="model/dataset of the root score, when there are several"
    )
    ceiling.add_argument(
        "--resample", action="store_true", help="draw the search anew per replica"
    )
    ceiling.add_argument(
        "--paired", action="store_true", help="one seed per replica, for every config"
    )
    ceiling.add_argument(
        "--minimize",
        action=argparse.BooleanOptionalAction,
        help="lower is better; default only for loss",
    )
    ceiling.add_argument("--workers", type=int, default=1)
    ceiling.set_defaults(func=cmd_ceiling)

    fidelity = sub.add_parser(
        "fidelity", help="fidelity of the runs of B against those of A (reference)"
    )
    fidelity.add_argument("a", help="runs folder of the reference backend")
    fidelity.add_argument("b", help="runs folder of the backend under test")
    fidelity.add_argument(
        "--metric",
        action="append",
        help="decide on it: a root score, time, bytes or time_to_target; "
        "default accuracy and time",
    )
    fidelity.add_argument(
        "--series", help="model/dataset of the root score, when there are several"
    )
    fidelity.add_argument("--target", help="metric=value for the time to target")
    fidelity.add_argument("--experiment", help="only the runs of this experiment")
    fidelity.add_argument(
        "--ceiling", help="output of onion_fl ceiling; default: the seeds of A"
    )
    fidelity.add_argument(
        "--minimize", action="append", help="a metric where lower is better"
    )
    fidelity.add_argument("--bootstrap", type=int, default=2000)
    fidelity.add_argument("--seed", type=int, default=0, help="of the bootstrap")
    fidelity.add_argument(
        "--calibration", help="the calibration_id every run of B must cite, alone"
    )
    fidelity.add_argument("--csv", help="also write the result as CSV rows")
    fidelity.add_argument("--html", help="also write an HTML report")
    fidelity.set_defaults(func=cmd_fidelity)

    cal = sub.add_parser(
        "calibrate", help="microbenchmarks that calibrate profiles and links"
    )
    steps = cal.add_subparsers(dest="step", required=True)
    measure = steps.add_parser("measure", help="run the microbenchmarks of a spec")
    measure.add_argument("spec")
    measure.add_argument("--out", required=True, help="a new session folder")
    measure.add_argument("--image")
    measure.add_argument("--project", help="prefix of the containers and networks")
    measure.add_argument("--subnet")
    fit = steps.add_parser("fit", help="sign the calibration of sessions of a spec")
    fit.add_argument("sessions", nargs="+", help="sessions of one spec (3 or more)")
    fit.add_argument("--out", default="calibrations")
    fit.add_argument(
        "--infrastructure-out", help="write the infrastructure citing the calibration"
    )
    check = steps.add_parser(
        "check", help="predict another session with a calibrated infrastructure"
    )
    check.add_argument("session")
    check.add_argument("--infrastructure", required=True)
    check.add_argument("--calibrations", default="calibrations")
    check.add_argument("--out", help="also write the result to this file")
    cal.set_defaults(func=cmd_calibrate)

    report = sub.add_parser("report", help="HTML report comparing runs")
    report.add_argument(
        "targets", nargs="+", help="experiment YAML files or runs folders"
    )
    report.add_argument("--out", default="report.html")
    report.add_argument("--metric", action="append")
    report.add_argument("--by", action="append")
    report.set_defaults(func=cmd_report)

    baseline = sub.add_parser(
        "baseline", help="classical baselines on the experiment's subject roles"
    )
    baseline.add_argument("experiment")
    baseline.add_argument("--models", nargs="+", default=["lr", "rf", "xgboost"])
    baseline.add_argument(
        "--cv", type=int, help="subject-level k-fold instead of the test role"
    )
    baseline.set_defaults(func=cmd_baseline)

    serve = sub.add_parser(
        "serve", help="Onion-FL Studio: the web app over the repository"
    )
    serve.add_argument(
        "--root", default=".", help="folder with topologies/, experiments/ and runs/"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    schema = sub.add_parser(
        "schema", help="JSON Schema of the experiment config and plugins"
    )
    schema.add_argument("--out")
    schema.set_defaults(func=cmd_schema)
    return p


def _utf8_output() -> None:
    """Windows consoles default to cp1252, which cannot draw the topology tree."""
    for stream in (sys.stdout, sys.stderr):
        if (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):  # not a reconfigurable text stream
                pass


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_output()
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, TopologyError, OSError, NotImplementedError) as exc:
        print(f"onion_fl {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
