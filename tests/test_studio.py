"""Tests for Onion-FL Studio: the API, the previews and the static app (issue #103).

The workspace holds a CSV format fixture and an experiment with the stub
trainer and a stub scorer, so nothing is trained or evaluated (docs/RULES.md).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from onion_fl.studio.api import create_app  # noqa: E402

TOPOLOGY = {
    "name": "two_fogs",
    "levels": ["global", "fog", "edge"],
    "root": {"id": "cloud"},
    "fog": {
        "defaults": {"link_up": {"profile": "wifi"}},
        "nodes": [{"id": "fog_a", "home": "demo"}, {"id": "fog_b", "home": "demo"}],
    },
    "edge": {"link_up": {"profile": "4g"}},
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for folder in ("topologies", "experiments", "datasets", "raw"):
        (tmp_path / folder).mkdir()
    rows = ["pp,cond,f1"] + [
        f"{s},{'NT'[i % 2]},{s + i}" for s in range(1, 11) for i in range(4)
    ]
    (tmp_path / "raw" / "t.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "datasets" / "demo.yaml").write_text(
        textwrap.dedent(
            f"""
            name: demo
            root: {(tmp_path / "raw").as_posix()}
            source: {{reader: csv, path: t.csv}}
            steps:
              - subject: {{column: pp}}
              - label: {{task: stress, column: cond, map: {{"N": 0, "T": 1}}}}
              - features: {{}}
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "topologies" / "two_fogs.yaml").write_text(
        yaml.safe_dump(TOPOLOGY), encoding="utf-8"
    )
    experiment = {
        "name": "demo_exp",
        "description": "fixture",
        "topology": "two_fogs",
        "data": {"datasets": {"demo": {}}, "roles": {"test": 0.2}},
        "learning": {
            "model": {"name": "modular_mlp", "adapter_width": 4, "trunk_hidden": [4]},
            "trainer": "stub",
        },
        "rounds": 2,
        "evaluation": {"global": {"every": 1}},
        "seeds": [0, 1],
        "sweep": {"data.placement.alpha": [0.0, 1.0]},
    }
    (tmp_path / "experiments" / "demo_exp.yaml").write_text(
        yaml.safe_dump(experiment), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def client(root: Path) -> TestClient:
    return TestClient(create_app(root))


@pytest.fixture
def recorded(root: Path) -> list[str]:
    """Two finished runs of the fixture experiment (stub trainer and scorer)."""
    from onion_fl.experiment.config import parse_experiment
    from onion_fl.experiment.runner import run_scenario
    from onion_fl.experiment.sweep import scenarios
    from onion_fl.studio.api import rooted

    raw = yaml.safe_load(
        (root / "experiments" / "demo_exp.yaml").read_text(encoding="utf-8")
    )
    config = rooted(parse_experiment(raw | {"sweep": {}}), root)

    def score(model, data):
        return {"accuracy": 0.5}, data.n_samples

    return [run_scenario(s, evaluate=score).name for s in scenarios(config)]


# --- schema and tutorial ------------------------------------------------------------------------


def test_the_schema_and_the_plugin_catalogue(client: TestClient) -> None:
    schema = client.get("/api/schema").json()

    assert "plugins" in schema and "properties" in schema


def test_the_tutorial_explains_every_axis(client: TestClient) -> None:
    axes = client.get("/api/tutorial").json()

    from onion_fl.experiment.config import REGISTRIES

    assert {axis["kind"] for axis in axes} == set(REGISTRIES)
    sharing = next(a for a in axes if a["kind"] == "sharing")
    assert sharing["title"] and sharing["explain"]
    assert {p["name"] for p in sharing["plugins"]} >= {"fedavg", "fedper", "zone"}


# --- topologies -------------------------------------------------------------------------------------


def test_the_topology_library(client: TestClient) -> None:
    (entry,) = client.get("/api/topologies").json()

    assert (
        entry["name"] == "two_fogs"
        and entry["leaves"] == 2
        and len(entry["topology_id"]) == 64
    )


def test_a_topology_with_its_graph_and_yaml(client: TestClient) -> None:
    body = client.get("/api/topologies/two_fogs").json()

    assert "fog_a" in body["yaml"]
    assert [n["id"] for n in body["graph"]["nodes"]] == ["cloud", "fog_a", "fog_b"]


def test_validating_a_topology_returns_its_graph_or_where_it_fails(
    client: TestClient,
) -> None:
    ok = client.post("/api/topologies/validate", json=TOPOLOGY)
    bad = client.post(
        "/api/topologies/validate", json=TOPOLOGY | {"levels": ["global"]}
    )

    assert ok.status_code == 200 and ok.json()["graph"]["name"] == "two_fogs"
    assert bad.status_code == 422 and "levels" in bad.json()["errors"][0]


def test_saving_a_topology(client: TestClient, root: Path) -> None:
    three = TOPOLOGY | {
        "name": "three",
        "fog": {"nodes": [{"id": f"f{i}"} for i in range(3)]},
    }

    response = client.put("/api/topologies/three", json=three)

    assert response.status_code == 200
    saved = yaml.safe_load(
        (root / "topologies" / "three.yaml").read_text(encoding="utf-8")
    )
    assert [n["id"] for n in saved["fog"]["nodes"]] == ["f0", "f1", "f2"]
    assert client.put("/api/topologies/broken", json={"levels": []}).status_code == 422
    assert not (root / "topologies" / "broken.yaml").exists()


@pytest.mark.parametrize("name", ["..", "../secrets", "a b", "x%2Fy"])
def test_names_never_leave_their_folder(client: TestClient, name: str) -> None:
    assert client.get(f"/api/topologies/{name}").status_code in (400, 404)
    assert client.put(f"/api/topologies/{name}", json=TOPOLOGY).status_code in (
        400,
        404,
        405,
    )


def test_an_unknown_topology_is_404(client: TestClient) -> None:
    assert client.get("/api/topologies/nope").status_code == 404


# --- experiments ----------------------------------------------------------------------------------------


def test_the_experiment_list_and_detail(client: TestClient) -> None:
    (entry,) = client.get("/api/experiments").json()
    detail = client.get("/api/experiments/demo_exp").json()

    assert entry == {
        "name": "demo_exp",
        "description": "fixture",
        "topology": "two_fogs",
        "scenarios": 2,
        "seeds": 2,
    }
    assert len(detail["scenarios"]) == 4
    assert {s["name"] for s in detail["scenarios"]} == {
        "data.placement.alpha=0.0",
        "data.placement.alpha=1.0",
    }


def test_validating_an_experiment_names_the_error_path(
    client: TestClient, root: Path
) -> None:
    raw = yaml.safe_load(
        (root / "experiments" / "demo_exp.yaml").read_text(encoding="utf-8")
    )

    ok = client.post("/api/experiments/validate", json=raw)
    bad = client.post("/api/experiments/validate", json=raw | {"rounds": 0})

    assert ok.status_code == 200 and len(ok.json()["scenarios"]) == 4
    assert bad.status_code == 422 and any("rounds" in e for e in bad.json()["errors"])


def test_the_plan_of_an_experiment(client: TestClient) -> None:
    previews = client.post("/api/experiments/demo_exp/plan").json()

    assert len(previews) == 4
    assert set(previews[0]["composition"]) == {"fog_a", "fog_b"}


def test_the_plan_shows_the_config_id_a_run_from_the_root_signs(
    client: TestClient, root: Path
) -> None:
    from onion_fl.experiment.config import load_experiment
    from onion_fl.experiment.sweep import scenarios

    previews = client.post("/api/experiments/demo_exp/plan").json()

    config = load_experiment(root / "experiments" / "demo_exp.yaml")
    assert [p["config_id"] for p in previews] == [
        s.config_id for s in scenarios(config)
    ]


def test_a_plan_without_data_says_what_is_missing(
    client: TestClient, root: Path
) -> None:
    (root / "raw" / "t.csv").unlink()

    response = client.post("/api/experiments/demo_exp/plan")

    assert response.status_code == 422 and "t.csv" in response.json()["errors"][0]


def test_launching_a_run_starts_the_command_line(
    client: TestClient, root: Path, monkeypatch
) -> None:
    from onion_fl.studio import api

    started = {}

    class FakeProcess:
        pid = 4242

        def __init__(self, args, cwd):
            started.update(args=args, cwd=cwd)

    monkeypatch.setattr(api.subprocess, "Popen", FakeProcess)

    response = client.post(
        "/api/experiments/demo_exp/run",
        json={"mode": "sim", "workers": 2, "scenario": "base"},
    )

    import sys

    assert response.json() == {"pid": 4242}
    experiment = str(root / "experiments" / "demo_exp.yaml")
    assert started["args"] == [
        sys.executable,
        "-m",
        "onion_fl",
        "run",
        experiment,
        "--mode",
        "sim",
        "--workers",
        "2",
        "--scenario",
        "base",
    ]
    assert started["cwd"] == str(root)


# --- runs -------------------------------------------------------------------------------------------------


def test_the_run_list_and_detail(client: TestClient, recorded: list[str]) -> None:
    runs = client.get("/api/runs").json()

    assert {r["run_id"] for r in runs} == set(recorded)
    assert all(
        r["status"] == "finished" and r["experiment"] == "demo_exp" for r in runs
    )
    detail = client.get(f"/api/runs/{recorded[0]}").json()
    assert detail["meta"]["run_id"] == recorded[0]
    assert detail["summary"]["rounds"] == 2
    assert set(detail["composition"]) == {"fog_a", "fog_b"}
    assert set(detail["roles"]["demo"]) >= {"test", "train"}


def test_events_come_in_pages_for_live_monitoring(
    client: TestClient, recorded: list[str]
) -> None:
    first = client.get(f"/api/runs/{recorded[0]}/events?after=0&limit=5").json()
    second = client.get(
        f"/api/runs/{recorded[0]}/events?after={first['next']}&limit=5"
    ).json()

    assert len(first["events"]) == 5 and first["next"] == 5
    assert second["events"][0] != first["events"][0]


def test_a_metric_series_per_round(client: TestClient, recorded: list[str]) -> None:
    series = client.get(
        f"/api/runs/{recorded[0]}/series?level=global&name=accuracy&model=global"
    ).json()

    assert [p["round"] for p in series] == [1, 2]
    assert {"mean", "min", "max"} <= set(series[0])


def test_comparing_runs(client: TestClient, recorded: list[str]) -> None:
    rows = client.get(
        "/api/compare?experiment=demo_exp&level=global&metric=accuracy&by=topology_id"
    ).json()

    assert rows and rows[0]["n"] == 2 and {"mean", "ci_low", "ci_high"} <= set(rows[0])


def test_an_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/api/runs/20260101T000000Z-000000000000").status_code == 404


# --- previews ------------------------------------------------------------------------------------------------


def test_the_sharing_preview_shows_what_crosses_each_link(client: TestClient) -> None:
    body = {"topology": TOPOLOGY, "sharing": "fedper", "datasets": ["a", "b"]}

    preview = client.post("/api/preview/sharing", json=body).json()

    edge_link = next(link for link in preview["links"] if link["child"] == "*")
    assert "trunk" in edge_link["groups"] and not any(
        g.startswith("head") for g in edge_link["groups"]
    )
    assert preview["held"]["global"] == ["adapter.a", "adapter.b", "trunk"]


def test_the_placement_preview_moves_with_alpha(client: TestClient) -> None:
    topology = TOPOLOGY | {
        "fog": {"nodes": [{"id": "fog_a", "home": "a"}, {"id": "fog_b", "home": "b"}]}
    }

    def mix(alpha):
        body = {
            "topology": topology,
            "placement": {"name": "mixing", "alpha": alpha},
            "datasets": {"a": 10, "b": 10},
        }
        return client.post("/api/preview/placement", json=body).json()

    segregated, mixed = mix(0.0), mix(1.0)

    assert (
        segregated["fog_a"]["datasets"] == {"a": 10}
        and segregated["fog_a"]["entropy"] == 0
    )
    assert mixed["fog_a"]["datasets"] == {"a": 5, "b": 5} and mixed["fog_a"][
        "entropy"
    ] == pytest.approx(1.0)


def test_the_link_preview_compares_profiles(client: TestClient) -> None:
    lan = client.post(
        "/api/preview/link", json={"profile": "lan", "bytes": 100_000}
    ).json()
    lora = client.post(
        "/api/preview/link", json={"profile": "lora", "bytes": 100_000}
    ).json()

    assert lora["transmission_up_s"] > lan["transmission_up_s"]
    assert lora["latency"]["p50"] > lan["latency"]["p50"]
    assert {"p50", "p95", "mean"} <= set(lan["latency"]) and 0 <= lan["loss"] <= 1


def test_previews_report_bad_input(client: TestClient) -> None:
    response = client.post(
        "/api/preview/sharing",
        json={"topology": TOPOLOGY, "sharing": "nope", "datasets": ["a"]},
    )

    assert response.status_code == 422 and "nope" in response.json()["errors"][0]


# --- the app ---------------------------------------------------------------------------------------------------


def test_the_single_page_app_is_served(client: TestClient) -> None:
    index = client.get("/")
    script = client.get("/app.js")

    assert index.status_code == 200 and "Onion-FL Studio" in index.text
    assert index.headers["cache-control"] == "no-cache"
    assert script.status_code == 200 and "fetch(" in script.text


def test_serve_is_a_command(monkeypatch, root: Path) -> None:
    import uvicorn

    from onion_fl.experiment.cli import main

    calls = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: calls.update(host=host, port=port)
    )

    assert main(["serve", "--root", str(root), "--port", "9999"]) == 0
    assert calls == {"host": "127.0.0.1", "port": 9999}


def test_json_files_only(client: TestClient, root: Path) -> None:
    (root / "runs").mkdir(exist_ok=True)
    (root / "runs" / "junk").mkdir()  # a folder without run.json is not a run

    assert client.get("/api/runs").json() == []
    assert json.loads(client.get("/api/topologies").text)


def test_series_can_pick_the_overall_or_one_dataset(
    client: TestClient, recorded: list[str]
) -> None:
    base = f"/api/runs/{recorded[0]}/series?level=global&name=accuracy&model=global"

    overall = client.get(f"{base}&dataset=*").json()
    demo = client.get(f"{base}&dataset=demo").json()
    nothing = client.get(f"{base}&dataset=wesad").json()

    assert [p["count"] for p in overall] == [1, 1]
    assert [p["count"] for p in demo] == [1, 1]
    assert nothing == []


def test_a_bad_launch_mode_is_refused(client: TestClient) -> None:
    response = client.post("/api/experiments/demo_exp/run", json={"mode": "quantum"})

    assert response.status_code == 400


def test_broken_files_are_left_out_of_the_lists(client: TestClient, root: Path) -> None:
    (root / "topologies" / "broken.yaml").write_text(
        "levels: [only]\n", encoding="utf-8"
    )
    (root / "experiments" / "broken.yaml").write_text(
        "name: broken\nrounds: 0\n", encoding="utf-8"
    )

    assert [t["name"] for t in client.get("/api/topologies").json()] == ["two_fogs"]
    assert [e["name"] for e in client.get("/api/experiments").json()] == ["demo_exp"]
    assert client.get("/api/experiments/broken").status_code == 422


@pytest.mark.parametrize(
    "text",
    ["name: [unclosed\n", "- a list\n- not a mapping\n"],
    ids=["syntax", "list"],
)
def test_files_that_are_not_yaml_mappings_are_reported(
    client: TestClient, root: Path, text: str
) -> None:
    for kind in ("topologies", "experiments"):
        (root / kind / "odd.yaml").write_text(text, encoding="utf-8")

    assert [t["name"] for t in client.get("/api/topologies").json()] == ["two_fogs"]
    assert [e["name"] for e in client.get("/api/experiments").json()] == ["demo_exp"]
    assert client.get("/api/experiments/odd").status_code == 422
    assert client.get("/api/topologies/odd").status_code == 422


def test_an_experiment_whose_sweep_breaks_is_reported(
    client: TestClient, root: Path
) -> None:
    raw = yaml.safe_load((root / "experiments" / "demo_exp.yaml").read_text("utf-8"))
    raw["sweep"] = {"rounds": [0]}  # valid as a sweep, not as a scenario
    (root / "experiments" / "swept.yaml").write_text(yaml.safe_dump(raw), "utf-8")

    response = client.get("/api/experiments/swept")

    assert response.status_code == 422 and "rounds" in response.json()["errors"][0]


def test_preview_errors_name_the_problem(client: TestClient) -> None:
    placement = client.post(
        "/api/preview/placement",
        json={"topology": TOPOLOGY, "placement": "nope", "datasets": {"demo": 1}},
    )
    link = client.post("/api/preview/link", json={"profile": "carrier_pigeon"})

    assert placement.status_code == 422 and "nope" in placement.json()["errors"][0]
    assert link.status_code == 422 and "carrier_pigeon" in link.json()["errors"][0]


def test_a_file_error_is_reported_without_the_local_path(
    client: TestClient, monkeypatch
) -> None:
    def missing(config):
        raise FileNotFoundError(
            2, "No such file or directory", "C:/private/place/x.csv"
        )

    monkeypatch.setattr("onion_fl.experiment.runner.plan", missing)
    response = client.post("/api/experiments/demo_exp/plan")

    text = json.dumps(response.json())
    assert response.status_code == 422
    assert "x.csv" in text and "private" not in text
