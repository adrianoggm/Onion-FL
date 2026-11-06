"""The Studio's versions view and operator controls (continuum #203).

The runs are the stream experiments of ``test_experiment_stream``: a small CSV
format fixture, the stub trainer and a stub scorer, so nothing learns
(docs/RULES.md). The live test runs a simulation in a thread and drives it
through the API, as the Studio does.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from onion_fl.continuum.control import ENV, read_state  # noqa: E402
from onion_fl.experiment.config import ConfigError, parse_experiment  # noqa: E402
from onion_fl.experiment.runner import run_scenario  # noqa: E402
from onion_fl.experiment.sweep import scenarios  # noqa: E402
from onion_fl.studio.api import create_app  # noqa: E402
from tests import test_experiment_stream as stream  # noqa: E402
from tests.test_experiment_stream import (  # noqa: E402
    PROMOTION,
    events,
    experiment,
    run,
    stub_score,
)

workspace = stream.workspace  # the fixture, shared
TOKEN = "a-token-of-sixteen-or-more"
BEARER = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def operators(workspace: Path, monkeypatch) -> Path:
    path = workspace / "operators.yaml"
    path.write_text(f"alice: {TOKEN}\n", encoding="utf-8")
    monkeypatch.setenv(ENV, str(path))
    return path


def test_the_versions_view_of_a_promotion_run(workspace: Path, monkeypatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    path = run(workspace, promotion=PROMOTION)
    client = TestClient(create_app(workspace))

    view = client.get(f"/api/runs/{path.name}/versions").json()

    lineage = json.loads((path / "bundle" / "lineage.json").read_text("utf-8"))
    assert view["champion"] == lineage["champion"]
    assert view["promotion"]["name"] == "champion_challenger"
    statuses = {v["v"]: v["status"] for v in view["versions"]}
    assert statuses == {v["v"]: v["status"] for v in lineage["versions"]}
    digests = {v["v"]: v["digest"] for v in view["versions"]}
    assert all(digests[v["v"]] == v["digest"] for v in lineage["versions"][1:])
    decided = [d for d in view["decisions"] if d["event"] != "candidate"]
    assert decided and all(d["by"] == "policy" for d in decided)
    promoted = next(v for v in view["versions"] if v["decision"] == "promoted")
    assert {"delta", "forgetting", "recent_challenger"} <= set(promoted["scores"])
    # every round names what it served, and the prequential rows are counted
    served = [r["version"] for r in view["rounds"]]
    assert served[0] == 0 and len(served) == len(events(path, "round.started"))
    assert any(v["prequential"] for v in view["versions"])
    arrived = sum(e["value"] for e in events(path, "data.arrived"))
    assert sum(e["arrived"] for e in view["edges"]) == arrived


def test_a_run_without_versions_has_an_empty_view(workspace: Path, monkeypatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    path = run(workspace)
    view = TestClient(create_app(workspace)).get(f"/api/runs/{path.name}/versions")

    body = view.json()
    assert body["champion"] is None and body["versions"] == []
    assert body["promotion"] is None and body["edges"]


def test_orders_fail_closed_without_operators(workspace: Path, monkeypatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    path = run(workspace, promotion=PROMOTION)
    client = TestClient(create_app(workspace))

    response = client.post(
        f"/api/runs/{path.name}/control", json={"order": "rollback"}, headers=BEARER
    )

    assert response.status_code == 403
    assert client.get(f"/api/runs/{path.name}/control").json()["operators"] is False


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-token-of-sixteen"}, {"Authorization": TOKEN}],
)
def test_an_order_needs_an_operators_token(
    workspace: Path, operators: Path, headers
) -> None:
    path = run(workspace, promotion=PROMOTION)
    client = TestClient(create_app(workspace))

    response = client.post(
        f"/api/runs/{path.name}/control", json={"order": "rollback"}, headers=headers
    )

    assert response.status_code == 401
    assert not (path / "control.jsonl").exists()


def test_a_finished_run_takes_no_orders(workspace: Path, operators: Path) -> None:
    path = run(workspace, promotion=PROMOTION)
    client = TestClient(create_app(workspace))

    assert read_state(path)["listening"] is False
    response = client.post(
        f"/api/runs/{path.name}/control", json={"order": "rollback"}, headers=BEARER
    )
    assert response.status_code == 409


def test_paused_needs_operators(workspace: Path, monkeypatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    (scenario,) = scenarios(parse_experiment(experiment(workspace)))

    with pytest.raises(ConfigError, match="paused"):
        run_scenario(scenario, evaluate=stub_score, paused=True)


def test_launching_paused_passes_the_flag_and_the_operators(
    workspace: Path, monkeypatch
) -> None:
    from onion_fl.studio import api

    monkeypatch.delenv(ENV, raising=False)
    (workspace / "experiments").mkdir()
    (workspace / "experiments" / "e.yaml").write_text(
        json.dumps(experiment(workspace, promotion=PROMOTION)), encoding="utf-8"
    )
    file = workspace / "operators.yaml"
    file.write_text(f"alice: {TOKEN}\n", encoding="utf-8")
    started = {}

    class FakeProcess:
        pid = 7

        def __init__(self, args, cwd, env=None):
            started.update(args=args, env=env)

    monkeypatch.setattr(api.subprocess, "Popen", FakeProcess)

    plain = TestClient(create_app(workspace))
    assert (
        plain.post("/api/experiments/e/run", json={"paused": True}).status_code == 400
    )
    client = TestClient(create_app(workspace, operators=file))
    assert (
        client.post("/api/experiments/e/run", json={"paused": True}).status_code == 200
    )
    assert started["args"][-1] == "--paused"
    assert started["env"][ENV] == str(file.resolve())


# --- acceptance: promote and roll back a live simulation from the Studio ------------


def wait(until, timeout: float = 30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = until()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out")


def test_the_studio_promotes_and_rolls_back_a_live_simulation(
    workspace: Path, operators: Path
) -> None:
    (scenario,) = scenarios(
        parse_experiment(experiment(workspace, promotion=PROMOTION))
    )
    done: dict = {}

    def simulate() -> None:
        try:
            done["path"] = run_scenario(scenario, evaluate=stub_score, paused=True)
        except BaseException as exc:  # reported by the test, not lost in the thread
            done["error"] = exc

    thread = threading.Thread(target=simulate, daemon=True)
    thread.start()
    client = TestClient(create_app(workspace))

    def live_run():
        runs = [r for r in client.get("/api/runs").json() if r["status"] == "running"]
        if runs:
            state = client.get(f"/api/runs/{runs[0]['run_id']}/control").json()
            if (state["state"] or {}).get("paused"):
                return runs[0]["run_id"]
        return None

    run_id = wait(live_run)
    control = f"/api/runs/{run_id}/control"

    def order(**body) -> None:
        response = client.post(control, json=body, headers=BEARER)
        assert response.status_code == 200, response.text

    # schedule a pause after a few rounds, resume, and wait for it
    order(order="pause", at=12.0)
    order(order="resume")

    def paused_at_12():
        state = client.get(control).json()["state"]
        return state["paused"] and state["t"] == 12.0

    wait(paused_at_12)
    view = client.get(f"/api/runs/{run_id}/versions").json()
    champion = view["champion"]
    assert champion > 0
    order(order="rollback")  # to the champion it replaced
    order(order="promote", version=champion)  # and back again

    def seen_while_paused():
        decisions = client.get(f"/api/runs/{run_id}/versions").json()["decisions"]
        return [d["event"] for d in decisions if d["by"] == "operator"] == [
            "rollback",
            "promoted",
        ]

    wait(seen_while_paused)  # the paused run wrote its events out
    assert client.get(control).json()["state"]["paused"]
    order(order="resume")
    thread.join(timeout=60)

    assert "error" not in done and done["path"].name == run_id
    path = done["path"]
    (rollback,) = events(path, "version.rollback")
    assert rollback["tags"]["previous"] == champion
    assert rollback["tags"]["reason"] == "operator"
    promoted = [
        e for e in events(path, "version.promoted") if e["tags"]["reason"] == "operator"
    ]
    assert [e["value"] for e in promoted] == [champion]
    assert promoted[0]["tags"]["superseded"] == rollback["value"]
    received = [e["tags"]["operator"] for e in events(path, "control.received")]
    assert received == ["alice"] * 5 and not events(path, "control.refused")
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished" and meta.get("run_hash")
    final = TestClient(create_app(workspace)).get(f"/api/runs/{run_id}/versions")
    statuses = {v["v"]: v["status"] for v in final.json()["versions"]}
    lineage = json.loads((path / "bundle" / "lineage.json").read_text("utf-8"))
    assert statuses == {v["v"]: v["status"] for v in lineage["versions"]}
