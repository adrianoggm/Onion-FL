from __future__ import annotations

import json
import re
from pathlib import Path

from onion_fl.observability.sinks import EXTRA_LABELS, LABELS, METRICS

DASHBOARD = (
    Path(__file__).resolve().parents[1]
    / "docker"
    / "grafana"
    / "provisioning"
    / "dashboards"
    / "json"
    / "onion-fl.json"
)


def dashboard() -> dict:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def expressions() -> list[str]:
    return [
        target["expr"]
        for panel in dashboard()["panels"]
        for target in panel.get("targets", [])
    ]


def test_every_query_uses_a_series_the_sink_exports() -> None:
    used = {
        name
        for expr in expressions()
        for name in re.findall(r"\bonionfl_[a-z_]+", expr)
    }

    assert used and used <= set(METRICS)


def test_every_label_in_a_query_exists_on_the_series() -> None:
    allowed = set(LABELS) | set(EXTRA_LABELS)
    for expr in expressions():
        matchers = re.findall(r"(\w+)\s*=~?\s*\"", expr)
        grouping = [
            g.strip()
            for group in re.findall(r"by \(([^)]*)\)", expr)
            for g in group.split(",")
        ]
        assert set(matchers) | set(grouping) <= allowed, expr


def test_queries_follow_the_run_and_topology_variables() -> None:
    names = {v["name"] for v in dashboard()["templating"]["list"]}

    assert names == {"run_id", "topology_id"}
    for expr in expressions():
        assert 'run_id=~"$run_id"' in expr and 'topology_id=~"$topology_id"' in expr, (
            expr
        )


def test_panels_have_unique_ids_and_the_prometheus_datasource() -> None:
    panels = dashboard()["panels"]
    ids = [p["id"] for p in panels]

    assert len(ids) == len(set(ids))
    for panel in panels:
        if panel["type"] != "row":
            assert panel["datasource"]["uid"] == "prometheus", panel["title"]
    assert dashboard()["uid"] == "onion-fl-main"
