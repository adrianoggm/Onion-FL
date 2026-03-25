from __future__ import annotations

import json
from pathlib import Path

from flower_basic.brokers import fog, sweet_fog


def test_broker_shutdown_runtime_skips_pushgateway() -> None:
    assert not hasattr(fog, "push_metrics_to_gateway")
    assert not hasattr(sweet_fog, "push_metrics_to_gateway")


def test_broker_region_panels_collapse_duplicate_series() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dashboard_path = (
        repo_root
        / "docker"
        / "grafana"
        / "provisioning"
        / "dashboards"
        / "json"
        / "flower-fl.json"
    )
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    panels = {panel["title"]: panel for panel in dashboard["panels"] if "title" in panel}

    assert panels["Clients per Fog Region"]["targets"][0]["expr"] == (
        "max by (region) (flower_fl_broker_clients_per_region)"
    )
    assert panels["Buffer Size per Region"]["targets"][0]["expr"] == (
        "max by (region) (flower_fl_broker_buffer_size)"
    )
