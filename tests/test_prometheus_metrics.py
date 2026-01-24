from __future__ import annotations

from unittest.mock import Mock

from flower_basic import prometheus_metrics


def test_get_metrics_port_from_env_priority(monkeypatch) -> None:
    monkeypatch.setenv("METRICS_PORT_SERVER", "9100")
    monkeypatch.setenv("METRICS_PORT", "9200")

    port = prometheus_metrics.get_metrics_port_from_env(default=8000, component="server")
    assert port == 9100


def test_get_metrics_port_from_env_fallback(monkeypatch) -> None:
    monkeypatch.delenv("METRICS_PORT_SERVER", raising=False)
    monkeypatch.setenv("METRICS_PORT", "9300")

    port = prometheus_metrics.get_metrics_port_from_env(default=8000, component="server")
    assert port == 9300


def test_start_metrics_server_starts_once(monkeypatch) -> None:
    monkeypatch.setattr(prometheus_metrics, "PROMETHEUS_AVAILABLE", True)
    monkeypatch.setattr(prometheus_metrics, "_metrics_server_started", False)
    start_http_server = Mock()
    monkeypatch.setattr(prometheus_metrics, "start_http_server", start_http_server)

    assert prometheus_metrics.start_metrics_server(port=9999) is True
    assert prometheus_metrics.start_metrics_server(port=9999) is True
    start_http_server.assert_called_once_with(9999)


def test_start_metrics_server_no_prometheus(monkeypatch) -> None:
    monkeypatch.setattr(prometheus_metrics, "PROMETHEUS_AVAILABLE", False)
    monkeypatch.setattr(prometheus_metrics, "_metrics_server_started", False)

    assert prometheus_metrics.start_metrics_server(port=9999) is False
