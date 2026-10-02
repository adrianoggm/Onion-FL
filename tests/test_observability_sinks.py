"""Tests for the Prometheus and OpenTelemetry sinks (issue #96).

The events come from a federation with the stub trainer and a stub scorer;
nothing is trained (docs/RULES.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry

from onion_fl.core.topology import parse_topology
from onion_fl.learning.model import (
    DataShape,
    ModularMLP,
    ModularMLPConfig,
    state_arrays,
)
from onion_fl.learning.trainers import trainers
from onion_fl.observability.run import Run
from onion_fl.observability.sinks import OtelSink, PrometheusSink
from onion_fl.roles import EdgeSpec, build_federation

SHAPE = DataShape(dataset="a", task="t", n_features=2, n_classes=2)
CONFIG = ModularMLPConfig(adapter_width=2, trunk_hidden=[2], dropout=0.0)
TOPOLOGY = parse_topology(
    {
        "name": "t",
        "levels": ["global", "fog", "edge"],
        "root": {"id": "cloud", "eval": {"every": 1}},
        "fog": {"nodes": [{"id": "fog_0"}]},
    }
)


def model() -> ModularMLP:
    return ModularMLP(CONFIG, [SHAPE], seed=0)


def record(tmp_path: Path, *sinks) -> Run:
    edges = {
        "fog_0": [
            EdgeSpec("e1", model(), trainer=trainers.create("stub")),
            EdgeSpec("e2", model(), trainer=trainers.create("stub")),
        ],
        "cloud": [
            EdgeSpec("t1", model(), data=0.5, train=False, tags={"dataset": "a"})
        ],
    }
    fed = build_federation(
        TOPOLOGY,
        edges,
        initial_state=state_arrays(model()),
        rounds=2,
        evaluate=lambda m, d: ({"accuracy": d}, 4),
    )
    run = Run(tmp_path, config={}, topology=TOPOLOGY, seed=0, sinks=sinks)
    run.attach(fed)
    fed.run()
    run.finish()
    return run


# --- Prometheus -------------------------------------------------------------------------------


@pytest.fixture
def prometheus(tmp_path: Path):
    sink = PrometheusSink(registry=CollectorRegistry())
    run = record(tmp_path, sink)
    return sink, run


def labels(
    run: Run, level: str, node: str, dataset: str = "", **extra
) -> dict[str, str]:
    return {
        "run_id": run.run_id,
        "topology_id": TOPOLOGY.topology_id,
        "level": level,
        "node": node,
        "dataset": dataset,
        **extra,
    }


def test_rounds_and_finish_are_exported(prometheus) -> None:
    sink, run = prometheus
    value = sink.registry.get_sample_value

    assert value("onionfl_round", labels(run, "global", "cloud")) == 2
    assert value("onionfl_round", labels(run, "fog", "fog_0")) == 2
    finished = {"run_id": run.run_id, "topology_id": TOPOLOGY.topology_id}
    assert value("onionfl_run_finished", finished) == 1


def test_metrics_keep_their_last_value_per_dataset(prometheus) -> None:
    sink, run = prometheus
    extra = {"name": "eval.accuracy", "model": "global", "source": "evaluators"}

    value = sink.registry.get_sample_value(
        "onionfl_metric", labels(run, "global", "cloud", "a", **extra)
    )

    assert value == 0.5


def test_traffic_is_counted_per_node_and_kind(prometheus) -> None:
    sink, run = prometheus
    value = sink.registry.get_sample_value

    updates = value("onionfl_messages_total", labels(run, "edge", "e1", kind="update"))
    size = value(
        "onionfl_message_bytes_total", labels(run, "edge", "e1", kind="update")
    )
    assert updates == 2 and size > 0


def test_diagnostics_are_exported_per_group(prometheus) -> None:
    sink, run = prometheus

    value = sink.registry.get_sample_value(
        "onionfl_diagnostic",
        labels(run, "fog", "fog_0", name="divergence_l2", group="trunk"),
    )

    assert value == pytest.approx(0.0)  # both edges add the same constant


def test_drops_and_failed_quorums_are_counted(tmp_path: Path) -> None:
    sink = PrometheusSink(registry=CollectorRegistry())
    run = Run(tmp_path, config={}, topology=TOPOLOGY, seed=0, sinks=[sink])
    run.nodes["fog_0"] = ("fog", "aggregator")

    for name in ("link.dropped", "message.dropped_offline", "round.quorum_failed"):
        run.on_event(
            {
                "t": 1.0,
                "node": "fog_0",
                "name": name,
                "value": 1,
                "tags": {"kind": "update"},
            }
        )

    value = sink.registry.get_sample_value
    assert (
        value(
            "onionfl_messages_dropped_total", labels(run, "fog", "fog_0", kind="update")
        )
        == 2
    )
    assert value("onionfl_quorum_failed_total", labels(run, "fog", "fog_0")) == 1


def test_the_metrics_endpoint_can_be_served() -> None:
    import urllib.request

    sink = PrometheusSink(registry=CollectorRegistry())
    server, thread = sink.serve(port=0, addr="127.0.0.1")
    try:
        port = server.server_address[1]
        body = (
            urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics").read().decode()
        )
    finally:
        server.shutdown()

    assert "onionfl_round" in body


# --- OpenTelemetry ----------------------------------------------------------------------------


@pytest.fixture
def spans(tmp_path: Path):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    record(tmp_path, OtelSink(provider, base_ns=0))
    return exporter.get_finished_spans()


def test_one_span_per_send_and_per_receive(spans) -> None:
    sends = [s for s in spans if s.name.startswith("send ")]
    receives = [s for s in spans if s.name.startswith("receive ")]

    assert sends and len(sends) == len(receives)


def test_a_receive_span_links_to_its_send_and_lasts_the_transit(spans) -> None:
    sends = {s.context.span_id: s for s in spans if s.name.startswith("send ")}
    receive = next(s for s in spans if s.name == "receive update")

    (link,) = receive.links
    send = sends[link.context.span_id]
    assert send.name == "send update"
    assert receive.start_time == send.start_time
    assert receive.end_time > receive.start_time  # virtual latency, in ns
    assert receive.attributes["msg_id"] == send.attributes["msg_id"]


def test_spans_carry_the_run_labels(spans) -> None:
    send = next(s for s in spans if s.name == "send update")

    assert send.attributes["node"] in {"e1", "e2"}
    assert send.attributes["level"] == "edge"
    assert send.attributes["round"] in (1, 2)
    assert send.attributes["bytes"] > 0
    assert "topology_id" in send.attributes


def test_the_declared_series_are_the_exported_ones(prometheus) -> None:
    from onion_fl.observability.sinks import METRICS

    sink, _ = prometheus
    exported = {
        f"{family.name}_total" if family.type == "counter" else family.name
        for family in sink.registry.collect()
    }

    assert exported == set(METRICS)
