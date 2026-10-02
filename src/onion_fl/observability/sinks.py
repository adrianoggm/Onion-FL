from __future__ import annotations

"""Live sinks: Prometheus and OpenTelemetry (spec §10.5).

Both receive the same events as ``events.jsonl`` (pass them to ``Run(sinks=…)``)
and only translate them; ``events.jsonl`` stays the source of truth.

Prometheus series, all labelled ``run_id, topology_id, level, node, dataset``:

=================================  ==============================================
``onionfl_metric``                 last value of a metric (+ ``name, model, source``)
``onionfl_diagnostic``             last value of a diagnostic (+ ``name, group``)
``onionfl_round``                  current round of a node
``onionfl_messages_total``         messages sent (+ ``kind``)
``onionfl_message_bytes_total``    encoded bytes sent (+ ``kind``)
``onionfl_messages_dropped_total``  messages lost or dropped offline (+ ``kind``)
``onionfl_quorum_failed_total``    rounds closed without quorum
``onionfl_run_finished``           1 once the run has finished (``run_id, topology_id``)
=================================  ==============================================

OpenTelemetry: one span per send and one per receive. The receive span starts
when the message was sent, ends when it arrives (virtual time, in ns from
``base_ns``) and links to the send span through the message's ``msg_id``.
"""

import time
from collections.abc import Mapping
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Link
from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

LABELS = ("run_id", "topology_id", "level", "node", "dataset")
METRICS = (
    "onionfl_metric",
    "onionfl_diagnostic",
    "onionfl_round",
    "onionfl_messages_total",
    "onionfl_message_bytes_total",
    "onionfl_messages_dropped_total",
    "onionfl_quorum_failed_total",
    "onionfl_run_finished",
)
EXTRA_LABELS = ("name", "model", "source", "group", "kind")


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


class PrometheusSink:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        base = list(LABELS)

        def gauge(name: str, doc: str, extra=()) -> Gauge:
            return Gauge(name, doc, [*base, *extra], registry=self.registry)

        def counter(name: str, doc: str, extra=()) -> Counter:
            return Counter(name, doc, [*base, *extra], registry=self.registry)

        self.metric = gauge(
            "onionfl_metric",
            "Last value of a metric event",
            ("name", "model", "source"),
        )
        self.diagnostic = gauge(
            "onionfl_diagnostic", "Last value of a diagnostic", ("name", "group")
        )
        self.round = gauge("onionfl_round", "Current round of a node")
        self.messages = counter("onionfl_messages", "Messages sent", ("kind",))
        self.bytes = counter("onionfl_message_bytes", "Encoded bytes sent", ("kind",))
        self.dropped = counter(
            "onionfl_messages_dropped", "Messages lost on a link or offline", ("kind",)
        )
        self.quorum_failed = counter(
            "onionfl_quorum_failed", "Rounds closed without quorum"
        )
        self.finished = Gauge(
            "onionfl_run_finished",
            "1 once the run has finished",
            ["run_id", "topology_id"],
            registry=self.registry,
        )

    def write(self, event: Mapping[str, Any]) -> None:
        tags = event.get("tags") or {}
        name, value = event["name"], event.get("value")
        base = {k: _text(event.get(k)) for k in LABELS if k != "dataset"} | {
            "dataset": _text(tags.get("dataset"))
        }
        if event["kind"] == "metric" and _number(value):
            extra = {
                "name": name,
                "model": _text(tags.get("model")),
                "source": _text(tags.get("source")),
            }
            self.metric.labels(**base, **extra).set(value)
        elif event["kind"] == "diagnostic" and _number(value):
            self.diagnostic.labels(
                **base, name=name[len("diagnostic.") :], group=_text(tags.get("group"))
            ).set(value)
        if (
            name in ("round.started", "round.participants")
            and event.get("round") is not None
        ):
            self.round.labels(**base).set(event["round"])
        elif name == "message.sent":
            self.messages.labels(**base, kind=_text(tags.get("kind"))).inc()
            self.bytes.labels(**base, kind=_text(tags.get("kind"))).inc(value or 0)
        elif name in ("link.dropped", "message.dropped_offline"):
            self.dropped.labels(**base, kind=_text(tags.get("kind"))).inc()
        elif name == "round.quorum_failed":
            self.quorum_failed.labels(**base).inc()
        elif name == "run.finished":
            self.finished.labels(
                run_id=base["run_id"], topology_id=base["topology_id"]
            ).set(1)

    def serve(self, port: int = 9464, addr: str = "0.0.0.0") -> Any:
        """Expose ``/metrics`` for Prometheus to scrape; returns the server and its thread."""
        return start_http_server(port, addr=addr, registry=self.registry)

    def close(self) -> None:
        pass


def otlp_provider(
    endpoint: str = "http://localhost:4320", service: str = "onion-fl"
) -> Any:
    """A tracer provider that exports over OTLP/HTTP, e.g. to the collector of ``docker/``."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


class OtelSink:
    def __init__(self, tracer_provider: Any = None, base_ns: int | None = None) -> None:
        self.provider = tracer_provider or trace.get_tracer_provider()
        self.tracer = self.provider.get_tracer("onion_fl")
        self.base_ns = time.time_ns() if base_ns is None else base_ns
        self._sent: dict[str, tuple[Any, int]] = {}

    def _ns(self, t_virtual: float) -> int:
        return self.base_ns + int(round(t_virtual * 1e9))

    def _attributes(self, event: Mapping[str, Any]) -> dict[str, Any]:
        tags = event.get("tags") or {}
        values = {
            "run_id": event.get("run_id"),
            "topology_id": event.get("topology_id"),
            "level": event.get("level"),
            "node": event.get("node"),
            "round": event.get("round"),
            "kind": tags.get("kind"),
            "msg_id": tags.get("msg_id"),
        }
        if event["name"] == "message.sent":
            values |= {"dst": tags.get("dst"), "bytes": event.get("value")}
        else:
            values["src"] = tags.get("src")
        return {k: v for k, v in values.items() if v is not None}

    def write(self, event: Mapping[str, Any]) -> None:
        name, tags = event["name"], event.get("tags") or {}
        msg_id = tags.get("msg_id")
        if msg_id is None:
            return
        if name == "message.sent":
            start = self._ns(event["t_virtual"])
            span = self.tracer.start_span(
                f"send {tags.get('kind')}",
                start_time=start,
                attributes=self._attributes(event),
            )
            span.end(end_time=start)
            self._sent[msg_id] = (span.get_span_context(), start)
        elif (
            name in ("message.delivered", "message.dropped_offline")
            and msg_id in self._sent
        ):
            context, start = self._sent.pop(msg_id)
            attributes = self._attributes(event) | {
                "delivered": name == "message.delivered"
            }
            span = self.tracer.start_span(
                f"receive {tags.get('kind')}",
                start_time=start,
                links=[Link(context)],
                attributes=attributes,
            )
            span.end(end_time=self._ns(event["t_virtual"]))
        elif name == "link.dropped":
            self._sent.pop(msg_id, None)

    def close(self) -> None:
        flush = getattr(self.provider, "force_flush", None)
        if flush is not None:
            flush()
