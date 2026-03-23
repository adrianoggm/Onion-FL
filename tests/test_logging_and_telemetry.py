from __future__ import annotations

import builtins
import re
from types import SimpleNamespace

from flower_basic import telemetry
from flower_basic.logging_utils import enable_timestamped_print


def _reset_timestamp_print(monkeypatch) -> None:
    monkeypatch.delattr(builtins, "_flower_timestamp_print", raising=False)


def test_enable_timestamped_print_prefix(monkeypatch) -> None:
    _reset_timestamp_print(monkeypatch)
    output: list[str] = []

    def fake_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        output.append(sep.join(str(a) for a in args) + end)

    monkeypatch.setenv("FLOWER_LOG_UTC", "1")
    monkeypatch.setattr(builtins, "print", fake_print, raising=False)

    enable_timestamped_print()
    builtins.print("hello")

    assert len(output) == 1
    line = output[0].rstrip("\n")
    assert re.match(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] hello$", line)


def test_enable_timestamped_print_idempotent(monkeypatch) -> None:
    _reset_timestamp_print(monkeypatch)
    output: list[str] = []

    def fake_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        output.append(sep.join(str(a) for a in args) + end)

    monkeypatch.setattr(builtins, "print", fake_print, raising=False)

    enable_timestamped_print()
    enable_timestamped_print()
    builtins.print("once")

    line = output[0].rstrip("\n")
    assert re.match(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] once$", line)
    assert line.count("[") == 1


class _MetricAdd:
    def __init__(self):
        self.calls = []

    def add(self, value, attributes=None):
        self.calls.append((value, attributes))


class _MetricRecord:
    def __init__(self):
        self.calls = []

    def add(self, value, attributes=None):
        raise AttributeError("no add")

    def record(self, value, attributes=None):
        self.calls.append((value, attributes))


def test_record_metric_add_and_record() -> None:
    metric = _MetricAdd()
    telemetry.record_metric(metric, 3, {"k": "v"})
    assert metric.calls == [(3, {"k": "v"})]

    metric_rec = _MetricRecord()
    telemetry.record_metric(metric_rec, 2, {"a": 1})
    assert metric_rec.calls == [(2, {"a": 1})]


def test_create_metric_helpers() -> None:
    class _Meter:
        def __init__(self):
            self.calls = []

        def create_counter(self, name, description=None, unit=None):
            self.calls.append(("counter", name, description, unit))
            return "counter"

        def create_histogram(self, name, description=None, unit=None):
            self.calls.append(("hist", name, description, unit))
            return "hist"

        def create_up_down_counter(self, name, description=None, unit=None):
            self.calls.append(("gauge", name, description, unit))
            return "gauge"

    meter = _Meter()
    assert telemetry.create_counter(meter, "c", "d") == "counter"
    assert telemetry.create_histogram(meter, "h", "d") == "hist"
    assert telemetry.create_gauge(meter, "g", "d") == "gauge"
    assert [call[0] for call in meter.calls] == ["counter", "hist", "gauge"]


class _DummySpan:
    def __init__(self):
        self.attrs: dict[str, object] = {}

    def set_attribute(self, key, value):
        self.attrs[key] = value


class _DummyTracer:
    def __init__(self):
        self._instrumentation_scope = SimpleNamespace(name="dummy-service")
        self.last_kwargs = None
        self.last_span: _DummySpan | None = None

    def start_as_current_span(self, name, **kwargs):
        self.last_kwargs = kwargs
        span = _DummySpan()
        self.last_span = span

        class _CM:
            def __enter__(self_inner):
                return span

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _CM()


def test_start_span_sets_peer_and_attrs(monkeypatch) -> None:
    tracer = _DummyTracer()
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    with telemetry.start_span(
        tracer, "op", attributes={"a": 1}, peer_service="svc"
    ) as span:
        assert span is tracer.last_span

    assert tracer.last_span is not None
    assert tracer.last_span.attrs["service.name"] == "dummy-service"
    assert tracer.last_span.attrs["peer.service"] == "svc"
    assert tracer.last_span.attrs["net.peer.name"] == "svc"
    assert tracer.last_span.attrs["a"] == 1


def test_start_client_server_producer_consumer(monkeypatch) -> None:
    tracer = _DummyTracer()
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    with telemetry.start_client_span(tracer, "c", "svc") as span:
        assert span is tracer.last_span
    with telemetry.start_server_span(tracer, "s") as span:
        assert span is tracer.last_span
    with telemetry.start_producer_span(tracer, "p", "svc") as span:
        assert span is tracer.last_span
    with telemetry.start_consumer_span(tracer, "n", source_service="svc") as span:
        assert span is tracer.last_span


def test_start_span_with_context_uses_parent(monkeypatch) -> None:
    tracer = _DummyTracer()
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    parent_ctx = object()

    with telemetry.start_span_with_context(
        tracer, "op", parent_ctx, attributes={"x": "y"}, peer_service="svc"
    ) as span:
        assert span is tracer.last_span

    assert tracer.last_kwargs is not None
    assert tracer.last_kwargs.get("context") is parent_ctx
    assert tracer.last_span.attrs["peer.service"] == "svc"
    assert tracer.last_span.attrs["x"] == "y"


def test_inject_extract_no_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    monkeypatch.setattr(telemetry, "_propagator", None)

    assert telemetry.inject_trace_context() == {}
    assert telemetry.extract_trace_context({}) is None


def test_start_linked_spans_with_custom_trace_context(monkeypatch) -> None:
    tracer = _DummyTracer()
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    monkeypatch.setattr(telemetry, "inject_trace_context", lambda *a, **k: {"t": "1"})
    monkeypatch.setattr(telemetry, "extract_trace_context", lambda *_: "ctx")

    with telemetry.start_linked_producer_span(tracer, "op", "target") as (
        _span,
        trace_ctx,
    ):
        assert trace_ctx == {"t": "1"}

    with telemetry.start_linked_consumer_span(
        tracer, "op", {"traceparent": "x"}, source_service="svc"
    ) as span:
        assert span is tracer.last_span


def test_init_otel_returns_none_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    tracer, meter = telemetry.init_otel("svc")
    assert tracer is None
    assert meter is None
