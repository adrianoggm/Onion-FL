from __future__ import annotations

"""Lightweight OpenTelemetry helpers (optional).

If OTEL_EXPORTER_OTLP_ENDPOINT is not set or opentelemetry is not installed,
all helpers degrade to no-ops.

Environment variables (loaded from .env if available):
    OTEL_EXPORTER_OTLP_ENDPOINT: Base endpoint for traces (e.g., http://localhost:4318)
    OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: Optional separate endpoint for metrics
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from enum import Enum

# Load environment variables from .env file (docker/.env or project root .env)
_env_loaded = False
try:
    from dotenv import load_dotenv

    # Try to find .env in common locations - be more aggressive
    possible_env_paths = [
        Path(__file__).resolve().parents[3] / ".env",  # project/.env (most common)
        Path(__file__).resolve().parents[3] / "docker" / ".env",  # project/docker/.env
        Path.cwd() / ".env",  # cwd/.env
        Path.cwd() / "docker" / ".env",  # cwd/docker/.env
        Path.home() / "flower-basic" / ".env",  # home/flower-basic/.env
    ]

    for env_path in possible_env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            _env_loaded = True
            # Debug: print which .env was loaded
            if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
                break
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.propagate import inject, extract
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    OTEL_AVAILABLE = True
    _propagator = TraceContextTextMapPropagator()
except Exception:  # pragma: no cover
    OTEL_AVAILABLE = False
    _propagator = None
    # Define SpanKind enum for when OTEL is not available
    class SpanKind(Enum):
        INTERNAL = 0
        SERVER = 1
        CLIENT = 2
        PRODUCER = 3
        CONSUMER = 4


# Cache for tracers per service name
_tracers: dict[str, Any] = {}
_meters: dict[str, Any] = {}
_meter_providers: dict[str, Any] = {}  # Keep references for shutdown
_tracer_provider: Any = None
_provider_initialized = False


def init_otel(service_name: str, endpoint: str | None = None):
    """Initialize OTEL tracer/meter for a specific service.

    Each service gets its own tracer with its service name as an attribute.
    The TracerProvider is shared but each tracer is named differently.

    Environment variables:
        OTEL_EXPORTER_OTLP_ENDPOINT: Base endpoint for traces (e.g., http://localhost:4318)
        OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: Optional separate endpoint for metrics
    """
    global _provider_initialized

    if not OTEL_AVAILABLE:
        print("[OTEL] OpenTelemetry not available (packages not installed)")
        return None, None

    # Return cached tracer if already initialized for this service
    if service_name in _tracers:
        return _tracers[service_name], _meters.get(service_name)

    endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    metrics_endpoint_base = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")

    if not endpoint:
        print("[OTEL] No endpoint configured, telemetry disabled")
        return None, None

    print(f"[OTEL] Initializing telemetry for {service_name} -> {endpoint}")

    endpoint = endpoint.rstrip("/")
    traces_endpoint = f"{endpoint}/v1/traces"

    tracer = None
    meter = None

    # Initialize the global provider only once per process
    # The first service to call init_otel sets the service name for the entire process
    try:
        global _tracer_provider
        if not _provider_initialized:
            # Use the actual service name for this process
            resource = Resource.create({"service.name": service_name})
            _tracer_provider = TracerProvider(resource=resource)
            _tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=traces_endpoint))
            )
            trace.set_tracer_provider(_tracer_provider)
            _provider_initialized = True
            print(
                f"[OTEL] Provider initialized for '{service_name}' -> {traces_endpoint}"
            )

        # Get a tracer for this specific service
        tracer = trace.get_tracer(service_name)
        _tracers[service_name] = tracer
        print(f"[OTEL] Tracer '{service_name}' ready")
    except Exception as e:
        print(f"[OTEL] Failed to initialize tracer: {e}")

    # Initialize meter only if metrics endpoint is configured
    if metrics_endpoint_base:
        metrics_endpoint_base = metrics_endpoint_base.rstrip("/")
        metrics_endpoint = f"{metrics_endpoint_base}/v1/metrics"
        try:
            if service_name not in _meters:
                metric_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=metrics_endpoint),
                    export_interval_millis=5000,
                )
                resource = Resource.create({"service.name": service_name})
                meter_provider = MeterProvider(
                    resource=resource, metric_readers=[metric_reader]
                )
                _meter_providers[service_name] = (
                    meter_provider  # Keep reference for shutdown
                )
                metrics.set_meter_provider(meter_provider)
                meter = metrics.get_meter(service_name)
                _meters[service_name] = meter
                print(f"[OTEL] Meter initialized -> {metrics_endpoint}")
            else:
                meter = _meters[service_name]
        except Exception as e:
            print(f"[OTEL] Failed to initialize meter: {e}")
    else:
        print(
            "[OTEL] Metrics disabled (set OTEL_EXPORTER_OTLP_METRICS_ENDPOINT to enable)"
        )

    return tracer, meter


def shutdown_telemetry():
    """Shutdown telemetry and flush all pending metrics/traces.

    Call this before process exit to ensure all metrics are exported.
    """
    if not OTEL_AVAILABLE:
        return

    print("[OTEL] Shutting down telemetry...")

    # Shutdown all meter providers (flush pending metrics)
    for name, provider in _meter_providers.items():
        try:
            if hasattr(provider, "shutdown"):
                provider.shutdown()
                print(f"[OTEL] Meter provider '{name}' shutdown complete")
        except Exception as e:
            print(f"[OTEL] Failed to shutdown meter provider '{name}': {e}")

    # Shutdown tracer provider (flush pending traces)
    if _tracer_provider is not None:
        try:
            if hasattr(_tracer_provider, "shutdown"):
                _tracer_provider.shutdown()
                print("[OTEL] Tracer provider shutdown complete")
        except Exception as e:
            print(f"[OTEL] Failed to shutdown tracer provider: {e}")


def create_counter(meter, name: str, description: str, unit: str = "1"):
    """Create a counter metric (monotonically increasing value)."""
    if meter is None:
        return None
    try:
        return meter.create_counter(name, description=description, unit=unit)
    except Exception:
        return None


def create_histogram(meter, name: str, description: str, unit: str = "s"):
    """Create a histogram metric (distribution of values)."""
    if meter is None:
        return None
    try:
        return meter.create_histogram(name, description=description, unit=unit)
    except Exception:
        return None


def create_gauge(meter, name: str, description: str, unit: str = "1"):
    """Create an observable gauge metric (current value).

    Note: Observable gauges require a callback function to be registered.
    For simple gauges, use UpDownCounter instead.
    """
    if meter is None:
        return None
    try:
        return meter.create_up_down_counter(name, description=description, unit=unit)
    except Exception:
        return None


def record_metric(metric, value, attributes: dict[str, Any] | None = None):
    """Record a value on a metric (counter, histogram, or gauge)."""
    if metric is None:
        return
    try:
        if attributes:
            metric.add(value, attributes)
        else:
            metric.add(value)
    except AttributeError:
        # For histograms, use record instead of add
        try:
            if attributes:
                metric.record(value, attributes)
            else:
                metric.record(value)
        except Exception:
            pass
    except Exception:
        pass


@contextmanager
def start_span(
    tracer,
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: SpanKind | None = None,
    peer_service: str | None = None,
):
    """Start a span with the tracer, adding service name as attribute.
    
    Args:
        tracer: The OpenTelemetry tracer instance
        name: Name of the span
        attributes: Additional attributes to set on the span
        kind: SpanKind (CLIENT, SERVER, PRODUCER, CONSUMER, INTERNAL)
              - Use CLIENT when calling another service
              - Use SERVER when handling incoming requests
              - Use PRODUCER when sending messages (MQTT)
              - Use CONSUMER when receiving messages
        peer_service: Name of the service being called (required for dependencies)
    
    For Jaeger Service Dependencies to work, you must:
    1. Set kind=SpanKind.CLIENT when making outgoing calls
    2. Set peer_service to the name of the service being called
    """
    if tracer is None:
        yield None
        return
    
    # Use INTERNAL as default kind if none specified (avoid None which causes export error)
    if OTEL_AVAILABLE:
        span_kind = kind if kind is not None else SpanKind.INTERNAL
    else:
        span_kind = None
    
    with tracer.start_as_current_span(name, kind=span_kind) as span:
        # Add service name from tracer's instrumentation scope
        if hasattr(tracer, "_instrumentation_scope") and tracer._instrumentation_scope:
            span.set_attribute("service.name", tracer._instrumentation_scope.name)
        
        # Add peer service for dependency tracking (critical for Jaeger architecture view)
        if peer_service:
            span.set_attribute("peer.service", peer_service)
            span.set_attribute("net.peer.name", peer_service)
        
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span


@contextmanager
def start_client_span(tracer, name: str, target_service: str, attributes: dict[str, Any] | None = None):
    """Start a CLIENT span for outgoing calls to another service.
    
    This is a convenience wrapper around start_span that automatically sets
    the correct SpanKind and peer.service attributes for dependency tracking.
    
    Args:
        tracer: The OpenTelemetry tracer instance
        name: Name of the span (e.g., "send_model_update")
        target_service: Name of the service being called (e.g., "fl-server")
        attributes: Additional attributes
    
    Example:
        with start_client_span(tracer, "send_weights", "fl-server") as span:
            client.send(weights)
    """
    # Check tracer first, not OTEL_AVAILABLE
    if tracer is None:
        yield None
        return
    
    kind = SpanKind.CLIENT if OTEL_AVAILABLE else None
    with start_span(
        tracer,
        name,
        attributes=attributes,
        kind=kind,
        peer_service=target_service
    ) as span:
        yield span


@contextmanager
def start_server_span(tracer, name: str, attributes: dict[str, Any] | None = None):
    """Start a SERVER span for incoming requests.
    
    Args:
        tracer: The OpenTelemetry tracer instance
        name: Name of the span (e.g., "handle_client_update")
        attributes: Additional attributes
    
    Example:
        with start_server_span(tracer, "aggregate_round") as span:
            aggregate_results()
    """
    # Check tracer first, not OTEL_AVAILABLE
    if tracer is None:
        yield None
        return
    
    kind = SpanKind.SERVER if OTEL_AVAILABLE else None
    with start_span(
        tracer,
        name,
        attributes=attributes,
        kind=kind,
    ) as span:
        yield span


@contextmanager
def start_producer_span(tracer, name: str, target_service: str, attributes: dict[str, Any] | None = None):
    """Start a PRODUCER span for sending messages (e.g., MQTT publish).
    
    Args:
        tracer: The OpenTelemetry tracer instance
        name: Name of the span (e.g., "mqtt_publish")
        target_service: Name of the service receiving the message
        attributes: Additional attributes
    
    Example:
        with start_producer_span(tracer, "publish_weights", "mqtt-broker") as span:
            mqtt_client.publish(topic, payload)
    """
    # Check tracer first, not OTEL_AVAILABLE
    if tracer is None:
        yield None
        return
    
    kind = SpanKind.PRODUCER if OTEL_AVAILABLE else None
    with start_span(
        tracer,
        name,
        attributes=attributes,
        kind=kind,
        peer_service=target_service
    ) as span:
        yield span


@contextmanager  
def start_consumer_span(tracer, name: str, source_service: str | None = None, attributes: dict[str, Any] | None = None):
    """Start a CONSUMER span for receiving messages (e.g., MQTT subscribe).
    
    Args:
        tracer: The OpenTelemetry tracer instance
        name: Name of the span (e.g., "mqtt_receive")
        source_service: Name of the service that sent the message (optional)
        attributes: Additional attributes
    
    Example:
        with start_consumer_span(tracer, "receive_weights", "fl-client") as span:
            process_message(msg)
    """
    # Check tracer first, not OTEL_AVAILABLE
    if tracer is None:
        yield None
        return
    
    attrs = attributes.copy() if attributes else {}
    if source_service:
        attrs["messaging.source.name"] = source_service
    
    kind = SpanKind.CONSUMER if OTEL_AVAILABLE else None
    with start_span(
        tracer,
        name,
        attributes=attrs,
        kind=kind,
    ) as span:
        yield span


# =============================================================================
# Context Propagation for Distributed Tracing via MQTT
# =============================================================================
# These functions enable trace context to be passed between services via MQTT
# messages, allowing Jaeger to show service dependencies.


def inject_trace_context(carrier: dict[str, Any] | None = None) -> dict[str, str]:
    """Inject current trace context into a carrier dict for propagation.
    
    Call this when SENDING a message to another service. The returned dict
    contains W3C Trace Context headers (traceparent, tracestate) that should
    be included in the MQTT message payload.
    
    Args:
        carrier: Optional existing dict to inject into. If None, creates new dict.
    
    Returns:
        Dict with trace context headers (traceparent, tracestate)
    
    Example:
        with start_producer_span(tracer, "publish_update", "fog-broker") as span:
            trace_ctx = inject_trace_context()
            payload = {"weights": weights, "trace_context": trace_ctx}
            mqtt_client.publish(topic, json.dumps(payload))
    """
    if not OTEL_AVAILABLE or _propagator is None:
        return {}
    
    carrier = carrier if carrier is not None else {}
    try:
        inject(carrier)
    except Exception:
        pass
    return carrier


def extract_trace_context(carrier: dict[str, Any]) -> Any:
    """Extract trace context from a carrier dict received from another service.
    
    Call this when RECEIVING a message from another service. Returns a context
    object that can be used with start_span_with_context().
    
    Args:
        carrier: Dict containing trace context headers (from inject_trace_context)
    
    Returns:
        OpenTelemetry context object (or None if extraction fails)
    
    Example:
        payload = json.loads(msg.payload)
        trace_ctx = payload.get("trace_context", {})
        parent_ctx = extract_trace_context(trace_ctx)
        with start_span_with_context(tracer, "receive_update", parent_ctx) as span:
            process_update(payload)
    """
    if not OTEL_AVAILABLE or _propagator is None or not carrier:
        return None
    
    try:
        return extract(carrier)
    except Exception:
        return None


@contextmanager
def start_span_with_context(
    tracer,
    name: str,
    parent_context: Any,
    attributes: dict[str, Any] | None = None,
    kind: SpanKind | None = None,
    peer_service: str | None = None,
):
    """Start a span with an extracted parent context for distributed tracing.
    
    Use this when processing a message received from another service to create
    a child span that continues the trace from the sending service.
    
    Args:
        tracer: The OpenTelemetry tracer instance
        name: Name of the span
        parent_context: Context extracted via extract_trace_context()
        attributes: Additional attributes to set on the span
        kind: SpanKind (typically SERVER or CONSUMER for incoming messages)
        peer_service: Name of the calling service (for dependency tracking)
    
    Example:
        trace_ctx = payload.get("trace_context", {})
        parent_ctx = extract_trace_context(trace_ctx)
        with start_span_with_context(
            tracer, "process_update", parent_ctx, 
            kind=SpanKind.SERVER, peer_service="swell-client"
        ) as span:
            process_update(payload)
    """
    if tracer is None:
        yield None
        return
    
    # Use INTERNAL as default kind if none specified
    if OTEL_AVAILABLE:
        span_kind = kind if kind is not None else SpanKind.INTERNAL
    else:
        span_kind = None
    
    # Start span with parent context if available
    ctx_manager = tracer.start_as_current_span(
        name,
        context=parent_context,
        kind=span_kind
    ) if parent_context and OTEL_AVAILABLE else tracer.start_as_current_span(
        name,
        kind=span_kind
    )
    
    with ctx_manager as span:
        # Add service name from tracer's instrumentation scope
        if hasattr(tracer, "_instrumentation_scope") and tracer._instrumentation_scope:
            span.set_attribute("service.name", tracer._instrumentation_scope.name)
        
        # Add peer service for dependency tracking
        if peer_service:
            span.set_attribute("peer.service", peer_service)
            span.set_attribute("net.peer.name", peer_service)
        
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span


@contextmanager
def start_linked_consumer_span(
    tracer,
    name: str,
    trace_context: dict[str, Any] | None,
    source_service: str | None = None,
    attributes: dict[str, Any] | None = None,
):
    """Start a CONSUMER span linked to a parent trace from another service.
    
    Convenience wrapper that extracts context and creates a properly linked span.
    
    Args:
        tracer: The OpenTelemetry tracer instance
        name: Name of the span (e.g., "receive_update")
        trace_context: Dict with trace context from the message payload
        source_service: Name of the service that sent the message
        attributes: Additional attributes
    
    Example:
        payload = json.loads(msg.payload)
        with start_linked_consumer_span(
            tracer, "receive_update", 
            payload.get("trace_context"),
            source_service="swell-client"
        ) as span:
            process_update(payload)
    """
    if tracer is None:
        yield None
        return
    
    parent_ctx = extract_trace_context(trace_context) if trace_context else None
    
    attrs = attributes.copy() if attributes else {}
    if source_service:
        attrs["messaging.source.name"] = source_service
        attrs["peer.service"] = source_service
    
    kind = SpanKind.CONSUMER if OTEL_AVAILABLE else None
    
    with start_span_with_context(
        tracer,
        name,
        parent_ctx,
        attributes=attrs,
        kind=kind,
        peer_service=source_service
    ) as span:
        yield span


@contextmanager
def start_linked_producer_span(
    tracer,
    name: str,
    target_service: str,
    attributes: dict[str, Any] | None = None,
):
    """Start a PRODUCER span and return trace context for propagation.
    
    Convenience wrapper that creates a producer span and provides the trace
    context to inject into the outgoing message.
    
    Args:
        tracer: The OpenTelemetry tracer instance
        name: Name of the span (e.g., "publish_update")
        target_service: Name of the service receiving the message
        attributes: Additional attributes
    
    Yields:
        Tuple of (span, trace_context_dict)
    
    Example:
        with start_linked_producer_span(tracer, "publish_update", "fog-broker") as (span, trace_ctx):
            payload = {"weights": weights, "trace_context": trace_ctx}
            mqtt_client.publish(topic, json.dumps(payload))
    """
    if tracer is None:
        yield None, {}
        return
    
    kind = SpanKind.PRODUCER if OTEL_AVAILABLE else None
    
    with start_span(
        tracer,
        name,
        attributes=attributes,
        kind=kind,
        peer_service=target_service
    ) as span:
        trace_ctx = inject_trace_context()
        yield span, trace_ctx


@contextmanager
def start_linked_client_span(
    tracer,
    name: str,
    target_service: str,
    trace_context: dict[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
):
    """Start a CLIENT span linked to a parent trace for outgoing calls.
    
    Use this when making an outgoing call (e.g., gRPC, HTTP) and you want to
    continue a trace received from another service.
    
    Args:
        tracer: The OpenTelemetry tracer instance
        name: Name of the span (e.g., "forward_to_server")
        target_service: Name of the service being called
        trace_context: Dict with trace context from a previous message (optional)
        attributes: Additional attributes
    
    Example:
        # Forward request to server, continuing trace from received message
        with start_linked_client_span(
            tracer, "forward_to_server", "server-swell",
            trace_context=stored_trace_context
        ) as span:
            grpc_call(server)
    """
    if tracer is None:
        yield None
        return
    
    parent_ctx = extract_trace_context(trace_context) if trace_context else None
    
    kind = SpanKind.CLIENT if OTEL_AVAILABLE else None
    
    with start_span_with_context(
        tracer,
        name,
        parent_ctx,
        attributes=attributes,
        kind=kind,
        peer_service=target_service
    ) as span:
        yield span
