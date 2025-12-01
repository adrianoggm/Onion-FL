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

# Load environment variables from .env file (docker/.env or project root .env)
try:
    from dotenv import load_dotenv

    # Try to find .env in common locations
    possible_env_paths = [
        Path(__file__).resolve().parents[3] / "docker" / ".env",  # project/docker/.env
        Path(__file__).resolve().parents[3] / ".env",  # project/.env
        Path.cwd() / "docker" / ".env",  # cwd/docker/.env
        Path.cwd() / ".env",  # cwd/.env
    ]

    for env_path in possible_env_paths:
        if env_path.exists():
            load_dotenv(env_path)
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

    OTEL_AVAILABLE = True
except Exception:  # pragma: no cover
    OTEL_AVAILABLE = False


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
    global _tracer_provider
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
def start_span(tracer, name: str, attributes: dict[str, Any] | None = None):
    """Start a span with the tracer, adding service name as attribute."""
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        # Add service name from tracer's instrumentation scope
        if hasattr(tracer, "_instrumentation_scope") and tracer._instrumentation_scope:
            span.set_attribute("service.name", tracer._instrumentation_scope.name)
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span
