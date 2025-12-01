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
from typing import Any, Dict, Optional, Tuple

# Load environment variables from .env file (docker/.env or project root .env)
try:
    from dotenv import load_dotenv
    
    # Try to find .env in common locations
    possible_env_paths = [
        Path(__file__).resolve().parents[3] / "docker" / ".env",  # project/docker/.env
        Path(__file__).resolve().parents[3] / ".env",              # project/.env
        Path.cwd() / "docker" / ".env",                            # cwd/docker/.env
        Path.cwd() / ".env",                                       # cwd/.env
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
_tracers: Dict[str, Any] = {}
_meters: Dict[str, Any] = {}
_provider_initialized = False


def init_otel(service_name: str, endpoint: Optional[str] = None):
    """Initialize OTEL tracer/meter for a specific service.
    
    Each service gets its own tracer with its service name as an attribute.
    The TracerProvider is shared but each tracer is named differently.
    
    Environment variables:
        OTEL_EXPORTER_OTLP_ENDPOINT: Base endpoint for traces (e.g., http://localhost:4318)
        OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: Optional separate endpoint for metrics
    """
    global _provider_initialized
    
    if not OTEL_AVAILABLE:
        print(f"[OTEL] OpenTelemetry not available (packages not installed)")
        return None, None
    
    # Return cached tracer if already initialized for this service
    if service_name in _tracers:
        return _tracers[service_name], _meters.get(service_name)
    
    endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    metrics_endpoint_base = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    
    if not endpoint:
        print(f"[OTEL] No endpoint configured, telemetry disabled")
        return None, None

    print(f"[OTEL] Initializing telemetry for {service_name} -> {endpoint}")
    
    endpoint = endpoint.rstrip("/")
    traces_endpoint = f"{endpoint}/v1/traces"

    tracer = None
    meter = None

    # Initialize the global provider only once per process
    # The first service to call init_otel sets the service name for the entire process
    try:
        if not _provider_initialized:
            # Use the actual service name for this process
            resource = Resource.create({"service.name": service_name})
            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=traces_endpoint))
            )
            trace.set_tracer_provider(tracer_provider)
            _provider_initialized = True
            print(f"[OTEL] Provider initialized for '{service_name}' -> {traces_endpoint}")
        
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
                meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                metrics.set_meter_provider(meter_provider)
                meter = metrics.get_meter(service_name)
                _meters[service_name] = meter
                print(f"[OTEL] Meter initialized -> {metrics_endpoint}")
            else:
                meter = _meters[service_name]
        except Exception as e:
            print(f"[OTEL] Failed to initialize meter: {e}")
    else:
        print(f"[OTEL] Metrics disabled (set OTEL_EXPORTER_OTLP_METRICS_ENDPOINT to enable)")
        
    return tracer, meter


def create_counter(meter, name: str, description: str):
    if meter is None:
        return None
    try:
        return meter.create_counter(name, description=description)
    except Exception:
        return None


@contextmanager
def start_span(tracer, name: str, attributes: Optional[Dict[str, Any]] = None):
    """Start a span with the tracer, adding service name as attribute."""
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        # Add service name from tracer's instrumentation scope
        if hasattr(tracer, '_instrumentation_scope') and tracer._instrumentation_scope:
            span.set_attribute("service.name", tracer._instrumentation_scope.name)
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span
