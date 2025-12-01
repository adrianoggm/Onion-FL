from __future__ import annotations

"""Lightweight OpenTelemetry helpers (optional).

If OTEL_EXPORTER_OTLP_ENDPOINT is not set or opentelemetry is not installed,
all helpers degrade to no-ops.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

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


def init_otel(service_name: str, endpoint: Optional[str] = None):
    """Initialize OTEL tracer/meter if exporter endpoint is provided.
    
    Environment variables:
        OTEL_EXPORTER_OTLP_ENDPOINT: Base endpoint for traces (e.g., http://localhost:4318)
        OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: Optional separate endpoint for metrics
            If not set, metrics export is disabled to avoid 404 errors with Jaeger-only setups.
    """
    if not OTEL_AVAILABLE:
        print(f"[OTEL] OpenTelemetry not available (packages not installed)")
        return None, None
    
    endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    metrics_endpoint_base = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    
    if not endpoint:
        print(f"[OTEL] No endpoint configured, telemetry disabled")
        return None, None

    print(f"[OTEL] Initializing telemetry for {service_name} -> {endpoint}")
    
    # Ensure endpoint doesn't have trailing slash
    endpoint = endpoint.rstrip("/")
    traces_endpoint = f"{endpoint}/v1/traces"

    resource = Resource.create({"service.name": service_name})
    tracer = None
    meter = None

    # Initialize tracer
    try:
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=traces_endpoint))
        )
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(service_name)
        print(f"[OTEL] Tracer initialized -> {traces_endpoint}")
    except Exception as e:
        print(f"[OTEL] Failed to initialize tracer: {e}")

    # Initialize meter only if metrics endpoint is configured
    if metrics_endpoint_base:
        metrics_endpoint_base = metrics_endpoint_base.rstrip("/")
        metrics_endpoint = f"{metrics_endpoint_base}/v1/metrics"
        try:
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=metrics_endpoint),
                export_interval_millis=5000,
            )
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(meter_provider)
            meter = metrics.get_meter(service_name)
            print(f"[OTEL] Meter initialized -> {metrics_endpoint}")
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
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span
