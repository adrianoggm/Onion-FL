"""Prometheus metrics for Flower FL system.

This module provides Prometheus metrics for monitoring the federated learning
system in Grafana. Metrics are exposed via HTTP endpoint that Prometheus scrapes.

Usage:
    from flower_basic.prometheus_metrics import (
        start_metrics_server,
        FL_ROUNDS, FL_ACCURACY, FL_LOSS, ...
    )
    
    # Start HTTP server to expose metrics (call once at startup)
    start_metrics_server(port=8000)
    
    # Update metrics during FL execution
    FL_ROUNDS.inc()
    FL_ACCURACY.set(0.85)
"""

from __future__ import annotations

import os
import threading
from typing import Optional

# Try to import prometheus_client, gracefully degrade if not available
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
        REGISTRY,
        CollectorRegistry,
        push_to_gateway,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class _DummyMetric:
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
        def set(self, value): pass
        def observe(self, value): pass
        def labels(self, **kwargs): return self
        def add(self, value): pass
    
    def Counter(*args, **kwargs): return _DummyMetric()
    def Gauge(*args, **kwargs): return _DummyMetric()
    def Histogram(*args, **kwargs): return _DummyMetric()
    def start_http_server(*args, **kwargs): pass
    def push_to_gateway(*args, **kwargs): pass
    REGISTRY = None
    CollectorRegistry = None


# Pushgateway configuration
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "localhost:9091")


# =============================================================================
# Server Metrics
# =============================================================================

# Total number of FL rounds completed
FL_ROUNDS = Counter(
    'flower_fl_rounds_total',
    'Total number of federated learning rounds completed',
    ['server']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Current global model accuracy (0-1)
FL_ACCURACY = Gauge(
    'flower_fl_global_accuracy', 
    'Current global model accuracy',
    ['server']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Current global model loss
FL_LOSS = Gauge(
    'flower_fl_global_loss',
    'Current global model loss',
    ['server']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Number of active clients connected
FL_ACTIVE_CLIENTS = Gauge(
    'flower_fl_active_clients',
    'Number of active clients (fog bridges) connected to server',
    ['server']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Total aggregations performed by server
FL_AGGREGATIONS = Counter(
    'flower_fl_aggregations_total',
    'Total number of model aggregations performed',
    ['server']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Server round duration
FL_ROUND_DURATION = Histogram(
    'flower_fl_round_duration_seconds',
    'Duration of each FL round in seconds',
    ['server'],
    buckets=[1, 2, 5, 10, 20, 30, 60, 120, 300, 600]
) if PROMETHEUS_AVAILABLE else _DummyMetric()


# =============================================================================
# Broker Metrics
# =============================================================================

# Clients per fog region
BROKER_CLIENTS_PER_REGION = Gauge(
    'flower_fl_broker_clients_per_region',
    'Number of clients connected per fog region',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Aggregations per region
BROKER_AGGREGATIONS = Counter(
    'flower_fl_broker_aggregations_total',
    'Total aggregations performed per fog region',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Buffer size per region (updates waiting to be aggregated)
BROKER_BUFFER_SIZE = Gauge(
    'flower_fl_broker_buffer_size',
    'Current buffer size (pending updates) per fog region',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Updates received per region
BROKER_UPDATES_RECEIVED = Counter(
    'flower_fl_broker_updates_received_total',
    'Total updates received from clients per region',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Partials published per region
BROKER_PARTIALS_PUBLISHED = Counter(
    'flower_fl_broker_partials_published_total',
    'Total partial aggregates published per region',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Client contribution tracking
BROKER_CLIENT_CONTRIBUTION = Gauge(
    'flower_fl_broker_client_contribution',
    'Number of samples contributed by each client',
    ['client_id', 'region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()


# =============================================================================
# Client Metrics  
# =============================================================================

# Training samples per client/region
CLIENT_TRAIN_SAMPLES = Gauge(
    'flower_fl_client_train_samples',
    'Number of training samples per client',
    ['client_id', 'region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Validation samples per client/region
CLIENT_VAL_SAMPLES = Gauge(
    'flower_fl_client_val_samples',
    'Number of validation samples per client',
    ['client_id', 'region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Test samples per client/region
CLIENT_TEST_SAMPLES = Gauge(
    'flower_fl_client_test_samples',
    'Number of test samples per client',
    ['client_id', 'region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Training rounds completed per client
CLIENT_TRAINING_ROUNDS = Counter(
    'flower_fl_client_training_rounds_total',
    'Total training rounds completed per client',
    ['client_id', 'region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Training duration histogram
CLIENT_TRAINING_DURATION = Histogram(
    'flower_fl_client_training_duration_seconds',
    'Duration of client training per round in seconds',
    ['client_id', 'region'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120]
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Client local loss
CLIENT_LOCAL_LOSS = Gauge(
    'flower_fl_client_local_loss',
    'Local training loss per client',
    ['client_id', 'region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Client local accuracy
CLIENT_LOCAL_ACCURACY = Gauge(
    'flower_fl_client_local_accuracy',
    'Local validation accuracy per client',
    ['client_id', 'region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()


# =============================================================================
# Fog Bridge Metrics
# =============================================================================

# Fog bridge aggregation count
FOG_BRIDGE_AGGREGATIONS = Counter(
    'flower_fl_fog_bridge_aggregations_total',
    'Total aggregations performed by fog bridge',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Fog bridge clients received
FOG_BRIDGE_CLIENTS_RECEIVED = Counter(
    'flower_fl_fog_bridge_clients_received_total',
    'Total client updates received by fog bridge',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()


# =============================================================================
# Fog Region Aggregated Metrics
# =============================================================================

# Average accuracy per fog region (aggregated from clients)
FOG_REGION_ACCURACY = Gauge(
    'flower_fl_fog_region_accuracy',
    'Average validation accuracy per fog region (weighted by samples)',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Average loss per fog region
FOG_REGION_LOSS = Gauge(
    'flower_fl_fog_region_loss',
    'Average training loss per fog region (weighted by samples)',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Total samples processed per fog region
FOG_REGION_SAMPLES = Gauge(
    'flower_fl_fog_region_samples',
    'Total samples processed per fog region',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# L2 norm of aggregated model weights per fog region (centroid magnitude)
FOG_REGION_MODEL_NORM = Gauge(
    'flower_fl_fog_region_model_norm',
    'L2 norm of the aggregated model weights per fog region (centroid magnitude)',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Model weight mean per fog region
FOG_REGION_MODEL_MEAN = Gauge(
    'flower_fl_fog_region_model_mean',
    'Mean of all aggregated model weights per fog region',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()

# Model weight std per fog region  
FOG_REGION_MODEL_STD = Gauge(
    'flower_fl_fog_region_model_std',
    'Standard deviation of aggregated model weights per fog region',
    ['region']
) if PROMETHEUS_AVAILABLE else _DummyMetric()


# =============================================================================
# Server Management
# =============================================================================

_metrics_server_started = False
_metrics_lock = threading.Lock()


def start_metrics_server(port: int = 8000) -> bool:
    """Start the Prometheus metrics HTTP server.
    
    Args:
        port: Port to expose metrics on (default: 8000)
    
    Returns:
        True if server started successfully, False otherwise
    """
    global _metrics_server_started
    
    if not PROMETHEUS_AVAILABLE:
        print("[METRICS] prometheus_client not installed, metrics disabled")
        return False
    
    with _metrics_lock:
        if _metrics_server_started:
            print(f"[METRICS] Server already running")
            return True
        
        try:
            start_http_server(port)
            _metrics_server_started = True
            print(f"[METRICS] Prometheus metrics server started on port {port}")
            return True
        except Exception as e:
            print(f"[METRICS] Failed to start metrics server: {e}")
            return False


def get_metrics_port_from_env(default: int = 8000, component: str = "default") -> int:
    """Get metrics port from environment variable.
    
    Environment variables checked (in order):
    - METRICS_PORT_{COMPONENT} (e.g., METRICS_PORT_SERVER)
    - METRICS_PORT
    - Default value
    
    Args:
        default: Default port if not configured
        component: Component name (SERVER, BROKER, CLIENT, FOG_BRIDGE)
    
    Returns:
        Port number to use
    """
    # Try component-specific port first
    component_var = f"METRICS_PORT_{component.upper()}"
    port_str = os.getenv(component_var)
    if port_str:
        try:
            return int(port_str)
        except ValueError:
            pass
    
    # Fall back to generic METRICS_PORT
    port_str = os.getenv("METRICS_PORT")
    if port_str:
        try:
            return int(port_str)
        except ValueError:
            pass
    
    return default


# =============================================================================
# Convenience Functions
# =============================================================================

def record_server_round(server_id: str = "swell", accuracy: float = 0.0, loss: float = 0.0):
    """Record completion of a server round with metrics.
    
    Args:
        server_id: Identifier for the server
        accuracy: Global accuracy after this round
        loss: Global loss after this round
    """
    FL_ROUNDS.labels(server=server_id).inc()
    FL_ACCURACY.labels(server=server_id).set(accuracy)
    FL_LOSS.labels(server=server_id).set(loss)


def record_aggregation(server_id: str = "swell", num_clients: int = 0):
    """Record a model aggregation event.
    
    Args:
        server_id: Identifier for the server
        num_clients: Number of clients that participated
    """
    FL_AGGREGATIONS.labels(server=server_id).inc()
    FL_ACTIVE_CLIENTS.labels(server=server_id).set(num_clients)


def record_broker_update(region: str, client_id: str, num_samples: int):
    """Record a client update received by the broker.
    
    Args:
        region: Fog region ID
        client_id: Client identifier
        num_samples: Number of samples in the update
    """
    BROKER_UPDATES_RECEIVED.labels(region=region).inc()
    BROKER_CLIENT_CONTRIBUTION.labels(client_id=client_id, region=region).set(num_samples)


def record_broker_aggregation(region: str, buffer_size: int = 0):
    """Record a fog-level aggregation by the broker.
    
    Args:
        region: Fog region ID
        buffer_size: Current buffer size after aggregation
    """
    BROKER_AGGREGATIONS.labels(region=region).inc()
    BROKER_PARTIALS_PUBLISHED.labels(region=region).inc()
    BROKER_BUFFER_SIZE.labels(region=region).set(buffer_size)


def record_client_data(
    client_id: str, 
    region: str, 
    train_samples: int = 0,
    val_samples: int = 0,
    test_samples: int = 0
):
    """Record client dataset sizes.
    
    Args:
        client_id: Client identifier
        region: Fog region ID
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
    """
    CLIENT_TRAIN_SAMPLES.labels(client_id=client_id, region=region).set(train_samples)
    CLIENT_VAL_SAMPLES.labels(client_id=client_id, region=region).set(val_samples)
    CLIENT_TEST_SAMPLES.labels(client_id=client_id, region=region).set(test_samples)


def record_client_training(
    client_id: str,
    region: str,
    duration_seconds: float,
    loss: float = 0.0,
    accuracy: float = 0.0
):
    """Record a client training round.
    
    Args:
        client_id: Client identifier
        region: Fog region ID
        duration_seconds: Training duration in seconds
        loss: Local training loss
        accuracy: Local validation accuracy
    """
    CLIENT_TRAINING_ROUNDS.labels(client_id=client_id, region=region).inc()
    CLIENT_TRAINING_DURATION.labels(client_id=client_id, region=region).observe(duration_seconds)
    CLIENT_LOCAL_LOSS.labels(client_id=client_id, region=region).set(loss)
    CLIENT_LOCAL_ACCURACY.labels(client_id=client_id, region=region).set(accuracy)


def set_broker_clients(region: str, num_clients: int):
    """Set the number of connected clients for a region.
    
    Args:
        region: Fog region ID  
        num_clients: Number of connected clients
    """
    BROKER_CLIENTS_PER_REGION.labels(region=region).set(num_clients)


def push_metrics_to_gateway(job: str, grouping_key: dict = None):
    """Push all current metrics to Pushgateway for persistence.
    
    This should be called before the process exits to ensure metrics
    are persisted even after the process terminates.
    
    Args:
        job: Job name for grouping in Pushgateway (e.g., "flower-server", "flower-client")
        grouping_key: Optional dict for additional grouping (e.g., {"region": "fog_0"})
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    try:
        grouping = grouping_key or {}
        push_to_gateway(PUSHGATEWAY_URL, job=job, registry=REGISTRY, grouping_key=grouping)
        print(f"[METRICS] Pushed metrics to Pushgateway ({PUSHGATEWAY_URL}) for job={job}")
    except Exception as e:
        # Don't fail if Pushgateway is not available
        print(f"[METRICS] Could not push to Pushgateway: {e}")
