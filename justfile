# Justfile for Flower Basic - Federated Learning Demo
# Install: `cargo install just` or `brew install just`
# Usage: `just` or `just <recipe>`

set shell := ["bash", "-c"]
set dotenv-load := true

# Color output
GREEN := '\033[92m'
BLUE := '\033[94m'
YELLOW := '\033[93m'
NC := '\033[0m'

# Default recipe
default:
    @echo "🌸 Flower Basic - Federated Learning Demo"
    @echo ""
    @just --list

# ============================================================================
# 🚀 SWELL FEDERATED DEMO (RECOMENDADO - ONE COMMAND SETUP)
# ============================================================================

# 🎯 One-command setup: Start everything for SWELL federated demo
@swell-demo:
    echo "{{GREEN}}🚀 Starting SWELL Federated Demo with Observability Stack{{NC}}"
    echo ""
    just docker-up
    sleep 3
    just swell-prepare
    sleep 2
    echo ""
    echo "{{GREEN}}✅ Setup complete! Starting federated system...{{NC}}"
    echo ""
    just swell-launch

# Full SWELL demo with physiology-only (lighter setup)
@swell-demo-light:
    echo "{{GREEN}}🚀 Starting SWELL Federated Demo (physiology-only){{NC}}"
    just docker-up
    sleep 3
    just swell-prepare-physio
    sleep 2
    just swell-launch-physio

# ============================================================================
# 🐳 DOCKER & INFRASTRUCTURE
# ============================================================================

# Start obs stack (MQTT + Jaeger + Prometheus + Grafana)
@docker-up:
    echo "{{BLUE}}📦 Starting Docker observability stack...{{NC}}"
    cd docker && docker-compose -f docker-compose.otel.yml up -d
    echo "{{GREEN}}✅ Docker containers up{{NC}}"
    echo ""
    echo "{{YELLOW}}📊 Observability Dashboard:{{NC}}"
    echo "   • Grafana:     http://localhost:3000 (admin/admin)"
    echo "   • Jaeger:      http://localhost:16686"
    echo "   • Prometheus:  http://localhost:9090"
    echo "   • MQTT:        localhost:1883"

# Stop all Docker containers
@docker-down:
    echo "{{BLUE}}🛑 Stopping Docker containers...{{NC}}"
    cd docker && docker-compose -f docker-compose.otel.yml down
    echo "{{GREEN}}✅ Containers stopped{{NC}}"

# Clean Docker (stop + remove volumes)
@docker-clean:
    echo "{{BLUE}}🧹 Cleaning Docker (including volumes)...{{NC}}"
    cd docker && docker-compose -f docker-compose.otel.yml down -v
    echo "{{GREEN}}✅ Docker cleaned{{NC}}"

# View Docker logs
@docker-logs:
    cd docker && docker-compose -f docker-compose.otel.yml logs -f

# ============================================================================
# 📊 SWELL PREPARATION
# ============================================================================

# Prepare SWELL splits (full modalities: computer + facial + posture + physiology)
@swell-prepare:
    echo "{{BLUE}}📊 Preparing SWELL federated splits (full modalities)...{{NC}}"
    python scripts/prepare_swell_federated.py --config configs/swell_federated.example.yaml
    echo "{{GREEN}}✅ SWELL splits prepared at federated_runs/swell/example_manual/{{NC}}"

# Prepare SWELL splits (physiology only - lighter)
@swell-prepare-physio:
    echo "{{BLUE}}📊 Preparing SWELL federated splits (physiology-only)...{{NC}}"
    python scripts/prepare_swell_federated.py --config configs/swell_federated_10runs.yaml
    echo "{{GREEN}}✅ SWELL splits prepared at federated_runs/swell/10_executions_physiology/{{NC}}"

# ============================================================================
# 🚀 LAUNCH FEDERATED SYSTEM
# ============================================================================

# Launch full federated system (full modalities)
@swell-launch:
    echo "{{GREEN}}🚀 Launching federated system (full modalities)...{{NC}}"
    export MQTT_BROKER=localhost MQTT_PORT=1883
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4320
    export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://localhost:4320
    python scripts/run_architecture_from_config.py \
      --config configs/federated_architecture.example.yaml \
      --manifest federated_runs/swell/example_manual/manifest.json \
      --launch \
      --delay 0.1

# Launch federated system (physiology-only)
@swell-launch-physio:
    echo "{{GREEN}}🚀 Launching federated system (physiology-only)...{{NC}}"
    export MQTT_BROKER=localhost MQTT_PORT=1883
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4320
    export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://localhost:4320
    python scripts/run_architecture_from_config.py \
      --config configs/federated_architecture.example.yaml \
      --manifest federated_runs/swell/10_executions_physiology/manifest.json \
      --launch \
      --delay 0.1

# Launch SWELL with stale updates accepted (recommended baseline for comparison)
@swell-launch-accept:
    echo "{{GREEN}}🚀 Launching SWELL with stale policy=accept{{NC}}"
    export MQTT_BROKER=localhost MQTT_PORT=1883
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4320
    export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://localhost:4320
    python scripts/run_architecture_from_config.py \
      --config configs/federated_architecture_accept.yaml \
      --manifest federated_runs/swell/10_executions_physiology/manifest.json \
      --launch \
      --delay 0.1

# Launch SWELL with stale updates dropped when they do not match the expected round
@swell-launch-strict:
    echo "{{GREEN}}🚀 Launching SWELL with stale policy=strict{{NC}}"
    export MQTT_BROKER=localhost MQTT_PORT=1883
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4320
    export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://localhost:4320
    python scripts/run_architecture_from_config.py \
      --config configs/federated_architecture_strict.yaml \
      --manifest federated_runs/swell/10_executions_physiology/manifest.json \
      --launch \
      --delay 0.1

# One-command demo for the accept policy using the same physiology manifest
@swell-demo-accept:
    echo "{{GREEN}}🚀 Starting SWELL demo with stale policy=accept{{NC}}"
    just docker-up
    sleep 3
    just swell-prepare-physio
    sleep 2
    just swell-launch-accept

# One-command demo for the strict policy using the same physiology manifest
@swell-demo-strict:
    echo "{{GREEN}}🚀 Starting SWELL demo with stale policy=strict{{NC}}"
    just docker-up
    sleep 3
    just swell-prepare-physio
    sleep 2
    just swell-launch-strict

# ============================================================================
# 🧪 TESTING & EVALUATION
# ============================================================================

# Run all tests
@test:
    echo "{{BLUE}}🧪 Running tests...{{NC}}"
    python run_tests.py

# Run tests with coverage
@test-cov:
    echo "{{BLUE}}🧪 Running tests with coverage...{{NC}}"
    python -m pytest --cov=src --cov-report=html --cov-report=term-missing

# Evaluate WESAD baseline
@eval-wesad:
    echo "{{BLUE}}📊 Evaluating WESAD baseline...{{NC}}"
    python scripts/evaluate_wesad_baseline.py

# Evaluate SWELL baseline
@eval-swell:
    echo "{{BLUE}}📊 Evaluating SWELL baseline...{{NC}}"
    python scripts/evaluate_swell_baseline.py

# Evaluate multimodal baseline
@eval-multimodal:
    echo "{{BLUE}}📊 Evaluating multimodal baseline...{{NC}}"
    python scripts/evaluate_multimodal_baseline.py

# Run multi-dataset demo
@demo-multidataset:
    echo "{{BLUE}}🎬 Running multi-dataset demo...{{NC}}"
    python scripts/demo_multidataset_fl.py

# ============================================================================
# 🔧 DEVELOPMENT & QUALITY
# ============================================================================

# Install dev environment
@install-dev:
    echo "{{BLUE}}📦 Installing development environment...{{NC}}"
    python -m venv .venv
    source .venv/bin/activate && pip install -e ".[dev,test]"
    echo "{{GREEN}}✅ Dev environment ready{{NC}}"

# Format code
@format:
    echo "{{BLUE}}🎨 Formatting code...{{NC}}"
    black .
    isort .
    echo "{{GREEN}}✅ Code formatted{{NC}}"

# Lint code
@lint:
    echo "{{BLUE}}🔍 Linting code...{{NC}}"
    ruff check .
    echo "{{GREEN}}✅ Linting complete{{NC}}"

# Type check
@type-check:
    echo "{{BLUE}}📋 Type checking...{{NC}}"
    mypy src/ || true
    echo "{{GREEN}}✅ Type check complete{{NC}}"

# Run all quality checks
@quality: format lint type-check
    echo "{{GREEN}}✅ All quality checks passed{{NC}}"

# ============================================================================
# 🧹 CLEANUP
# ============================================================================

# Clean Python cache
@clean-cache:
    echo "{{BLUE}}🧹 Cleaning Python cache...{{NC}}"
    find . -type d -name __pycache__ -exec rm -rf {} + || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name ".pytest_cache" -delete
    find . -type d -name ".mypy_cache" -exec rm -rf {} + || true
    echo "{{GREEN}}✅ Cache cleaned{{NC}}"

# Clean all (cache + test artifacts)
@clean-all: clean-cache
    echo "{{BLUE}}🧹 Cleaning all artifacts...{{NC}}"
    rm -rf .coverage htmlcov/ dist/ build/ *.egg-info
    echo "{{GREEN}}✅ All cleaned{{NC}}"

# ============================================================================
# 📖 UTILITY COMMANDS
# ============================================================================

# Show available ports and services
@status:
    echo "{{BLUE}}📊 Service Status:{{NC}}"
    echo ""
    echo "MQTT Broker:    localhost:1883"
    echo "Grafana:        http://localhost:3000"
    echo "Jaeger:         http://localhost:16686"
    echo "Prometheus:     http://localhost:9090"
    echo "Flower Server:  http://localhost:8080"
    echo ""
    echo "{{YELLOW}}Checking active services:{{NC}}"
    netstat -tuln 2>/dev/null | grep -E "(1883|3000|8080|16686|9090)" || echo "   (No services currently running)"

# Show current setup info
@info:
    echo "{{GREEN}}🌸 Flower Basic - System Information{{NC}}"
    echo ""
    echo "{{BLUE}}Python Environment:{{NC}}"
    python --version
    echo "Location: $(python -c 'import sys; print(sys.executable)')"
    echo ""
    echo "{{BLUE}}Docker:{{NC}}"
    docker --version || echo "   Docker not installed"
    docker-compose --version || echo "   Docker-compose not installed"
    echo ""
    echo "{{BLUE}}Available Data:{{NC}}"
    ls -1 federated_runs/swell/ 2>/dev/null || echo "   No federated runs yet"

# Clear all metrics from Prometheus
@metrics-clear:
    echo "{{BLUE}}🧹 Clearing Prometheus metrics...{{NC}}"
    curl -X DELETE http://localhost:9091/metrics/job/flower-client 2>/dev/null || echo "   (Client metrics not found)"
    curl -X DELETE http://localhost:9091/metrics/job/flower-broker 2>/dev/null || echo "   (Broker metrics not found)"
    curl -X DELETE http://localhost:9091/metrics/job/flower-server 2>/dev/null || echo "   (Server metrics not found)"
    echo "{{GREEN}}✅ Metrics cleared{{NC}}"

# ============================================================================
# 🎯 QUICK WORKFLOWS
# ============================================================================

# Full setup + run (one command for everything)
@full-setup: install-dev quality swell-demo
    echo "{{GREEN}}✅ Full setup complete!{{NC}}"

# Dev setup (install + format + lint)
@dev-setup: install-dev quality
    echo "{{GREEN}}✅ Development environment ready{{NC}}"
    echo ""
    echo "{{YELLOW}}Next steps:{{NC}}"
    echo "   • just swell-demo          - Run full SWELL federated demo"
    echo "   • just test                - Run tests"
    echo "   • just docker-down         - Stop services"
