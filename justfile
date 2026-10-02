# Onion-FL recipes. Install just: `cargo install just`, `brew install just` or `winget install Casey.Just`.
# Usage: `just` lists them; `just <recipe> [args]`.

set shell := ["bash", "-c"]
set dotenv-load := true

python := if os_family() == "windows" { ".venv/Scripts/python" } else { ".venv/bin/python" }
onion := python + " -m onion_fl"

default:
    @just --list

# --- setup and quality ---------------------------------------------------------------

# Create .venv (Python 3.11, CPU torch) and install the package with dev extras
install-dev:
    uv venv .venv --python 3.11
    uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python .venv -e ".[dev]"

# Lint, format check and tests: what a PR needs
check:
    {{python}} -m ruff check .
    {{python}} -m ruff format --check .
    {{python}} -m pytest -q

test *args:
    {{python}} -m pytest -q {{args}}

test-cov:
    {{python}} -m pytest -q --cov=onion_fl --cov-report=term-missing

format:
    {{python}} -m ruff format .
    {{python}} -m ruff check --fix .

# --- experiments ---------------------------------------------------------------------

# Dry run: composition per fog and parameter groups per link, no training
plan experiment="experiments/mix_ab.yaml":
    {{onion}} plan {{experiment}}

# Run an experiment in simulation (parallel scenarios with workers > 1)
run experiment="experiments/mix_ab.yaml" workers="1":
    {{onion}} run {{experiment}} --workers {{workers}}

# Run an experiment for real: one process per fog over MQTT (needs `just docker-up`)
run-real experiment="experiments/swell_reference.yaml":
    {{onion}} run {{experiment}} --mode real

# HTML report comparing the runs of an experiment
report experiment="experiments/mix_ab.yaml" out="report.html":
    {{onion}} report {{experiment}} --out {{out}}

# Classical baselines on the subject roles of an experiment
baseline experiment="experiments/mix_ab.yaml":
    {{onion}} baseline {{experiment}}

# Prepare the cache of a dataset and show its card
data dataset="swell":
    {{onion}} data prepare {{dataset}}
    {{onion}} data inspect {{dataset}}

topology file="topologies/four_fogs.yaml":
    {{onion}} topology show {{file}}

# --- infrastructure ------------------------------------------------------------------

# MQTT broker, OTEL collector, Jaeger, Prometheus and Grafana (see docker/README.md)
docker-up:
    cd docker && docker compose -f docker-compose.otel.yml up -d
    @echo "Grafana http://localhost:3000 · Jaeger http://localhost:16686 · Prometheus http://localhost:9090 · MQTT localhost:1883"

docker-down:
    cd docker && docker compose -f docker-compose.otel.yml down

# Also removes the volumes
docker-clean:
    cd docker && docker compose -f docker-compose.otel.yml down -v

docker-logs service="mosquitto":
    cd docker && docker compose -f docker-compose.otel.yml logs -f {{service}}

# Remove caches and build leftovers (not data/ or runs/)
clean:
    find . -type d -name "__pycache__" -not -path "./.venv/*" -prune -exec rm -rf {} +
    rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
