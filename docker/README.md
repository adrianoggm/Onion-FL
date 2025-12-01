# Docker Compose for Flower-Basic Federated Learning# Docker Compose for Flower-Basic Federated Learning# Docker Compose for Flower-Basic Federated Learning# Docker Compose for Fog-Cloud Demo



This folder contains Docker Compose configurations for running the Flower-Basic federated learning system with full observability support.



## 📁 Folder StructureThis folder contains Docker Compose configurations for running the Flower-Basic federated learning system with full observability support.



```

docker/

├── docker-compose.yml           # Main FL system (Server, Clients, Brokers)## 📁 Folder StructureThis folder contains Docker Compose configurations for running the Flower-Basic federated learning system with full observability support.This Compose file starts the full SWELL fog–cloud hierarchy:

├── docker-compose.otel.yml      # OpenTelemetry + MQTT observability stack

├── .env.template                # Environment variables template

├── .env                         # Local environment variables (git-ignored)

├── otel-collector-config.yaml   # OTEL Collector configuration```

├── prometheus.yml               # Prometheus scrape configuration

├── README.md                    # This filedocker/

├── mosquitto/

│   └── mosquitto.conf           # MQTT broker configuration├── docker-compose.yml           # Main FL system (MQTT, Server, Clients, Brokers)## 📁 Folder Structure- Mosquitto MQTT broker

└── grafana/

    └── provisioning/├── docker-compose.otel.yml      # OpenTelemetry + MQTT observability stack

        ├── datasources/         # Auto-configured data sources

        └── dashboards/          # Pre-built FL dashboards├── otel-collector-config.yaml   # OTEL Collector configuration- Central server (`flower_basic.servers.swell`)

```

├── prometheus.yml               # Prometheus scrape configuration

---

├── README.md                    # This file```- Fog bridges (one per fog region)

## ⚙️ Environment Configuration

├── mosquitto/

### Setup

│   └── mosquitto.conf           # MQTT broker configurationdocker/- Fog broker (regional aggregator)

1. Copy the template to create your local `.env` file:

└── grafana/

```bash

cd docker    └── provisioning/├── docker-compose.yml           # Main FL system (MQTT, Server, Clients, Brokers)- SWELL clients (2 per fog, by default)

cp .env.template .env

```        ├── datasources/         # Auto-configured data sources



2. Edit `.env` with your specific values (optional - defaults work for local dev).        └── dashboards/          # Pre-built FL dashboards├── docker-compose.otel.yml      # OpenTelemetry observability stack



> **Note**: The `.env` file is ignored by git and will not be committed.```



### Available Variables├── otel-collector-config.yaml   # OTEL Collector configuration## Prerequisites



| Variable | Default | Description |---

|----------|---------|-------------|

| `MQTT_BROKER` | localhost | MQTT broker hostname |├── prometheus.yml               # Prometheus scrape configuration- Prepare SWELL federated splits and manifest (e.g. `federated_runs/swell/example_manual/manifest.json` with `fog_0`/`fog_1` dirs).

| `MQTT_PORT` | 1883 | MQTT broker port |

| `INPUT_DIM` | 113 | Model input dimension |## 🚀 Quick Start

| `ROUNDS` | 3 | Number of FL rounds |

| `FOG_K` | 2 | Updates per region before aggregate |├── README.md                    # This file- Docker and Docker Compose installed.

| `OTEL_EXPORTER_OTLP_ENDPOINT` | http://localhost:4318 | OTLP endpoint |

| `GF_SECURITY_ADMIN_USER` | admin | Grafana admin username |### Option 1: Full Stack (MQTT + Observability)

| `GF_SECURITY_ADMIN_PASSWORD` | admin | Grafana admin password |

└── grafana/

---

This is the recommended option for development and testing:

## 🚀 Quick Start

    └── provisioning/## Usage

### Option 1: Full Stack (MQTT + Observability)

```bash

This is the recommended option for development and testing:

cd docker        ├── datasources/         # Auto-configured data sourcesFrom repo root:

```bash

cd dockerdocker compose -f docker-compose.otel.yml up -d

docker compose -f docker-compose.otel.yml up -d

``````        └── dashboards/          # Pre-built FL dashboards```bash



This starts:

- **Mosquitto** MQTT broker on port `1883`

- **Jaeger** tracing UI on port `16686`This starts:```cd docker

- **Prometheus** metrics on port `9090`

- **Grafana** dashboards on port `3000`- **Mosquitto** MQTT broker on port `1883`

- **OTEL Collector** for telemetry pipeline

- **Jaeger** tracing UI on port `16686`docker compose up --build

### Option 2: FL System Only (from docker-compose.yml)

- **Prometheus** metrics on port `9090`

```bash

cd docker- **Grafana** dashboards on port `3000`---```x x   x         

docker compose up --build

```- **OTEL Collector** for telemetry pipeline



### Option 3: Run Python Components Locally with Docker Services



```bash### Option 2: FL System Only (from docker-compose.yml)

cd docker

## 🚀 Quick StartEnvironment overrides (examples):

# Start MQTT + Observability stack

docker compose -f docker-compose.otel.yml up -d```bash



# Then run your Python FL components locallycd docker- `INPUT_DIM` (default 113)

$env:OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

python -m flower_basic.servers.swell --mqtt-broker localhost --rounds 5docker compose up --build

```

```### Option 1: FL System Only (No Observability)- `ROUNDS` (default 3)

---



## 🔭 Services & Access URLs

### Option 3: Run Python Components Locally with Docker Services- `MANIFEST_PATH` (default `/app/federated_runs/swell/example_manual/manifest.json`)

The `docker-compose.otel.yml` provides:



| Service | Port | URL | Description |

|---------|------|-----|-------------|```bash```bash- `FOG0_NODE_DIR`, `FOG1_NODE_DIR` for client data dirs

| **Mosquitto** | 1883 | `mqtt://localhost:1883` | MQTT broker for FL |

| **Jaeger** | 16686 | http://localhost:16686 | Distributed tracing UI |cd docker

| **Prometheus** | 9090 | http://localhost:9090 | Metrics queries |

| **Grafana** | 3000 | http://localhost:3000 | Dashboards |cd docker- `K` (broker aggregation, default 2)

| **OTEL Collector** | 4318 | http://localhost:4318 | OTLP HTTP endpoint |

| **OTEL Health** | 13133 | http://localhost:13133 | Collector health check |# Start MQTT + Observability stack



---docker compose -f docker-compose.otel.yml up -ddocker compose up --build



## 🏗️ Architecture



```# Then run your Python FL components locally```Services use `mqtt-broker=mosquitto` (internal network). The repo is bind-mounted into `/app` so local edits are visible.

┌─────────────────────────────────────────────────────────────────┐

│                    Flower-Basic Components                       │$env:OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │

│  │  Server  │  │  Client  │  │  Broker  │  │Fog Bridge│        │python -m flower_basic.servers.swell --mqtt-broker localhost --rounds 5

│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │

│       │             │             │             │               │```

│       ├─────────────┴─────────────┴─────────────┤               │

│       │                                         │               │### Option 2: FL System with Full ObservabilityTo enable OpenTelemetry export, set `OTEL_EXPORTER_OTLP_ENDPOINT` in the `environment` section or when invoking `docker compose`:

│       ▼                                         ▼               │

│  ┌──────────┐                          ┌────────────────┐       │---

│  │Mosquitto │ ◄─── MQTT Messages ────► │ OTEL Collector │       │

│  │  (MQTT)  │                          │  (Telemetry)   │       │```bash

│  └──────────┘                          └───────┬────────┘       │

│                                          ┌─────┴─────┐          │## 🔭 Services & Access URLs

│                                          ▼           ▼          │

│                                    ┌──────────┐ ┌──────────┐    │```bashOTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4318 docker compose up

│                                    │  Jaeger  │ │Prometheus│    │

│                                    │ (Traces) │ │(Metrics) │    │The `docker-compose.otel.yml` provides:

│                                    └──────────┘ └────┬─────┘    │

│                                                      │          │cd docker```

│                                                      ▼          │

│                                                ┌──────────┐     │| Service | Port | URL | Description |

│                                                │ Grafana  │     │

│                                                │(Dashboard)│    │|---------|------|-----|-------------|

│                                                └──────────┘     │

└─────────────────────────────────────────────────────────────────┘| **Mosquitto** | 1883 | `mqtt://localhost:1883` | MQTT broker for FL |

```

| **Jaeger** | 16686 | http://localhost:16686 | Distributed tracing UI |# Start observability stack firstStop with `docker compose down`. Remove containers/images with `docker compose down --rmi local`.

---

| **Prometheus** | 9090 | http://localhost:9090 | Metrics queries |

## 🔧 Using with Python Components

| **Grafana** | 3000 | http://localhost:3000 | Dashboards (admin/admin) |docker compose -f docker-compose.otel.yml up -d

### 1. Start the Docker Stack

| **OTEL Collector** | 4318 | http://localhost:4318 | OTLP HTTP endpoint |

```bash

cd docker| **OTEL Health** | 13133 | http://localhost:13133 | Collector health check |# Then start FL system with telemetry enabled

docker compose -f docker-compose.otel.yml up -d

```OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4318 docker compose up --build



### 2. Configure Environment Variables---```



```powershell

# PowerShell

$env:OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"## 🏗️ Architecture### Option 3: Observability Stack Only (for local development)

$env:MQTT_BROKER="localhost"

$env:MQTT_PORT="1883"

```

``````bash

```bash

# Bash/Linux/Mac┌─────────────────────────────────────────────────────────────────┐cd docker

export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

export MQTT_BROKER=localhost│                    Flower-Basic Components                       │docker compose -f docker-compose.otel.yml up -d

export MQTT_PORT=1883

```│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │



### 3. Run FL Components│  │  Server  │  │  Client  │  │  Broker  │  │Fog Bridge│        │# Then run your Python FL components locally with:



```bash│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │# export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Start server

python -m flower_basic.servers.swell --mqtt-broker localhost --rounds 5│       │             │             │             │               │```



# Start fog broker│       ├─────────────┴─────────────┴─────────────┤               │

python -m flower_basic.brokers.fog --mqtt-broker localhost

│       │                                         │               │---

# Start clients

python -m flower_basic.clients.swell --mqtt-broker localhost --node_dir ./data│       ▼                                         ▼               │

```

│  ┌──────────┐                          ┌────────────────┐       │## 🔭 OpenTelemetry Observability Stack

---

│  │Mosquitto │ ◄─── MQTT Messages ────► │ OTEL Collector │       │

## 🔧 OpenTelemetry in Python Code

│  │  (MQTT)  │                          │  (Telemetry)   │       │The `docker-compose.otel.yml` provides a complete observability solution:

Your code already uses `telemetry.py`. Here's how it works:

│  └──────────┘                          └───────┬────────┘       │

```python

from flower_basic.telemetry import init_otel, create_counter, start_span│                                          ┌─────┴─────┐          │| Service | Port | Description |



# Initialize at service startup│                                          ▼           ▼          │|---------|------|-------------|

TRACER, METER = init_otel("my-service-name")

│                                    ┌──────────┐ ┌──────────┐    │| **Jaeger** | [16686](http://localhost:16686) | Distributed tracing UI |

# Create counters for metrics

rounds_counter = create_counter(METER, "training.rounds", "Training rounds completed")│                                    │  Jaeger  │ │Prometheus│    │| **Prometheus** | [9090](http://localhost:9090) | Metrics storage & queries |



# Trace operations with spans│                                    │ (Traces) │ │(Metrics) │    │| **Grafana** | [3000](http://localhost:3000) | Visualization dashboards |

with start_span(TRACER, "federated_round", {"round": round_num}) as span:

    # Your code here│                                    └──────────┘ └────┬─────┘    │| **OTEL Collector** | 4318 (HTTP), 4317 (gRPC) | Telemetry pipeline |

    aggregated = aggregate_weights(client_updates)

│                                                      │          │

# Increment counter

if rounds_counter:│                                                      ▼          │### Access URLs

    rounds_counter.add(1, {"client_id": client_id})

```│                                                ┌──────────┐     │



---│                                                │ Grafana  │     │- **Jaeger UI**: http://localhost:16686 - View distributed traces



## 🐛 Troubleshooting│                                                │(Dashboard)│    │- **Prometheus**: http://localhost:9090 - Query metrics



### MQTT connection refused│                                                └──────────┘     │- **Grafana**: http://localhost:3000 (login: `admin` / `admin`)



1. Check Mosquitto is running:└─────────────────────────────────────────────────────────────────┘- **OTEL Health**: http://localhost:13133 - Collector health check

   ```bash

   docker compose -f docker-compose.otel.yml ps```

   ```

### Architecture

2. Test MQTT connection:

   ```bash---

   mosquitto_pub -h localhost -p 1883 -t test -m "hello"

   ``````



### No telemetry data appearing## 🔧 Using with Python Components┌─────────────────────────────────────────────────────────────────┐



1. Check OTEL Collector is running:│                    Flower-Basic Components                       │

   ```bash

   curl http://localhost:13133### 1. Start the Docker Stack│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │

   ```

│  │  Server  │  │  Client  │  │  Broker  │  │Fog Bridge│        │

2. Verify environment variable:

   ```powershell```bash│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │

   echo $env:OTEL_EXPORTER_OTLP_ENDPOINT

   ```cd docker│       │             │             │             │               │



3. Check OTEL Collector logs:docker compose -f docker-compose.otel.yml up -d│       └─────────────┴─────────────┴─────────────┘               │

   ```bash

   docker compose -f docker-compose.otel.yml logs otel-collector```│                           │                                      │

   ```

│                    OTLP (HTTP/gRPC)                             │

### Reset Everything

### 2. Configure Environment Variables│                           ▼                                      │

```bash

docker compose -f docker-compose.otel.yml down -v│              ┌────────────────────────┐                         │

docker network prune

docker compose -f docker-compose.otel.yml up -d```powershell│              │   OTEL Collector       │                         │

```

# PowerShell│              │  (Processing & Routing)│                         │

---

$env:OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"│              └───────────┬────────────┘                         │

## 📊 Pre-built Grafana Dashboards

$env:MQTT_BROKER="localhost"│                    ┌─────┴─────┐                                │

Access at: http://localhost:3000 → Dashboards → Flower FL

$env:MQTT_PORT="1883"│                    ▼           ▼                                │

Includes:

- Total aggregations counter```│              ┌──────────┐ ┌──────────┐                          │

- Client training rounds

- Broker message throughput│              │  Jaeger  │ │Prometheus│                          │

- Per-client training activity

```bash│              │ (Traces) │ │(Metrics) │                          │

---

# Bash/Linux/Mac│              └──────────┘ └────┬─────┘                          │

## 🔗 Related Documentation

export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318│                                │                                 │

- [OpenTelemetry Python](https://opentelemetry-python.readthedocs.io/)

- [Jaeger Documentation](https://www.jaegertracing.io/docs/)export MQTT_BROKER=localhost│                                ▼                                 │

- [Prometheus Documentation](https://prometheus.io/docs/)

- [Grafana Documentation](https://grafana.com/docs/)export MQTT_PORT=1883│                          ┌──────────┐                           │

- [Eclipse Mosquitto](https://mosquitto.org/documentation/)

```│                          │ Grafana  │                           │

│                          │(Dashboards)│                         │

### 3. Run FL Components│                          └──────────┘                           │

└─────────────────────────────────────────────────────────────────┘

```bash```

# Start server

python -m flower_basic.servers.swell --mqtt-broker localhost --rounds 5---



# Start fog broker## 🔧 Using OpenTelemetry in Python Code

python -m flower_basic.brokers.fog --mqtt-broker localhost

### 1. Initialize Telemetry

# Start clients

python -m flower_basic.clients.swell --mqtt-broker localhost --node_dir ./data```python

```from flower_basic.telemetry import init_otel, create_counter, start_span



---# Initialize at service startup

TRACER, METER = init_otel("my-service-name")

## 🔧 OpenTelemetry in Python Code

# Create counters for metrics

Your code already uses `telemetry.py`. Here's how it works:rounds_counter = create_counter(METER, "training.rounds", "Training rounds completed")

```

```python

from flower_basic.telemetry import init_otel, create_counter, start_span### 2. Trace Operations with Spans



# Initialize at service startup```python

TRACER, METER = init_otel("my-service-name")# Wrap operations in spans for tracing

with start_span(TRACER, "federated_round", {"round": round_num}) as span:

# Create counters for metrics    # Your code here

rounds_counter = create_counter(METER, "training.rounds", "Training rounds completed")    aggregated = aggregate_weights(client_updates)

    

# Trace operations with spans    # Add events to span

with start_span(TRACER, "federated_round", {"round": round_num}) as span:    if span:

    # Your code here        span.add_event("aggregation_complete", {"num_clients": len(client_updates)})

    aggregated = aggregate_weights(client_updates)```



# Increment counter### 3. Record Metrics

if rounds_counter:

    rounds_counter.add(1, {"client_id": client_id})```python

```# Increment counter

if rounds_counter:

---    rounds_counter.add(1, {"client_id": client_id, "region": region})

```

## 🌫️ Fog-Cloud FL System (docker-compose.yml)

### 4. Environment Variable

The main compose file starts the full SWELL fog–cloud hierarchy:

Set this to enable telemetry export:

### Services

```bash

- **Mosquitto**: MQTT broker for FL communication# For Docker (from host to container network)

- **Server**: Central FL server (`flower_basic.servers.swell`)export OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4318

- **Fog Bridges**: Bridge between Flower and MQTT (one per fog region)

- **Fog Broker**: Regional aggregator# For local development

- **SWELL Clients**: FL clients (2 per fog region by default)export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

```

### Environment Variables

---

| Variable | Default | Description |

|----------|---------|-------------|## 🌫️ Fog-Cloud FL System (docker-compose.yml)

| `INPUT_DIM` | 113 | Model input dimension |

| `ROUNDS` | 3 | Number of FL rounds |The main compose file starts the full SWELL fog–cloud hierarchy:

| `MANIFEST_PATH` | `/app/federated_runs/swell/example_manual/manifest.json` | Path to manifest |

| `FOG0_NODE_DIR` | `/app/federated_runs/swell/example_manual/fog_0` | Fog 0 data directory |### Services

| `FOG1_NODE_DIR` | `/app/federated_runs/swell/example_manual/fog_1` | Fog 1 data directory |

| `K` | 2 | Broker aggregation parameter |- **Mosquitto**: MQTT broker for FL communication

| `OTEL_EXPORTER_OTLP_ENDPOINT` | (none) | OTLP endpoint for telemetry |- **Server**: Central FL server (`flower_basic.servers.swell`)

- **Fog Bridges**: Bridge between Flower and MQTT (one per fog region)

### Usage Examples- **Fog Broker**: Regional aggregator

- **SWELL Clients**: FL clients (2 per fog region by default)

```bash

# Basic run with full FL stack### Prerequisites

docker compose up --build

1. Prepare SWELL federated splits and manifest:

# Custom configuration   ```

ROUNDS=10 INPUT_DIM=128 docker compose up --build   federated_runs/swell/example_manual/manifest.json

   federated_runs/swell/example_manual/fog_0/

# With telemetry (after starting OTEL stack)   federated_runs/swell/example_manual/fog_1/

OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4318 docker compose up --build   ```

```

2. Docker and Docker Compose installed

---

### Environment Variables

## 🐛 Troubleshooting

| Variable | Default | Description |

### MQTT connection refused|----------|---------|-------------|

| `INPUT_DIM` | 113 | Model input dimension |

1. Check Mosquitto is running:| `ROUNDS` | 3 | Number of FL rounds |

   ```bash| `MANIFEST_PATH` | `/app/federated_runs/swell/example_manual/manifest.json` | Path to manifest |

   docker compose -f docker-compose.otel.yml ps| `FOG0_NODE_DIR` | `/app/federated_runs/swell/example_manual/fog_0` | Fog 0 data directory |

   ```| `FOG1_NODE_DIR` | `/app/federated_runs/swell/example_manual/fog_1` | Fog 1 data directory |

| `K` | 2 | Broker aggregation parameter |

2. Test MQTT connection:| `OTEL_EXPORTER_OTLP_ENDPOINT` | (none) | OTLP endpoint for telemetry |

   ```bash

   # Install mosquitto-clients if needed### Usage Examples

   mosquitto_pub -h localhost -p 1883 -t test -m "hello"

   ``````bash

# Basic run

### No telemetry data appearingdocker compose up --build



1. Check OTEL Collector is running:# Custom configuration

   ```bashROUNDS=10 INPUT_DIM=128 docker compose up --build

   curl http://localhost:13133  # Should return OK

   ```# With telemetry (after starting OTEL stack)

OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4318 docker compose up --build

2. Verify environment variable:

   ```powershell# Detached mode

   echo $env:OTEL_EXPORTER_OTLP_ENDPOINTdocker compose up -d --build

   ```

# View logs

3. Check OTEL Collector logs:docker compose logs -f server

   ```bashdocker compose logs -f client_fog0_c1

   docker compose -f docker-compose.otel.yml logs otel-collector

   ```# Stop and cleanup

docker compose down

### Jaeger shows no tracesdocker compose down --rmi local  # Also remove images

```

1. Ensure your Python code initializes telemetry:

   ```python---

   TRACER, METER = init_otel("service-name")

   # TRACER should not be None## 🐛 Troubleshooting

   ```

### No telemetry data appearing

2. Check Jaeger service dropdown - select your service name

1. Check OTEL Collector is running:

### Docker network issues   ```bash

   curl http://localhost:13133  # Should return OK

```bash   ```

# Reset Docker networks

docker compose -f docker-compose.otel.yml down2. Verify environment variable is set:

docker network prune   ```bash

docker compose -f docker-compose.otel.yml up -d   echo $OTEL_EXPORTER_OTLP_ENDPOINT

```   ```



---3. Check OTEL Collector logs:

   ```bash

## 📊 Pre-built Grafana Dashboards   docker compose -f docker-compose.otel.yml logs otel-collector

   ```

The setup includes a pre-configured "Flower Federated Learning" dashboard with:

### Jaeger shows no traces

- Total aggregations counter

- Client training rounds1. Ensure your Python code initializes telemetry:

- Broker message throughput   ```python

- Aggregation rate over time   TRACER, METER = init_otel("service-name")

- Per-client training activity   # TRACER should not be None

   ```

Access at: http://localhost:3000 → Dashboards → Flower FL

2. Check Jaeger service dropdown - select your service name

---

### Grafana dashboards empty

## 🔗 Related Documentation

1. Wait 30-60 seconds for metrics to populate

- [OpenTelemetry Python](https://opentelemetry-python.readthedocs.io/)2. Verify Prometheus is scraping:

- [Jaeger Documentation](https://www.jaegertracing.io/docs/)   - Go to http://localhost:9090/targets

- [Prometheus Documentation](https://prometheus.io/docs/)   - All targets should be "UP"

- [Grafana Documentation](https://grafana.com/docs/)

- [Eclipse Mosquitto](https://mosquitto.org/documentation/)### Docker network issues


```bash
# Reset Docker networks
docker compose down
docker compose -f docker-compose.otel.yml down
docker network prune
```

---

## 📊 Pre-built Grafana Dashboards

The setup includes a pre-configured "Flower Federated Learning" dashboard with:

- Total aggregations counter
- Client training rounds
- Broker message throughput
- Aggregation rate over time
- Per-client training activity

Access at: http://localhost:3000 → Dashboards → Flower FL

---

## 🔗 Related Documentation

- [OpenTelemetry Python](https://opentelemetry-python.readthedocs.io/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
