# Infrastructure stack

`docker-compose.otel.yml` runs what an Onion-FL run talks to: the MQTT broker for real runs, and the tracing and metrics backends. Onion-FL itself runs on the host.

```bash
just docker-up       # or: cd docker && docker compose -f docker-compose.otel.yml up -d
just docker-down
just docker-clean    # also removes volumes
```

| Service | Host port | Role |
|---|---|---|
| Mosquitto | 1883 (websockets 9001) | MQTT broker for `onion_fl run --mode real` (`mosquitto/mosquitto.conf`, anonymous, no TLS) |
| OTEL Collector | **4320** (OTLP HTTP), 4319 (OTLP gRPC) | Receives the spans of the `otel` sink, derives span metrics, exports to Jaeger and Prometheus |
| Jaeger | 16686 (UI) | Traces: one span per send and per receive, linked through the message id |
| Prometheus | 9090 | Scrapes the `prometheus` sink on the host (`host.docker.internal:9464`) and the collector (`prometheus.yml`) |
| Grafana | 3000 (admin/admin) | "Onion-FL" dashboard (`grafana/provisioning/dashboards/json/onion-fl.json`) |

## A run against the stack

Point the links at the broker and turn the sinks on in the experiment:

```yaml
# topologies/<name>.yaml: every link that should go over MQTT
fog:
  defaults:
    link_up: {transport: {name: mqtt, broker: "localhost:1883", qos: 1}}
edge:
  link_up: {transport: {name: mqtt, broker: "localhost:1883", qos: 1}}

# experiments/<name>.yaml
runtime: {mode: real, timeout: 3600, heartbeat: 10}
sinks:
  - {name: prometheus, port: 9464}            # live series for Grafana
  - {name: otel, endpoint: "http://localhost:4320"}
```

```bash
onion_fl run experiments/<name>.yaml --mode real
```

`run --mode real` checks that the broker answers, then starts one process per aggregator (plus the root with the test evaluators). They rebuild the same scenario, exchange messages on `onionfl/<run_id>/<node>/inbox` and stop when the coordinator finishes; the launcher merges their events into `runs/<run_id>/`. Without `--mode real` the same experiment runs in simulation and needs no broker.

## Port trap

Host port 4318 is Jaeger itself, not the collector. Spans sent there skip the collector, so no span metrics are derived. Use `http://localhost:4320` for the `otel` sink's `endpoint`.

## Tests against a broker

The real-runtime tests skip without a broker. To run them locally:

```bash
docker run -d --name mosquitto -p 1883:1883 eclipse-mosquitto:2 mosquitto -c /mosquitto-no-auth.conf
ONIONFL_MQTT=localhost:1883 pytest tests/test_runtime_real.py tests/test_runtime_equivalence.py
```

The CI starts the same container before the tests (`.github/workflows/ci.yml`). In Git Bash on Windows, prefix the `docker run` with `MSYS_NO_PATHCONV=1`, or the container path of the config file is rewritten.
