# Observability stack

`docker-compose.otel.yml` runs the infrastructure that a federated run uses: the MQTT broker plus tracing and metrics. The federated processes themselves run on the host (`just swell-launch-*`).

```bash
just docker-up       # or: cd docker && docker-compose -f docker-compose.otel.yml up -d
just docker-down
just docker-clean    # also removes volumes
```

| Service | Host port | Role |
|---|---|---|
| Mosquitto | 1883 (websockets 9001) | MQTT broker for all FL traffic (`mosquitto/mosquitto.conf`, anonymous, no TLS) |
| OTEL Collector | **4320** (OTLP HTTP), 4319 (OTLP gRPC) | Receives traces and metrics, derives span metrics, exports to Jaeger and Prometheus |
| Jaeger | 16686 (UI) | Traces |
| Prometheus | 9090 | Scrapes the server (`:8000`), broker (`:8001`), clients (`:8100-8124`), the collector and Pushgateway (`prometheus.yml`) |
| Pushgateway | 9091 | Server and client metrics pushed at shutdown |
| Grafana | 3000 (admin/admin) | "Flower FL Dashboard", provisioned from `grafana/provisioning/` |

## Port trap

Host port 4318 is Jaeger itself, not the collector. Traces sent there skip the collector, so no span metrics or OTLP metrics are produced. Point both `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` at `http://localhost:4320`, as the `just` recipes and `.env.template` do.

A `.env` in the repo root or in `docker/` overrides the recipes, because the launcher and `telemetry.py` both load it.
