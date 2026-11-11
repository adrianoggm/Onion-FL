# Host image of the emulated backend (I10): Python with the runtime
# dependencies of onion_fl and iproute2 for tc and routes. The onion_fl source
# is mounted at /onionfl/src when a run starts, so the image does not change
# with the code. The ranges below are those of pyproject.toml (a test checks
# it); the backend builds the image with NUMERIC pinned to the versions of the
# environment that launches the runs, so that both compute the same bits.
FROM python:3.11-slim-bookworm
ARG TORCH="torch>=2.2,<3"
ARG NUMERIC="numpy>=1.25,<3 pandas>=1.5,<3 scikit-learn>=1.2,<2"
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      iproute2 \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      "$TORCH" \
 && pip install --no-cache-dir $NUMERIC \
      "numpy>=1.25,<3" \
      "pandas>=1.5,<3" \
      "scikit-learn>=1.2,<2" \
      "paho-mqtt>=2.0,<3" \
      "pyyaml>=6,<7" \
      "opentelemetry-sdk>=1.24,<2" \
      "opentelemetry-exporter-otlp>=1.24,<2" \
      "prometheus-client>=0.17,<1" \
      "pydantic>=2.12,<3"
ENV PYTHONPATH=/onionfl/src PYTHONUNBUFFERED=1
CMD ["sleep", "infinity"]
