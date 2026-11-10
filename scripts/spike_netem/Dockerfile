# Measurement host for the I9 spike: tc, ping and iperf3 from Debian's apt.
# net-tools only for Containernet: Mininet brings interfaces up with ifconfig.
FROM debian:bookworm-slim
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      iproute2 iputils-ping iperf3 procps python3 python3-numpy \
      net-tools \
 && rm -rf /var/lib/apt/lists/*
CMD ["sleep", "infinity"]
