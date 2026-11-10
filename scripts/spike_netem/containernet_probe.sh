#!/bin/sh
# I9 spike: can Containernet's Open vSwitch run on this Docker host?
# Run inside the official image (see README.md). Docker Desktop's VM has the
# openvswitch module, but the image cannot modprobe it without /lib/modules:
#   docker run --rm -i --privileged --pid=host --network host \
#     -v /var/run/docker.sock:/var/run/docker.sock \
#     -v /lib/modules:/lib/modules:ro \
#     --entrypoint sh containernet/containernet:latest -s < containernet_probe.sh
head -2 /etc/os-release
ovs-vsctl --version | head -1
service openvswitch-switch start 2>&1 | tail -3
timeout 20 ovs-vsctl show
timeout 20 ovs-vsctl add-br probe0 && ip link show probe0 && ovs-vsctl del-br probe0
docker version --format 'docker client {{.Client.Version}} server {{.Server.Version}}'
