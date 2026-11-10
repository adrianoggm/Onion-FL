#!/bin/sh
# Entrypoint for the Containernet image: start Open vSwitch, then the lab.
# $1 is the shaping set (rate | loss) or "cpu"; see containernet_topo.py.
set -e
service openvswitch-switch start
exec python3 /spike/containernet_topo.py "$1"
