# Distributed backend on real hardware (I14, #199)

**Status: pending.** The software side of the `distributed` backend is built and tested on Docker containers acting as machines (`tests/test_runtime_equivalence.py`). Nothing here has been run on real hardware yet: no Raspberry Pi, no physical Linux server, no phone. The issue's acceptance (an experiment on at least three real machines, one of them a Raspberry Pi, giving `sim`'s final model with full quorum and no losses, and the mobile spike's conclusion written) stays open until the steps below are done and their results committed.

## 1. The machines

| Host role | Machine | How it joins |
|---|---|---|
| cloud, fog | Linux server | Python 3.11 + `onion_fl`, or the image of `src/onion_fl/infrastructure/distributed.Dockerfile` |
| fog or edge | Development PC (Windows 11) | WSL2 with `sshd` and a venv, or the same image in Docker Desktop with port 22 published |
| edge | Raspberry Pi (ARM64, 64-bit OS) | Python 3.11 + `onion_fl` in a venv, or the image built on the Pi |

Each machine needs `sh`, `tar`, an SSH server that accepts the launcher's key, and the launcher's commit of `onion_fl` (the preflight compares source digests, so a different checkout is refused). The data cache is sent by the launcher; the raw datasets are not needed on the machines.

An inventory for that lab (no secrets: keys are paths, here from the environment):

```yaml
# inventories/lab.yaml
machines:
  cloud-1: {address: 192.168.1.10, user: onion, key: "${ONIONFL_KEY}", python: .venv/bin/python, root: onion-fl}
  fog-1:   {address: 192.168.1.11, user: onion, key: "${ONIONFL_KEY}", python: .venv/bin/python, root: onion-fl}
  field-raspberry_pi-1:
    address: 192.168.1.20
    user: pi
    key: "${ONIONFL_KEY}"
    python: onion-fl/.venv/bin/python
    root: onion-fl
clock: {reference: ntp, server: pool.ntp.org, max_offset: 0.05}
```

and in the experiment: `backend: {name: distributed, inventory: inventories/lab.yaml}`, with an `infrastructure` whose hosts are those names and `services: {mqtt_broker: cloud-1}` (Mosquitto running on the server) or gRPC links.

## 2. Raspberry Pi: what to check and how

1. **OS and Python.** A 64-bit OS (`uname -m` gives `aarch64`); Python 3.11 or newer (`python3 --version`; on Raspberry Pi OS Bookworm it is 3.11).
2. **PyTorch and the dependencies.** In a venv: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -e .` from the launcher's commit. Check `python -c "import torch, numpy, pandas, sklearn, paho.mqtt, grpc; print(torch.__version__)"`. Record what fails and what had to be built from source.
3. **Same numeric versions.** The preflight warns when torch, numpy, scipy, pandas or scikit-learn differ from the launcher's. A stub run must still give `sim`'s model bit for bit (the coordinator takes the launcher's initial model, and the stub trainer only uses numpy); with a real trainer, write down whether the bits differ and by how much.
4. **Clock.** `timedatectl` shows `System clock synchronized: yes`; the preflight's offset (in `run.json`, `distributed.machines.<host>.offset` and `ntp`) must stay under `max_offset`.
5. **`train_rate`.** Run the calibration of I12 on the Pi with the benchmark's models and record `train_rate` per (profile, model) in the infrastructure's `raspberry_pi` profile; note the temperature and whether it throttled (`vcgencmd get_throttled`).
6. **The run.** `onion_fl plan` (machines and hosts), then `onion_fl run --backend distributed` with the stub trainer and full quorum; then `onion_fl fidelity runs_sim runs_distributed` (reference first) must say the functional fidelity is complete and the model bitwise equal. Commit the run with its ids (docs/RULES.md).

## 3. Mobile spike (optional)

Question: can an Android phone host an `onion_fl node`? iOS is out of scope.

1. Install Termux (from F-Droid), then `pkg install python openssh tar` and start `sshd` (port 8022).
2. Try `pip install numpy pandas pyyaml pydantic paho-mqtt`; then PyTorch: there is no official Android wheel, so try Termux's `python-torch` package if present, or note that it is missing.
3. If the imports work, add the phone to the inventory (`port: 8022`, `python: python`) as an edge host and run the stub experiment; the preflight shows what is missing.
4. Write the conclusion in this file: either the phone ran a node (with its versions and `train_rate`), or why phones stay out (for example, no PyTorch build), and close the spike.

## 4. Network

- The machines must reach each other on the broker's port (1883) or on each one's `grpc_port` (50051 by default); the preflight checks both from every machine and names the pair that fails.
- On the development PC under WSL2, published ports and the `advertise` address need care: with NAT networking, use the Windows host's LAN address and a port proxy, or WSL's mirrored networking.

## 5. The distributed continuum (C9, #164)

**Status: run on containers only.** `results/continuum_distributed/` has a C8 scenario on five containers acting as machines, against its simulation, with an edge host killed and resumed from its edges' saved state and an edge host that joins late. The acceptance of #164 on real machines needs the same on the lab above:

1. The inventory of §1 with three hosts of edges, and `experiments/continuum_distributed.yaml` with its `infrastructure` pointing at the lab (hosts, assignment) and `backend: {name: distributed, inventory: inventories/lab.yaml, start_in: 60, checkpoints: true}`.
2. `start_in` must exceed the slowest machine's start (on the containers a process took about 40 s to build the scenario); the Pi's will be longer. A round-1 deadline that fits a cold first training avoids its late updates.
3. **A real failure.** Power the Pi off (or `sudo reboot`) during the run rather than `kill -KILL`. The launcher does not yet start a node again on its own after a reboot: give `restarts` a `down` longer than the reboot, or start `onion_fl node --resume` on the Pi by hand (`python -m onion_fl node --id <host> --run <run_id> --config runs/<run_id>/scenario.json --resume`, in its `root`). Check that its edges' `node.resumed` events say `restored` and that `python scripts/continuum_distributed.py resumed` shows each edge's `examples` equal to the simulation's.
4. Restarting a fog or the root is not supported: put the aggregators on machines that stay up.
