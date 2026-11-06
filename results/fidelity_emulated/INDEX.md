# Fidelity of the simulation against the emulated backend (issue #195)

`experiments/fidelity_emulated.yaml` was run on 2026-11-06 with real SWELL-KW data (physiology features, per minute), once with `backend: emulated` and once with `backend: sim`. Then `onion_fl fidelity` compared them, with the emulated runs as the reference (A) and the simulation as the backend under test (B).

```bash
onion_fl run experiments/fidelity_emulated.yaml --backend emulated   # into runs_emulated/
onion_fl run experiments/fidelity_emulated.yaml --backend sim        # into runs_sim/
onion_fl fidelity runs_emulated runs_sim --metric macro_f1 --metric time \
  --series global/swell --csv fidelity.csv --html fidelity.html > fidelity.json
```

The two runs differ only in `backend` and `paths.runs`. The emulated runs used `image: onionfl-a195-emulated:latest` and `project: onionfl-a195`, which only name the containers.

## Setup

- **Federation.** One cloud and two fogs. There are 20 SWELL training edges, mixed over the fogs (α = 1), and 5 test subjects evaluated at the root.
- **Training.** FedAvg for 10 rounds. Each round is 5 local epochs of the `standard` trainer (lr 0.01, batch 32) on a `modular_mlp` (adapter 32, trunk [16]). Seeds 0–2.
- **Infrastructure.** `infrastructures/emulated_lab.yaml`:
  - a cloud VM, which also hosts the MQTT broker;
  - two fog servers, each on its own site;
  - six edge devices. `colocated` puts 4 edges on each device, so 5 of them are used.
  - The links are field–fog_a (10 ms, 50 Mbit/s), field–fog_b (15 ms, 20 Mbit/s), fog_a–dc (40 ms, 10 Mbit/s) and fog_b–dc (60 ms, 10 Mbit/s).
- **Emulation.** 13 containers per run: 8 hosts, 4 routers and the broker.
  - The hosts get their profiles' CPU and memory limits: the cloud 2 CPUs and 4 GB, each fog 1 CPU and 2 GB, each device 1 CPU and 2 GB.
  - Docker Desktop 28.3.2 on WSL2, Windows 11, Intel Core i7-14650HX. The containers shared the machine with other workloads.
  - Setting up the containers took 6–14 s per run. It is not part of the run's time.
- **Profiles are not calibrated** (#197). The only compute rate in the simulation is the devices' `train_rate: 2000` samples/s, and it is a guess. The performance errors below therefore measure an uncalibrated simulation. They do not measure the simulator itself.

## Identity

| Identity | Value |
|---|---|
| `topology_id` | `464141d81d17d3a9da4622bec8ecd06fe6a655e8d44cb1bbbf98993b9dfcfe9c` |
| `infrastructure_id` | `da875674ad22d77f7aabcc6d9ac2fe642e9e8be204889668bb4adc5f238df43e` |
| `data_id` | `7949a27b972ad85bf284ca213fcbfe77c610276f373755f866de29850c0bfc53` |
| `config_id` emulated | `9c5825053e7b01f3745295663ca262295b29d6175e078c2fd23d9d1bb4b649ed` |
| `config_id` sim | `0875a24b6653f60c53713f952da89614a7ed462489f9f06b2106b2b7c1947479` |
| code | commit `67fdded`, clean tree, for all six runs |

The two `config_id`s differ because `backend` is part of the config. `onion_fl fidelity` pairs runs by `design_id`, which leaves `backend` and `paths` out, and by seed. The `run_id` and `run_hash` of every run are in its `run.json`.

## Results

**Functional.**
- The protocol events of every node matched in all 3 pairs: rounds started and closed, and the end of the run.
- The final models match within the tolerance of `test_runtime_equivalence` (largest difference 1.2–2.1 × 10⁻⁷). They are **not identical bit for bit**.
  - The torch builds for Windows (the simulation) and for Linux (the containers) compute training with other floating-point kernels.
  - The JSON codec writes the floats with as many digits as they need, so the bytes per link also differ, by less than 0.02 % (7 212 737 against 7 213 964 bytes for seed 0).
- The final test macro-F1 is the same in both backends for every seed: 0.4734, 0.4764 and 0.4721.
- With the `stub` trainer, the same check is exact: the same final model bit for bit and the same bytes per logical link and round. `tests/test_runtime_equivalence.py::test_the_emulated_backend_matches_the_simulation` checks that, and it passed here. With real training, bit-for-bit equality needs the same torch build on both sides.

**Performance.** Relative error (B − A) / A, the mean over pairs with a 95 % bootstrap interval (`fidelity.json`):

| Quantity | Emulated (A) | Sim (B) | Relative error | 95 % CI |
|---|---|---|---|---|
| Run duration (s) | 91.6 | 17.6 | −0.79 | [−0.87, −0.72] |
| Round time at the root (s) | 7.47 | 1.72 | −0.72 | [−0.80, −0.64] |
| Link latency (s) | 0.874 | 0.118 | −0.75 | [−0.79, −0.72] |
| Edge compute time per round (s) | 1.24 | 0.70 | −0.06 (mean of signed errors); 0.58 absolute | [−0.39, 0.11]; [0.54, 0.63] |

**Decision.** There is one configuration, so there is no ranking to compare, and 3 seeds are fewer than the 5 that decision fidelity needs (I1 §12). `fidelity.json` reports that, and its top-1 and regret are trivial.

## Reading

- **The emulated backend reproduces the federation functionally.** The protocol and the scores are the simulation's, and with the stub trainer the final model is identical bit for bit.
- **The uncalibrated simulation is about 4–5 times faster than the emulation**, in round time and in the run's duration.
  - The largest gap is link latency: 0.87 s emulated against 0.12 s simulated.
    - Both backends route an edge's message through the broker on the cloud, so it crosses the fog–cloud link twice: about 100 ms of propagation each way. A message is about 12 kB of JSON on average (7.2 MB in 621 messages per run), a few milliseconds at these rates. The simulation's 0.12 s is that.
    - The other ~0.75 s is not explained by the links. A plausible cause, not measured here: a real process handles its events one at a time, so a delivery to a device that is busy training its 4 edges on 1 CPU waits until the training ends, and the latency is measured when the message is handled. TCP, the broker and the co-located containers add some time too.
  - Per edge, compute time is close on average (−6 %) but varies a lot (58 % absolute error). The simulation charges a fixed 2000 samples/s. The containers share the laptop's cores, and the spike measured 12–16 % lost when CPU-limited hosts run side by side.
- **Next step.** Calibrate `train_rate` and the links with the hosts co-located as in the experiment (#197), then validate with other loads (#198).

## Files

- `runs_emulated/<run_id>/` and `runs_sim/<run_id>/`: `run.json`, `summary.json` and `model.npz` of each run. The `emulation` entry of an emulated `run.json` lists every tc command its containers ran.
- `fidelity.json`, `fidelity.csv` and `fidelity.html`: the output of `onion_fl fidelity`.

`events.jsonl` is left out. `host` and `pid` are removed from `run.json`, and so is `config.paths`, which named scratch folders on this machine. As a result, `verify_run` cannot check these copies. It passed on the original run folders when `onion_fl fidelity` read them.
