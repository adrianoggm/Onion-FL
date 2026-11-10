# FEMNIST on FedScale devices (new framework, real data)

`experiments/femnist_fedscale.yaml` was run in `sim` on 2026-11-10, issue #253. It shows that a federation runs on devices sampled from FedScale's client traces and cites them; it is not a fidelity result, because the rates come from a trace, not a calibration (`docs/RULES.md` §3).

```bash
onion_fl traces import fedscale data/FedScale
onion_fl run experiments/femnist_fedscale.yaml
```

## Setup

- **Data.** FEMNIST as in `results/femnist_reference/`: the 200 fixed writers, 40 reserved for the test with role seed 0, 160 training edges.
- **Devices.** `infrastructures/fedscale_field.yaml`: 160 phones sampled with seed 0 from FedScale's traces (commit `7ec441c`, Apache-2.0), one edge per phone (`one_per_device`). Each trains at `1000 / (3 · computation)` samples/s (1.7–22 samples/s), sends at its client's bandwidth and is up only inside its availability client's windows.
- **Federation.** `femnist_fedscale`: one fog under the cloud; the fog closes at its 900 s deadline with whoever answered (quorum 1).
- **Training.** FedAvg, `modular_mlp` (adapter 128, trunk [128], dropout 0.2), 20 rounds of 1 local epoch (Adam, lr 0.001, batch 32), seed 0. Hyperparameters as in the reference, not tuned.

| Identity | Value |
|---|---|
| `topology_id` | `18086603f5b1b16c80d6f511ca71272027578268e89353a6f8052f07f69b6322` |
| `config_id` | `d93fc2177303c5c6cac4570d5beea97ad47f0cb48b4c7a5e9d150a61c8ce96a9` |
| `infrastructure_id` | `84cd172bdff7bbb813b9a2ea8aa8236394abec5a2124763cbad3f6723eaae7e9` |
| `data_id` | `7d8238e25baf310766da9c287196bd4e42a45a3a709873d21edc89830d10ad20` |
| `run_id` | `20261110T121250Z-fd3675c2f84b` |
| `run_hash` | `7abcb32702d1472f97eebff8d452f14c40234695c0c7a92dd2b5d706906fcb89` |
| code | commit `87603c3`, clean tree |
| `client_device_capacity` | SHA-256 `477f6104…9e53` (`capacity.csv` `c76bca57…7b9e`) |
| `client_behave_trace` | SHA-256 `d0b6f81a…1662` (`behaviour.jsonl` `8677d515…ede`) |

`run.json` holds the full hashes and the 160 clients drawn under `device_traces`.

## What it shows

- **The phones' availability shapes the run.** The federation starts at 71.4 h of virtual time, when the last phone first comes online and says hello; the 20 rounds then take 4.8 h, closed at the deadline. Between 19 and 46 phones (median 30) train in each round, and 2,671 messages are dropped by offline phones.
- **The model still learns:** test accuracy 0.075 after round 1, 0.36 at round 9 and 0.438 at round 20, against 0.49 at round 20 for the reference with every writer in every round (`results/femnist_reference/`, seed 0). One seed, so no interval.
- **Reproducible.** The same config run twice gave the same summary (same final metrics, virtual end time and message counts).

## Files

- `20261110T121250Z-fd3675c2f84b/`: `run.json`, `summary.json` and `model.npz` (the events, 8 MB, are not committed).
