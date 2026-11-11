# CIFAR-10 FedAvg with an MLP (new framework, real data)

`experiments/cifar10_mlp.yaml` was run on 2026-11-10 on CIFAR-10, using the official python version from the University of Toronto (`cifar-10-python.tar.gz`, MD5 `c58f30108f718f92721af3b95e74349a`, checked). Issue #247.

**This is not the NIID-Bench reproduction.** NIID-Bench trains a simple CNN, which needs torch_module presets (#245). This MLP stands in until then. The CNN run, which is the issue's acceptance check (±3 points), is pending #245.

```bash
onion_fl run experiments/cifar10_mlp.yaml --workers 6     # OMP_NUM_THREADS=4 per worker
onion_fl report runs --out report.html --metric accuracy --metric macro_f1 --by scenario
```

## Setup

- **Data.** There are 50,000 training images in 10 classes, as 3,072 pixels in [0, 1], standardised with the training clients' statistics.
- **Test.** The test is the official test batch: 10,000 images, 1,000 per class. The `partition` step makes it the subject `test`, and `data.roles.overrides.cifar10.test: [test]` reserves it. It is never part of a client. `tests/test_datasets_descriptors.py` checks that none of its images is an exact copy of a training image.
- **Clients.** There are 10 synthetic clients, drawn by `datasets/cifar10.yaml` with `partition_seed` 0, which is not the training seed. The three training seeds therefore see the same clients.
  - `dirichlet_label`: Dir(0.5) per class, as NIID-Bench draws it (table below).
  - `iid`: 5,000 random images per client.
- **Federation.** All 10 clients take part in every round, under one `fedavg` fog with quorum 1.0, on lossless `lan` links with the `npz` codec. The root takes evaluators only, so the fog stands in for NIID-Bench's single server, and a tree of `fedavg` is a flat FedAvg.
- **Training.**
  - Model: `modular_mlp` with a 256-unit adapter, a [128] trunk and no dropout, about 820k parameters.
  - Schedule: NIID-Bench's, 50 rounds of 10 local epochs with batch 64 and SGD at lr 0.01. NIID-Bench adds momentum 0.9, which the standard trainer lacks.
  - Seeds: 0–2.
- **Hyperparameters.** They were set before the runs and not tuned. No validation subjects were used.

| Dir(0.5) client | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Images | 5,309 | 5,109 | 6,544 | 5,291 | 4,675 | 5,360 | 4,323 | 5,217 | 2,369 | 5,803 |
| Classes with ≥ 1 image | 8 | 7 | 4 | 9 | 10 | 9 | 10 | 8 | 10 | 8 |
| Largest class share | 0.60 | 0.64 | 0.55 | 0.40 | 0.53 | 0.32 | 0.45 | 0.56 | 0.35 | 0.39 |

| Identity | Dir(0.5) | IID |
|---|---|---|
| `config_id` | `d0b012259c2a7fccf5aeb1f0746d0a9734f7a4fb93988317c7ac5d84279d27a0` | `71c75dfce3ed304ef04afd5ea750910536e70978b7ea87c50289b4d8b37686f0` |
| `topology_id` | `60f08e8ddbcfa14cc1e8f3aacbffc97e7a2765ee375a4c063b7a7333953289ee` | same |
| `data_id` | `9b532eefac6c222371863d1f210b9b855917f293b53f92f10f21a1c926f5a23f` | `44a91b813d3dab9641d6d0782b5a9c21559099c35ea8cf6bc3e873ea235d22fe` |
| code | commit `195443d`, clean tree | same |

| Partition | Seed | `run_id` | `run_hash` |
|---|---|---|---|
| Dir(0.5) | 0 | `20261110T171948Z-ead1c50d8ddd` | `7bffd40fc701ce07eab2e9809bff62d4d9cfefd1df850babacbda4a4ce811acc` |
| Dir(0.5) | 1 | `20261110T171948Z-e91861646e13` | `1b69f028f498a7c6e33f806c62040bf0b23b4104088260640cffcd62791224f9` |
| Dir(0.5) | 2 | `20261110T171948Z-4ff5a0823c83` | `52d137a3dbae569276bc8456c52e5fc519dd98dfcf4af7701109215de492ab72` |
| IID | 0 | `20261110T171948Z-a0d609647150` | `4f3b245f0bf7d6948366d84965c0c74dd5c391003a6744db9df7873bf681e1b9` |
| IID | 1 | `20261110T171948Z-755b16842878` | `0d09c285f3fed0b9ff9bf668bef501f9067c417de41dba5f3ff513d3fc9c2b5a` |
| IID | 2 | `20261110T171948Z-26f7a238039c` | `5276f960dbab3739f20e9d5a0a9d47d5e661b1e3118c709533d39359e7e577fc` |

## Results on the official test batch

| Model | Dir(0.5) accuracy | Dir(0.5) macro-F1 | IID accuracy | IID macro-F1 |
|---|---|---|---|---|
| Federated MLP, round 50 (mean ± 95% CI over 3 seeds, Student t) | 0.477 ± 0.003 | 0.474 ± 0.003 | 0.534 ± 0.011 | 0.534 ± 0.010 |
| NIID-Bench FedAvg, simple CNN (Li et al., 2022, Table III; not run here) | 0.682 ± 0.007 | — | 0.704 ± 0.002 | — |
| Constant predictor (balanced test) | 0.100 | 0.018 | 0.100 | 0.018 |

- **The model learns:** 0.48 and 0.53 accuracy against 0.10 for a constant answer. Non-IID costs it 5.7 points, against 2.2 points for NIID-Bench's CNN.
- **There is no NIID-Bench MLP to compare with.** NIID-Bench uses an MLP (32-16-8) only on its tabular datasets (adult, rcv1, covtype). On CIFAR-10 it reports the CNN alone.
- **The gap with NIID-Bench, 20.5 points on Dir(0.5) and 17.0 on IID, is the model, and it is not a failed reproduction.**
  - An MLP on raw pixels has no convolutions, so nothing in it is shared across image positions. Our IID run, which is close to centralised training, levels off at 0.53–0.54.
  - The missing momentum also slows the local steps.
  - Only the CNN run (#245) can test the issue's ±3 points.
- **The runs overfit after about round 10.** The test accuracy levels off from about round 20, while the test loss goes up.
  - Dir(0.5), seed 0: accuracy 0.453 at round 10 and 0.476 at round 50; test loss 1.73 at both rounds 5 and 10, then 2.51 at round 50.
  - IID, seed 0: test loss lowest at round 5 (1.41), 2.41 at round 50, with accuracy 0.538 from round 20 on.
  - Ten local epochs on 5,000 images fit the training data (training loss 0.10–0.15 at round 50).
  - The table reports round 50 as declared, not the best round. The best round is at most 0.010 higher.
- **Time.** Each run took about 71 minutes of wall time, with six runs in parallel on 24 logical cores and 4 torch threads each. Its 1,236 messages came to 3.8 GB encoded.

The constant predictor's macro-F1 is its class's F1 (2 × 0.1 / 1.1 = 0.18) divided by the 10 classes. The Dir(0.5) class shares come from the cached partition (`data/cache/cifar10/`), and `data.composition` in the events records the same counts.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`) and `summary.json` of each run.
- `report.html`: accuracy and macro-F1 per round for each partition, mean ± CI over the seeds.

`events.jsonl` (2.5 MB per run), `model.npz` (3.3 MB per run) and the continuation bundle are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies.
