# CIFAR-10 FedAvg with NIID-Bench's CNN (new framework, real data)

`experiments/cifar10_cnn.yaml` was run on 2026-11-11 on CIFAR-10, with the same official python version as [`cifar10_mlp`](../cifar10_mlp/INDEX.md) (`cifar-10-python.tar.gz`, MD5 `c58f30108f718f92721af3b95e74349a`). Issue #247.

**The reproduction misses NIID-Bench by 18 and 15 points, and the cause is identified but not yet tested.** NIID-Bench trains CIFAR-10 with data augmentation; this run has none, and it overfits.

```bash
onion_fl run experiments/cifar10_cnn.yaml --workers 6     # OMP_NUM_THREADS=4 per worker
onion_fl report runs --out report.html --metric accuracy --metric macro_f1 --by scenario
```

## Setup

It is `cifar10_mlp` with two changes, set before the runs and not tuned:

- **Model:** the `torch_module` preset `niid_bench_cnn`, NIID-Bench's simple CNN (conv 5×5/6, pool, conv 5×5/16, pool, dense 120 and 84). Its first convolution is the adapter, the rest of the body the trunk and the last linear layer the head.
- **Optimizer:** SGD with momentum 0.9, as NIID-Bench's paper states for its runs.

Everything else is the MLP run's:
- the same 10 synthetic clients (Dir(0.5) and IID, `partition_seed` 0);
- the official test batch (10,000 images);
- one `fedavg` server with quorum 1.0 on lossless links;
- 50 rounds of 10 local epochs, batch 64, lr 0.01;
- seeds 0–2.

The pixels are standardised per pixel with the training clients' statistics. NIID-Bench feeds them in [0, 1].

| Identity | Dir(0.5) | IID |
|---|---|---|
| `config_id` | `c6d9b6a957f56e4dccc315975762f339d8eb93a05319ce05e46d3964f26f0e69` | `b3cf5129005ef08897fe643b2b636e51f5ef684d621d842f32c45a89ec1e8f8a` |
| `topology_id` | `60f08e8ddbcfa14cc1e8f3aacbffc97e7a2765ee375a4c063b7a7333953289ee` | same |
| `data_id` | `9b532eefac6c222371863d1f210b9b855917f293b53f92f10f21a1c926f5a23f` | `44a91b813d3dab9641d6d0782b5a9c21559099c35ea8cf6bc3e873ea235d22fe` |
| code | commit `8b8ec67`, clean tree | same |

| Partition | Seed | `run_id` | `run_hash` |
|---|---|---|---|
| Dir(0.5) | 0 | `20261111T173005Z-0c3f41996ff8` | `6778a780755870b9e1db49dacfb88c524c61f1a9cf00991cf61e8cc89509ca5c` |
| Dir(0.5) | 1 | `20261111T173006Z-93842db8a711` | `3e174e8b7bece0ddd3ef424964fe20fc2990ff1a6a08e8531f1348f26ef3fdbd` |
| Dir(0.5) | 2 | `20261111T173006Z-b390c7df8ac0` | `2f5e8fe30362ddc46c2500fdb4d0f910e058ffe258456335d72072cd07164d09` |
| IID | 0 | `20261111T173006Z-e97730e73ff8` | `fd04ac773748559d2764f5da72ad12f97171ae43f1ecc71dc905c038b64c9739` |
| IID | 1 | `20261111T173008Z-db260cc552b4` | `d25122980d2cc09cd710449795cc822db98b8d425cc0972823432a3f50880839` |
| IID | 2 | `20261111T173005Z-660f41828ed6` | `fb40615980d6d3f4811adaba1a2a2dc9b37e7414b03e555e33ade45865478f8f` |

## Results on the official test batch

| Model | Dir(0.5) accuracy | Dir(0.5) macro-F1 | IID accuracy | IID macro-F1 |
|---|---|---|---|---|
| Federated NIID-Bench CNN, round 50 (mean ± 95% CI over 3 seeds, Student t) | 0.500 ± 0.007 | 0.498 ± 0.006 | 0.553 ± 0.009 | 0.552 ± 0.009 |
| Federated MLP, round 50 ([`cifar10_mlp`](../cifar10_mlp/INDEX.md)) | 0.477 ± 0.003 | 0.474 ± 0.003 | 0.534 ± 0.011 | 0.534 ± 0.010 |
| NIID-Bench FedAvg, simple CNN (Li et al., 2022, Table III; not run here) | 0.682 ± 0.007 | — | 0.704 ± 0.002 | — |

- **The CNN beats the MLP, but by 2 points rather than the 20 NIID-Bench implies.** It misses NIID-Bench by 18.2 points on Dir(0.5) and 15.1 on IID, far outside the issue's ±3.
- **It overfits early.** IID reaches 0.57 by round 5 and peaks at 0.573 at round 14 (every seed), then falls to 0.553; Dir(0.5) peaks at 0.51. The test loss at round 50 is 3.2 (Dir(0.5)) and 3.7–3.8 (IID), while accuracy stays flat. The table reports round 50, as declared; the best round is at most 0.02 higher, so choosing it would not close the gap.
- **The likely cause is the missing data augmentation, checked in NIID-Bench's code but not tested here.**
  - NIID-Bench's `utils.py` (`get_dataloader`, CIFAR-10) trains with a reflect pad of 4, `RandomCrop(32)` and `RandomHorizontalFlip()`, and tests without them.
  - Its `experiments.py` builds SGD with `weight_decay=args.reg`, default `1e-5`. This run has neither.
  - With 10 local epochs on about 5,000 images per client, a model that sees the same unaugmented images 500 times overfits. That matches the rising test loss.
  - The framework has no training-time augmentation. Adding it, then rerunning this experiment, is what can show whether it closes the gap.
- **Time.** Each run took about 105 minutes of wall time (six in parallel, 24 logical cores, 4 torch threads each). Its 1,236 messages came to 289 MB encoded.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`) and `summary.json` of each run.
- `report.html`: accuracy and macro-F1 per round for each partition, mean ± CI over the seeds.

`events.jsonl`, `model.npz` and the continuation bundle are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies.
