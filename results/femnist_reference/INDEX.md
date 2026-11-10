# FEMNIST FedAvg reference (new framework, real data)

`experiments/femnist_reference.yaml` was run on 2026-11-05 on FEMNIST from the Flower Datasets copy of LEAF ([flwrlabs/femnist](https://huggingface.co/datasets/flwrlabs/femnist), Parquet SHA-256 `cdb389c8…522b`), written as LEAF JSON by `scripts/femnist_from_flower.py`. Issue #206.

```bash
onion_fl run experiments/femnist_reference.yaml --workers 3
onion_fl report experiments/femnist_reference.yaml --out report.html --metric accuracy --metric macro_f1
onion_fl baseline experiments/femnist_reference.yaml --models lr > baselines.json
```

## Setup

- **Data.** The descriptor's 200 fixed writers, with 44,451 images and 62 classes. Each writer is one subject and one edge.
- **Roles.** 40 writers, drawn with role seed 0, are reserved for the test (8,665 images). The other 160 train, with 35,786 images.
- **Preprocessing.** The pixels are standardised with the training writers' statistics, and constant pixels are dropped.
- **Federation.** Four fogs under one cloud (`femnist_four_fogs`, lossless `lan` links, `npz` codec), with the training edges spread uniformly (40 per fog).
- **Training.** FedAvg at both levels with `modular_mlp` (adapter 128, trunk [128], dropout 0.2) for 30 rounds of 1 local epoch (Adam, lr 0.001, batch 32), seeds 0–2.
- **Hyperparameters.** They were set before the run and not tuned. No validation writers were used.

| Identity | Value |
|---|---|
| `topology_id` | `80d45cbee3ed481093da8f2e29f7fde652ac1049eec791d7406cd7e3e1ac2067` |
| `config_id` | `befb8635851ce15488af0e7e050c40103b839a5e112a0c65f08f917300902494` |
| `data_id` | `7d8238e25baf310766da9c287196bd4e42a45a3a709873d21edc89830d10ad20` |
| code | commit `a3fa271`, clean tree |

| Seed | `run_id` | `run_hash` |
|---|---|---|
| 0 | `20261105T210025Z-e36f7580e6cf` | `6851a6a285aa5a63afdf0f50e6abd3171d2da3e03fa4658e6cc5f0c4d631ece7` |
| 1 | `20261105T210025Z-a53ff493c857` | `86a67ae71ebc0d26358be5f1f5ecc50cf1c7d5c8cbb331f00fb762a774786c53` |
| 2 | `20261105T210025Z-422b67ac4f7e` | `1b248de25e8a175c45f882d7634ddaf102370bd24cd763407dee81537b4241e2` |

## Results on the test writers

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Federated MLP, round 30 (mean ± 95% CI over 3 seeds) | 0.519 ± 0.003 | 0.153 ± 0.002 |
| Centralised logistic regression | 0.661 | 0.472 |
| Always "1" (the training bag's majority class) | 0.060 | 0.002 |

- **The federated model clearly beats the majority class:** 0.519 against 0.060 in accuracy, nearly nine times as high. It is not a constant predictor.
- **Seed 0's test accuracy was still rising at round 30:** 0.24 at round 5, 0.38 at round 10, 0.49 at round 20 and 0.52 at round 30. The reference is under-trained by design: it shows that the signal is there, not the best result.
- **Centralised logistic regression does better** on the same split: 0.661 accuracy and 0.472 macro-F1. Thirty rounds of one local epoch are not enough for the MLP; more rounds or local epochs, chosen on validation writers, are a separate experiment.
- **The low macro-F1 comes from the letters,** most of which are rare: the ten digits make up about half the images, and several letters have fewer than 50 test images.
- **Time.** Each run took about 6.5 minutes of wall time for 160 edges and 30 rounds. Its 12,852 messages came to 5.1 GB encoded.

The majority row is computed from the roles: class "1" is the most frequent in the training bag, and it is 516 of the 8,665 test images. A constant predictor's macro-F1 is its class's F1 (0.112) divided by the 62 classes present.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`), `summary.json` and `model.npz` of each seed.
- `report.html`: accuracy and macro-F1 per round, mean ± CI over the seeds.
- `baselines.json`: centralised logistic regression on the same split.

`events.jsonl` (19 MB per run) and the continuation bundle are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies.
