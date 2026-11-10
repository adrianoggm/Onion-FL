# LEAF's CNN on FEMNIST (new framework, real data)

`experiments/femnist_cnn.yaml` was run on 2026-11-10 on the same FEMNIST extract as [the FedAvg reference](../femnist_reference/INDEX.md). Issue #245.

```bash
onion_fl run experiments/femnist_cnn.yaml --workers 1
onion_fl report experiments/femnist_cnn.yaml --out report.html --metric accuracy --metric macro_f1
```

## Setup

It is `experiments/femnist_reference.yaml` with only the model changed: `{name: torch_module, module: leaf_femnist_cnn}` instead of `modular_mlp`.

- **Data and roles.** The same 200 writers, `data_id` and role seed, so the same 40 test writers (8,665 images) and 160 training writers (35,786 images).
- **Preprocessing.** Unchanged: pixels standardised with the training writers' statistics, constant pixels dropped (698 of the 784 kept). `torch_module` puts each kept `x_<i>` column back at pixel `i` of the 28x28 image and leaves the 86 dropped pixels at 0.
- **Model.** LEAF's FEMNIST CNN (Caldas et al., 2018): 5x5 convolutions of 32 and 64 filters, each followed by 2x2 max pooling, a dense layer of 2048 and a 62-class head. The first convolution is the adapter (`adapter.femnist`), the rest of the body the shared trunk, the last layer the head (`head.character`). 6,603,710 parameters, 58 times the MLP's 113,982.
- **Federation and training.** Unchanged: four fogs under one cloud (`femnist_four_fogs`, `npz` codec), FedAvg at both levels, 30 rounds of 1 local epoch (Adam, lr 0.001, batch 32), seeds 0–2. No hyperparameter was tuned for the CNN.

| Identity | Value |
|---|---|
| `topology_id` | `80d45cbee3ed481093da8f2e29f7fde652ac1049eec791d7406cd7e3e1ac2067` |
| `config_id` | `0dd4b57bba9ae7e1c9770be376d30161754a49c873e735e86ca565e18e925c0b` |
| `data_id` | `7d8238e25baf310766da9c287196bd4e42a45a3a709873d21edc89830d10ad20` |
| code | commit `8ab0ba5`, clean tree |

| Seed | `run_id` | `run_hash` |
|---|---|---|
| 0 | `20261110T194738Z-c20094805c97` | `b2c0090b82755cd96f8163e51982dc9cfecbe4c9cad70218a7774145fc056a87` |
| 1 | `20261110T213011Z-c391efc4a77d` | `8a7fc0691be63305f28d818efab9f983a85977c5cbe88a5f2f3099f7894e17c3` |
| 2 | `20261110T224046Z-9cbe82b852a1` | `386ceed9180b0b68b8063505febc713039cdc91c05ce288fe03c19ac7081b928` |

## Results on the test writers

Round 30, mean ± 95% CI (Student's t) over 3 seeds:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Federated LEAF CNN (`torch_module`, this run) | **0.712 ± 0.025** | **0.413 ± 0.043** |
| Federated MLP (`modular_mlp`, [reference](../femnist_reference/INDEX.md)) | 0.519 ± 0.003 | 0.153 ± 0.002 |
| Centralised logistic regression (reference's `baselines.json`) | 0.661 | 0.472 |
| Always "1" (the training bag's majority class) | 0.060 | 0.002 |

Per seed, accuracy 0.704, 0.708, 0.723 and macro-F1 0.397, 0.410, 0.431.

- **The CNN clearly beats the MLP** on the same test writers: +0.19 accuracy and 2.7 times the macro-F1, with intervals far apart. It also passes centralised logistic regression in accuracy, though not in macro-F1.
- **It learns faster too.** Test accuracy at rounds 5, 10, 20 and 30 was 0.41, 0.53, 0.65 and 0.70 for seed 0, against 0.24, 0.38, 0.49 and 0.52 for the MLP's seed 0. It was still rising at round 30.
- **Macro-F1 stays low** for the reason given in the reference: most letters are rare. The wider interval comes from those rare classes.
- **Cost.** The model is 26 MB in float32, so the 12,852 messages of a run came to 292 GB encoded (5.1 GB for the MLP). A run took 65 to 102 minutes of wall time on CPU, one seed at a time, and its process held 17 to 26 GB of memory: every edge keeps its own model and copies of the state. Running the seeds in parallel did not fit in this machine's 32 GB.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`) and `summary.json` of each seed.
- `report.html`: accuracy and macro-F1 per round, mean ± CI over the seeds.

`events.jsonl` (32 MB per run), the continuation bundle (5.1 GB per run) and `model.npz` (26 MB per run) are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. The SHA-256 of each final `model.npz` was:

| Seed | `model.npz` SHA-256 |
|---|---|
| 0 | `688b1dc147a6098a6ffab7e8486e8380e5b019a2a3e72cc76f4a549aea5f4032` |
| 1 | `f47f420ffd215f89fd9a59424d7b8ffcb5ebf99fa9585d2e7b2d37fffb9c7f58` |
| 2 | `6b775d6e18e546a2b606d4b7862c5c0c3022ce5d190c3276e3286e636fabfbb0` |
