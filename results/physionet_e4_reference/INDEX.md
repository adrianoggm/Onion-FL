# PhysioNet Wearable Device: centralised references (new framework, real data)

`experiments/physionet_e4_reference.yaml` (`stress_binary`) and `experiments/physionet_e4_exercise_reference.yaml` (`stress_vs_exercise`) were run on 2026-11-10 on the PhysioNet "Wearable Device Dataset from Induced Stress and Structured Exercise Sessions" v1.0.1 (ODC-By 1.0). The data was the official zip, checked against PhysioNet's `SHA256SUMS.txt`. Issue #240.

```bash
onion_fl run experiments/physionet_e4_reference.yaml --workers 3
onion_fl run experiments/physionet_e4_exercise_reference.yaml --workers 3
onion_fl report experiments/physionet_e4_reference.yaml --out report_binary.html --metric accuracy --metric macro_f1
onion_fl report experiments/physionet_e4_exercise_reference.yaml --out report_exercise.html --metric accuracy --metric macro_f1
onion_fl baseline experiments/physionet_e4_reference.yaml --models lr rf > baselines_binary.json
onion_fl baseline experiments/physionet_e4_exercise_reference.yaml --models lr rf > baselines_exercise.json
onion_fl data inspect physionet_e4 > inspect_binary.json
onion_fl data inspect physionet_e4 --option label=three_class > inspect_exercise.json
```

## Setup

- **Data.** `datasets/physionet_e4.yaml`: Empatica E4 at the native rates, 60 s windows with half overlap, and the 30 harmonized E4 features (`acc_x`, `acc_y`, `acc_z`, `bvp`, `eda` and `temp`, each with mean, std, min, max and median). The labels come from the protocol tags. The sessions that `data_constraints.txt` flags are dropped or cut, as the descriptor's header lists.
  - `stress_binary`: 35 participants and 1,446 windows (1,243 rest, 203 stress).
  - `stress_vs_exercise`: 36 participants and 2,643 windows (plus 1,197 exercise). f07, whose stress session is dropped, has exercise windows only.
- **Dataset card.** `data inspect` gives no warning: no feature reaches |r| > 0.95 with the label (`inspect_*.json`).
  - The largest |r| is 0.17 (`bvp_median`) for `stress_binary`.
  - Against the one-vs-rest classes of `stress_vs_exercise`, it is 0.46 for exercise (`bvp_median`) and 0.15 for stress (`acc_z_min`).
- **Roles.** Nine participants are reserved for the test, the same in both experiments: S04, S09, S10, S11, S16, S18, f10, f15 and f17. They are the role-seed-0 draw of a quarter of the 35 `stress_binary` participants, listed explicitly. Six did protocol V1 and three V2.
  - The test holds 319 windows for `stress_binary` (269 rest, 50 stress) and 610 for `stress_vs_exercise` (269, 50, 291).
  - The other 26 participants train (27 with f07 in `stress_vs_exercise`). No validation participants were used.
- **Preprocessing.** The features are standardised with the training participants' statistics, and constant features are dropped.
- **Training.** The `pooled` placement puts every training participant in one edge, under the single fog of `one_fog` (lossless `lan` links). The run is therefore the centralised baseline, with the same model and trainer as a federated run.
  - Model: `modular_mlp` (adapter 64, trunk [64], dropout 0.2).
  - 10 rounds of 5 local epochs, which makes 50 epochs: Adam, lr 0.001, batch 32.
  - Seeds 0–4. The hyperparameters were set before the runs and not tuned.
- **Classical baselines.** `onion_fl baseline` runs logistic regression and a random forest (seed 0) on the same split and the same features.

| Identity | `stress_binary` | `stress_vs_exercise` |
|---|---|---|
| `topology_id` | `18086603f5b1b16c80d6f511ca71272027578268e89353a6f8052f07f69b6322` | same |
| `config_id` | `dbff9b1a8649848fbef455a27493173d943c8c293d2837fc10d6ab1a407e371e` | `099b613750504f877338e6906d9edfd8851c7101edc5780b4fe9479f9f611d8e` |
| `data_id` | `c6dbe7c6bf22fbd1bf5250557116c7c9b124514e59f31becae41b2d51b21e76e` | `0de0516fecbda48d3a84e0daacca117d05b3594e0a32247d7fbb1f3122b3599f` |
| code | commit `bb5781d`, clean tree | same |

| Task | Seed | `run_id` | `run_hash` | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| `stress_binary` | 0 | `20261110T124743Z-608c07032dde` | `f3dabae571211d9390a7bd90714afebb397fbd5fd9947c182c94e4ecc7a0ebd9` | 0.871 | 0.633 |
| `stress_binary` | 1 | `20261110T124743Z-b153ce60d175` | `9b920ff7d10558dd0bc9f7dc07e92f4ffac00bfdb4114f5145619349d8c4ded9` | 0.865 | 0.671 |
| `stress_binary` | 2 | `20261110T124743Z-0078cf24ce14` | `b781ee3818eeef063f624978d851b6ad1fe3f399b78d815ee8b051efa40ee54f` | 0.859 | 0.668 |
| `stress_binary` | 3 | `20261110T124752Z-5a196384ebff` | `17a95a09ed29ddc82a4d5640d4f32f010ce874b0832e6931dedee68b6c78e230` | 0.843 | 0.598 |
| `stress_binary` | 4 | `20261110T124752Z-87af3528a11f` | `99310eea3e9114b101195783f85716829583177ee93e8d38902879a881fb1fca` | 0.834 | 0.569 |
| `stress_vs_exercise` | 0 | `20261110T124835Z-2754b7853b95` | `7458a8f7d27e24f9d6aee945140b2bf1cac23f851636d061f35a4d24246ef8a6` | 0.811 | 0.628 |
| `stress_vs_exercise` | 1 | `20261110T124835Z-4100fbb3057c` | `075aad0f10bcf6f44eb4b1a503c7d8fb21538f52fb21f29bfb25cdfa79b0a5a2` | 0.836 | 0.641 |
| `stress_vs_exercise` | 2 | `20261110T124835Z-a60debbf207e` | `8280139995d0e3b7f2ed6dac450b681cd911d1d956ac88905334ed77a01b0d8e` | 0.800 | 0.637 |
| `stress_vs_exercise` | 3 | `20261110T124847Z-8f2fd6e40a83` | `082ed5d86d2bcd6d2a455503a943aef75218e1f2dc27a813171e685eac855360` | 0.831 | 0.573 |
| `stress_vs_exercise` | 4 | `20261110T124847Z-95b43f57297f` | `dc73d06a96e49be8e2259c13b2d7cf26d8ca1dd659c353b514caf73a5d37af96` | 0.848 | 0.603 |

## Results on the test participants

The MLP rows give the mean ± 95% CI over the 5 seeds (Student's t, 4 degrees of freedom), at round 10.

| Model | `stress_binary` accuracy | `stress_binary` macro-F1 | `stress_vs_exercise` accuracy | `stress_vs_exercise` macro-F1 |
|---|---|---|---|---|
| Pooled MLP | 0.855 ± 0.019 | 0.628 ± 0.055 | 0.825 ± 0.024 | 0.616 ± 0.035 |
| Logistic regression | 0.837 | 0.548 | 0.708 | 0.524 |
| Random forest | 0.787 | 0.514 | 0.854 | 0.662 |
| Always "rest" (the training majority) | 0.843 | 0.457 | 0.441 | 0.204 |

- **Stress against rest is hard on unseen participants.** The MLP's accuracy is barely above always answering "rest", which scores 0.843 because only 16% of the test windows are stress. Its macro-F1 (0.628 ± 0.055) is clearly above the majority's 0.457 and above logistic regression's 0.548. It is still far from WESAD's 0.91 accuracy for centralised logistic regression, and the protocols differ.
- **Stress windows are few.** At 60 s, only the Stroop (V1) and the TMCT give stress windows: the opinion and subtract tasks last 30 s. That leaves 3–6 stress windows per V1 participant and 6–13 per V2 participant. `window_seconds: 30` gives 560 stress windows instead of 203; it has not been run.
- **Exercise separates well, but not from rest alone.** In `stress_vs_exercise`, the random forest (0.662) and the MLP (0.616 ± 0.035) are far above the majority's 0.204 macro-F1. The exercise windows come from separate recordings (the aerobic sessions), so part of what separates them may be the session, not only the effort. The task exists because exercise raises EDA and HR like stress does; the session effect is a limitation of this design.
- **The seeds disagree.** The MLP's macro-F1 spans 0.569–0.671 for `stress_binary`, so a difference smaller than about 0.05 against another method is not measurable with these 5 seeds.
- **Logistic regression and the forest are one run each**: logistic regression is deterministic, and the forest is seed 0 only. Their numbers have no interval.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`), `summary.json` and `model.npz` of each run.
- `report_binary.html`, `report_exercise.html`: accuracy and macro-F1 per round, mean ± CI over the seeds.
- `baselines_binary.json`, `baselines_exercise.json`: logistic regression and random forest on the same split.
- `inspect_binary.json`, `inspect_exercise.json`: the dataset cards.

`events.jsonl` (about 0.4 MB per run) and the continuation bundle are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies; it passed on the ten original run folders.
