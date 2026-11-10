# Contribution scores under noisy edges (new framework, real data)

`experiments/contribution_noise.yaml` was run on 2026-11-10 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #230.

```bash
onion_fl run experiments/contribution_noise.yaml --workers 3
onion_fl explain runs/*                      # explain.json per run
python scripts/contribution_auc.py runs --out results/contribution_noise
```

## Setup

- **Topology and placement.** `four_fogs_swell_wesad_lossless`, `mixing` placement at α = 0.5, one subject per edge, 6 to 8 training edges per fog. Every aggregator runs FedAvg and the `contribution` diagnostic (on by default; every fog has ≤ 10 children, so its Shapley values are exact).
- **Training.** SGD, lr 0.3, 10 local epochs, 10 rounds, seeds 0–4; as `techniques_robustness`, with half the rounds.
- **Noisy edges.** In each dataset a seeded 20% of the training edges (6 per run, listed in the `data.attack` event) either flip their labels (`label_flip`: y → n_classes − 1 − y, real labels, noisy data) or send Gaussian noise instead of their update (`gaussian`, σ = 0.01 per coordinate). σ was fixed before running and not tuned on the scores. A third scenario has no attack.
- **Statistic, fixed before running** (`scripts/contribution_auc.py` at commit `f257ac4`). The fogs' scores only. An edge's score in a round is its mean over the groups it sends (`all`), or its `trunk` score. AUC = P(honest score > noisy score) over the pairs in the same fog and round, ties ½; 0.5 is chance. Mean over seeds with a 95% t interval (t = 2.78, 5 seeds).
- **Added after the first look** (exploratory): `edge_auc_*`, the same AUC with each edge's score averaged over the 10 rounds (pairs within a fog).
- **Code.** Commit `f257ac4`, clean tree, for all 15 runs. `topology_id` `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa`, `data_id` `9fd573754157d68b111f037ecd891bdb62abcbff1d5e5e818da789dde470257c`.

| Scenario | `config_id` |
|---|---|
| no attack | `27eda288eeb1885e6ea209cbf51d3acea663e1c5786e7a22631a5f20be19e4b1` |
| `label_flip` | `d2254d008132c860230fcd017471ec453a38a0f78a38b596b48b9fed7313b0c1` |
| `gaussian` | `1c2bcf0acfca9848a29da55e15d89f20b28260aadf136b88b72d05ba5098cb7d` |

`run_id` and `run_hash` of each run are in `runs.csv` and in its folder.

## Results (`summary.csv`, `runs.csv`)

Primary statistic, per fog and round:

| Noisy edges | Score | AUC (all groups) | AUC (trunk) |
|---|---|---|---|
| `gaussian` | Shapley | 0.772 ± 0.054 | 0.755 ± 0.122 |
| `gaussian` | leave-one-out | 0.780 ± 0.052 | 0.758 ± 0.118 |
| `label_flip` | Shapley | 0.697 ± 0.238 | 0.704 ± 0.231 |
| `label_flip` | leave-one-out | 0.704 ± 0.240 | 0.702 ± 0.232 |

Exploratory, per edge over the run (all groups): Shapley 0.906 ± 0.028 (`gaussian`) and 0.731 ± 0.236 (`label_flip`).

- **Gaussian noise ranks below the honest edges in every seed** (per-seed Shapley AUC 0.70–0.82; per edge 0.88–0.94). Its mean Shapley score is −0.011 ± 0.005 against +0.089 for honest edges.
- **Label flipping ranks lower on average, but not in every seed.** Per-seed Shapley AUC: 0.57, 0.79, 0.43, 0.90, 0.79 (seeds 0–4). With seed 2 the flipped edges beat the honest ones in most pairs of the same fog and round (AUC 0.43), although their mean score is lower (+0.051 against +0.077). The interval includes 0.5, so for label flipping this experiment does not establish a lower score.
- **Honest edges score low too** (≈ 0.08 on average): with one subject per edge and α = 0.5 the honest updates are far from aligned, so the score separates "agrees with the others" from "does not", not good data from bad.
- **Cost to the model.** Global macro-F1 after 10 rounds: 0.613 ± 0.016 without attack, 0.610 ± 0.030 with `gaussian`, 0.594 ± 0.069 with `label_flip`.

## Explainability (`explain.csv`, `explain.json` per run)

Permutation importance of the final global model on the `val` subjects (2 per dataset: SWELL 10 and 25, WESAD S4 and S11), macro-F1, 5 permutations per feature within each subject. Without attack, over the 5 seeds:

- **SWELL** (baseline 0.531 ± 0.057): `snkeystrokes` 0.051 ± 0.027, `snchars` 0.037 ± 0.028, `snmousedistance` 0.016 ± 0.023; the rest within ±0.03 of 0.
- **WESAD** (baseline 0.983 ± 0.015): `bvp_0_median` 0.170 ± 0.047; every other feature ≤ 0.011.

## Caveats

- Five seeds, one topology, one placement, one noise level each. The result for `label_flip` is not conclusive.
- The `val` subjects are only two per dataset, so the importances describe those subjects.
- On WESAD the model leans almost entirely on the window median of the BVP signal, and scores 0.98 on the two val subjects. That deserves a look for a subject- or session-level signal (docs/RULES.md §2) before citing WESAD baselines; it was not investigated here.
- `events.jsonl` (2.5 MB per run) is not committed, so `run_hash` cannot be recomputed from this folder; `runs.csv` holds the scores derived from it.
- The `edge_auc_*` columns came from a later version of the script (committed after the runs); the runs and the primary statistic did not change.
