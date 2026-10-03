# Robust aggregation under attack (new framework, real data)

`experiments/techniques_robustness.yaml` was run on 2026-10-03 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #149.

```bash
onion_fl run experiments/techniques_robustness.yaml --workers 3
onion_fl report <folder with the 72 runs of commit 30ea239> --out report.html --metric macro_f1 --by scenario
```

## Setup

- **Topology and placement.** `four_fogs_swell_wesad_lossless`, the latencies and bandwidths of `four_fogs_swell_wesad` without message loss, so that lost rounds do not mix with the attack. The `mixing` placement is at α = 0.5. Each edge holds one subject, and the fogs hold 6 or 8 training edges.
- **Aggregators.** Every fog (a leaf aggregator) runs the scenario's aggregator; the cloud runs FedAvg. Bulyan picks its θ = n − 2f candidates with Krum one at a time, removing each pick before the next (El Mhamdi et al., 2018), then trims coordinate-wise around the median.
- **Training.** SGD with lr 0.3, 10 local epochs, 20 rounds, seeds 0–2.
  - 0.3 was FedAvg's rate on validation in the first, lossy drift selection.
  - The later lossless drift selection preferred 0.1 for FedAvg (0.654 against 0.640, at α = 0; `../techniques_drift/validation.csv`).
  - The rate was not re-selected for this setting (α = 0.5); every scenario here shares it.
- **Attacks.**
  - In each dataset, 20% of the edges, picked with the run seed, flip the sign of their update: they send x − s·(y − x).
  - At s = 1 the honest majority still dominates the mean. At s = 5 the averaged update, 0.8Δ − 0.2·5Δ, points against learning. s was set by that arithmetic, not tuned on test scores.
- **Defence parameters.** They come from the assumed 20% attacker fraction, not from test scores.
  - The 20% is drawn per dataset, not per fog, so a fog holds 0 to 4 attackers (4 of 8 in `fog_a2` with seed 2; `diagnostic.selection` events). The parameters below assume about 1–2.
  - The trimmed mean cuts β = 0.2 per side.
  - Krum and Multi-Krum tolerate f = 2, lowered to 1 in the 6-edge fogs (Krum needs n ≥ 2f + 3).
  - Bulyan needs n ≥ 4f + 3, so f = 1, which the 6-edge fogs lower to 0. About half of Bulyan's selections ran at f = 0, i.e. without protection (`selections_f0` in `run_metrics.csv`).
- **Clipping bound.** `norm_clip` uses 1.0, the median edge update norm under FedAvg without attack in the first, lossy run of this comparison (seed 0: 464 updates, median 1.02, interquartile range 0.82–1.34). It is a training statistic.
- **Selection aggregators keep every dataset's keys.**
  - When a fog drops every edge that holds a key (a dataset's adapter), that key comes from its best-ranked holders, which are reported as its sources.
  - A dropped edge can therefore still shape the model through a rescued key. The edges that shaped nothing are reported apart, as excluded.
  - Krum keeps one edge per fog, so it rescues the other dataset's keys in most rounds.
- **Code.** Commit `30ea239`, clean tree, for all 72 runs. See "Reproducibility" below.

`topology_id` `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa`, `data_id` `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e`. Each run's `config_id` is in its `run.json`.

## Results (global macro-F1, mean ± 95% CI over 3 seeds; `run_metrics.csv`)

| Fog aggregator | SWELL, no attack | WESAD, no attack | SWELL, s = 1 | WESAD, s = 1 | SWELL, s = 5 | WESAD, s = 5 |
|---|---|---|---|---|---|---|
| FedAvg | 0.578 ± 0.109 | 0.751 ± 0.000 | 0.564 ± 0.115 | 0.744 ± 0.097 | 0.244 ± 0.000 | 0.392 ± 0.000 |
| Median | 0.587 ± 0.036 | 0.763 ± 0.069 | 0.554 ± 0.075 | 0.752 ± 0.344 | 0.571 ± 0.109 | 0.450 ± 0.963 |
| Trimmed mean (β = 0.2) | 0.584 ± 0.066 | 0.751 ± 0.000 | 0.562 ± 0.111 | 0.736 ± 0.118 | 0.429 ± 0.266 | 0.316 ± 0.515 |
| Krum (f = 2) | 0.582 ± 0.036 | 0.743 ± 0.021 | 0.558 ± 0.066 | 0.677 ± 0.396 | 0.549 ± 0.143 | 0.486 ± 0.768 |
| Multi-Krum (f = 2) | 0.599 ± 0.064 | 0.788 ± 0.099 | 0.576 ± 0.119 | 0.738 ± 0.169 | 0.564 ± 0.104 | 0.445 ± 1.006 |
| Geometric median | 0.587 ± 0.062 | 0.751 ± 0.000 | 0.585 ± 0.147 | 0.773 ± 0.247 | 0.569 ± 0.156 | 0.648 ± 0.752 |
| Bulyan (f = 1) | 0.587 ± 0.043 | 0.754 ± 0.014 | 0.568 ± 0.081 | 0.759 ± 0.185 | 0.503 ± 0.280 | 0.213 ± 0.147 |
| Norm clip (1.0) | 0.589 ± 0.059 | 0.751 ± 0.000 | 0.574 ± 0.125 | 0.766 ± 0.148 | 0.575 ± 0.132 | 0.777 ± 0.127 |

**Intervals wider than [0, 1].** With 3 seeds the t interval uses t = 4.30, so seeds that disagree give half-widths near 1. Under s = 5, the WESAD scores per seed (0, 1, 2) are:

| Fog aggregator | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Median | 0.316 | 0.886 | 0.146 |
| Krum | 0.668 | 0.661 | 0.129 |
| Multi-Krum | 0.248 | 0.910 | 0.175 |
| Geometric median | 0.756 | 0.882 | 0.306 |

**Detection** by the aggregators that select, from the `diagnostic.selection` events, summed over fogs, rounds and seeds.

- **Selection** counts the edges that lost the selection on the keys every child holds. Recall is malicious dropped / malicious aggregated, and precision is malicious dropped / all dropped.
- **Exclusion** counts only the dropped edges that shaped nothing: no rescued key came from them. It is the effective detection. Recall is malicious excluded / malicious aggregated, and precision is malicious excluded / all excluded.

| Aggregator | Attack | Selection recall | Selection precision | Exclusion recall | Exclusion precision |
|---|---|---|---|---|---|
| Krum | s = 1 | 0.947 | 0.237 | 0.792 | 0.237 |
| Krum | s = 5 | 1.000 | 0.242 | 0.912 | 0.265 |
| Multi-Krum | s = 1 | 0.367 | 0.367 | 0.256 | 0.321 |
| Multi-Krum | s = 5 | 0.746 | 0.742 | 0.638 | 0.783 |
| Bulyan | s = 1 | 0.208 | 0.312 | 0.097 | 0.236 |
| Bulyan | s = 5 | 0.349 | 0.500 | 0.229 | 0.644 |

`run_metrics.csv` also counts, per run, how many rescued-key sources were malicious (`malicious_rescue_sources`).

**Lost rounds.** The links lose nothing, yet some s = 5 runs lose cloud rounds. In those runs the attack drives some edges to non-finite weights; they roll back and send an empty update, and a fog with quorum 1.0 then fails the round. Lost cloud rounds per seed (0, 1, 2):

- FedAvg: 8, 10, 11.
- Bulyan: 8, 1, 8.
- Krum: 0, 10, 7.
- Trimmed mean: 3, 3, 7.
- Multi-Krum: 0, 1, 2.
- Median: 0, 0, 1.
- Every other run: none.

## Reading

- **The strong attack (s = 5) collapses FedAvg**, to SWELL 0.244 and WESAD 0.392 in every seed. The trimmed mean does not stop it either (0.429 and 0.316). It trims one value per side in these fogs, and a fog can hold up to four attackers.
- **In these runs, norm clipping is the most stable defence against the strong attack on both datasets.** It keeps SWELL near the clean scenario (0.575 against 0.589) and avoids the WESAD degradation seen with the other robust aggregators (0.777). Clipped to the honest median norm, a ×5 reversed update weighs no more than an honest one, and the 80% honest majority wins.
- **The median, Krum, Multi-Krum and the geometric median hold SWELL under the strong attack (0.549–0.571), but their WESAD scores depend on the seed.** Each fog holds few WESAD edges at α = 0.5, so one bad selection decides the WESAD keys of a round, and how many attackers land in each fog changes with the seed.
- **Bulyan degrades WESAD the most (0.213).** About half of its selections ran at f = 0 because the 6-edge fogs cannot satisfy n ≥ 4f + 3. Its one-at-a-time selection also drops fewer attackers than the single Krum ranking it used before (selection recall 0.35 against 0.55 at s = 5, commit `5995082`).
- **Selection detects the strong attackers, but rescued keys let some back in.**
  - Multi-Krum drops 75% of the malicious updates from its selection and fully excludes 64%, with 78% of what it excludes being malicious.
  - Krum's selection recall of 1.0 falls to 0.91 once rescued keys count. Its precision of 0.24–0.27 is near the attacker base rate, because it keeps a single update per fog.
  - The weak attackers (s = 1) are much harder to tell apart.
- **The weak attack (s = 1) barely hurts FedAvg** (SWELL 0.578 → 0.564), so it does not separate the defences.
- **Next step.** Three seeds leave the WESAD intervals under attack very wide; more seeds would be needed to rank the selection aggregators there.

## Reproducibility

- **Across #148's fixes.** The comparison first ran at commit `5995082`, before #148's fixes were merged in. All 72 runs were repeated at the merge `0a4a6ad` on a clean tree, and gave the same final models (bit for bit), final metrics and lost cloud rounds. #148 changes what edges announce and send, but not what a standard-trainer run computes.
- **Across #149's review fixes.** The runs above are from `30ea239`, which adds the excluded and rescued-source diagnostics and Bulyan's one-at-a-time selection. The 63 runs without Bulyan give the same final models (bit for bit), metrics and lost rounds as at `5995082`. The 9 Bulyan runs changed, as expected.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of each of the 72 runs.
- `run_metrics.csv`: per run, the global macro-F1 per dataset and `quorum_failed`. At the fogs, it also counts:
  - dropped, excluded, malicious-dropped, malicious-excluded and malicious-present updates;
  - clipped updates;
  - selections, and selections at f = 0;
  - rescued keys, and their malicious sources.
- `report.html`: macro-F1 per round by scenario.

`events.jsonl` is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
