# Technique comparisons with ten seeds and paired contrasts (#293)

The README's claims of the form "X beats FedAvg" rested on 3 seeds and on intervals that did not overlap. With 3 seeds, the smallest possible sign-flip p-value is 0.25, so none of them could be significant. Four experiments were rerun here with seeds 0–9, unchanged otherwise: `techniques_drift`, `techniques_server`, `techniques_personalisation` and `mix_swell_wesad`. The four experiments have the same `config_id`s as before, because the seeds are not part of a config's identity. Every arm is paired with the reference on the runs that share a seed.

```bash
onion_fl run experiments/<experiment>.yaml --workers 5      # the four, at 049242f
python scripts/paired_techniques.py runs --out results/techniques_paired
```

## Method

- **Metric.** Final global macro-F1 at round 20 on the test subjects (SWELL 1, 2, 12, 18, 19; WESAD S6, S7, S10), per dataset. For the edges, the per-edge scores combined by samples: `local/*` is the model each edge trained, and `personal/*` is the personal model of Ditto and APFL.
- **Reference.** FedAvg in each experiment (`standard` trainer, `replace` server, `fedavg` sharing), and α = 0.5 for the placement sweep. Personal models are compared with FedAvg's `local/*`, as the README table does.
- **Statistics** (`onion_fl.observability.stats`):
  - Δ is the mean paired difference, with a 95 % paired-bootstrap CI, and `d_z` is Cohen's paired d;
  - `p_adj` is the exact sign-flip p (1024 sign patterns), Holm-adjusted over the arms of one experiment and series;
  - `p_tost` is the TOST p for a margin of ± 0.02, the margin of the degradation study (#290).
- **Verdict.** *equivalent* when TOST < 0.05; else *different* when p_adj < 0.05; else *inconclusive*.
- **Smallest p.** With 10 pairs, the smallest raw p is 2/1024 ≈ 0.002, so p_adj is at least 0.010–0.012 with 5–6 arms.

## Results

Δ is the arm's mean minus the reference's. The full table, with the bootstrap CIs, `sd_diff`, Wilcoxon and the TOST p, is `contrasts.csv`.

### Drift (`techniques_drift`, segregated, lossless, FedAvg = `standard`)

| Technique | SWELL Δ (p_adj) | WESAD Δ (p_adj) | Edge, trained Δ (p_adj) |
|---|---|---|---|
| FedProx | −0.002 equivalent | −0.001 equivalent | +0.001 equivalent |
| SCAFFOLD | **−0.080** (0.010) different, d_z −2.7 | **+0.073** (0.010) different, d_z +2.7 | **−0.028** (0.010) different |
| FedNova | +0.014 (0.010) equivalent | +0.007 equivalent | +0.009 equivalent |
| FedDyn | +0.036 (0.065) inconclusive | **+0.121** (0.010) different, d_z +4.7 | **+0.032** (0.039) different |
| MOON | −0.001 equivalent | −0.002 equivalent | −0.009 equivalent |

FedAvg's means are SWELL 0.549, WESAD 0.751 and edge 0.584.

### Server side (`techniques_server`, FedAvg = `replace`)

| Server side | SWELL Δ (p_adj) | WESAD Δ (p_adj) |
|---|---|---|
| FedAvgM | **−0.058** (0.012) different | **+0.051** (0.023) different |
| FedAdam | **+0.058** (0.012) different | **+0.084** (0.016) different |
| FedYogi | **+0.048** (0.012) different | **+0.118** (0.012) different |
| FedAdagrad | **+0.068** (0.012) different | **+0.132** (0.012) different, d_z +3.3 |
| FedBuff-style buffering | **−0.125** (0.012) different | −0.021 (0.16) inconclusive |
| FedBuff-style + FedAsync mixing | **−0.077** (0.016) different | +0.012 (0.30) inconclusive |

FedAvg's means are SWELL 0.549 and WESAD 0.751.

### Personalisation (`techniques_personalisation`, α = 0.5, FedAvg = `fedavg` + `standard`)

| Technique | Global SWELL Δ | Global WESAD Δ | Edge, trained Δ | Edge, personal Δ |
|---|---|---|---|---|
| FedPer | **−0.076** different | **−0.055** different | −0.003 equivalent | — |
| LG-FedAvg | **−0.333** different | **−0.308** different | **+0.061** (0.012) different, d_z +2.3 | — |
| Ditto | 0.000 equivalent | 0.000 equivalent | 0.000 equivalent | **+0.019** (0.018) different |
| APFL | 0.000 equivalent | 0.000 equivalent | 0.000 equivalent | **+0.054** (0.004) different, d_z +2.7 |
| FedRep | **−0.101** different | **−0.244** different | −0.013 inconclusive | — |
| FedBABU | −0.011 (0.023) equivalent | +0.001 equivalent | −0.010 equivalent | — |

- **Reference values.** FedAvg's means are global SWELL 0.617, global WESAD 0.731, edge 0.638. Every significant row here has p_adj ≤ 0.012 unless shown.
- **Ditto and APFL train FedAvg's global model unchanged.** Their global and `local/*` scores equal FedAvg's bit for bit.
- **The global models of FedPer, LG-FedAvg and FedRep are not meaningful scores.** They keep groups on the edge. The rows are given for completeness.
- **FedBABU's −0.011 on SWELL** is detectable (p_adj 0.023) but within the margin.
- **Ditto's personal gain is small.** Its CI, [+0.007, +0.032], crosses the margin.

### Placement (`mix_swell_wesad`, reference α = 0.5)

| α | SWELL Δ | WESAD Δ | Both datasets Δ |
|---|---|---|---|
| 0.0 (segregated) | −0.005 equivalent | +0.013 inconclusive | +0.000 equivalent |
| 1.0 (mixed) | −0.004 equivalent | +0.005 inconclusive | −0.001 equivalent |

## What changes in the README

- **Drift.**
  - FedDyn and SCAFFOLD beat FedAvg on WESAD: confirmed.
  - "On SWELL no technique separates from FedAvg" is replaced:
    - SCAFFOLD is **worse** than FedAvg on SWELL (−0.080) and on the edges (−0.028);
    - FedDyn is better on the edges (+0.032), and inconclusive on SWELL;
    - FedProx, FedNova and MOON are equivalent to FedAvg everywhere.
- **Server.**
  - FedAdagrad, FedYogi and FedAdam beat FedAvg on WESAD: confirmed.
  - "On SWELL they overlap" is replaced: with 10 paired seeds, all three also beat FedAvg on SWELL.
  - FedAvgM helps on WESAD and hurts on SWELL.
  - Both FedBuff-style variants are worse on SWELL.
- **Personalisation.**
  - APFL's personal model and LG-FedAvg beat FedAvg on the edges: confirmed.
  - Ditto's personal model is better by a small margin.
  - "The global model is not hurt" holds for Ditto and APFL, which are bit for bit FedAvg's, and for FedBABU, which is equivalent within ± 0.02.
- **Placement.** "No measurable effect" becomes:
  - equivalent within ± 0.02 on SWELL and on both datasets together;
  - inconclusive on WESAD.

## Underpowered at n = 10

These *inconclusive* verdicts are no evidence of a tie:

| Contrast | sd_diff | Power at 0.02 | Seeds for 0.02 |
|---|---|---|---|
| Drift, SWELL, FedDyn | 0.044 | 0.25 | 40 |
| Server, WESAD, FedBuff-style | 0.037 | 0.33 | 29 |
| Server, WESAD, FedBuff-style + FedAsync | 0.030 | 0.48 | 20 |
| Personalisation, edge trained, FedRep | 0.017 | 0.91 | 8 |
| Placement, WESAD, α = 0 | 0.039 | 0.31 | 32 |
| Placement, WESAD, α = 1 | 0.040 | 0.29 | 34 |

FedRep's edge contrast is well powered, so it is close to a tie: its CI is [−0.023, −0.003].

## Not rerun

`techniques_drift_lossy`, `techniques_robustness` and `techniques_privacy` keep their 3 seeds. Their README comparisons are descriptive, and the README says so.

## Identity

| | |
|---|---|
| code | commit `049242f`, clean tree, for all 230 runs (`run_metrics.csv`: `commit`, `dirty`) |
| experiments | `techniques_drift` (6 `config_id`s), `techniques_server` (7), `techniques_personalisation` (7), `mix_swell_wesad` (3); seeds 0–9 |
| topologies | `ff55fe4d…` (drift, server), `9033d1cf…` (personalisation, placement) |
| runs | `<experiment>/<run_id>/`: signed `run.json` and `summary.json` of every run. Models and event files are not committed |
| tables | `run_metrics.csv` (one row per run and series, with `run_id`, `run_hash`, `config_id`, `topology_id`, `data_id`); `contrasts.csv` |

- **Five runs were rerun.** They are the `mix_swell_wesad` runs that started while two untracked files were in the tree, so they were marked `dirty`.
- **They were rerun at `049242f` with a clean tree.** Their models and scores are identical bit for bit, and the replaced runs are not committed.
- **The 3-seed results remain where they were.** They are in `results/<experiment>/`, cited by their own commits.
