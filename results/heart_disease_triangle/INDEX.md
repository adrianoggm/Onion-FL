# Fed-Heart-Disease: FLamby's triangle (new framework, real data)

`experiments/heart_disease_triangle.yaml` was run on 2026-11-10 on the four UCI Heart Disease files that FLamby downloads (MD5 sums checked), written with FLamby's preprocessing and train/test split by `scripts/heart_disease_flamby.py`. Issue #251.

```bash
onion_fl run experiments/heart_disease_triangle.yaml --workers 3
python scripts/score_test_subjects.py results/heart_disease_triangle/*/ > per_center.json
onion_fl report experiments/heart_disease_triangle.yaml --out report.html --metric accuracy --metric macro_f1
onion_fl baseline experiments/heart_disease_triangle.yaml --models lr > baselines.json
```

## Setup

- **Data.** Four hospitals, 740 patients and FLamby's 13 features. Each hospital's train split is one client, with 199 (Cleveland), 172 (Hungarian), 30 (Switzerland) and 85 (VA) patients.
- **Test.** The official FLamby test split of each hospital: 104, 89, 16 and 45 patients (254 in all). It is the same in every arm and is scored per hospital (docs/RULES.md §2). No validation set and no tuning were used.
- **Preprocessing.** The features are standardised with the training rows of each arm. Constant features are dropped: `chol` is 0 in every Switzerland row, so it is dropped in that local arm.
- **Model and trainer.** All arms use the same model and trainer: `modular_mlp` with one hidden layer (adapter 32, no trunk, no dropout) and a two-class head, trained for 50 rounds of 1 local epoch with Adam (lr 0.001, batch 4). Each arm sees every training row 50 times, which is FLamby's pooled and local recipe (50 epochs). Seeds 0–4.
- **Arms.**
  - **Pooled:** the four train splits in one edge.
  - **Local:** one hospital's train split alone, with the other three excluded.
  - **FedAvg:** one edge per hospital, `fog_us` = Cleveland + VA and `fog_eu` = Hungarian + Switzerland, with FedAvg at both levels.
- **Topologies.** Pooled and local run on `heart_disease_one_fog`. With `pooled` placement on two fogs, the fog left without edges never registers, and the run never starts.

| Identity | Value |
|---|---|
| `topology_id`, pooled and local (`heart_disease_one_fog`) | `54cd664efd6fd812626a9751ba7689a8a66ae5dbc2e8f7b37e2af6529051ee8b` |
| `topology_id`, FedAvg (`heart_disease_two_fogs`) | `6f49bb9fa961df752b183adf750c8f704fd98c021131b10d593772b71cd56ded` |
| `config_id`, pooled | `7594b0d894e56210fa99eac0b7c4054801b250b7ce0f1cba661214f8a129f3a5` |
| `config_id`, local Cleveland | `3874d24cbcc7e6945af6cc631fa42ce04614f313eca3431a55b825364df04087` |
| `config_id`, local Hungarian | `5ede5bd7e1d475a8ec22a601da97f075b24a6306c0e3dbd05125985945d121ba` |
| `config_id`, local Switzerland | `ed135709f41f6cf3f181672bf55b2da094de3239419d4e449ba3347b1898ae39` |
| `config_id`, local VA | `b7031f33ddf35aeb051f8772726e59dbe1e70ce1e0c7628ff65bfeb07dbbebf4` |
| `config_id`, FedAvg | `6704195151b665edfa4c58145cdd6ae3f06ab334065313769fdf252db2d9f10f` |
| `data_id` | `a0e7a725e8dc42ff8f6e540d50dc8b4571b1c2b6dba72aebd538cdcd03f60b01` |
| code | commit `0fa2b87`, clean tree |

## Results on each hospital's test split

Accuracy of the final global model (round 50), as the mean ± 95% CI (Student's t) over 5 seeds. "Mean of hospitals" is the unweighted mean of the four columns. "All 254" is the run's own `global/*` accuracy over the four test splits together.

| Arm | Cleveland (104) | Hungarian (89) | Switzerland (16) | VA (45) | Mean of hospitals | All 254 |
|---|---|---|---|---|---|---|
| Pooled | 0.756 ± 0.026 | 0.807 ± 0.012 | 0.887 ± 0.065 | 0.831 ± 0.042 | 0.820 ± 0.018 | 0.795 ± 0.014 |
| Local Cleveland | 0.744 ± 0.014 | 0.710 ± 0.033 | 0.650 ± 0.118 | 0.711 ± 0.065 | 0.704 ± 0.042 | 0.720 ± 0.015 |
| Local Hungarian | 0.742 ± 0.013 | 0.757 ± 0.023 | 0.362 ± 0.101 | 0.582 ± 0.023 | 0.611 ± 0.026 | 0.695 ± 0.013 |
| Local Switzerland | 0.462 ± 0.000 | 0.371 ± 0.000 | 0.938 ± 0.000 | 0.778 ± 0.000 | 0.637 ± 0.000 | 0.516 ± 0.000 |
| Local VA | 0.592 ± 0.115 | 0.499 ± 0.123 | 0.713 ± 0.104 | 0.738 ± 0.030 | 0.635 ± 0.049 | 0.593 ± 0.088 |
| FedAvg (2 fogs) | 0.796 ± 0.031 | 0.796 ± 0.029 | 0.613 ± 0.101 | 0.782 ± 0.023 | 0.747 ± 0.039 | 0.782 ± 0.024 |
| Centralised logistic regression (`baselines.json`) | — | — | — | — | — | 0.795 |

- **FedAvg comes close to pooled over the 254 test patients:** 0.782 ± 0.024 against 0.795 ± 0.014, and the intervals overlap. It beats every local-only arm, the best of which is Cleveland's (0.720 ± 0.015).
- **On Switzerland, FedAvg does worse than pooled** (0.613 ± 0.101 against 0.887 ± 0.065), so FedAvg's mean of hospitals stays below pooled's (0.747 against 0.820). With only 16 test patients, one patient moves that column by 0.0625.
- **The local Switzerland model is a constant predictor.** Its 30 training patients all have the disease, so it answers "disease" for every patient, whatever the seed. Its scores are the disease rates of the test splits: 0.462 on Cleveland and 0.938 on Switzerland (15 of 16).
- **Small test sets make the intervals wide.** The Switzerland column is the extreme case, with 16 patients, and 5 seeds only measure the training variance on a fixed split.

## Comparison with FLamby

FLamby publishes the per-seed numbers behind its benchmark figure: `flamby/results/results_benchmark_fed_heart_disease/heart_seed{42..46}.csv` on GitHub (main branch, read on 2026-11-10). Below is the mean over its 5 seeds, computed from those files.

| FLamby arm | Cleveland | Hungarian | Switzerland | VA | Mean of hospitals | All 254 |
|---|---|---|---|---|---|---|
| Pooled | 0.783 | 0.764 | 0.750 | 0.644 | 0.735 | 0.795 |
| Local 0 (Cleveland) | 0.746 | 0.760 | 0.725 | 0.658 | 0.722 | 0.784 |
| Local 1 (Hungarian) | 0.746 | 0.746 | 0.525 | 0.498 | 0.629 | 0.727 |
| Local 2 (Switzerland) | 0.481 | 0.391 | 0.725 | 0.671 | 0.567 | 0.483 |
| Local 3 (VA) | 0.577 | 0.467 | 0.800 | 0.702 | 0.637 | 0.611 |
| FedAvg (100 local steps) | 0.731 | 0.719 | 0.650 | 0.640 | 0.685 | 0.748 |

- **The protocols are not the same.**
  - FLamby's model is a logistic regression with a sigmoid and BCE; here it is a one-hidden-layer MLP with a two-class softmax.
  - FLamby's FedAvg is flat (four clients under one server), with 15 rounds of 100 local SGD steps at lr 0.001. Here it is hierarchical (two fogs), with 50 rounds of one Adam epoch.
  - FLamby scales each federated or local client with its own train statistics, and its pooled arm with the pooled ones. Here every arm is scaled with the statistics of its training rows: the pooled ones for FedAvg.
  - FLamby's seeds are 42–46; ours are 0–4.
- **Over all 254 patients, our pooled arm matches FLamby's** (0.795 and 0.795), and so does the centralised logistic regression (0.795). The ordering of the local arms is the same: Cleveland is best, then Hungarian, then VA and Switzerland.
- **Our FedAvg is closer to pooled than FLamby's** (0.782 against 0.748), but it trains with Adam and hierarchically. FLamby's tuned SGD rate (lr 0.001) was not reproduced, so this is not a reproduction of its FedAvg number.
- **Per hospital the two differ more,** for example on VA (0.831 against 0.644 for pooled), with 45 test patients. FLamby's local Switzerland arm is not a constant predictor either: 0.481 on Cleveland, where the disease rate is 0.462. The split is the same (row by row), so the model, the scaling and the seeds are what differ. Which of them causes the gap was not measured.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`), `summary.json` and `model.npz` of each run.
- `per_center.json`: the final model of each run scored on each hospital's test split, by `scripts/score_test_subjects.py`. The script rebuilds the split from `run.json` and checks that its weighted mean equals the run's `global/*` accuracy.
- `report.html`: accuracy and macro-F1 per round, mean ± CI over the seeds, per arm.
- `baselines.json`: centralised logistic regression on the pooled train splits.

| Arm | Seed | `run_id` | `run_hash` |
|---|---|---|---|
| Pooled | 0 | `20261110T121712Z-ab32658ced50` | `dc8c0a03e6e48eabc6fe41b95db143fec2f44e2bdb0e9fbf79f7ec1773dd0e59` |
| Pooled | 1 | `20261110T121712Z-91043e5c0225` | `de51dae6a734744351fa9967d6c4f37451bcd543d06ca8c3a3784dde67502aea` |
| Pooled | 2 | `20261110T121712Z-e1767bfd1637` | `10b1c187601ec05e6c020fd7f8ad0e424a0aebbc65192533aa523d227f21df8d` |
| Pooled | 3 | `20261110T121736Z-37946e0bc0fb` | `2b4b2839e64d7e96225191249c42ca6213542bff5483c21f4abf71866344ff2e` |
| Pooled | 4 | `20261110T121736Z-d870f968f21e` | `03e585e5176e33070212259f6d1fe044ce9826373bfe1c25642176b0e13eed75` |
| Local Cleveland | 0 | `20261110T121736Z-94667d79dff8` | `90addf7178a32e9e30dd1f85a9f08e40b335c252a626583f5aefa6fedfc25fab` |
| Local Cleveland | 1 | `20261110T121744Z-ee0033e1aecd` | `13f5578c9682856bca108223f44c507d7dfcd971629bda7ce3c0bd1015d19a97` |
| Local Cleveland | 2 | `20261110T121751Z-4c055f12137f` | `ee7f9a4b5e3f1e885e2290963049e07b98285b6b9d345d0470ef7099d5b4c70f` |
| Local Cleveland | 3 | `20261110T121753Z-c2b4c4c417b0` | `eb10152244b24a5b07fea12f6f6d8270438d4544dd270139c7dc4c74fd1bb8af` |
| Local Cleveland | 4 | `20261110T121753Z-2403cee5aaae` | `89473025827c8745bda056f8aa73706ae87a873c2c078e366b0e3486202962f9` |
| Local Hungarian | 0 | `20261110T121802Z-f11c30b9a5bc` | `89285af40b290e1ac46800231681c8a30a82b9c588788dda7d1683f9ee4fd2ac` |
| Local Hungarian | 1 | `20261110T121804Z-08999a642c3c` | `4e2ec4c4d210a07baaa5c269c0d912f52ca6825ae73ae0b92b3f89aeb8f85d7f` |
| Local Hungarian | 2 | `20261110T121805Z-ca7a3e1a1ac3` | `09eb7eeefec687acfe7a5ca827cbc1c8cd62c0be2332d504e86cef0ea0ecf70b` |
| Local Hungarian | 3 | `20261110T121814Z-c4e105293076` | `9797e022620446be96ee1fe331a6b6963e2cd7470d0c144de307cf6570cba17d` |
| Local Hungarian | 4 | `20261110T121815Z-023a5f942f27` | `57d2843bbc56ea077b7d6302a79a5c753c1329eb3a363e0c95a2441fc3c1d1ec` |
| Local Switzerland | 0 | `20261110T121815Z-1def2351bf8c` | `6b6f3012af1fd0e89c7fc2d7c783f320d7c2f9e0cdf87ee08bb4f3f12a29bee2` |
| Local Switzerland | 1 | `20261110T121819Z-b81fafafeaef` | `fdb03513f6c4c69881c86b8d9ddd9fd49fba9f73c6e6e1d8cc90f2e996067761` |
| Local Switzerland | 2 | `20261110T121822Z-3621ecf9697f` | `6feed0036e8cc909f5ee31ca0963ed8c800df7e46f2fafa99c362117e914518c` |
| Local Switzerland | 3 | `20261110T121823Z-f3db70654a35` | `d0652486c965cd7a412e6cc9f9e8662d598419ec3139a3eb027566cfaafebd51` |
| Local Switzerland | 4 | `20261110T121825Z-8c0e4c79885e` | `b6b2e4d7b2197f4b7a87570eca72d653ac9375644a31b62dd3e3bc8a8693bfad` |
| Local VA | 0 | `20261110T121826Z-5b0d36d707b5` | `9366198022159dde36f2b9befd83ca1f4b7a063ced6c4ea32113f8a856a84394` |
| Local VA | 1 | `20261110T121827Z-44620b829487` | `045c4853722591616491f1001663867b60f9dbd79ce83c8a82acd7c90805d94e` |
| Local VA | 2 | `20261110T121829Z-fb2a23b6ce21` | `ef229da059f4dc48563c47674cdcf9ced2003f3f599fbf33aefd820e7457fd66` |
| Local VA | 3 | `20261110T121832Z-76ee05038ca5` | `e3306957f7639c2e78253d4d8fb8db9ce50b01e89e077a8db9bf1b9ba5573db5` |
| Local VA | 4 | `20261110T121833Z-911efdd0e5de` | `230b07285722ded246e1027aaa3ceae58454f1bf9257026745a2f8456b4b329c` |
| FedAvg | 0 | `20261110T121833Z-964727ae41c0` | `0b9e774fdbf8c8a23380f9510e3035a2177c28fcef7e913bffc25df51ed9f11e` |
| FedAvg | 1 | `20261110T121837Z-8ff8d9ece379` | `ef4caea59e34d0bf89fca10cb6678d88495e4c826df8d7791b0eaeb4aec2da13` |
| FedAvg | 2 | `20261110T121837Z-73421de0cefa` | `dabb188fe74fd19b5b03f02fd78864eec4a6e98ef60ca93fc0a582a65c053d4e` |
| FedAvg | 3 | `20261110T121853Z-e15105bd2ded` | `49a40a5d9a199991217388daa9a6715c8d1c744789f480b5207d5c3307bcbedc` |
| FedAvg | 4 | `20261110T121857Z-bc27b89c8861` | `7657e0b8ddb1cd620f230e4180c39eed735a516736ea325e8186baac305dabf4` |

`events.jsonl` (1.5–3.4 MB per run) and the continuation bundle are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. Two executions of the experiment on 2026-11-10 gave identical per-hospital scores for all 30 runs. The first was on an uncommitted tree that differed only in the formatting of `scripts/heart_disease_flamby.py`; the second, at commit `0fa2b87` with a clean tree, gave these runs.
