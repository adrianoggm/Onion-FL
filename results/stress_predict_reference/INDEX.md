# Stress-Predict centralised reference (new framework, real data)

`datasets/stress_predict.yaml` on Stress-Predict ([italha-d/Stress-Predict-Dataset](https://github.com/italha-d/Stress-Predict-Dataset), MIT, repository archive of commit `f49d2ea`, downloaded on 2026-11-10). Cite Iqbal et al., 2022, *Sensors* 22:8135. Issue #242.

```bash
onion_fl data inspect stress_predict > card.json
for s in 0 1 2 3 4; do
  onion_fl baseline experiments/stress_predict_reference.yaml --models lr rf --cv 5 --seed $s > baseline_seed$s.json
done
```

## Setup

- **Data.** 34 subjects (S02–S35; S01 is left out, as by the authors), 3,425 windows of 60 s with half overlap, 2,274 rest and 1,151 stress (34%). Each subject has 92 to 114 windows.
- **Features.** The 30 harmonized E4 features: mean, std, min, max and median of `acc_x`, `acc_y`, `acc_z`, `bvp`, `eda` and `temp`, each read at its native rate. No time, subject or phase column is a feature. The card (`card.json`) reports no missing values and no feature whose correlation with the label is above 0.95.
- **Labels.** From `Processed_data/Time_logs.xlsx`: 1 for the Stroop test, the interview and the hyperventilation; 0 for the baseline, the three rests, the questionnaire and the final rest. Consent and the gaps between phases have no class, so the windows that touch them are dropped.
- **Evaluation.** Subject-level 5-fold cross-validation (`onion_fl baseline --cv 5`): every subject is tested once, by a model trained on the other folds' subjects. The scaler is fitted on the training folds only. The seed draws the folds and seeds the models, so the five seeds are five different partitions of the subjects.
- **Models.** Logistic regression and random forest with their default parameters (`onion_fl.baselines`), nothing tuned. XGBoost is not installed in this environment, so it was not run.

| Identity | Value |
|---|---|
| `config_id` | `6a48914d5f1a0466b3d32f1608d227883819feca94283431b4aa2b57ab22adb4` |
| `data_id` | `4d64cbd9f1cb2b807c0e88ea88b42f8193ad444aaa6b762a9ca71874d1be3839` (cache digest `52d9aa9e…a9d`) |
| code | commit `e8ddfd9`, clean tree |

No federated run was made, so there is no `run_id` or `run_hash` here: these are classical baselines, identified by the config, the data and the commit.

## Labels against the authors' processed CSV

`Processed_data/Improved_All_Combined_hr_rsp_binary.csv` gives one label per second (0 rest, 1 task) for the same 34 subjects. Only its `Participant`, `Time(sec)` and `Label` columns were read, to compare; none of its columns is used by the descriptor. `label_agreement.json` has the counts per subject.

- **Per second:** 107,315 of its 112,516 seconds fall in a phase with a class; the other 5,201 (4.6%) fall between two phases. On those 107,315 seconds the labels agree on 96.3% (3,952 differ: 2,178 rest in the CSV that we call stress, 1,774 the other way round).
- **Per window:** for 96.2% of the 3,425 windows our label equals the majority of the CSV's labels inside it.
- **Why they differ.** The authors took the task boundaries from the button presses (`tags_S<n>.csv`, to the second), and the time log is at minute resolution. Where the tags are wrong (S06, S17, S18, S30: some presses are stamped 12 hours early, and those of S06 are shifted by minutes), the CSV's boundaries fall on whole minutes of the time log. S06, where they used the time log throughout, agrees on every second. The largest differences are S03 (12.9% of its seconds) and S02 (10.6%), whose time logs are one or two minutes off the presses.
- **Corrections in the reader.** The time log's clock is 12-hour without AM/PM: hours before 08:00 are taken as afternoon (the sessions ran from 09:05 to 16:15). The times are Irish time, which is UTC from 7 February to 9 March 2022, and they match the E4's Unix stamps. In S33 the baseline ends a minute after the Stroop test starts, so the reader ends it where the Stroop test starts.

## Results (subject-level 5-fold CV, mean ± 95% CI over seeds 0–4)

| Model | Macro-F1 | Balanced accuracy | Accuracy |
|---|---|---|---|
| Random forest | 0.568 ± 0.017 | 0.574 ± 0.014 | 0.666 ± 0.012 |
| Logistic regression | 0.521 ± 0.007 | 0.547 ± 0.004 | 0.666 ± 0.006 |
| Always "rest" (the training folds' majority) | 0.399 | 0.500 | 0.664 |

Macro-F1 per seed: random forest 0.570, 0.560, 0.549, 0.574, 0.585; logistic regression 0.519, 0.520, 0.515, 0.521, 0.530. The intervals use Student's t with 4 degrees of freedom on the five per-seed means (each the mean over five folds).

- **The features carry some signal on unseen subjects, but little.** Both models beat the majority class on macro-F1 and balanced accuracy, and the random forest beats logistic regression. Neither beats it on accuracy: they trade rest windows for stress windows.
- **The intervals measure the partition, not the subjects.** Every seed tests the same 34 subjects, so the CI only says how much the result depends on which subjects share a fold. Across the folds of one seed the macro-F1 varies more (std 0.041 for logistic regression and 0.054 for the random forest at seed 0).
- **This is a floor, not a tuned result.** Five window statistics per channel, no EDA peaks, no HRV features from IBI and no tuning. The tasks are short (2 to 11 minutes) and the hyperventilation is a physiological stressor, not a psychological one.

## Files

- `card.json`: `onion_fl data inspect stress_predict`.
- `label_agreement.json`: the comparison with the authors' CSV, per subject.
- `baseline_seed<S>.json`: `onion_fl baseline` output for each seed, with the test subjects and scores of every fold.
