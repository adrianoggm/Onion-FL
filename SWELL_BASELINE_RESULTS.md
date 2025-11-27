# SWELL Baseline

* Train samples: 2500
* Test samples: 639
* Features: 16

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| logistic_regression | 0.679 | 0.431 |
| random_forest | 0.671 | 0.571 |

**5-fold Group Cross-Validation (accuracy +/- std / macro-F1 +/- std)**
- logistic_regression: 0.669 +/- 0.014 / 0.422 +/- 0.014
- random_forest: 0.670 +/- 0.014 / 0.580 +/- 0.013