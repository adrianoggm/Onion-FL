# WESAD Baseline

* Train samples: 572
* Test samples: 287
* Features: 30

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| logistic_regression | 0.930 | 0.921 |
| random_forest | 0.962 | 0.959 |

**5-fold Group Cross-Validation (accuracy +/- std / macro-F1 +/- std)**
- logistic_regression: 0.829 +/- 0.056 / 0.811 +/- 0.074
- random_forest: 0.809 +/- 0.111 / 0.786 +/- 0.127