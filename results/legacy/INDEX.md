# Resultados históricos (antes del refactor a `onion_fl`)

Todos son **baselines centralizados**: ninguno es un resultado federado, porque las ejecuciones federadas escribían en `federated_runs/`, que nunca se ha commiteado.

Las fechas son las del contenido, sacadas del historial de git. Lee las salvedades antes de citar cualquier cifra.

| Carpeta | Experimento | Script de origen | Fecha | Salvedades |
|---|---|---|---|---|
| `subject_cv_results/` | Validación cruzada de 5 folds por sujeto (GroupKFold), LR y RF, sobre WESAD, SWELL (modalidad computer) y WESAD+SWELL | `scripts/run_subject_cv.py` | 2025-09-28 | ⚠️ Las filas de SWELL y WESAD+SWELL son **anteriores a la corrección de la fuga de `blok`** (commit `002246f`, 2025-10-25), y su ~0,99 no es válido. Las de WESAD sí lo son. |
| `advanced_ml_results/wesad_baseline_results.json` | WESAD binario, separación por sujetos (7/3/5), 22 features de muñeca | `scripts/evaluate_wesad_baseline.py` | 2025-09-27 | Válido. RF tiene una accuracy de test de 0,828, pero una F1 de la clase estrés de solo 0,393. |
| `advanced_ml_results/swell_baseline_results.json`, `swell_dataset_analysis.json` | SWELL con 4 modalidades, separación por sujetos (50/20/30), 163 features | `scripts/evaluate_swell_baseline.py` | 2025-12-04 | Muestra aleatoria de 50.000 filas. El bloque "validation" es una copia del de test. |
| `advanced_ml_results/cv_results.json`, `ultra_powerful_results.json` y las figuras | SWEET selection1, 3 clases, 5 folds por sujeto, varios modelos | `advanced_ml_comparison.py`, `ultra_powerful_ml.py` (borrados) | 2025-12-14 | Todos los modelos quedan en la tasa de la clase mayoritaria (0,552) o por debajo. En `cv_results.json`, `mean_f1` es la F1 ponderada, no la macro. |
| `extreme_deep_results/` | SWEET selection1, XGBoost y MLP profundos | `extreme_deep_models.py` (borrado) | 2025-12-14 | El "mejor" modelo, Pyramid16, predice siempre la clase 0. |
| `hypertuning_results/` | SWEET selection1, RandomizedSearchCV (5 folds por sujeto) | `hyperparameter_tuning.py` (borrado) | 2025-12-14 | El mejor resultado, 0,554, es igual a la tasa de la clase mayoritaria. |
| `baseline_models/sweet/` | Baselines de SWEET selection1: SweetMLP (`.pth`, metadata, history) y XGBoost (`xgboost_tuned_model.json`, `scaler.json`, `training_report.json`) | `scripts/prepare_sweet_baseline.py`, `scripts/train_sweet_baseline_selection1.py` | 2025-12-14 | El scaler de XGBoost se ajustó con todo selection1 antes de la validación cruzada (fuga leve). Ningún cliente federado carga estos modelos. |
| `comparativa_completa/` | Comparación descriptiva de WESAD y SWELL (features, correlaciones, balance de clases, valores ausentes por sujeto) | `comparativa_wesad_swell.py`, `missing_por_cliente.py` (en la misma carpeta) | 2026-02-08 | La tabla HRV de SWELL dice 0 % de ausentes, pero el fichero de features de fisiología tiene un 15,2 %. El número de sujetos de SWELL es 23 o 25 según la fuente. |
| `swell_plots/` | Figuras y CSV del análisis de SWELL (importancias, correlaciones, distribuciones) y dos figuras de SWEET | `scripts/evaluate_swell_baseline.py` y otros | 2025-12-04 | HR y RMSSD tienen una correlación de 1,000, lo que apunta a que los centinelas 999 no se limpiaron. |

Los resultados que se generen a partir de ahora van a `results/` (fuera de `legacy/`) o a `runs/<run_id>/` en el framework nuevo.
