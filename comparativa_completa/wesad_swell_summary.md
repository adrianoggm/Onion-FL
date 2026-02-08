# WESAD vs SWELL - Resumen

| Metrica | WESAD | SWELL |
| --- | --- | --- |
| Sujetos | 15 | 23 |
| Muestras | 859 | 410322 |
| Features | 30 | 34 |
| Ratio nulos | 0.000000 | 0.000000 |

## Distribucion de clases
- WESAD: {'0': 557, '1': 302}
- SWELL: {'0': 222240, '1': 188082}

## Etiquetas
- WESAD: {0: 'baseline/no_stress', 1: 'stress'}
- SWELL: {'no stress': 0, 'control': 0, 'neutral': 0, 'baseline': 0, '0': 0, 'n': 0, 'time pressure': 1, 'interruption': 1, 'interruptions': 1, 'combined': 1, 'stress': 1, '1': 1, '2': 1, '3': 1, 't': 1, 'i': 1, 'r': 1}
- Figura: `wesad_swell_label_tables.png`

## Features y correlacion
- WESAD: ver `wesad_feature_list.txt` y `wesad_feature_correlations.csv`
- SWELL (fisiologico): ver `swell_physiology_feature_list.txt` y `swell_physiology_feature_correlations.csv`
- Tabla compartida: `wesad_swell_feature_table.(csv|md|tex|png)`

## SWELL: sujetos detectados (raw rri)
- 1, 10, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 3, 4, 5, 6, 7, 9

## Features compartidas
- Conteo: 0
- Ejemplos: (ninguna)

## Correlacion media absoluta (pares de features)
- WESAD: 0.1908 (mediana 0.1161)
- SWELL: 0.3143 (mediana 0.2320)

## Top correlaciones feature-label
- WESAD: bvp_0_median (-0.430), eda_0_std (0.409), eda_0_max (0.383), eda_0_mean (0.365), eda_0_median (0.364)
- SWELL: MEAN_RR (0.266), MEDIAN_RR (0.250), HR (-0.247), pNN25 (0.220), TP (0.198)