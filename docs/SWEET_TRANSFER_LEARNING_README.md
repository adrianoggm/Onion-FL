# SWEET Transfer Learning: Federated Fine-Tuning

Este documento describe el sistema de transfer learning implementado para el dataset SWEET, que entrena un modelo base en selection1 (102 sujetos) y luego lo fine-tunea en un entorno federado usando selection2 (140 sujetos).

## 🎯 Objetivo

Implementar aprendizaje federado con transfer learning:
1. **Pre-entrenamiento**: Entrenar XGBoost en selection1 (centralizado)
2. **Fine-tuning Federado**: Mejorar el modelo en selection2 distribuido entre 3 nodos fog

## 📊 Datasets

### Selection1 (Pre-training)
- **Sujetos**: 102
- **Muestras**: ~3,927
- **Uso**: Entrenamiento centralizado del modelo base
- **Ubicación**: `data/SWEET/selection1/users`

### Selection2 (Federated Fine-tuning)
- **Sujetos**: 140
- **Muestras**: Variable por sujeto
- **Uso**: Fine-tuning federado distribuido
- **Ubicación**: `data/SWEET/selection2/users`
- **Distribución**: 3 nodos fog (~47 sujetos cada uno)

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────┐
│  Phase 1: Centralized Pre-training         │
│  (Selection1 - 102 subjects)               │
│                                             │
│  XGBoost Model (55.44% accuracy)           │
│  + StandardScaler                           │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Phase 2: Federated Fine-tuning            │
│  (Selection2 - 140 subjects)               │
│                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │ Fog 0   │  │ Fog 1   │  │ Fog 2   │   │
│  │ ~47 subj│  │ ~46 subj│  │ ~47 subj│   │
│  └─────────┘  └─────────┘  └─────────┘   │
│       │             │             │        │
│       └─────────────┴─────────────┘        │
│                     ↓                       │
│          Central Aggregation Server        │
└─────────────────────────────────────────────┘
```

## 🚀 Uso Rápido

### Flujo Completo Automático

```bash
# Ejecutar todo el pipeline de transfer learning
python scripts/run_sweet_transfer_learning.py

# Con opciones personalizadas
python scripts/run_sweet_transfer_learning.py \
  --rounds 20 \
  --mqtt-broker localhost \
  --mqtt-port 1883 \
  --enable-prometheus \
  --enable-telemetry
```

### Flujo Manual (Paso a Paso)

#### 1. Extracción de Selection2 (si no se ha hecho)

```bash
python scripts/extract_sweet_selection2.py
```

**Salida**: `data/SWEET/selection2/users/` con 140 carpetas de usuarios

#### 2. Pre-entrenamiento en Selection1

```bash
python scripts/train_sweet_baseline_selection1.py
```

**Salida**:
- `baseline_models/sweet/xgboost_tuned_model.json`: Modelo pre-entrenado
- `baseline_models/sweet/scaler.json`: StandardScaler ajustado
- `baseline_models/sweet/training_report.json`: Métricas de entrenamiento

**Métricas esperadas**:
- Mean CV Accuracy: ~55.44% ± 0.5%
- Método: 5-fold subject-level stratified CV
- Features: 258 características fisiológicas

#### 3. Preparación de Splits Federados (Selection2)

```bash
python scripts/prepare_sweet_federated_transfer.py
```

**Salida**: `federated_runs/sweet/transfer_selection2/`
```
transfer_selection2/
├── manifest.json                    # Configuración y metadata
├── scaler_global.json              # Scaler para selection2
├── pretrained_model.json           # Copia del modelo pre-entrenado
├── pretrained_scaler.json          # Copia del scaler de selection1
├── fog_0/
│   ├── train.npz                   # Datos agregados del nodo
│   ├── val.npz
│   ├── test.npz
│   └── subject_XXXX/               # Datos por sujeto
│       ├── train.npz
│       ├── val.npz
│       └── test.npz
├── fog_1/...
└── fog_2/...
```

#### 4. Iniciar MQTT Broker

```bash
# Usando Mosquitto (debe estar instalado)
mosquitto -c mosquitto.conf

# O usando Docker
docker run -it -p 1883:1883 eclipse-mosquitto
```

#### 5. Ejecutar Fine-tuning Federado

```bash
python scripts/run_sweet_federated_demo.py \
  --config configs/sweet_federated_transfer.yaml \
  --num-rounds 10 \
  --mqtt-broker localhost \
  --mqtt-port 1883
```

## ⚙️ Configuración

El archivo `configs/sweet_federated_transfer.yaml` controla todos los parámetros:

```yaml
dataset:
  data_dir: data/SWEET/selection2/users
  label_strategy: ordinal_3class     # 0=low, 1=medium, 2=high stress
  elevated_threshold: 2.0
  min_samples_per_subject: 5

split:
  train: 0.6                         # 60% entrenamiento
  val: 0.2                           # 20% validación
  test: 0.2                          # 20% test
  seed: 42
  scaler: global                     # Escalado global
  strategy: per_subject              # Split por sujeto

transfer_learning:
  pretrained_model_path: baseline_models/sweet/xgboost_tuned_model.json
  pretrained_scaler_path: baseline_models/sweet/scaler.json
  freeze_initial_weights: false      # Permitir actualización de todos los pesos
  fine_tune_lr_multiplier: 0.1       # Learning rate reducido (10%)

federation:
  mode: auto                         # Asignación automática de sujetos
  num_fog_nodes: 3
  per_node_percentages: [0.34, 0.33, 0.33]
  output_dir: federated_runs/sweet
  run_name: transfer_selection2
  ensure_min_train_per_node: true
```

## 📈 Modelo Base

### XGBoost Hyperparameters (Optimizados)

```python
{
  "max_depth": 4,
  "n_estimators": 300,
  "learning_rate": 0.01,
  "subsample": 0.6,
  "colsample_bytree": 0.6,
  "reg_alpha": 1.0,           # L1 regularization
  "reg_lambda": 2,            # L2 regularization
  "gamma": 0.3,               # Min split loss
  "min_child_weight": 1,
  "objective": "multi:softmax",
  "num_class": 3              # ordinal_3class
}
```

### Estrategia de Etiquetas

**ordinal_3class**: Balance optimizado de clases

| Original | ordinal_3class | Descripción |
|----------|----------------|-------------|
| 1        | 0              | Low stress  |
| 2        | 1              | Medium stress |
| 3, 4, 5  | 2              | High stress |

**Ratio de balance**: 4.28:1 (vs 1507:1 en 5-class original)

## 🔬 Características Extraídas

### Distribución por Modalidad
- **ECG**: 84 características (32.6%)
- **EDA**: 84 características (32.6%)
- **TEMP**: 45 características (17.4%)
- **ACC**: 45 características (17.4%)

### Ejemplos de Features
```
ECG_mean, ECG_std, ECG_psd_vlf, ECG_hrv_rmssd, ...
EDA_mean, EDA_std, EDA_scl_mean, EDA_scr_peaks, ...
TEMP_mean, TEMP_std, TEMP_slope, ...
ACC_X_mean, ACC_Y_std, ACC_Z_energy, ...
```

**Total**: 258 características fisiológicas

## 📊 Rendimiento Esperado

### Selection1 Baseline (Pre-training)
- **Accuracy**: 55.44% ± 0.5%
- **Método**: 5-fold subject-level CV
- **Sujetos**: 102
- **Muestras**: ~3,927

### Selection2 Fine-tuning (Esperado)
- **Objetivo**: Mejorar accuracy mediante fine-tuning federado
- **Distribución**: 3 nodos con ~47, 46, 47 sujetos
- **Rounds**: 10-20 rounds federados

## 🔧 Troubleshooting

### Error: "Manifest not found"
```bash
# Solución: Ejecutar preparación de datos
python scripts/prepare_sweet_federated_transfer.py
```

### Error: "MQTT connection refused"
```bash
# Solución: Verificar que Mosquitto esté corriendo
mosquitto -c mosquitto.conf

# O verificar puerto
netstat -an | grep 1883
```

### Error: "Baseline model not found"
```bash
# Solución: Entrenar modelo base
python scripts/train_sweet_baseline_selection1.py
```

### Error: "Selection2 data not found"
```bash
# Solución: Extraer selection2
python scripts/extract_sweet_selection2.py
```

## 📁 Estructura de Archivos

```
flower-basic/
├── configs/
│   └── sweet_federated_transfer.yaml       # Configuración del sistema
├── data/
│   └── SWEET/
│       ├── selection1/users/               # 102 sujetos (pre-training)
│       └── selection2/users/               # 140 sujetos (fine-tuning)
├── baseline_models/
│   └── sweet/
│       ├── xgboost_tuned_model.json        # Modelo pre-entrenado
│       ├── scaler.json                      # Scaler de selection1
│       └── training_report.json             # Métricas de baseline
├── federated_runs/
│   └── sweet/
│       └── transfer_selection2/             # Splits y resultados federados
├── scripts/
│   ├── extract_sweet_selection2.py          # [1] Extracción de datos
│   ├── train_sweet_baseline_selection1.py   # [2] Pre-entrenamiento
│   ├── prepare_sweet_federated_transfer.py  # [3] Preparación federada
│   ├── run_sweet_federated_demo.py          # [4] Demo federado
│   └── run_sweet_transfer_learning.py       # [Master] Flujo completo
└── src/
    └── flower_basic/
        ├── datasets/
        │   ├── sweet_samples.py             # Carga de datos SWEET
        │   └── sweet_federated.py           # Splits federados + transfer learning
        ├── clients/
        │   └── sweet.py                     # Cliente federado SWEET
        └── servers/
            └── sweet.py                     # Servidor federado SWEET
```

## 🎓 Conceptos Clave

### Transfer Learning
- **Pre-training**: Modelo aprende patrones generales en selection1
- **Fine-tuning**: Modelo se adapta a selection2 manteniendo conocimiento previo
- **Ventaja**: Mejor generalización con menos datos locales

### Federated Learning
- **Descentralización**: Cada nodo fog entrena con sus datos locales
- **Agregación**: Servidor combina actualizaciones de todos los nodos
- **Privacidad**: Datos nunca salen de los nodos fog

### Subject-Level Splitting
- **Objetivo**: Prevenir data leakage entre splits
- **Método**: Cada sujeto completo pertenece a un solo split (train/val/test)
- **Ventaja**: Evaluación realista de generalización

## 📚 Referencias

### Scripts Relacionados
- `scripts/diagnose_sweet.py`: Análisis del dataset
- `scripts/analyze_sweet_sensors.py`: Análisis de sensores
- `scripts/hyperparameter_tuning.py`: Búsqueda de hiperparámetros
- `scripts/advanced_ml_comparison.py`: Comparación de modelos ML

## 🤝 Inspiración

Este sistema está inspirado en la implementación existente de SWELL:
- `src/flower_basic/datasets/swell_federated.py`: Template de splits federados
- `src/flower_basic/clients/swell.py`: Cliente MQTT federado
- `src/flower_basic/servers/swell.py`: Servidor de agregación

## 📝 Notas Adicionales

### Limitaciones
- **Accuracy ceiling**: ~55.5% debido a correlaciones débiles (<0.11) entre features y labels
- **Class imbalance**: Incluso con ordinal_3class, existe desbalance 4.28:1
- **Subject variability**: Alta variabilidad inter-sujeto en respuestas fisiológicas

### Mejoras Futuras
- Implementar SMOTE o class weighting para balance
- Probar feature selection para reducir dimensionalidad
- Explorar ensemble methods combinando múltiples modelos
- Implementar personalización por nodo (personalized federated learning)

---

**Última actualización**: 2024
**Mantenedor**: Equipo Flower-Basic
**Licencia**: [TBD]
