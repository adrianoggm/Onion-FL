# SWEET Federated Learning Setup

Sistema de aprendizaje federado para el dataset SWEET con distribución **70/20/10** (train/val/test), compatible con la infraestructura existente de SWELL.

## 📊 Configuración de Splits

- **70% Entrenamiento**: Datos para entrenar modelo local en cada cliente
- **20% Validación**: Evaluación durante entrenamiento
- **10% Test**: Evaluación final del modelo global

Los splits se aplican **por sujeto** para garantizar independencia entre splits.

## 🏗️ Arquitectura

La implementación reutiliza completamente la infraestructura de SWELL:

- **`BaseMQTTComponent`**: Base común para clientes y bridges
- **Telemetry & Prometheus**: Métricas compartidas entre datasets
- **Pattern FedAvg**: Mismo algoritmo de agregación
- **Split strategies**: Subject-based splitting consistente (70/20/10)
- **Broker Fog**: Totalmente retrocompatible con clientes SWEET y SWELL

## 📁 Archivos Creados

### Core
- `src/flower_basic/datasets/sweet_federated.py` - Preparación de splits federados
- `src/flower_basic/sweet_model.py` - Modelo MLP para SWEET
- `src/flower_basic/clients/sweet.py` - Cliente federado SWEET
- `src/flower_basic/servers/sweet.py` - Servidor federado SWEET

### Scripts
- `scripts/prepare_sweet_baseline.py` - **NUEVO**: Entrenar modelo baseline desde selection1 (70/20/10)
- `scripts/prepare_sweet_federated.py` - Preparación de splits federados (70/20/10)
- `scripts/run_sweet_federated_demo.py` - Demo completo
- `scripts/test_sweet_federated.py` - **NUEVO**: Verificar configuración correcta

### Config
- `configs/sweet_federated.example.yaml` - Configuración de ejemplo

## 🚀 Uso Rápido

### 0. Verificar Configuración (Recomendado)

```bash
python scripts/test_sweet_federated.py
```

Este script verifica que el sistema está correctamente configurado con splits 70/20/10.

### 1. (Opcional) Entrenar Modelo Baseline

Si quieres entrenar un modelo baseline desde `selection1` para usar como punto de partida:

```bash
python scripts/prepare_sweet_baseline.py \
    --data-dir data/SWEET/selection1 \
    --output-dir baseline_models/sweet \
    --epochs 50
```

Esto crea un modelo baseline entrenado con splits 70/20/10 que puede usarse para:
- Fine-tuning en federated learning
- Comparación de rendimiento
- Inicialización de pesos

### 2. Preparar Datos Federados

```bash
python scripts/prepare_sweet_federated.py \
    --config configs/sweet_federated.example.yaml
```

Esto crea:
```
federated_runs/sweet/demo_3nodes/
├── manifest.json
├── scaler_global.json
├── fog_0/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── subject_user0001/
│       ├── train.npz
│       ├── val.npz
│       └── test.npz
├── fog_1/
│   └── ...
└── fog_2/
    └── ...
```

### 3. Ejecutar Demo Completo

```bash
# Asegúrate de que mosquitto está corriendo
mosquitto -c mosquitto.conf

# Ejecuta el demo (preparará datos automáticamente si no existen)
python scripts/run_sweet_federated_demo.py \
    --config configs/sweet_federated.example.yaml \
    --num-rounds 10
```

### 4. Ejecución Manual (más control)

```bash
# Preparar manifest + splits y lanzar toda la arquitectura
python scripts/run_sweet_architecture.py \
    --config configs/sweet_architecture_5nodes.yaml \
    --dispatch-config \
    --launch

# Si ya existe manifest, reutilizarlo
python scripts/run_sweet_architecture.py \
    --config configs/sweet_architecture_5nodes.yaml \
    --manifest federated_runs/sweet/auto_5nodes/manifest.json \
    --launch
```

## ⚙️ Configuración

### Archivo YAML

```yaml
dataset:
  data_dir: "data/SWEET/sample_subjects"
  label_strategy: "binary"  # "binary" o "ordinal"
  elevated_threshold: 2.0
  min_samples_per_subject: 5

split:
  seed: 42
  train: 0.6
  val: 0.2
  test: 0.2
  scaler: "global"
  strategy: "per_subject"

federation:
  mode: "manual"  # "manual" o "auto"
  num_fog_nodes: 3
  
  manual_assignments:
    fog_0: ["user0001", "user0002"]
    fog_1: ["user0003", "user0004"]
    fog_2: ["user0005", "user0006"]
```

### Estrategias de Labels

**Binary** (por defecto):
- `MAXIMUM_STRESS >= 2.0` → 1 (stress)
- `MAXIMUM_STRESS < 2.0` → 0 (no stress)

**Ordinal**:
- Usa valores raw de `MAXIMUM_STRESS` (0-4)

### Modos de Federación

**Manual**:
- Define explícitamente qué sujetos van a cada nodo
- Útil para experimentación controlada

**Auto**:
- Distribución automática balanceada
- Usa `per_node_percentages` para control proporcional

## 🔄 Compatibilidad con SWELL

Ambos sistemas comparten:

| Componente | SWELL | SWEET |
|------------|-------|-------|
| Base MQTT | ✅ `BaseMQTTComponent` | ✅ `BaseMQTTComponent` |
| Telemetry | ✅ OpenTelemetry | ✅ OpenTelemetry |
| Prometheus | ✅ Shared metrics | ✅ Shared metrics |
| Model API | ✅ `get_parameters()` | ✅ `get_parameters()` |
| Aggregation | ✅ FedAvg | ✅ FedAvg |
| Split strategy | ✅ Subject-based | ✅ Subject-based |

**La infraestructura reutiliza el mismo patrón MQTT/Flower que SWELL**, pero cada ejecución debe usar un único workflow por run.

## 📊 Métricas

### Prometheus (puerto 9090+)

- `fl_client_training_rounds_total`
- `fl_client_training_duration_seconds`
- `fl_client_local_accuracy`
- `fl_rounds_total`
- `fl_global_accuracy`
- `fl_aggregation_duration_seconds`

### OpenTelemetry

Traces completos de:
- Training loops
- MQTT publish/subscribe
- Aggregation operations

## 🧪 Testing

```bash
# Verificar imports
python -c "from flower_basic.datasets import plan_and_materialize_sweet_federated"
python -c "from flower_basic.sweet_model import SweetMLP"
python -c "from flower_basic.clients.sweet import SweetFLClientMQTT"
python -c "from flower_basic.servers.sweet import MQTTFedAvgSweet"

# Test preparación
python scripts/prepare_sweet_federated.py \
    --config configs/sweet_federated.example.yaml
```

## 📝 Diferencias con SWELL

| Aspecto | SWELL | SWEET |
|---------|-------|-------|
| Features | 20 (multimodal) | 50+ (stress metrics) |
| Subjects | 25 | Variable (sample_subjects) |
| Labels | Binary (stress/no-stress) | Binary/Ordinal |
| Source | Curated NPZ | CSV per subject |
| Missing values | Handled | Filtered per config |

## 🔧 Troubleshooting

### Error: "Subject split not found"

Verifica que `min_samples_per_subject` no sea demasiado alto:

```yaml
dataset:
  min_samples_per_subject: 5  # Reduce si es necesario
```

### Error: "MQTT connection refused"

Inicia mosquitto:

```bash
mosquitto -c mosquitto.conf
```

### Error: "Input dim mismatch"

Verifica el número de features en `manifest.json`:

```bash
cat federated_runs/sweet/demo_3nodes/manifest.json | grep num_features
```

## 🎯 Próximos Pasos

1. **Fog Bridge**: Crear `fog_bridge_sweet.py` similar a SWELL
2. **Multi-dataset**: Integrar en `demo_multidataset_fl.py`
3. **Comparison**: Añadir a `compare_models.py`
4. **Baseline**: Extender `evaluate_sweet_sample_baseline.py` con CV

## 📚 Referencias

- Documentación SWELL: Ver `src/flower_basic/clients/swell.py`
- Baseline SWEET: Ver `scripts/evaluate_sweet_sample_baseline.py`
- Config examples: Ver `configs/swell_federated.example.yaml`
