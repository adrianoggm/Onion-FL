# Resumen de Cambios: Sistema SWEET Federated (70/20/10)

## 📋 Cambios Realizados

### 1. Actualización de Splits a 70/20/10

#### Archivos Modificados:

**`src/flower_basic/datasets/sweet_federated.py`**
- ✅ `FederatedConfigSWEET`: Cambiado `split_train: 0.6 → 0.7`, `split_test: 0.2 → 0.1`
- ✅ `_read_config()`: Defaults actualizados a 0.7/0.2/0.1

**`src/flower_basic/datasets/sweet_samples.py`**
- ✅ `load_sweet_sample_dataset()`: Cambiado `train_fraction: 0.6 → 0.7`

**`configs/sweet_federated.example.yaml`**
- ✅ Actualizado a `train: 0.7, val: 0.2, test: 0.1`
- ✅ Añadidos comentarios explicativos sobre distribución 70/20/10

### 2. Nuevos Scripts Creados

**`scripts/prepare_sweet_baseline.py`** (NUEVO)
- Entrena modelo baseline desde datos de `selection1`
- Usa split 70/20/10 por sujeto
- Guarda modelo, metadata e historial de entrenamiento
- Útil para fine-tuning en federated learning

**`scripts/test_sweet_federated.py`** (NUEVO)
- Verifica que el sistema está correctamente configurado
- Comprueba splits 70/20/10 en todos los componentes
- Útil para debugging y validación

### 3. Documentación Actualizada

**`SWEET_FEDERATED_README.md`**
- ✅ Añadida sección sobre distribución 70/20/10
- ✅ Documentados nuevos scripts
- ✅ Añadida guía de uso del modelo baseline
- ✅ Actualizado flujo de trabajo completo

## 🔧 Retrocompatibilidad

### ✅ Mantenida en:

1. **Broker Fog** (`src/flower_basic/brokers/fog.py`)
   - ❌ **NO modificado** - Ya es agnóstico al tipo de cliente
   - Funciona con SWEET y SWELL sin cambios
   - Procesa mensajes MQTT genéricos

2. **Servidores** (`src/flower_basic/servers/`)
   - ❌ **NO modificado** - `sweet.py` ya existe y es compatible
   - ❌ **NO modificado** - `swell.py` usa misma infraestructura
   - Ambos publican/reciben en mismos topics MQTT

3. **Clientes** (`src/flower_basic/clients/`)
   - ❌ **NO modificado** - `sweet.py` ya existe y hereda de `BaseMQTTComponent`
   - ❌ **NO modificado** - `swell.py` usa misma base
   - Ambos implementan misma interfaz

4. **Modelos** (`src/flower_basic/`)
   - ❌ **NO modificado** - `sweet_model.py` ya existe
   - ❌ **NO modificado** - `swell_model.py` usa misma estructura

## 📊 Estructura de Splits por Dataset

| Dataset | Train | Val | Test | Archivo de Config |
|---------|-------|-----|------|-------------------|
| SWELL   | 50%   | 20% | 30%  | `swell_federated.example.yaml` |
| **SWEET**   | **70%**   | **20%** | **10%**  | `sweet_federated.example.yaml` |

**Nota**: Los porcentajes son por sujeto (subject-based split).

## 🎯 Nuevas Capacidades

### Modelo Baseline desde Selection1

```bash
# 1. Entrenar baseline con datos de selection1
python scripts/prepare_sweet_baseline.py \
    --data-dir data/SWEET/selection1 \
    --output-dir baseline_models/sweet \
    --epochs 50

# Salida:
#   baseline_models/sweet/
#   ├── baseline_model.pth          # Pesos del modelo
#   ├── baseline_metadata.json      # Arquitectura, features, accuracy
#   └── training_history.json       # Historial completo
```

### Verificación de Configuración

```bash
# 2. Verificar que todo está configurado correctamente
python scripts/test_sweet_federated.py

# Verifica:
#   ✓ Dataset carga con splits 70/20/10
#   ✓ FederatedConfigSWEET tiene valores correctos
#   ✓ Config YAML tiene splits actualizados
```

### Flujo Completo

```bash
# 3. Preparar splits federados (70/20/10)
python scripts/prepare_sweet_federated.py \
    --config configs/sweet_federated.example.yaml

# 4. Ejecutar federated learning
python scripts/run_sweet_federated_demo.py \
    --config configs/sweet_federated.example.yaml \
    --num-rounds 10
```

## 📝 Archivos NO Modificados

Los siguientes archivos **NO fueron modificados** para mantener retrocompatibilidad:

### Infraestructura Core (compartida SWELL/SWEET)
- `src/flower_basic/brokers/fog.py`
- `src/flower_basic/clients/baseclient.py`
- `src/flower_basic/telemetry.py`
- `src/flower_basic/prometheus_metrics.py`

### Modelos y Clientes (ya existían)
- `src/flower_basic/sweet_model.py`
- `src/flower_basic/clients/sweet.py`
- `src/flower_basic/servers/sweet.py`

### Scripts existentes (funcionan sin cambios)
- `scripts/prepare_sweet_federated.py` (usa config YAML actualizado)
- `scripts/run_sweet_federated_demo.py` (usa config YAML actualizado)

## 🚀 Cómo Usar

### Para Selection1 (datos de usuarios reales)

```bash
# Entrenar modelo baseline con selection1
python scripts/prepare_sweet_baseline.py \
    --data-dir data/SWEET/selection1 \
    --output-dir baseline_models/sweet \
    --label-strategy binary \
    --threshold 2.0 \
    --epochs 50

# Resultado: Modelo baseline listo para fine-tuning
```

### Para Sample Subjects (subset de prueba)

```bash
# Verificar configuración
python scripts/test_sweet_federated.py

# Preparar splits federados
python scripts/prepare_sweet_federated.py \
    --config configs/sweet_federated.example.yaml

# Ejecutar demo
python scripts/run_sweet_federated_demo.py \
    --config configs/sweet_federated.example.yaml \
    --num-rounds 10
```

## ✅ Checklist de Implementación

- [x] Actualizar splits default a 70/20/10 en `sweet_federated.py`
- [x] Actualizar splits default a 70/20/10 en `sweet_samples.py`
- [x] Actualizar config ejemplo `sweet_federated.example.yaml`
- [x] Crear script `prepare_sweet_baseline.py` para selection1
- [x] Crear script `test_sweet_federated.py` para validación
- [x] Actualizar documentación `SWEET_FEDERATED_README.md`
- [x] Verificar retrocompatibilidad (broker fog, servidores, clientes)
- [x] Mantener infraestructura compartida sin cambios

## 🔍 Verificación

Para verificar que todo funciona correctamente:

```bash
# 1. Verificar configuración
python scripts/test_sweet_federated.py

# Debe mostrar:
# ✓ PASS - Dataset Loading
# ✓ PASS - Federated Config
# ✓ PASS - Config File
# ✓ ALL TESTS PASSED
```

## 📚 Información del Dataset SWEET

### Features Utilizadas

```
ECG Features (ventana 5 min, incremento 1 min):
- ECG_mean_heart_rate: Frecuencia cardíaca media
- ECG_sdnn: Desviación estándar RR intervals
- ECG_rmssd: Root mean square successive differences
- ECG_LF: Low frequency (0.04-0.15 Hz)
- ECG_HF: High frequency (0.15-0.4 Hz)
- ECG_LFHF: Ratio LF/HF
- ('std', 'ACC'): Desviación estándar acelerómetro
```

### Labels

```
MAXIMUM_STRESS (1-5):
- 1 = Sin estrés
- 5 = Extremadamente estresado

Estrategias:
- binary: >= threshold → estresado (1), else no estresado (0)
- ordinal: valores raw 1-5
```

### Splits por Usuario

Cada usuario se divide en:
- 70% datos para entrenamiento local
- 20% datos para validación durante entrenamiento
- 10% datos para test final

**Importante**: Los usuarios se asignan completamente a train, val o test para evitar data leakage.

## 🎓 Resumen

Este sistema implementa federated learning para SWEET con:
1. ✅ Splits **70/20/10** (train/val/test) por sujeto
2. ✅ **Retrocompatibilidad total** con infraestructura SWELL existente
3. ✅ **Modelo baseline** entrenable desde selection1
4. ✅ **Scripts de verificación** para debugging
5. ✅ **Documentación completa** actualizada

**Cambios mínimos**: Solo se modificaron archivos específicos de SWEET y configuración. El broker fog, base clients, telemetry y Prometheus metrics permanecen sin cambios, permitiendo usar SWEET y SWELL simultáneamente en la misma arquitectura federada.
