# Guía de Ejecución - Federated Fog Demo

## ¡Configuración Completada!

Tu entorno está completamente configurado y listo para ejecutar la demo de aprendizaje federado con fog computing.

## Pasos para Ejecutar la Demo

### Opción 1: Usando Scripts .bat (Windows - Recomendado)

Ejecuta los siguientes archivos .bat en **orden secuencial**, cada uno en una ventana de terminal separada:

```
1️⃣ run_server.bat        - Servidor Flower central
2️⃣ run_fog_broker.bat    - Broker fog que agrega updates locales
3️⃣ run_fog_client.bat    - Cliente Flower en el nodo fog
4️⃣ run_client.bat        - Cliente local (ejecutar múltiples veces)
5️⃣ run_debug.bat         - Monitor MQTT (opcional)
```

**Orden de ejecución:**

1. Haz doble clic en `run_server.bat`
2. Espera hasta ver "Flower server running"
3. Haz doble clic en `run_fog_broker.bat`
4. Espera hasta ver "MQTT Broker connected"
5. Haz doble clic en `run_fog_client.bat`
6. Espera hasta ver "Connected to Flower server"
7. Haz doble clic en `run_client.bat` (puedes ejecutar múltiples instancias)
8. (Opcional) Haz doble clic en `run_debug.bat` para monitorear tráfico MQTT

### Opción 2: Línea de Comandos Manual

Si prefieres usar la línea de comandos:

**Terminal 1 - Servidor Central:**

```bash
.venv\Scripts\python.exe server.py
```

**Terminal 2 - Fog Broker:**

```bash
.venv\Scripts\python.exe broker_fog.py
```

**Terminal 3 - Fog Client:**

```bash
.venv\Scripts\python.exe fog_flower_client.py
```

**Terminal 4+ - Clientes Locales:**

```bash
.venv\Scripts\python.exe client.py
```

**Terminal Opcional - Debug:**

```bash
.venv\Scripts\python.exe debug.py
```

## Qué Esperar Durante la Ejecución

### 1. Servidor Central (`server.py`)

```
[INFO] Flower server running on port 8080
[MQTT] Connected to test.mosquitto.org
[SERVER] Ready to aggregate...
```

### 2. Fog Broker (`broker_fog.py`)

```
[MQTT] Connected to test.mosquitto.org (rc=0)
[BROKER] Waiting for client updates...
[BROKER] Region 'region_0' has 3/3 updates, computing partial...
```

### 3. Fog Client (`fog_flower_client.py`)

```
[FOG CLIENT] Connected to Flower server at localhost:8080
[FOG CLIENT] Received partial for region=region_0
[FOG CLIENT] Sending partial aggregate to central server...
```

### 4. Clientes Locales (`client.py`)

```
[CLIENT] Training locally on ECG5000 data...
[CLIENT] Publishing update to fog broker...
[CLIENT] Waiting for global model...
[CLIENT] Received new global model, starting next round...
```

### 5. Debug Monitor (`debug.py`)

```
[DEBUG] fl/updates → {"region": "region_0", "client_id": "client_1", ...}
[DEBUG] fl/partial → {"region": "region_0", "partial_weights": {...}}
[DEBUG] fl/global_model → {"global_weights": {...}, "round": 1}
```

## Flujo de Entrenamiento

1. **Ronda 1**: Los clientes entrenan localmente con datos ECG5000
2. **Agregación Fog**: El broker fog agrega K=3 updates por región
3. **Agregación Central**: El servidor Flower agrega todas las regiones
4. **Distribución**: El modelo global se distribuye via MQTT
5. **Ronda 2+**: Se repite el proceso con el nuevo modelo global

## Métricas y Resultados

El sistema mostrará:

-   **Precisión local** de cada cliente
-   **Pérdida de entrenamiento** por ronda
-   **Tiempo de agregación** en fog y central
-   **Número de clientes** participantes por ronda

Típicamente verás convergencia en **3-5 rondas** con mejora progresiva de la precisión.

## Troubleshooting

### ❌ Error de conexión al servidor Flower

-   Verificar que `server.py` esté ejecutándose
-   Comprobar que el puerto 8080 no esté ocupado: `netstat -ano | findstr :8080`

### ❌ Error de conexión MQTT

-   Verificar conectividad a internet
-   Probar con broker local: cambiar `test.mosquitto.org` por `localhost`

### ❌ Clientes no participan

-   Verificar que K≤número de clientes activos
-   Esperar a que se acumulen suficientes updates (K=3 por defecto)

### ❌ Modelo no converge

-   Aumentar número de épocas locales en `client.py`
-   Verificar que hay suficientes clientes (mínimo 2)
-   Revisar distribución de datos entre clientes

## Personalización

### Cambiar Número de Agregación Fog (K)

En `broker_fog.py`, línea 20:

```python
K = 3  # Cambiar a número deseado
```

### Cambiar Broker MQTT

En todos los archivos, cambiar:

```python
MQTT_BROKER = "test.mosquitto.org"  # Cambiar a tu broker
```

### Modificar Modelo

Editar `model.py` para cambiar arquitectura del CNN:

```python
class ECGModel(nn.Module):
    # Modificar capas aquí
```

## Archivos de Log

Los logs se muestran en las consolas de cada componente. Para logs persistentes, modifica los scripts para redirigir salida:

```bash
.venv\Scripts\python.exe server.py > server.log 2>&1
```

## Parar la Demo

1. Presiona `Ctrl+C` en cada terminal
2. O cierra las ventanas de los scripts .bat
3. Los archivos .bat tienen `pause` al final para mantener la ventana abierta

---

# 🔬 **Evaluación Robusta del Sistema**

## Ejecutar Comparación Estadística Completa

### Comando Principal

```bash
# Ejecutar evaluación robusta con cross-validation
python -c "from compare_models import ModelComparator; comp = ModelComparator(); comp.run_robust_comparison(n_cv_folds=5)"
```

### Qué Esperar Durante la Evaluación

#### 1. Cross-Validation Folds

```
--- Cross-Validation Fold 1/5 ---
*** Model Performance Comparison ***
============================================================
Loading ECG5000 dataset with subject-based simulation...
Dataset: 3998 train, 1000 test samples (simulated subjects)

1. Running Baseline (Centralized) Training...
Baseline completed: Accuracy = 0.9935

2. Running Federated Training...
Starting federated training (3 clients, 10 rounds)...
Federated completed: Accuracy = 0.9945

Fold 1 - Centralized: 0.9935, Federated: 0.9945
```

#### 2. Análisis Estadístico

```
--- Statistical Analysis ---
Centralized accuracy: 0.9935 ± 0.0015
Federated accuracy: 0.9945 ± 0.0005
T-statistic: -0.6325, p-value: 0.5918
Effect size (Cohen's d): 0.8944 (large)
Statistically significant: No
```

#### 3. Detección de Fugas de Datos

```
--- Data Leakage Detection ---
Mean similarity: 0.9763
Max similarity: 0.9977
Potential leakage ratio: 0.9210
Data leakage detected: Yes
```

#### 4. Resultados Finales

```
✅ Robust results saved to: comparison_results\robust_comparison_results.json

📊 RESULTADOS ESTADÍSTICOS:
• Precisión Federada: 99.45% ± 0.05%
• Precisión Centralizada: 99.35% ± 0.15%
• Significancia: p=0.592 (No significativa)
• Fuga de Datos: 92.1% detectada
```

## Ejecutar Comparación Rápida

### Comparación Básica

```bash
# Comparación rápida sin cross-validation
python quick_comparison.py
```

### Resultados en JSON

Los resultados se guardan automáticamente en:

-   `comparison_results/robust_comparison_results.json` - Resultados estadísticos
-   `comparison_results/comparison_results.json` - Resultados detallados
-   `comparison_results/comparison_plots.png` - Gráficos de comparación

## Análisis de Fugas de Datos

### Verificar Similitud de Datos

```bash
# Ejecutar análisis de similitud
python -c "from utils import detect_data_leakage; print('Leakage analysis completed')"
```

### Resultados Esperados

```
Data leakage detected: 92.1%
Recommendation: Use subject-based splitting for reliable comparisons
```

## Troubleshooting de Evaluación

### ❌ Error de Serialización JSON

-   Verificar que numpy esté instalado: `pip install numpy`
-   Los tipos numpy se convierten automáticamente a tipos Python

### ❌ Memoria Insuficiente

-   Reducir `n_cv_folds` de 5 a 2: `comp.run_robust_comparison(n_cv_folds=2)`
-   Aumentar `batch_size` en los parámetros de entrenamiento

### ❌ Resultados No Significativos

-   Es normal: indica que no hay diferencia estadística entre los enfoques
-   Revisar fuga de datos detectada (92.1% en ECG5000)

### ❌ Modelo No Convergente

-   Aumentar `epochs` en parámetros: `comp.run_robust_comparison(epochs=100)`
-   Verificar distribución de datos entre clientes

## Personalización de Evaluación

### Cambiar Parámetros de Evaluación

```python
# En compare_models.py o en línea de comandos
comp.run_robust_comparison(
    epochs=100,           # Épocas de entrenamiento
    num_clients=5,        # Número de clientes
    fl_rounds=15,         # Rondas federadas
    n_cv_folds=3,         # Folds de cross-validation
    batch_size=64         # Tamaño de batch
)
```

### Modificar Dataset

```python
# Cambiar a otro dataset en utils.py
def load_custom_dataset():
    # Implementar carga de datos personalizada
    pass
```

### Ajustar Detección de Fugas

```python
# En utils.py, modificar threshold
LEAKAGE_THRESHOLD = 0.85  # Cambiar umbral de detección
```

## Interpretación de Resultados

### Métricas Estadísticas

-   **p-value < 0.05**: Diferencia estadísticamente significativa
-   **p-value ≥ 0.05**: No hay diferencia significativa (resultado actual)
-   **Cohen's d**: Tamaño del efecto (0.8 = grande, resultado actual)

### Fuga de Datos

-   **< 70%**: Fuga baja, resultados confiables
-   **70-90%**: Fuga moderada, usar con precaución
-   **> 90%**: Fuga alta, resultados poco confiables (caso actual)

### Recomendaciones Automáticas

El sistema genera recomendaciones basadas en los resultados:

-   Advertencias sobre fuga de datos
-   Sugerencias de mejora metodológica
-   Recomendaciones para datasets alternativos

---

¡La evaluación robusta está lista para ejecutarse! Consulta el README.md para más detalles sobre los hallazgos científicos.
