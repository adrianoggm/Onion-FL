# Onion-FL: qué tiene que ser capaz de hacer

> Revisada el 2026-11-04; se integra con la issue I1 (#185). Complementa la spec de infraestructura `2026-11-03-onion-fl-infrastructure-design.md` (I1); no la sustituye. Recoge lo que exige la auditoría bibliográfica del gap GS-20261103-700-R (carpeta `Doctorado/research/audit-runs/2026-11-03-onion-fl/`).

## 1. Por qué esta spec

La auditoría cambió el encaje científico de Onion-FL:

- **Ya está publicado** que el error del tiempo total de un entrenamiento de épocas fijas puede ser < 2 % en simulación, frente a máquinas HPC con límites de CPU y red inyectados, no dispositivos edge (FedDES, ACM SEC 2025, 10.1145/3769102.3770613). También que, en FL **síncrono** con servidor central, el modelo analítico rondas × tiempo por ronda conserva el ranking de configuraciones frente al despliegue (τ = 0,93/0,91), mejor que la simulación (τ = 0,61 en FLAIR) (Valli et al., CCGrid 2026, 10.1109/CCGRID68966.2026.00052). También está publicado que asignar perfiles de dispositivo independientes de los datos cambia qué configuración gana (Maeng et al., RecSys 2022, 10.1145/3523227.3546759; REFL, EuroSys 2023, 10.1145/3552326.3567485).
- **Sigue abierto** (sin FATAL en la auditoría, pendiente de piloto): un **diagnóstico ex ante** de cuándo falla el predictor factorizado al elegir configuración en FL jerárquico, cuando un **deadline** acopla qué clientes participan con cuánto pesan sus datos. Explicaría por qué FLASH (WWW 2021) ve un efecto «trivial» del acoplamiento datos–dispositivo y Maeng uno grande.
- **Plataformas competidoras** (Flight, FedDES, FederNet, FLEET, Valli): ninguna combina jerarquía Edge–Fog–Cloud con validación frente a hardware real. Ese es el hueco del artículo de ingeniería.

Por eso Onion-FL persigue dos líneas, con requisitos distintos:

| Línea | Pregunta | Necesita | No necesita |
|---|---|---|---|
| **A. Científica** | ¿Predice un diagnóstico barato, antes de desplegar, cuándo el predictor factorizado elige una configuración equivocada en FL jerárquico con deadline y acoplamiento datos–dispositivo? | Simulación con acoplamiento controlado, deadline, jerarquía, techo de ruido y métricas de decisión (§3, N1–N8) | Emulación ni despliegue (salvo para comprobar que los tiempos de la simulación son creíbles) |
| **B. Ingeniería** | ¿Puede una plataforma ejecutar la misma federación jerárquica en simulación, emulación y despliegue, con fidelidad medida frente a hardware real? | Todo lo de I1 (backends, infraestructura, calibración, fidelidad) | El diagnóstico de A |

**Regla:** A no espera a B. El piloto de A (§5) se hace con el backend `sim` actual más N1–N8.

## 2. Capacidades: inventario completo

Los identificadores de capacidad llevan el prefijo `CAP-` al citarse fuera de esta spec (`CAP-D4`, `CAP-N1`…), para no chocar con los de las issues (`[S1]` #182, `[M1]` #193, E5/E6). Estado: **Hoy** = ya está en el código; **I1 §x** = diseñado en la spec I1; **Nuevo** = lo añade esta spec. Columna A/B: para qué línea es imprescindible.

### 2.1 Datos y clientes

| ID | Capacidad | Estado | A/B |
|---|---|---|---|
| D1 | Cliente lógico por sujeto, evaluación con sujetos disjuntos, sin columnas meta como features | Hoy | A, B |
| D2 | Separar cliente lógico, partición y nodo físico | I1 §4 | B |
| D3 | Asignación de clientes a dispositivos reproducible y con semilla propia | I1 §5 | A, B |
| D4 | **Asignación con correlación controlada** entre la velocidad del dispositivo y un estadístico de los datos del cliente (N1) | Nuevo | A |
| D5 | **Dataset con más clientes naturales** que SWELL/WESAD, real, no sintético (N2) | Nuevo | A |

### 2.2 Federación y protocolo

| ID | Capacidad | Estado | A/B |
|---|---|---|---|
| F1 | Árbol de niveles arbitrarios, agregador por nodo, quórum, deadline por nodo | Hoy | A, B |
| F2 | Participación `all` y `fraction`; actualizaciones tardías `drop` o `next_round` con ponderación por antigüedad; cierre al quórum (FedBuff) | Hoy | A |
| F3 | **Participación dependiente del sistema**: seleccionar hijos por velocidad estimada (estilo Oort) o por probabilidad de cumplir el deadline (N3) | Nuevo | A |
| F4 | **Frecuencias de agregación por nivel** (κ₁ rondas edge→fog por cada fog→cloud) explícitas y barribles | Nuevo, y no se puede expresar hoy: el fog reenvía hacia arriba cada ronda que cierra (`Aggregator._closed`). Exige subrondas locales en el fog, que tocan la evaluación por zona, las tardías y los streams de C3. Fuera del primer piloto (§5) | A (posterior) |
| F5 | Equivalencia FedAvg plano = jerárquico cuando el fog agrega cada ronda con pesos por muestras (test) | Hoy a nivel de agregador (`test_hierarchical_fedavg_equals_flat_fedavg`); falta el test de extremo a extremo con una federación | A, B |

### 2.3 Sistema: cómputo, red, disponibilidad

| ID | Capacidad | Estado | A/B |
|---|---|---|---|
| S1 | Cómputo `samples_per_second` y `measured`; disponibilidad `always`, `bernoulli`, `schedule`, `crash_at`; perfiles de enlace | Hoy | A, B |
| S2 | Perfiles con nombre, coste de agregar, contención, memoria | I1 §7 | B (A solo coste de agregar) |
| S3 | Red física con sitios, caminos, ancho de banda compartido, red dinámica | I1 §8 | B |
| S4 | **Variabilidad del tiempo de cómputo por cliente** (distribución, no solo tasa fija) para que P(cumplir deadline) sea < 1 sin ser 0 o 1 (N4). La latencia de red ya es aleatoria hoy (normal, lognormal, jitter); el cómputo es determinista | Nuevo | A |
| S5 | Fallos de host y enlace, failover | I1 §9 | B |

### 2.4 Ejecución

| ID | Capacidad | Estado | A/B |
|---|---|---|---|
| E1 | `sim` (reloj virtual) y `local`/MQTT con el mismo modelo final | Hoy | A, B |
| E2 | Backends `emulated` y `distributed`, contrato común | I1 §10–11 | B |
| E3 | gRPC | I1 §3 | B |

### 2.5 Medición

| ID | Capacidad | Estado | A/B |
|---|---|---|---|
| M1 | Eventos JSONL, ejecución firmada, `compare`, report | Hoy | A, B |
| M2 | Fidelidad funcional, de rendimiento y de decisión (τ-b, top-1, regret, IC bootstrap) | I1 §12 | A, B |
| M3 | **Tiempo hasta objetivo** (primer instante en que la métrica de validación alcanza un umbral), con censura si no se alcanza | I1 §12 lo nombra; Nuevo como métrica de primera clase (N5) | A |
| M4 | **Techo de ruido**: réplicas con semilla y HPO re-aleatorizadas de la misma configuración, y regret/top-1 entre réplicas (N6) | Nuevo | A |
| M5 | **Predictor factorizado** rondas × tiempo por ronda (Luo 2021 / Valli 2026) como «backend» de decisión más (N7) | Nuevo | A |
| M6 | **Diagnóstico ex ante** del sesgo inducido por el deadline, calculado sin entrenar la federación completa (N8) | Nuevo | A |
| M7 | Calibración y validación separadas | I1 §13 | B |

### 2.6 Reproducibilidad

| ID | Capacidad | Estado | A/B |
|---|---|---|---|
| R1 | `topology_id`, `config_id`, `run_id`, `run_hash`; semillas por nodo y por rol | Hoy | A, B |
| R2 | `infrastructure_id`, manifiesto con backend y entorno | I1 §5 | B |
| R3 | La semilla del acoplamiento (N1) y la de HPO (N6) son propias y entran en `config_id` | Nuevo | A |

## 3. Capacidades nuevas en detalle

### N1. Asignación con acoplamiento controlado (D4)

- Depende del registro `assignments` de I3 (I1 §5).
- Plugin de `assignments` `coupled`: ordena clientes por un estadístico `s_i` (nº de muestras, divergencia de la distribución de etiquetas frente a la global, o pérdida inicial) y dispositivos por velocidad, y los empareja con una correlación de rango objetivo `rho ∈ [-1, 1]` mediante una cópula gaussiana con semilla propia. `rho = 0` = asignación independiente (la de hoy y la de casi todos los simuladores).
- `plan` informa del Spearman efectivo conseguido y del estadístico usado.
- **Aceptación:** con 1.000 semillas, el Spearman medio está a ±0,05 de `rho`; con `rho = 0` coincide en distribución con `profile_mix`; cambiar `rho` no cambia los datos ni los sujetos de test.

### N2. Dataset con más clientes (D5)

- Con 15 (WESAD) o 25 (SWELL) clientes, el techo de ruido probablemente tapa las diferencias entre configuraciones (lo apunta la auditoría GS-30-R: FedAvg y FedProx empatan dentro de una desviación en pfl-research).
- Hace falta un lector del formato JSON de LEAF y una **submuestra fija de escritores** declarada en el descriptor: el FEMNIST completo tiene unos 3.500 escritores y nunca hemos simulado miles de edges, así que la escala se comprueba antes.
- Añadir un descriptor de un dataset federado **natural y real** con cientos de clientes, compatible con la regla «sin datos sintéticos»: FEMNIST (LEAF, por escritor) es el candidato obvio; FLAIR o Fed-ISIC2019 si el modelo lo permite. Requiere comprobar que `modular_mlp` da señal útil (no un predictor constante, lección de C3–C5).
- **Aceptación:** descriptor en `datasets/`, sujetos de test reservados, y una ejecución FedAvg de referencia cuya métrica supera claramente a la clase mayoritaria.

### N3. Participación dependiente del sistema (F3)

- Plugins de `participation`:
  - `fastest_k`: los k hijos con menor tiempo estimado;
  - `deadline_feasible`: los hijos cuya probabilidad estimada de cumplir el deadline supera un umbral;
  - `utility` (estilo Oort, simplificado): puntuación = utilidad estadística × penalización por lentitud.
- La estimación usa solo lo que un coordinador real sabe (tiempos de rondas anteriores), nunca el perfil verdadero.
- **Aceptación:** tests con stub trainer: con perfiles conocidos, la selección es la esperada y determinista por semilla.

### N4. Tiempo de ronda estocástico por cliente (S4)

- Plugin de cómputo `lognormal_rate` (media y dispersión por perfil) y, opcional, `trace` (reproducir tiempos medidos).
- Sin esto, cuando el cómputo domina, cada cliente cumple casi siempre o casi nunca el deadline y el acoplamiento se reduce a un filtro casi determinista (la red ya aporta algo de variabilidad, pero poca si los enlaces son rápidos).
- La duración la saca el modelo de cómputo de la semilla del nodo, para que la misma semilla dé los mismos tiempos: hoy `duration(samples, wall_s)` no recibe semilla, así que la interfaz cambia (con compatibilidad para los modelos actuales).
- **Aceptación:** la fracción de clientes que cumple el deadline en simulación coincide con la calculada analíticamente para la distribución (test).

### N5. Tiempo hasta objetivo (M3)

- Métrica derivada de los eventos: primer instante (reloj del backend) en que la métrica de validación supera un umbral, con el umbral fijado antes de ejecutar (por ejemplo, el 95 % del máximo de la mejor configuración en una ejecución piloto separada).
- Ejecuciones que no lo alcanzan: censuradas a la duración máxima, y se reportan como tales.
- **Aceptación:** test sobre una curva conocida; `compare` la muestra con IC bootstrap.

### N6. Techo de ruido (M4)

- Necesita una búsqueda aleatoria de hiperparámetros con presupuesto fijo, que hoy no existe (solo hay barridos).
- `onion_fl ceiling <experimento>`: para cada configuración, R réplicas con semilla de entrenamiento y, opcionalmente, búsqueda de HPO re-aleatorizada (presupuesto fijo, por ejemplo 16 configuraciones aleatorias); calcula top-1 y regret **entre réplicas** de la misma «verdad».
- Es el denominador de toda conclusión de A: un regret del predictor solo cuenta si supera el regret entre réplicas.
- **Aceptación:** con un experimento stub de configuraciones idénticas, el top-1 entre réplicas es el esperado por azar.

### N7. Predictor factorizado (M5)

- Dos componentes que se estiman por separado, como en Luo 2021 y Valli 2026:
  - rondas hasta objetivo, medidas en simulación **sin deadline ni acoplamiento** (lo que daría un estudio solo de datos);
  - tiempo por ronda, calculado analíticamente con los modelos de cómputo y red (max sobre los participantes de cómputo + comunicación, más agregación).
- El predictor elige la configuración que minimiza su producto. Depende del arnés de fidelidad (I7). Se trata como un backend más en el arnés de fidelidad (`onion_fl fidelity predictor sim`).
- Variante con corrección de participación no ignorable (estilo FLOSS, arXiv 2507.23115) como baseline adicional.
- **Aceptación:** en síncrono sin deadline, su ganador coincide con el de `sim` en un experimento de control (reproduce el resultado de Valli).

### N8. Diagnóstico ex ante (M6)

- Calculado **antes** de entrenar la federación completa, con: los modelos de cómputo y red (probabilidad `p_i` de que el cliente i cumpla el deadline), y un estadístico barato de los datos de cada cliente (`d_i`: divergencia de su distribución de etiquetas, o norma del gradiente local en el modelo inicial, una pasada).
- Candidato principal: `D = Cov_i(p_i, w_i · d_i)` normalizada, donde `w_i` es el peso de agregación. Intuición: si los clientes que no llegan son los que más aportan algo distinto, el modelo que se entrena no es el que asume el predictor.
- Baselines obligatorios del diagnóstico: la tasa de participación media sola, y la corrección estilo FLOSS.
- **Aceptación:** se calcula en segundos para cualquier configuración del barrido y se guarda en el `run.json`.

## 4. Lo que la plataforma debe poder demostrar (línea B)

Para que el artículo de ingeniería no se quede en «otro simulador», Onion-FL tiene que enseñar, con artefactos firmados:

1. La **misma federación jerárquica** (mismo `topology_id` y datos) ejecutada en `sim`, `emulated` y `distributed`, con el modelo final idéntico en condiciones deterministas (extensión de `test_runtime_equivalence`).
2. **Fidelidad de rendimiento** frente a hardware real con calibración y validación separadas (I1 §12–13), incluyendo agregación en fog, que FedDES y Valli no tienen (topología en estrella).
3. **Fidelidad de decisión** en régimen síncrono (comparable con Valli) **y** con deadline/asincronía (lo que Valli deja como trabajo futuro).
4. Comparativa honesta con Flight, FedDES, FederNet, FLEET y Valli: tabla de capacidades citada a texto completo, no a abstract.

## 5. Piloto de la línea A (decide si el gap existe)

Todo en `sim`, sin hardware.

- **Datos:** FEMNIST (N2) como principal; SWELL/WESAD solo si el paso 0 muestra que el ruido lo permite.
- **Federación:** plana, 2 fogs, 4 fogs; κ₁ = 1 (κ₁ > 1 no existe todavía, F4; entra en un segundo piloto si el primero sale positivo).
- **Algoritmos:** dos de familias distintas (FedAvg y FedAdam o SCAFFOLD; no FedAvg frente a FedProx, que empatan dentro del ruido).
- **Régimen:** síncrono sin deadline (control) / deadline con `drop` / FedBuff.
- **Acoplamiento:** `rho ∈ {-0.6, 0, 0.6}` sobre nº de muestras y sobre divergencia de etiquetas.
- **Pasos:**
  0. Techo de ruido (N6) con 5 semillas × 2 réplicas de HPO. **Si el top-1 entre réplicas < 0,6, se para:** la pregunta no es identificable con estos datos.
  1. Verdad = `sim` con acoplamiento y deadline. Predictor = N7 (con y sin corrección FLOSS). Simulación ingenua = `sim` con `rho = 0`.
  2. Regret de time-to-target del predictor y de la simulación ingenua frente a la verdad, comparado con el techo.
  3. AUC del diagnóstico (N8) para predecir «regret > techo», frente a sus baselines.
- **Criterios de caída (fijados antes de ejecutar):**
  - regret ≈ 0 (dentro del techo) también con deadline y `|rho| = 0,6` → el gap cae;
  - el diagnóstico no supera a la tasa de participación sola (AUC con IC solapado) → cae la contribución del diagnóstico, queda un resultado empírico menor.
- **Coste:** del orden de cientos de ejecuciones cortas en CPU/GPU modesta; días, no semanas.

## 6. Cambios a la spec I1 (aplicados el 2026-11-04)

1. **§1, pregunta 1:** dejar de presentarla como abierta en general; citar Valli (CCGrid 2026) y FedDES, y acotar la novedad a jerarquía + deadline/asincronía + validación frente a hardware real.
2. **§12, decisión:** añadir el techo de ruido (N6) como requisito previo de cualquier métrica de decisión, y el predictor factorizado (N7) como comparador obligatorio.
3. **§16, orden:** permitir que N1–N8 avancen en paralelo a v0.8.0, sin esperar a los backends `emulated`/`distributed`.

## 7. Decisiones de la revisión

| Pregunta | Decisión |
|---|---|
| ¿FEMNIST con `modular_mlp` o con un tronco convolucional? | `modular_mlp` sobre los 784 píxeles para el piloto: el criterio de N2 (superar claramente a la clase mayoritaria, ~1/62) lo comprueba. Un tronco convolucional sería un plugin de modelo nuevo, posterior |
| ¿Qué estadístico por defecto? | En N1, el nº de muestras: es el peso de FedAvg, así que mide justo el sesgo de dejar fuera a los lentos; la divergencia de etiquetas es el segundo valor del barrido. En N8, la divergencia de etiquetas, porque el nº de muestras ya está en w_i |
| ¿Se puede expresar κ₁? | No (F4). Se saca del primer piloto |
| Hardware real | Raspberry Pi, el PC de desarrollo, un servidor Linux y, opcionalmente, dos móviles (I1 §17) |

**Issues.** N1–N8 son las issues A1–A8 (#205–#212), y el piloto de §5 es A9 (#213).
