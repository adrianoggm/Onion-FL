# Onion-FL: diseño del framework de experimentación federada

- **Fecha:** 2026-10-02
- **Estado:** pendiente de revisión
- **Alcance de esta especificación:** subproyectos 1 a 4 más el transporte MQTT (primer corte). Los subproyectos 5, 6 y 7 tendrán su propia especificación.

---

## 1. Objetivo

Onion-FL pasa de ser un prototipo SWELL edge-fog-cloud a convertirse en una **herramienta de investigación de topologías de aprendizaje federado**. Con ella, un investigador puede declarar por configuración, sin tocar código:

- la topología (un árbol de cualquier profundidad);
- los datasets y en qué nodos edge cae cada sujeto;
- la técnica de aprendizaje y qué se comparte en cada nivel;
- el transporte y la red de cada enlace.

A partir de esa declaración, la herramienta ejecuta los experimentos en simulación o en modo real y produce métricas comparables por dataset, por nivel (edge, fog, global) y por ronda.

**Caso de uso guía.** Se toman dos datasets A y B y cuatro fogs. Primero, dos fogs solo con clientes de A y dos solo con clientes de B. Después, los mismos clientes repartidos de forma mezclada entre los cuatro fogs. El framework debe mostrar qué ocurre en cada caso. Que los resultados mejoren o empeoren depende de la técnica de aprendizaje; lo que el framework debe garantizar es la **instrumentación** que permite estudiarlo.

**Criterio de éxito.** Un experimento se describe solo en YAML (escenarios × semillas), se ejecuta con un comando y produce resultados marcados con la topología que los generó, guardados en disco y comparables entre escenarios y topologías.

---

## 2. Decisiones tomadas

| # | Decisión | Elección |
|---|---|---|
| D1 | Compartir conocimiento entre datasets | Modelo modular (adaptador por dataset, tronco común, cabeza por tarea) con alcance de compartición configurable |
| D2 | Ritmo de las rondas | Lo marca el coordinador; quórum y plazo por nodo |
| D3 | Nodo fog | Un nodo propio por fog (no un broker central con bridges) |
| D4 | Protocolos | Transporte enchufable y configurable por enlace (memoria, MQTT, gRPC, Flower…) |
| D5 | Código existente | Refactor progresivo de este repositorio, borrando todo lo inservible |
| D6 | Topologías | Árboles de cualquier profundidad; las interfaces no impiden grafos generales en el futuro |
| D7 | Tiempo en simulación | Reloj virtual de eventos discretos, en un solo proceso |
| D8 | Reparto de clientes | Cualquier estrategia por config: lista explícita, mezcla α, Dirichlet β, sesgo de etiquetas, o una propia |
| D9 | Evaluación | En los tres niveles (edge, fog, global), configurable |
| D10 | Visualización de resultados | Informe HTML, API de análisis con pandas, Grafana en vivo y Jaeger |
| D11 | Primer corte | Incluye el transporte MQTT, para poder borrar el runtime antiguo al terminarlo |
| D12 | Nombre del paquete | `flower_basic` pasa a llamarse `onion_fl` |
| D13 | Limpieza | Se borran la ruta ECG5000, los scripts rotos o duplicados, el CI, Docker y metadatos que no funcionan, y los docs y diagramas obsoletos |
| D14 | Modelo de programación de un nodo | Máquina de estados dirigida por mensajes y temporizadores |
| D15 | Datos | El edge no conoce ningún dataset. Cada dataset se describe en un YAML de ingesta, y solo los formatos exóticos necesitan un lector en código |
| D16 | Front | Subproyecto propio, que se construye justo después del núcleo. El núcleo nace con lo que el front necesita (§11) |
| D17 | Firma de resultados | Huellas SHA-256 (`topology_id`, `config_id`, `data_id`, `run_hash`) más `run_id` con fecha UTC y sha de la ejecución |
| D18 | Tests sin datos | Entrenador de prueba para los tests de protocolo. La regla de "no usar datos sintéticos" se aplica a todo lo que evalúe aprendizaje |
| D19 | Flujo de trabajo | Issues de GitHub y ramas `task/#N` que entran por PR en `develop`. `main` solo recibe versiones funcionales etiquetadas |

---

## 3. Subproyectos y orden

| # | Subproyecto | Especificación |
|---|---|---|
| 1 | Núcleo: mensajes, nodos, topología, registros y runtime de simulación | Esta |
| 2 | Aprendizaje enchufable | Esta |
| 3 | Datos: ingesta declarativa, caché, roles de sujeto y repartos | Esta |
| 4 | Instrumentación y experimentos | Esta |
| — | Modo real con transporte MQTT | Esta (D11) |
| 7 | Front: biblioteca y editor de topologías, monitor y tutorial | Siguiente |
| 5 | Transportes gRPC y Flower, emulación avanzada y métricas por enlace | Posterior |
| 6 | Despliegue real: contenedores, distribución de la config y coordinación | Posterior |

---

## 4. Principios de diseño

1. **La lógica federada no conoce el transporte.** Los nodos intercambian mensajes tipados y un transporte se encarga de entregarlos.
2. **Cada eje experimental es un plugin elegido por config.** Esto incluye reparto, transporte, codec, agregador, entrenador, optimizador de servidor, compartición, modelo, inicialización, dataset, participación, staleness, métricas, diagnósticos y sinks. El framework trae implementaciones de serie, y se registran otras sin tocar el núcleo, con un nombre o con una ruta del tipo `paquete.modulo:Clase`.
3. **El edge no sabe de datasets.** En `roles/`, `learning/` y `runtime/` no aparece el nombre de ningún dataset, y hay un test que lo comprueba.
4. **Se planifica con funciones puras y la entrada/salida queda en los bordes.** Construir la topología, aplicar el reparto, resolver la compartición y montar el plan de una ejecución no tiene efectos secundarios. Gracias a eso es posible la previsualización del front.
5. **Hay determinismo.** La misma config y la misma semilla producen la misma ejecución en simulación.
6. **La instrumentación es parte central del diseño.** Todo lo que ocurre emite eventos etiquetados.

---

## 5. Núcleo

### 5.1 Conceptos

- **`Message`:** dataclass inmutable con estos campos:
  - `kind`: `hello`, `global_model`, `update`, `eval_request`, `eval_report` o `control`;
  - `src` y `dst`;
  - `round`;
  - `payload`: un estado de modelo por clave, el número de muestras por clave y métricas;
  - `meta`: contexto de traza, bytes codificados e instante de envío.
- **Codec:** se encarga de la serialización y es un plugin. Vienen de serie `json` y uno binario (`npz`). Cada enlace usa uno, y el tamaño codificado cuenta para el modelo de red y para las métricas de comunicación.
- **`Node`:** máquina de estados con `on_start(ctx)`, `on_message(msg, ctx)` y `on_timer(nombre, ctx)`. No tiene hilos, sockets ni reloj propio.
- **`Context`:** la única vía de un nodo hacia fuera:
  - `send(msg)` para enviar mensajes (el destino es `msg.dst`, y `msg.src` debe ser el propio nodo);
  - `set_timer(retardo, nombre)` y `cancel_timer(nombre)` para los temporizadores;
  - `now()` para consultar el reloj;
  - `rng` para obtener aleatoriedad con semilla derivada del experimento y del id del nodo;
  - `emit(nombre, value=…, **tags)` para la instrumentación;
  - `compute(muestras)` para declarar el trabajo hecho en el manejador. El runtime lo convierte en tiempo ocupado con el modelo de cómputo del nodo, y los mensajes enviados en ese manejador salen al terminar.
- **`Runtime`:** contiene los nodos, entrega los mensajes, dispara los temporizadores y lleva el reloj. Hay dos implementaciones: `SimRuntime` (§9.1) y `RealRuntime` (§9.2).
- **`Transport`:** `start()`, `send(dirección, bytes)`, `on_receive(callback)` y `stop()`. Cada enlace elige el suyo en la topología.
- **`Topology`:** árbol validado. Cada nodo tiene `id`, `parent`, `level`, rol y ajustes, y cada enlace tiene transporte, codec y perfil de red. Expone `topology_id` (§10.1) y `to_graph()` (§11).
- **Registros:** asocian un nombre con una fábrica y unos metadatos (título, descripción y modelo pydantic de parámetros con rangos y valores por defecto, más un texto explicativo). Existe uno por cada eje del principio 2.

### 5.2 Estructura del paquete

```
src/onion_fl/
├── core/           message, codec, node, context, topology, registry, ids
├── roles/          coordinator, aggregator, edge (incluye el modo evaluador)
├── runtime/        sim (reloj virtual, enlaces, cómputo, disponibilidad), real
├── transports/     mqtt (en simulación, el transporte en memoria es la capa de enlaces de runtime/sim)
├── learning/       model, sharing, aggregators, trainers, server_optimizers, init
├── data/           contract, ingest (lectores y pasos), cache, roles, placement
├── observability/  events, sinks/{jsonl,prometheus,otel}, diagnostics, metrics, analysis, report
└── experiment/     config (pydantic), sweep, plan, cli
```

### 5.3 Qué se reaprovecha

| Código actual | Pasa a |
|---|---|
| `runtime_protocol.py` (envelopes, serialización) | `core/message`, `core/codec` |
| `training/local.py` | `learning/trainers` (`standard`) |
| `datasets/swell.py`, `sweet_samples.py`, `wesad.py` | Lógica de lectura dentro de `data/ingest` (lector `wesad_pickle`) y los YAML de `datasets/` |
| `brokers/federated_base.weighted_average` | `learning/aggregators.fedavg`, ahora por clave |
| Política de staleness `accept`/`strict` | Plugin de staleness (`next_round`/`drop`) |
| `telemetry.py`, `prometheus_metrics.py` | `observability/sinks/otel`, `observability/sinks/prometheus` |
| `federated_architecture.py` (parseo y validación) | `experiment/config`, `core/topology` |
| `docker/docker-compose.otel.yml`, Grafana | Se mantienen, con métricas y paneles adaptados a las etiquetas nuevas |

---

## 6. Roles y protocolo de rondas

### 6.1 Roles

| Rol | Comportamiento |
|---|---|
| **Coordinator** (raíz) | Inicializa el modelo con el plugin `init`. Espera a que su subárbol se registre (`hello`) o a que venza un plazo. En cada ronda *r*: selecciona qué hijos participan, envía `global_model(r)`, arma el plazo, cierra la ronda, agrega, aplica el optimizador de servidor, emite eventos y lanza las evaluaciones que pida la config. Se detiene al llegar a `rounds` o al cumplirse un criterio de parada. |
| **Aggregator** (cualquier nivel) | Al recibir `global_model(r)` de su padre, inyecta los grupos de parámetros cuyo alcance termina en su nivel (§7.2), selecciona qué hijos participan y reenvía el modelo. Después recoge los `update(r)`, cierra la ronda, agrega con su plugin, calcula los diagnósticos y envía `update(r)` a su padre con las muestras por clave. Guarda el modelo de su zona. |
| **Edge** | Al recibir `global_model(r)`, carga las claves compartidas y conserva sus grupos locales. Entrena con el plugin de entrenador y responde con `update(r)`, que contiene solo los grupos que permite la política de compartición, junto con las muestras por clave y las métricas de entrenamiento. Evalúa en local según la config. |
| **Evaluador** | Es un edge con `train: false` y sujetos reservados. Solo responde a `eval_request` con `eval_report`. |

### 6.2 Cierre de ronda

- **Parámetros por nodo o por nivel:**
  - `quorum`: fracción o número de hijos;
  - `deadline`: tiempo desde el reenvío del global.
- **Cuándo se cierra:** cuando responden todos los participantes, o cuando vence el plazo con quórum. Si vence sin quórum, el nodo envía un `update` vacío y emite `round.quorum_failed`.
- **Updates tardíos:** se gestionan con el plugin de staleness. Vienen de serie `drop` y `next_round(weighting)`; este último aplica una ponderación por antigüedad que es a su vez un plugin.
- **Participación:** se gestiona con un plugin. Vienen de serie `all` y `fraction(p)` con `rng`.

### 6.3 Pesos jerárquicos

Cada `update` lleva el número de muestras por clave. Cada agregador:
- promedia cada clave solo entre los hijos que la contienen;
- envía hacia arriba la suma de muestras de cada clave.

Con participación completa y `fedavg` en todos los niveles, el resultado equivale a un FedAvg plano sobre todas las muestras edge. Esto sustituye el peso fijo `num_samples=1000` actual.

### 6.4 Errores

- Un mensaje malformado o de un remitente desconocido se rechaza y emite `message.rejected`.
- Si falla el entrenamiento de un edge, se emite `edge.train_failed` y ese edge cuenta como ausente a efectos del quórum.
- Los errores de configuración se detectan antes de arrancar (§12.3). En ejecución, ningún error de un nodo detiene la simulación.

### 6.5 Orden de agregación

Los agregadores ordenan las contribuciones por el id del nodo emisor antes de agregar. Así el resultado no depende del orden de llegada, y simulación y modo real coinciden (§9.3).

---

## 7. Aprendizaje

### 7.1 Modelo modular

Las claves de los parámetros llevan espacio de nombres:

```
adapter.<dataset>.* | adapter.*   entrada: n_features del dataset → ancho común (o uno común)
trunk.* | trunk.<dataset>.*
head.<tarea>.* | head.<dataset>.*
```

- La config del modelo `modular_mlp` decide:
  - la anchura del adaptador, y `adapters: per_dataset | shared` (uno común exige las mismas features en todos los datasets);
  - las capas ocultas del tronco;
  - `trunk: shared | per_dataset`;
  - `heads: per_task | per_dataset`;
  - el dropout.
- Cada nodo instancia solo las partes que le corresponden, dimensionadas a partir del `meta` de sus datos (§8.1).
- Los coordinadores y los agregadores no necesitan instanciar un modelo para agregar: trabajan con diccionarios de arrays.

### 7.2 Compartición

Una política asigna a cada grupo de parámetros un alcance:

| Alcance | Significado |
|---|---|
| `global` | Se agrega hasta el coordinador |
| `level:<nombre>` | Se agrega hasta ese nivel; el agregador de ese nivel lo guarda y lo inyecta en el global que reenvía a sus hijos |
| `local` | No sale del edge |

Un grupo cruza el enlace entre un hijo y su padre, en los dos sentidos, si su alcance es `global` o `level:X` con X en el nivel del padre o por encima. `level:<nombre>` solo admite niveles de agregación intermedios (ni la raíz ni el edge). `traffic()` devuelve qué grupos cruzan cada enlace, para la previsualización del front.

**Presets:**

| Preset | Configuración | Equivale a |
|---|---|---|
| `fedavg` | Todo `global` | — |
| `fedper` | Tronco y adaptadores `global`, cabezas `local` | — |
| `zone(level="fog")` | Tronco y adaptadores `global`, cabezas `level:<level>` | — |
| `independent` | Todo `global`; exige `trunk` y `heads` por dataset | Un modelo por dataset sobre la misma infraestructura |
| `harmonized` | Todo `global`; exige `adapters: shared` | Un único modelo FedAvg sobre features armonizadas |

### 7.3 Plugins de serie

| Eje | Interfaz | De serie |
|---|---|---|
| Agregador | `aggregate(contribuciones) -> contribución` (por clave) | `fedavg`, `mean`, `median`, `trimmed_mean(β)` |
| Optimizador de servidor | `apply(global, agregado) -> global` | `replace`, `fedavgm`, `fedadam` |
| Entrenador | `train(modelo, datos, recibido, ctx) -> TrainResult` | `standard` (épocas, batch, lr, optimizador, grupos congelados), `fedprox(μ)` |
| Inicialización | `init(modelo, ctx)` | `random(seed)`, `checkpoint(ruta, grupos)` |

`checkpoint` permite partir de un modelo preentrenado, por ejemplo el tronco de un baseline centralizado. Así queda cubierta la transferencia de aprendizaje que hoy no está conectada en SWEET.

Una contribución contiene `{estado: clave → array, pesos: clave → muestras, meta}`. `TrainResult` informa de pérdida, muestras procesadas y lotes.

---

## 8. Datos

### 8.1 Contrato

```
SubjectData:  X: float32[n, f]   y: int64[n]
meta:         dataset, subject, task, n_classes, feature_names, n_samples, class_counts
```

### 8.2 Ingesta declarativa

Cada dataset se describe en `datasets/<nombre>.yaml`:

| Elemento | Qué define |
|---|---|
| `source` | Lector y ruta, admitiendo patrones `{subject}` y glob |
| `subject` | Columna, regex o ruta de donde sale el sujeto |
| `label` | Columna, más mapa, umbral o intervalos, opcionalmente uniendo con otro fichero por clave o por tiempo; `strategies` con nombre para elegir en el experimento |
| `features` | Listas `include`/`exclude` o regex |
| `steps` | Pasos de transformación |

**Lectores de serie:** `csv`, `excel`, `parquet`, `pickle`, `npz` y `wesad_pickle`; este último es el único con código específico.

**Pasos de serie:** `replace`, `join`, `window` (tamaño, solape y estadísticos por canal) y `select`.

Se escriben `datasets/swell.yaml`, `datasets/sweet.yaml` y `datasets/wesad.yaml`, con los mismos resultados de lectura que los cargadores actuales salvo en las correcciones del apartado 8.6.

### 8.3 Caché

- `onion_fl data prepare <ds> [--option …]` escribe `data/cache/<ds>/<hash-opciones>/subject_<id>.npz` más `meta.json`.
- Su huella forma parte del `data_id`.
- `onion_fl data inspect <ds>` muestra la ficha del dataset: sujetos, muestras, balance de clases y valores ausentes. Avisa si alguna feature tiene una correlación absoluta mayor que 0,95 con la etiqueta.

### 8.4 Roles de sujeto

Son fijos para todo el experimento y no dependen del reparto:

| Rol | Uso |
|---|---|
| `test` | Sujetos reservados por dataset. Son evaluadores para la evaluación global e idénticos en todos los escenarios. |
| `val` | Sujetos reservados opcionales que el reparto asigna como evaluadores de zona. |
| `train` | La bolsa que se reparte. `local_val` separa una parte de cada sujeto para la validación local en el edge. |

- Los roles se eligen por proporción con semilla o con listas.
- `subjects_per_client: n` agrupa varios sujetos en un mismo cliente.
- El escalado y la imputación se ajustan solo con la bolsa de entrenamiento, con tres opciones: `global` (por dataset), `local` (en cada edge) o `none`.

### 8.5 Repartos

Un plugin de reparto recibe la bolsa de entrenamiento (dataset, sujeto, muestras e histograma de clases), los agregadores hoja, sus parámetros y un `rng`, y devuelve la asignación.

| Reparto | Definición |
|---|---|
| `explicit` | Listas por agregador. |
| `mixing(α)` | Cada agregador hoja tiene un dataset de casa (declarado, heredado o asignado por turnos). El dataset *d* reparte sus sujetos con `w_f(d) = (1−α)·casa_f(d) + α/F`, donde `casa_f(d) = 1/|fogs de casa de d|` si *d* es el dataset de casa de *f* y 0 en otro caso. Los redondeos se reparten por mayor resto, con desempate por `rng`. Un dataset sin ningún fog de casa se reparte siempre de forma uniforme. Con α=0 sale segregado; con α=1, todos los fogs tienen la misma proporción de cada dataset. |
| `dirichlet(β)` | `w_f(d) ~ Dir(β·1_F)` para cada dataset, y luego los mismos redondeos. |
| `label_skew(β)` | Se fijan proporciones de clase objetivo por agregador, muestreadas de `Dir(β)`, y se asignan sujetos de forma voraz para acercarse a ellas. |
| `pooled` | Todos los sujetos de entrenamiento en un único edge. Es el baseline centralizado comparable, con el mismo modelo y el mismo entrenador. |

Al montar un escenario se emite su composición: sujetos, muestras y clases por agregador, y la entropía de la mezcla de datasets.

### 8.6 Correcciones respecto al código actual

- La imputación y el filtrado de varianza de SWELL dejan de calcularse sobre todo el dataset antes del split.
- Se elimina la semilla basada en `hash()` de SWEET.
- Ya no se exige que existan los cuatro CSV de SWELL cuando solo se usa una modalidad.
- Se excluyen de forma explícita las columnas meta (`blok`, …).

---

## 9. Runtimes, red y transporte MQTT

### 9.1 SimRuntime

- **Cola de eventos** ordenada por (tiempo virtual, secuencia), con tres tipos de evento: `deliver`, `timer` y `compute_done`.
- **Entrega:** `t + latencia + bytes / ancho_de_banda + jitter`. Con probabilidad `p` el mensaje se pierde y se emite `link.dropped`.
- **Perfil de enlace:** latencia (fija, normal o lognormal), ancho de banda de subida y de bajada, pérdidas y jitter. Hay presets `lan`, `wifi`, `4g` y `lora`, y se pueden fijar por enlace o por nivel.
- **Cómputo:** el entrenamiento real se ejecuta al recibir el modelo y su duración virtual es `muestras × épocas / samples_per_second` (perfil del dispositivo). Con `compute: measured` se usa el tiempo de pared medido, que no es determinista.
- **Disponibilidad:** `always`, `bernoulli(p)` por ronda, `schedule` o `crash_at(t)`.
- **Escala:** se ejecuta en un solo proceso, con el objetivo de simular cientos de edges en un portátil.

### 9.2 RealRuntime

- **Ejecución:**
  - `onion_fl node --id <id> --run <run_id> --config exp.yaml` arranca un solo nodo.
  - `onion_fl run --mode real` lanza localmente un proceso por grupo de nodos.
- **Arranque:** registro con `hello` antes de la ronda 1, en lugar de los `sleep` actuales.
- **Estado periódico:** `node.heartbeat` con ronda, cola, CPU y memoria.
- **`MqttTransport`:**
  - cada nodo se suscribe a `onionfl/<run_id>/<node_id>/inbox`;
  - el QoS y el broker se configuran por enlace;
  - reconecta solo;
  - mide bytes, mensajes y latencia de extremo a extremo, para lo que necesita relojes sincronizados por NTP (el informe lo indica).

### 9.3 Equivalencia entre modos

Con quórum completo, la misma semilla da el mismo modelo final en `SimRuntime` y con `RealRuntime` + MQTT, salvo redondeos de coma flotante. Un test lo verifica.

---

## 10. Instrumentación

### 10.1 Identidad y firma

| Identificador | Definición |
|---|---|
| `topology_id` | SHA-256 de la topología canónica: árbol, roles, niveles, enlaces, transportes, codecs y perfiles. No depende del nombre. |
| `config_id` | SHA-256 de la config resuelta del escenario. |
| `data_id` | SHA-256 de las huellas de las cachés usadas. |
| `code_version` | Commit de git, indicando si había cambios sin confirmar. |
| `run_id` | `<fecha UTC compacta>-<sha12>`, donde el sha se calcula sobre `config_id`, semilla, instante de inicio en ns, máquina, PID y un valor aleatorio. Es único aunque la misma topología se ejecute en paralelo, y se ordena por fecha. |
| `run_hash` | Al terminar, SHA-256 de `run.json` (con `started_at`, `finished_at`, máquina e identificadores) más las huellas de `events.jsonl` y `summary.json`. |

Firma de un resultado: `topology_id` + `config_id` + `run_id` + `run_hash`.

Cada ejecución escribe `runs/<run_id>/` con `run.json`, `events.jsonl` y `summary.json`.

### 10.2 Esquema de evento

```
{t_virtual, t_wall, run_id, topology_id, scenario, seed, round,
 level, node, role, kind, name, value, tags: {dataset, model_source, link, group, ...}}
```

- `kind` toma uno de estos valores: `metric`, `message`, `lifecycle`, `diagnostic` o `data`.
- `level` toma `edge`, el nombre del nivel de agregación (`fog`, `region`…) o `global`.

### 10.3 Evaluación

| Nivel | Qué se evalúa | Configuración |
|---|---|---|
| Edge | Modelo local tras entrenar y modelo recibido, sobre `local_val` | `every`, `models` |
| Fog (cualquier nivel de agregación) | Informes de sus hijos agregados por muestras, y modelo de zona sobre sus evaluadores `val` | `every`, `aggregate_children`, `holdout` |
| Global | Modelo global sobre los evaluadores `test`, por dataset | `every` |

Las métricas son un plugin. Vienen de serie `loss`, `accuracy`, `macro_f1`, `recall_per_class` y `confusion_matrix`.

### 10.4 Diagnósticos de serie

Los calcula cada agregador al agregar y se pueden ampliar con plugins.

| Diagnóstico | Qué mide |
|---|---|
| Divergencia | Coseno y L2 entre los Δ de los hijos (contribución − modelo enviado), por grupo de parámetros |
| Conflicto entre datasets | Coseno entre el Δ medio por dataset en los grupos compartidos, en cada agregador donde coexisten varios datasets |
| Deriva | Norma del cambio por grupo entre rondas, y distancia entre el modelo de zona y el global |
| Participación | Seleccionados, respondidos, quórums fallidos y updates tardíos |
| Comunicación | Bytes y mensajes por enlace y ronda, duración de ronda (virtual y de pared) y tiempo hasta el quórum |
| Equidad | Mínimo, máximo y desviación de las métricas de los edges por dataset |

### 10.5 Sinks

| Sink | Uso |
|---|---|
| `jsonl` | Fuente de verdad |
| `prometheus` | En vivo, con etiquetas `run_id`, `topology_id`, `level`, `node`, `dataset` |
| `otel` | Un span por envío y recepción, enlazados a través de `meta` |
| Informe HTML | `onion_fl report` |
| API de análisis | `load_runs` |

### 10.6 Análisis y comparativas

Hay una sola implementación, que comparten el informe, los notebooks y el front:

```python
runs = load_runs("runs/", experiment="mix_ab")
runs.metrics(level="global", name="accuracy")  # DataFrame con todas las etiquetas
runs.compare(level="fog", metric="accuracy", by=["topology_id", "dataset"], over="seed")
```

`compare` devuelve la media ± IC sobre las semillas, por ronda. En niveles con varios nodos añade el mínimo, el máximo y la dispersión entre nodos, de modo que se pueden comparar topologías de forma distinta.

---

## 11. Requisitos del núcleo para el front (subproyecto 7)

1. **Esquema autodescrito:** los modelos pydantic se exportan a JSON Schema (`onion_fl schema`) junto con los metadatos de cada plugin (título, descripción, parámetros y texto explicativo).
2. **Previsualización en seco:** `plan` y las funciones puras (topología, reparto y compartición) se pueden invocar sin ejecutar nada. Devuelven la composición por fog y qué grupos de parámetros circulan por cada enlace.
3. **`topology.to_graph()`:** JSON con nodos (rol, nivel, composición de datasets) y enlaces (transporte, codec y perfil).
4. **Estado en vivo:** cada ejecución mantiene en `run.json` su estado (`planned`, `running`, `finished`, `failed`), sus eventos se pueden leer mientras corre, y en modo real hay heartbeats.

La API `onion_fl serve` (FastAPI) y la interfaz forman parte del subproyecto 7.

---

## 12. Configuración y experimentos

### 12.1 Ficheros

| Fichero | Contenido |
|---|---|
| `datasets/<nombre>.yaml` | Ingesta (§8.2) |
| `topologies/<nombre>.yaml` | Árbol en forma compacta, con `levels`, `root`, un bloque por nivel con `defaults` y `nodes` y bloque `edge`; o en forma general, como lista de nodos. La compacta se compila a la general. Los edges los crea el reparto. |
| `experiments/<nombre>.yaml` | Topología, datos (datasets y opciones, roles, reparto), aprendizaje (modelo, compartición, entrenador, optimizador, inicialización), `rounds`, evaluación, runtime (modo y codec), `sweep`, `seeds` y `sinks` |

### 12.2 Barridos

`sweep` asocia rutas con puntos a listas de valores (`data.placement.alpha: [0, 0.5, 1]`, `topology: [four_fogs, regions]`). Un escenario es cada combinación del producto cartesiano de esos valores, combinada con cada semilla, y tiene su `config_id`. En simulación, los escenarios pueden ejecutarse en paralelo en varios procesos.

### 12.3 Validación

pydantic valida toda la config antes de arrancar y señala la ruta exacta de cada error. Cada plugin valida sus propios parámetros.

### 12.4 CLI `onion_fl`

| Comando | Qué hace |
|---|---|
| `data prepare <ds>` / `data inspect <ds>` | Prepara la caché / muestra la ficha |
| `topology show <fichero> [--graph]` | Dibuja el árbol, da su `topology_id` y exporta el grafo en JSON |
| `plan <experimento>` | Ejecución en seco |
| `run <experimento> [--scenario …] [--mode sim\|real]` | Ejecuta |
| `node --id <id> --run <run_id> --config <experimento>` | Arranca un nodo suelto en modo real |
| `report <experimento\|runs…>` | Genera el informe HTML |
| `baseline <experimento>` | Baselines clásicos (LR, RF, XGBoost) sobre los mismos roles de sujeto |
| `schema` | Exporta el JSON Schema |

---

## 13. Migración y flujo de trabajo

### 13.1 Ramas

- `develop` se crea desde `main`.
- Cada issue tiene su rama `task/#N` y entra en `develop` por PR con el CI en verde.
- `main` solo recibe merges de versión desde `develop`, con etiqueta semver y notas de versión en GitHub Releases.

### 13.2 Fases

Al final de cada fase, `develop` funciona y los tests pasan. Cada fase tiene su propio plan de implementación detallado y sus issues; el primer plan cubre las fases 0 y 1.

**Fase 0. Higiene**
- Crear `develop`.
- Borrar lo del apartado 13.3.
- Mover los resultados a `results/legacy/` con un `INDEX.md` que indique el experimento, el script de origen, la fecha y sus salvedades (por ejemplo, "anterior a la corrección de `blok`").
- Corregir `pyproject.toml`: licencia Apache-2.0, dependencias reales (incluidas `pyyaml`, `opentelemetry-*` y `prometheus-client`; `pydantic` entra en la fase 7, que es la primera que lo usa), Python ≥ 3.11 y URLs. `requirements.txt` se elimina y `pyproject.toml` queda como única lista de dependencias.
- Usar ruff para lint y formato, en lugar de black, isort y flake8. Se borran `.flake8` y `.isort.cfg` y se actualiza `.pre-commit-config.yaml`.
- Configurar el CI con `ruff check`, `ruff format --check` y `pytest`.
- Actualizar el `justfile`: `test` ejecuta `pytest` y desaparecen las recetas que dependen de lo borrado.
- Mantener `codeql.yml` y reactivarlo en GitHub.

**Fase 1. Renombrado:** `flower_basic` pasa a `onion_fl`.

**Fase 2. Núcleo:** `core/` y `runtime/sim`, cuya capa de enlaces es el transporte en memoria.

**Fase 3. Aprendizaje:** `learning/`.

**Fase 4. Datos:** `data/`, los YAML de SWELL, SWEET y WESAD, y la lógica de los baselines clásicos (`onion_fl.baselines`) sobre los roles de sujeto. Se borran `validations/` y los scripts de extracción y muestras que la ingesta deje sin uso.

**Fase 5. Roles:** `roles/` y la evaluación en los tres niveles.

**Fase 6. Observabilidad:** `observability/`, Prometheus y OTEL portados, y dashboard de Grafana actualizado.

**Fase 7. Experimentos:** `experiment/` y el CLI, incluido `baseline`. Se borran `scripts/evaluate_*`, `run_subject_cv.py`, `train_sweet_baseline_selection1.py`, `prepare_sweet_baseline.py` y `evaluation.py`.

**Fase 8. Modo real:** `runtime/real` + `transports/mqtt`, con el test de equivalencia (Mosquitto como servicio en el CI).

**Fase 9. Paridad y borrado.** Primero se reproduce funcionalmente la ejecución de referencia de SWELL: 3 fogs, fisiología, split `global` con los sujetos de test de `swell_federated_10runs.yaml`. "Funcionalmente" significa que el pipeline completo funciona sobre el mismo split. No se exige igualdad numérica, porque se corrige el peso de los fogs. Después se borra:
- `clients/`, `brokers/` y `servers/`;
- `federated_architecture.py`, `runtime_protocol.py`, `datasets/swell_federated.py`, `datasets/sweet_federated.py`, `swell_model.py`, `sweet_model.py`, `telemetry.py`, `prometheus_metrics.py` y `training/`;
- `scripts/run_*.py`, `scripts/prepare_*_federated.py` y las configs antiguas;
- sus tests.

Las recetas del `justfile` se actualizan.

**Fase 10. Documentación:** README, CLAUDE.md, un doc de arquitectura generado a partir de esta especificación y diagramas exportados con `topology show --graph`.

### 13.3 Inventario de borrado de la fase 0

- **Raíz:** `d.md`, `quick_comparison.py`, `quick_format.bat`, `format_code.py`, `run_tests.py`, `setup_dev.py`, `setup_dev_environment.py`, `setup_environment.py`, `Makefile`, `mosquitto.conf`, `run_multi_dataset_demo.py` y `FINAL_COMPREHENSIVE_COMPARISON.png` (duplicado del de `advanced_ml_results/`).
- **Ruta ECG5000:**
  - `src/flower_basic/` → `server.py`, `client.py`, `fog_flower_client.py`, `model.py`, `compare_models.py`, `baseline_model.py`, `utils.py`, `__main__.py`, `datasets/ecg5000.py` y sus exportaciones en los `__init__.py`;
  - la rama no-SWELL de `federated_architecture.py`;
  - `tests/` → `test_model.py`, `test_server.py`, `test_integration.py`, `test_fog_flower_client.py`, `test_baseline_comparison.py`, `test_utils.py` y la parte heredada de `test_mqtt_components.py`;
  - resultados → `quick_baseline_test/`, `quick_comparison_results/`, `comparativa_completa/comparison_report.*` y `comparativa_completa/comparison_plots.png`.
- **Scripts:**
  - rotos o heredados: `process_swell_rri.py`, `process_swell_labels.py`, `validate_sweet_transfer_setup.py`, `test_sweet_federated.py`, `dataset_rules.py`, `check_deprecated.py`, `check_deprecated_simple.py`;
  - demos: `demo_multidataset_fl.py`, `demo_simple_multidataset.py`;
  - barridos SWEET, cuyos resultados se conservan: `advanced_ml_comparison.py`, `ultra_powerful_ml.py`, `extreme_deep_models.py`, `hyperparameter_tuning.py`, `final_comparison_all_models.py`, `unified_comparison.py`;
  - transferencia SWEET sin conectar: `run_sweet_transfer_learning.py`, `prepare_sweet_federated_transfer.py` y su config `configs/sweet_federated_transfer.yaml`;
  - diagnóstico y análisis: `diagnose_sweet.py`, `diagnose_wesad.py`, `inspect_wesad_labels.py`, `analyze_sweet_sensors.py`, `plot_sweet_class_distribution.py`.
- **CI y metadatos:**
  - `.github/workflows/` → `nightly.yml`, `release.yml`, `publish.yml`, `release-notes.yml` y `README.md`;
  - `.github/` → `FUNDING.yml` y `stale.yml`;
  - `.devcontainer/` y `docker/docker-compose.yml`;
  - `CODEOWNERS` se corrige a `@adrianoggm`, y `pr-review.yml` se queda solo con el escaneo Trivy.
- **Docs:**
  - `docs/` → `EXECUTION_GUIDE.md`, `CHANGELOG.md` (lo sustituyen las notas de GitHub Releases), `SWEET_FEDERATED_README.md`, `SWEET_QUICKSTART.md`, `SWEET_TRANSFER_LEARNING_README.md`, `Context.md` y `Arquitecture.md` (los sustituye el doc de arquitectura de la fase 10);
  - `diagrams/` completo;
  - `docs/RULES.md` se reduce a las reglas propias del proyecto y absorbe `AI_DATASET_RULES.md`, que se borra;
  - `validations/validate_workflows.py`.

---

## 14. Tests

| Tipo | Qué cubre | En el CI |
|---|---|---|
| Unitarios | Agregación por clave; reparto (α=0 segregado, α=1 uniforme, conservación de sujetos, determinismo); alcances de compartición; estabilidad de `topology_id`; ida y vuelta de los codecs; cálculo de la entrega por enlace; validación de config | Sí |
| Protocolo (`SimRuntime`) | Rondas, quórum, plazos, staleness, participación, caídas y pérdidas, registro `hello`, evaluación en los tres niveles. Usa el **entrenador de prueba**, que no aprende y devuelve pesos deterministas, con edges sin datos. | Sí |
| Determinismo | La misma config y semilla dan la misma huella de `events.jsonl` (sin `t_wall`) | Sí |
| Desacople | En `roles/`, `learning/` y `runtime/` no aparece el nombre de ningún dataset registrado | Sí |
| Equivalencia entre modos | `SimRuntime` frente a `RealRuntime` + MQTT, con Mosquitto como servicio | Sí |
| Ingesta | Los YAML de SWELL, SWEET y WESAD sobre los datos reales | Local (se salta en el CI sin `data/`) |
| Paridad | Ejecución de referencia de SWELL | Local |

`docs/RULES.md` deja por escrito que el entrenador de prueba está permitido en los tests de protocolo, y que la prohibición de datos sintéticos se aplica a todo lo que entrene o evalúe aprendizaje.

---

## 15. Criterios de aceptación del primer corte

1. `onion_fl data prepare` funciona para `swell`, `sweet` y `wesad`, y `data inspect` avisa de las features sospechosas.
2. `onion_fl run experiments/mix_ab.yaml` ejecuta en simulación el barrido de α × compartición × semillas, y cada ejecución queda firmada (§10.1).
3. Hay métricas en los niveles edge, fog y global por dataset, y diagnósticos de divergencia, conflicto y comunicación, en `events.jsonl`.
4. `onion_fl report` y `load_runs().compare()` comparan escenarios y topologías por nivel.
5. El mismo experimento con `--mode real` sobre MQTT reproduce el modelo final de la simulación (§9.3).
6. La ejecución de referencia de SWELL funciona en el framework nuevo, y el runtime antiguo y todo el inventario de borrado han desaparecido.
7. El CI (`ruff`, `pytest` y el test con Mosquitto) está en verde en `develop`. Grafana muestra las métricas nuevas con sus etiquetas.

---

## 16. Fuera de alcance de esta especificación

- **Transportes gRPC y Flower** (subproyecto 5). Antes hay que hacer una prueba rápida de la API de mensajes de Flower (≥ 1.21) para usarlo como transporte sin que controle el bucle de rondas.
- **Despliegue real distribuido** (subproyecto 6).
- **Front** (subproyecto 7, siguiente especificación).
- Grafos generales y topologías P2P, estadísticas de preprocesado federadas, firma criptográfica, agregación segura y privacidad diferencial, y datos no tabulares (imágenes, audio). El contrato de datos y los registros no impiden añadirlos más adelante.

---

## 17. Decisiones a validar en esta revisión

Son propuestas que no se discutieron explícitamente:

1. Python ≥ 3.11, y ruff como única herramienta de lint y formato.
2. Los baselines clásicos (LR, RF, XGBoost y la validación cruzada por sujeto) se conservan como `onion_fl baseline` sobre la capa de datos nueva. Los scripts de barridos SWEET se borran y sus resultados se conservan en `results/legacy/`.
3. El baseline centralizado de aprendizaje federado se expresa con el reparto `pooled`.
4. El codec por defecto es `json`, por continuidad, y `npz` es la alternativa binaria de serie.
5. `CHANGELOG.md` se sustituye por las notas de GitHub Releases.
6. `validations/` se borra cuando `data inspect` cubra sus comprobaciones.
7. `pr-review.yml` se queda solo con el escaneo Trivy, y `dependabot.yml` se mantiene.
