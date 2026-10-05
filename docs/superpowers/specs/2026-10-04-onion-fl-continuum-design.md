# Onion-FL continuum: aprendizaje federado continuo

> Issue #156 (C1) del hito v0.7.0. Las issues #157–#163 (C2–C8) implementan esta spec; #164 (C9, v0.8.0) la lleva a nodos reales.

## 1. Objetivo

Convertir Onion-FL en un sistema federado **versionado** capaz de:
- aprender de **streams temporales**;
- **conservar** el conocimiento previo;
- incorporar información **etiquetada y no etiquetada**;
- **detectar** cuándo merece la pena actualizarse;
- **promover** solo los modelos que demuestren una mejora segura sobre el activo.

Todo v0.7.0 se desarrolla y se valida en simulación (`SimRuntime`, reloj virtual) con datos reales. El despliegue distribuido de extremo a extremo es C9 y depende de #105. Nada de v0.7.0 depende de E5/E6.

**Éxito** es poder responder con resultados confirmados (C8) a:
- cuánto mejora el modelo global a lo largo del tiempo frente a no actualizarlo y frente a reentrenar desde cero;
- cuánto olvida;
- qué aportan la memoria y los datos sin etiquetar;
- qué coste en rondas y bytes tiene.

## 2. Punto de partida

| Pieza | Hoy | Qué falta |
|---|---|---|
| Datos | `SubjectData` con `X`, `y` y metadatos; sujetos de test y validación reservados por dataset | Tiempo por fila; llegada en flujo; etiquetas diferidas |
| Preprocesado | Escalado, imputación y eliminación de columnas constantes ajustados en `data.roles`, sobre las partes de entrenamiento | Guardarlo como artefacto de la versión y congelarlo |
| Modelo inicial | `init: checkpoint` carga grupos de un `.npz` | Estado del servidor y de los edges, linaje, evolución del esquema |
| Rondas | El coordinador termina tras `rounds`; las políticas cubren participación, quórum, plazo, tardías y `close_at_quorum` (PR #165) | Rondas lanzadas por disparadores; ejecución sin fin |
| Evaluación | Test global, zona (validación), edge (`received`/`local`) | Evaluación prequential; ventanas reciente e histórica; versiones |
| Observabilidad | Eventos JSONL, `Run`, Prometheus/Grafana, Studio | Versiones, linaje, llegada de datos, deriva y controles |

## 3. Decisiones

| Tema | Decisión | Por qué |
|---|---|---|
| Unidad temporal | Cada observación lleva `observed_at → available_at → predicted_at (+ predicted_by_version) → label_available_at → trainable_at → first_consumed_by_version` | Hace explícito qué sabía el sistema y cuándo; evita fugas temporales |
| Dos relojes | Tiempo de datos (la semántica: retrasos, disparadores, bootstrap) separado del tiempo virtual de la simulación; `stream.speed` solo convierte uno en otro | Cambiar la velocidad acelera la simulación sin cambiar cuándo ocurre nada |
| Arranque | Un bootstrap temporal explícito: con lo anterior a t₀ se ajusta el preprocesado y se entrena v0; después se congela | Realista, y ninguna estadística del preprocesado usa observaciones posteriores a t₀ |
| Evaluación | Test-then-train (prequential): un ejemplo no entrena antes de que lo evalúe el modelo activo cuando llegó, y esa predicción se guarda para puntuarla cuando llegue la etiqueta | Es la medida estándar en streams y no tiene fugas |
| Validación | Un stream de validación independiente, solo para evaluar: nunca entrena | Da la ventana reciente y la histórica con las que se decide promover |
| Versiones | Champion (activo) y challenger (candidato) con una `PromotionPolicy` enchufable | Solo se promueve lo que mejora de forma segura; el rechazo y el rollback son baratos |
| Estado | Un `ModelBundle` con el estado global, el de cada edge, el preprocesado, el esquema y el linaje | Continuar no es solo cargar pesos: SCAFFOLD, FedDyn, MOON, Ditto/APFL y el replay tienen estado |
| Preprocesado | Artefacto de primera clase de cada versión, congelado por dataset; un dataset nuevo es una evolución explícita del esquema | Reajustarlo con datos futuros cambia la representación en silencio y filtra información |
| Memoria | Eje `continual.memory` con `none`, `fifo`, `reservoir` y `class_balanced` | Es la defensa básica contra el olvido y no toca la agregación |
| No etiquetados | Eje `unlabelled` con `pseudo_label` y `consistency` para empezar | Interfaz primero; el autosupervisado crece después como plugin |
| Disparadores | Plugins; el disparador local (edge) y el federativo (coordinador) son decisiones distintas | `trigger: drift` necesita saber qué deriva y quién decide |
| Asincronía | Rondas síncronas lanzadas por eventos; FedAsync/FedBuff (#150) serán otra política de consumo | No bloquear v0.7.0 por la asincronía |
| Simulación | Todo funciona en `SimRuntime`, y días de flujo pasan en minutos | Validar el comportamiento temporal sin infraestructura |

## 4. Protocolo temporal

### 4.1 Tiempo de una observación

Cada fila de un sujeto gana un tiempo `t` (segundos desde el inicio de su grabación) en `SubjectData`. Es un **metadato, nunca una feature** (docs/RULES.md).
- **SWELL:** de la columna `timestamp`, que hoy se excluye de las features.
- **WESAD:** del inicio de cada ventana.

Una observación recorre estos instantes, todos en tiempo de datos (§4.2):

| Instante | Significado |
|---|---|
| `observed_at` | `t` en el reloj del stream del edge |
| `available_at` | Cuando llega al edge (con `stream.latency`, 0 por defecto) |
| `predicted_at`, `predicted_by_version` | Cuándo y con qué versión la predijo el champion que el edge servía; la predicción se guarda |
| `label_available_at` | Cuando llega su etiqueta: `available_at + labels.delay`, o nunca si no está en la fracción etiquetada |
| `trainable_at` | Cuando el edge puede entrenar con ella: tras predecirla, sin etiqueta desde `available_at` y con etiqueta desde `label_available_at` |
| `first_consumed_by_version` | La primera versión candidata en cuyo entrenamiento entró; con replay puede entrar en varias, que se registran en los eventos |

### 4.2 Dos relojes y bootstrap

- **Dos relojes.**
  - El **tiempo de datos** es el de las grabaciones. En él se expresa toda la semántica: `labels.delay`, los disparadores por calendario, el bootstrap y el horizonte.
  - El **tiempo virtual** es el de `SimRuntime`. `stream.speed` solo convierte uno en otro: `speed: 60` hace que un minuto de datos pase en un segundo simulado. Pasar de 60 a 120 acelera la simulación y no cambia cuándo debería ocurrir una ronda.
- **Inicio de cada sujeto** (`stream.start`):
  - `aligned`: todos empiezan a la vez. Es el benchmark principal.
  - `staggered: <s>`: un sujeto cada `s` segundos de datos. Simula edges que se incorporan más tarde y queda como experimento secundario de heterogeneidad del sistema.
- **Horizonte.** `stream.horizon: session` termina cuando se agota la grabación de cada sujeto. Se deriva del máximo `t` disponible, no de una duración escrita a mano.
- **Bootstrap temporal** (`stream.bootstrap: 20m` o `{samples: 500}`):
  - lo ocurrido antes de t₀ es el histórico disponible antes de poner en marcha el continuum;
  - con él se ajusta el preprocesado, se entrena v0 y después se congela el preprocesado;
  - ninguna estadística del preprocesado puede usar observaciones posteriores a t₀;
  - un periodo de bootstrap, a diferencia de una sola primera ventana, puede abarcar varias condiciones y clases.

### 4.3 Regla de evaluación sin fugas

- Cuando un lote llega a un edge, primero lo **predice el champion** que el edge sirve, y la predicción se guarda con `predicted_at` y `predicted_by_version`.
- Cuando llega la etiqueta, se puntúa **esa predicción guardada**, no una nueva con el champion de ese momento. Eso mantiene el test-then-train con etiquetas diferidas.
- Solo después de predecirlo, el lote pasa al búfer de entrenamiento: sin etiqueta, o con ella cuando llegue.
- La etiqueta real de lo no etiquetado se conserva fuera del learner, solo para evaluar.
- **Los sujetos de validación también llegan en flujo, solo para evaluar:** forman un stream independiente que nunca entrena y que da la ventana reciente y la histórica de §5.2.
- Los sujetos de test **nunca** llegan a ningún edge.

### 4.4 Tipos de deriva

| Tipo | Qué cambia | Cómo se detecta (primeros detectores) |
|---|---|---|
| De datos | P(X) | Distancia entre la ventana reciente y la de referencia de las features del edge |
| De prior (de etiquetas) | P(Y) | Cambio en la proporción de clases de las etiquetas llegadas, o de las predichas |
| De concepto | P(Y\|X) | Error del champion sobre las etiquetas que van llegando, una vez descontado el cambio de P(Y) |
| De rendimiento | Una métrica observada | Caída del macro-F1 prequential frente a su media móvil |
| De cliente | Un edge respecto al resto | Divergencia de sus actualizaciones (`diagnostic.divergence_*`, ya existe) |

En SWELL y WESAD, las condiciones de estrés llegan por bloques dentro de cada sesión, así que hay **deriva de prior** (P(Y)) real, y probablemente de datos. El benchmark (C8) no necesita deriva sintética.

## 5. Versiones: champion y challenger

### 5.1 Ciclo de vida

```
active (champion) ──ronda──► candidate ──► validating ──► promoted ──► (nuevo champion; el anterior pasa a superseded)
                                              └──────────► rejected (el champion sigue activo)
rollback: un superseded vuelve a active (por orden del operador o de la política)
```

Cada versión tiene:
- un número monótono `v`, su hash de estado y su versión padre;
- la `run_id` y el `run_hash` de la ejecución que la produjo;
- el cursor temporal (la frontera de datos consumidos).

### 5.2 `PromotionPolicy`

Es enchufable y sin valores fijos en el código. Por ejemplo:

```yaml
promotion:
  name: champion_challenger
  metric: macro_f1
  min_delta: -0.01        # el challenger no puede empeorar más que esto en la ventana reciente
  max_forgetting: 0.03    # ni olvidar más que esto en la referencia
  min_samples: 500        # muestras mínimas en la ventana reciente para decidir
```

- **Dos conjuntos.**
  - **Ventana reciente (plasticidad):** lo último llegado al stream de validación.
  - **Referencia e historia (retención):** una muestra fija de los periodos anteriores del mismo stream.
- **El test nunca decide.** Los sujetos de test solo se usan para informar.
- **Dónde se sirve cada versión.** Los edges sirven el champion. El challenger no llega a producción antes de promoverse.

## 6. `ModelBundle` (C2)

Una versión se guarda como un directorio, no como un `.npz` suelto:

```
bundle/
├── model.npz            # estado global del modelo
├── server.npz           # optimizador de servidor y estado global del algoritmo (c de SCAFFOLD, h de FedDyn, momentos)
├── edges/<id>.npz       # estado de cada edge: c_i, memoria de FedDyn, modelo anterior de MOON, modelos personales, buffers de replay
├── preprocessing.json   # escalado, imputación y columnas eliminadas, por dataset
├── schema.json          # features, clases, tareas y grupos de parámetros por dataset
├── lineage.json         # versión, versión padre, run_id y run_hash padre, cursor temporal
└── config.yaml          # la config que la produjo
```

- **Dónde vive el estado de cada nodo.** En simulación, el bundle reúne el estado de todos los nodos. En real (C9), cada nodo guarda solo su parte: los buffers de un edge **no salen del edge**.
- **Arranque.**

  ```yaml
  init:
    name: run
    run: <run_id o ruta al bundle>
    restore: {model: true, preprocessing: true, server_state: true, edge_state: true}
  ```

  Verifica el `run_hash` del padre y registra el linaje en `run.json`. Reanudar con todo restaurado equivale a no haber parado; un test lo comprueba.
  - **Padre.** `run` es un `run_id`, una carpeta o `experiment:<nombre>[/<escenario>]`, que se resuelve a la última ejecución terminada de ese experimento con la misma semilla. Si hay varios escenarios, hay que nombrar uno.
  - **Alcance de la exactitud.** Es exacta con rondas síncronas, enlaces sin pérdidas y sin disponibilidad que dependa del reloj. Los streams de los enlaces, el reloj virtual y las tardías ya guardadas para la ronda siguiente empiezan de nuevo.
  - **Se rechaza** (al planificar) una continuación:
    - de un padre no terminado o con otra semilla;
    - con otro coordinador raíz;
    - con otro trainer y `edge_state`, o con otro optimizador de servidor y `server_state`;
    - con `restore.model` y un modelo que no encaja: otra estructura (`adapter_width`, `trunk_hidden`, `adapters`, `trunk`, `heads`…) o, en un dataset que el linaje ya conocía, otra tarea u otras clases;
    - con otro mecanismo de DP en los edges o en un agregador que el linaje ya conocía (σ, C y δ sí pueden cambiar);
    - con DP local y un edge nuevo que reúne sujetos que ya liberaron, porque su presupuesto volvería a cero (pasa al cambiar `subjects_per_client` o los sujetos de entrenamiento);
    - con SCAFFOLD o FedDyn, si `server_state` y `edge_state` no van juntos: c sin c_i, o h sin las correcciones de los edges, sesgaría la corrección para siempre;
    - con `restore.preprocessing` y `scaler: local`, porque el preprocesado local congelado aún no existe;
    - que cambia de rol a un sujeto que ya entrenó, validó o fue de test en el linaje (test → train, val → test…); los sujetos nuevos pueden entrar en cualquier rol;
    - en modo real, hasta C9.
  - **Qué restaura cada opción.**
    - `model` gobierna todos los pesos: el modelo global, las zonas, el agregado anterior, los modelos de los edges y la memoria del entrenador que guarda pesos (el modelo personal de Ditto, los de APFL, el modelo anterior de MOON).
    - `edge_state` gobierna el resto del estado de cada edge: la memoria del entrenador (c_i de SCAFFOLD, la corrección de FedDyn, α de APFL) y su stream aleatorio.
    - `server_state` gobierna el optimizador de servidor y el estado de algoritmo del modelo global (la c de SCAFFOLD, que viaja como claves auxiliares).
  - **Lo liberado se acumula.**
    - Cada contable de DP, central en el agregador o local en el edge, guarda el RDP acumulado en cada orden α: Σ_t α/(2σ_t²), con σ/2 en la DP local, porque su sensibilidad es 2C.
    - Continúa siempre, aunque cambie σ: lo liberado no se olvida.
    - Un edge con presupuesto conserva también su stream aleatorio, aun sin `edge_state`: uno nuevo repetiría el ruido del padre.
  - **El linaje tiene memoria.**
    - Por dataset, el bundle conserva el preprocesado, la tarea y las clases, y los sujetos que alguna vez entrenaron, validaron o fueron de test, aunque una generación no cargue ese dataset.
    - La comprobación de roles mira todo el linaje: A → B → C no olvida que un sujeto entrenó en A. Como un sujeto nunca cambia de rol, los roles del bundle siguen siendo una partición.
- **Edges que cambian.**
  - Un edge nuevo empieza sin estado de edge.
  - El estado de un nodo que ya no está (edge o agregador) pasa al bundle tal como esta ejecución lo conservó, con su presupuesto de DP, y se retoma si el nodo vuelve.
  - El estado global que depende del número de edges (por ejemplo, N en la fracción de SCAFFOLD) se recalcula con los `holders` registrados.
- **Evolución del esquema.**
  - Para un dataset existente, el preprocesado del bundle se congela.
  - Un dataset nuevo tiene su propio periodo de bootstrap explícito (§4.2): su preprocesado se ajusta únicamente sobre ese prefijo temporal y queda congelado después. Crea su adaptador (y su cabeza, si la tarea es nueva).
  - Los grupos cargados pueden congelarse o entrenarse con un lr menor, por grupo.

## 7. Streams y etiquetas (C3)

```yaml
stream:
  order: timestamp       # el orden real de las filas de cada sujeto
  batch_size: 32         # filas que llegan juntas
  speed: 60              # tiempo de datos → tiempo virtual (solo acelera)
  start: aligned         # o {staggered: 10m}
  horizon: session       # hasta el final de la grabación de cada sujeto
  bootstrap: 20m         # o {samples: 500}: el histórico antes de t₀
labels:
  fraction: 0.2          # fracción etiquetada, por edge
  delay: 30m             # retraso de la etiqueta, en tiempo de datos
```

- **Búfer de entrenamiento por edge.** Contiene lo reciente que el edge puede usar ahora (`trainable_at ≤ t`). La memoria de C4 es aparte.
- **Eventos.**
  - `data.arrived`: por edge, con cuántas filas llegaron sin etiqueta y cuántas con ella;
  - `data.labelled`: etiquetas que llegan tarde.
- **Determinismo.** La misma semilla da el mismo flujo, la misma fracción etiquetada y los mismos retrasos.
- **Decisiones de C3** (#158; el plan, `docs/superpowers/plans/2026-10-05-continuum-c3-streams.md`, da el detalle):
  - **Ritmo.** Hasta C6, las rondas llegan cada `stream.round_every` de tiempo de datos. La ejecución dura hasta una ronda después de la última llegada, con `rounds` como tope.
  - **Etiquetas.** `labels.fraction` y `labels.delay` valen para todas las filas, histórico incluido, así que con `fraction: 0` no entrena nada.
    - Cada edge etiqueta exactamente round(f·n) filas con una semilla propia.
    - La etiqueta de una fila del histórico cuenta su retraso desde que se observó.
  - **Bootstrap.** `{samples: N}` cuenta por edge, y se rechaza con varios sujetos por edge, igual que `staggered`.
  - **Dos puntuaciones prequential**, ambas sobre la predicción guardada al llegar:
    - `prequential`: toda llegada contra su verdad, solo en simulación;
    - `prequential_labelled`: solo las etiquetas que llegaron.
  - **Búfer.** Lo que pasó a ser entrenable en la última `stream.window` (por defecto `round_every`).
  - **Edges ociosos.** Un edge sin nada que entrenar responde ocioso, con sus puntuaciones. No cuenta para el quórum, y una ronda toda ociosa se cierra como `round.idle`.
  - **Tiempo de una ventana.** Una ventana de WESAD toma el tiempo de su final, porque solo es observable completa. `t` cuenta desde la primera observación de cada sujeto.
  - **Se rechaza con un stream, por ahora:**
    - el modo real;
    - `init: run`, porque el bundle no guarda aún el estado del stream;
    - `local_val`;
    - `scaler: local`.

## 8. Memoria e incremental (C4)

```yaml
continual:
  memory: {name: reservoir, capacity: 512}
  replay_ratio: 0.25     # fracción de cada lote de entrenamiento que sale de la memoria
```

- **Plugins.** `none`, `fifo`, `reservoir` y `class_balanced`.
- **Qué guarda la memoria.** Solo ejemplos etiquetados (con su etiqueta llegada) o pseudoetiquetados aceptados, nunca la etiqueta oculta.
- **Líneas base.** Ajuste fino solo con lo reciente, y ajuste fino más replay.
- **Replay y privacidad.**
  - El replay participa en el entrenamiento como cualquier dato. La privacidad de P3 se aplica a la **actualización liberada**: con DP local, el edge recorta y perturba la actualización resultante; con DP central, el fog o el servidor recortan y perturban la contribución recibida.
  - Cada liberación cuenta en la composición del presupuesto.
  - No se añade ruido a los ejemplos guardados.
- **Seguridad del almacenamiento.** La DP del protocolo no protege el búfer frente a quien comprometa físicamente el edge; eso es seguridad de almacenamiento. En C4 el búfer es local, no sale del edge y tiene capacidad (y opcionalmente TTL). El cifrado queda para después.

## 9. Datos sin etiquetar (C5)

```yaml
unlabelled:
  name: pseudo_label
  threshold: 0.9         # confianza mínima del modelo recibido
  weight: 0.5            # peso de la pérdida sobre los pseudoetiquetados
```

- **Plugins.** `pseudo_label` y `consistency` (perturbaciones de las features).
- **Más adelante.** Un objetivo autosupervisado para el adaptador y el tronco.
- **Diagnósticos.**
  - `pseudo_labels_seen`, `pseudo_labels_accepted`, `acceptance_rate` y `confidence`;
  - `pseudo_label_accuracy`, solo en simulación, con la etiqueta oculta.
- **Auditoría.** Cada pseudoetiqueta guarda `pseudo_label`, `confidence` y `generated_by_version`, para saber después qué champion generó una etiqueta errónea.

## 10. Disparadores y federación continua (C6)

```yaml
continuum:
  trigger:                # federativo: cuándo abre ronda el coordinador
    name: any
    of:
      - {name: schedule, every: 15m}                                    # tiempo de datos
      - {name: volume, samples: 500}
      - {name: drift, kind: performance, detector: {name: page_hinkley}}
  edge_trigger: {name: volume, samples: 64}   # local: cuándo un edge tiene una actualización
```

- **El coordinador no termina tras N rondas.** Termina al agotar `stream.horizon` o por orden del operador.
- **Duraciones en tiempo de datos.** Todas las duraciones de los disparadores están en tiempo de datos (§4.2).
- **Volumen y deriva llegan al coordinador como mensajes de estado.**
  - Los edges los envían cada `status_every` y los fogs los agregan.
  - Los mensajes no llevan datos: solo recuentos y estadísticos.
- **Un edge solo entra en una ronda si su disparador local se cumplió.** Es la participación de la ronda.
- **Rondas síncronas.** FedAsync y FedBuff (#150) se conectan después como política de consumo.

## 11. Observabilidad y control (C7)

- **Eventos.** `version.candidate`, `version.promoted`, `version.rejected` y `version.rollback`, con las métricas de la decisión; `trigger.fired` (con qué disparador); `drift.detected` (tipo, nodo y estadístico).
- **Prometheus/Grafana.** Series por versión, datos llegados por edge, promociones y deriva. El test del dashboard sigue limitado a las series exportadas.
- **Studio.**
  - Vista del continuum: línea de versiones con su linaje, métricas por versión (prequential, reciente y referencia), volumen y deriva por edge.
  - Controles para programar, pausar, promover y revertir.
- **API.** Los mismos controles, desde fuera del Studio.

## 12. Evaluación y métricas (C8)

| Métrica | Qué mide |
|---|---|
| Macro-F1 prequential (online) | Rendimiento del modelo servido sobre lo que va llegando |
| Área bajo la curva de rendimiento en el tiempo | Rendimiento acumulado |
| Macro-F1 final en test | Comparable con el resto de `results/` |
| Olvido | Caída en la referencia respecto a la mejor versión anterior |
| Tiempo de recuperación tras una deriva | Rondas o segundos hasta volver al nivel previo |
| Rondas, bytes y cómputo | Coste |
| Challengers promovidos y rechazados | Estabilidad del proceso |
| Eficiencia de etiquetas | Rendimiento frente a la fracción etiquetada |

Líneas base: congelado, reentrenar desde cero, warm start, warm start + replay, y warm start + replay + semisupervisado. Los hiperparámetros se eligen sobre validación y los resultados se confirman con su identidad.

## 13. Qué implementa cada issue

**Orden de implementación:** C2 → C3 → C4 → C6 → C5 → C7 → C8. Primero un continuum supervisado completo y medible (bundle, streams, memoria y federación continua); el semisupervisado (C5) es una extensión encima de esa base.

| Issue | Secciones |
|---|---|
| #157 C2 Model bundle | §6 |
| #158 C3 Streams | §4.1–4.3, §7 |
| #159 C4 Memoria | §8 |
| #160 C5 No etiquetados | §9 |
| #161 C6 Disparadores | §4.4, §10 |
| #162 C7 Versiones y observabilidad | §5, §11 |
| #163 C8 Benchmark | §12 |
| #164 C9 Distribuido | §6 (estado por nodo), §10–11 sobre los nodos de #105 |

## 14. Decisiones de la revisión

Las cuatro preguntas abiertas del primer borrador quedaron resueltas así:

| Pregunta | Decisión |
|---|---|
| ¿La validación llega en flujo? | Sí: un stream de validación independiente, **solo para evaluar** (§4.3); da la ventana reciente y la histórica |
| ¿Cómo se ajusta el escalado inicial? | Con un **bootstrap temporal explícito**, no con una sola primera ventana: lo anterior a t₀ ajusta el preprocesado y entrena v0 (§4.2) |
| ¿El replay pasa por la DP? | Sí, pero la DP se aplica a la **actualización liberada**, no a cada ejemplo del replay; la seguridad del búfer es aparte (§8) |
| ¿Qué escala tiene el benchmark? | La **sesión completa** (`horizon: session`, derivado de los datos) con `speed: 60`; tiempo de datos separado del virtual (§4.2) |

Además:
- **Deriva de prior.** Se añade como quinto tipo de deriva, P(Y) (§4.4).
- **Predicción guardada.** Cada predicción guarda `predicted_at` y `predicted_by_version`, y se puntúa esa misma cuando llega la etiqueta (§4.1, §4.3).
- **Varios consumos.** `first_consumed_by_version` sustituye a `consumed_by_version`, porque con replay un ejemplo entra en varias versiones.
- **Pseudoetiquetas auditables.** Cada una guarda `pseudo_label`, `confidence` y `generated_by_version` (§9).
- **Orden de implementación.** C2 → C3 → C4 → C6 → C5 → C7 → C8 (§13).
