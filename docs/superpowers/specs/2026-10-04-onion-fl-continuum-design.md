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
| Unidad temporal | Cada observación lleva `observed_at → available_at → label_available_at → trainable_at → consumed_by_version` | Hace explícito qué sabía el sistema y cuándo; evita fugas temporales |
| Evaluación | Test-then-train (prequential): un ejemplo no entrena antes de que lo evalúe el modelo activo cuando llegó | Es la medida estándar en streams y no tiene fugas |
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

Una observación recorre estos instantes:

| Instante | Significado |
|---|---|
| `observed_at` | `t` en el reloj del stream del edge |
| `available_at` | Cuando llega al edge (con `stream.latency`, 0 por defecto) |
| `label_available_at` | Cuando llega su etiqueta: `available_at + labels.delay`, o nunca si no está en la fracción etiquetada |
| `trainable_at` | Cuando el edge puede entrenar con ella: tras evaluarla (test-then-train), sin etiqueta desde `available_at` y con etiqueta desde `label_available_at` |
| `consumed_by_version` | La primera versión candidata en cuyo entrenamiento entró |

### 4.2 Reloj del stream

`stream.speed` convierte el tiempo de los datos al reloj virtual: por ejemplo, `speed: 60` hace que un minuto de datos pase en un segundo simulado. `stream.start` desplaza el inicio de cada sujeto:
- `aligned`: todos empiezan a la vez;
- `staggered: <s>`: un sujeto cada `s` segundos, que es como se simulan edges que se incorporan más tarde.

### 4.3 Regla de evaluación sin fugas

- Cuando un lote llega a un edge, primero lo **predice el champion** que el edge sirve. Esa predicción entra en las métricas prequential.
- Solo después pasa al búfer de entrenamiento: sin etiqueta, o con etiqueta cuando llegue.
- La etiqueta real de lo no etiquetado se conserva fuera del learner, solo para evaluar.
- Los sujetos de test **nunca** llegan a ningún edge.

### 4.4 Tipos de deriva

| Tipo | Qué cambia | Cómo se detecta (primeros detectores) |
|---|---|---|
| De datos | P(X) | Distancia entre la ventana reciente y la de referencia de las features del edge |
| De concepto | P(Y\|X) | Error del champion sobre las etiquetas que van llegando |
| De rendimiento | Una métrica observada | Caída del macro-F1 prequential frente a su media móvil |
| De cliente | Un edge respecto al resto | Divergencia de sus actualizaciones (`diagnostic.divergence_*`, ya existe) |

En SWELL y WESAD, las condiciones de estrés llegan por bloques dentro de cada sesión, así que hay **desplazamiento de etiquetas** real: el benchmark (C8) no necesita deriva sintética.

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
  - **Ventana reciente (plasticidad):** lo último llegado a los sujetos de validación.
  - **Referencia e historia (retención):** una muestra fija de los periodos anteriores, también de validación.
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
- **Edges que cambian.**
  - Un edge nuevo empieza sin estado de edge.
  - El estado de un edge que ya no está se conserva en el bundle, pero no se usa.
  - El estado global que depende del número de edges (por ejemplo, N en la fracción de SCAFFOLD) se recalcula con los `holders` registrados.
- **Evolución del esquema.**
  - Para un dataset existente, el preprocesado del bundle se congela.
  - Un dataset nuevo ajusta su preprocesado sobre su primera ventana de entrenamiento, que se congela después, y crea su adaptador (y su cabeza, si la tarea es nueva).
  - Los grupos cargados pueden congelarse o entrenarse con un lr menor, por grupo.

## 7. Streams y etiquetas (C3)

```yaml
stream:
  order: timestamp       # el orden real de las filas de cada sujeto
  batch_size: 32         # filas que llegan juntas
  speed: 60              # segundos de datos por segundo simulado
  start: aligned         # o {staggered: 600}
labels:
  fraction: 0.2          # fracción etiquetada, por edge
  delay: 2h              # retraso de la etiqueta, en tiempo de datos
```

- **Búfer de entrenamiento por edge.** Contiene lo reciente que el edge puede usar ahora (`trainable_at ≤ t`). La memoria de C4 es aparte.
- **Eventos.**
  - `data.arrived`: por edge, con cuántas filas llegaron sin etiqueta y cuántas con ella;
  - `data.labelled`: etiquetas que llegan tarde.
- **Determinismo.** La misma semilla da el mismo flujo, la misma fracción etiquetada y los mismos retrasos.

## 8. Memoria e incremental (C4)

```yaml
continual:
  memory: {name: reservoir, capacity: 512}
  replay_ratio: 0.25     # fracción de cada lote de entrenamiento que sale de la memoria
```

- **Plugins.** `none`, `fifo`, `reservoir` y `class_balanced`.
- **Qué guarda la memoria.** Solo ejemplos etiquetados (con su etiqueta llegada) o pseudoetiquetados aceptados, nunca la etiqueta oculta.
- **Líneas base.** Ajuste fino solo con lo reciente, y ajuste fino más replay.

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

## 10. Disparadores y federación continua (C6)

```yaml
continuum:
  trigger:                # federativo: cuándo abre ronda el coordinador
    name: any
    of:
      - {name: schedule, every: 6h}
      - {name: volume, samples: 500}
      - {name: drift, kind: performance, detector: {name: page_hinkley}}
  edge_trigger: {name: volume, samples: 64}   # local: cuándo un edge tiene una actualización
  horizon: 7d             # hasta dónde se reproduce el stream
```

- **El coordinador no termina tras N rondas.** Termina al agotar el `horizon` o por orden del operador.
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

## 14. Preguntas abiertas para la revisión

1. **Validación en flujo.** ¿Los sujetos de validación también llegan en flujo (ventana reciente) o son fijos? Propuesta: también en flujo, porque si no, la ventana reciente no existe.
2. **Escalado inicial.** Hoy `scaler: global` se ajusta con todas las partes de entrenamiento. En el continuum, ¿se ajusta solo con la primera ventana, o se acepta el ajuste global inicial como "modelo base"? Propuesta: la primera ventana, para que la versión 0 no vea el futuro.
3. **Privacidad del replay.** La memoria guarda datos crudos en el edge. Con `privacy` (P3), ¿se exige que el replay pase por el mismo recorte y ruido? Propuesta: sí, como cualquier actualización.
4. **Escala del benchmark.** ¿`horizon` cubre una sesión completa por sujeto (SWELL ≈ 3 h, WESAD ≈ 2 h) con `speed` de 60?
