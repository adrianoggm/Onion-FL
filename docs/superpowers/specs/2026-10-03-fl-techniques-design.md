# Técnicas de aprendizaje federado: diseño (subproyecto 8)

Petición: ampliar las técnicas de ML federado que se pueden elegir, para compararlas entre sí a todos los niveles (topología, reparto, seguridad, privacidad). Hay que empezar por las más importantes.

Las **familias de modelos** (lineales, 1D-CNN/LSTM sobre señal cruda, árboles federados) son el subproyecto 9 y tendrán su propia especificación. Este documento cubre los **algoritmos**: cómo entrenan los edges, cómo combinan los agregadores y cómo actualiza el servidor, con el `modular_mlp` actual.

## 1. Objetivo

- Elegir cada técnica por nombre en la config, como cualquier plugin. Cada una se puede barrer en un `sweep` y mezclar por nivel; por ejemplo, SCAFFOLD en los edges, Krum en los fogs y FedYogi en la nube.
- Medir lo que distingue a cada familia:
  - el rendimiento personal frente al global;
  - la precisión bajo ataque y la detección de nodos maliciosos;
  - el presupuesto de privacidad ε;
  - el tráfico.
- Cerrar cada fase con una comparación sobre datos reales (SWELL + WESAD) confirmada en `results/`.

Punto de partida, ya implementado:

| Eje | Plugins |
|---|---|
| Agregadores | `fedavg`, `mean`, `median`, `trimmed_mean` |
| Optimizadores de servidor | `replace`, `fedavgm`, `fedadam` |
| Entrenadores | `standard`, `fedprox` |
| Compartición | `fedavg`, `fedper`, `zone`, `independent`, `harmonized`, `custom` |
| Participación y asincronía | `all`, `fraction`; `drop`, `next_round`; pesos `constant`, `polynomial` |

## 2. Decisiones

| Decisión | Elección | Por qué |
|---|---|---|
| Forma de cada técnica | Plugins sobre los ejes existentes y dos ejes nuevos (`attack`, `privacy`) | Cada técnica se mezcla por nivel y se barre como cualquier otro plugin. Una «estrategia» por algoritmo, como en Flower, obligaría a rehacer los nodos y no permitiría mezclar por nivel |
| Estado extra (variables de control, actualizaciones normalizadas) | Claves auxiliares `<algoritmo>/<clave del parámetro>` dentro del mismo `Payload` | Viajan por donde viaja su parámetro sin tocar mensajes ni agregadores |
| Estado del cliente entre rondas | Lo guarda el entrenador; cada edge tiene el suyo durante toda la ejecución, como ya ocurre | No cambia el ciclo de vida de los nodos |
| Presupuesto de privacidad | ε por RDP del mecanismo gaussiano compuesto, en forma cerrada, sin amplificación por submuestreo | Cota superior conservadora sin dependencias. Opacus queda como mejora si hace falta afinar |
| Orden | P1 personalización, P2 deriva no IID, P3 robustez y privacidad, P4 servidor y asincronía | Prioridad pedida. Cada fase es una issue, un PR y una comparación con datos reales |
| Hito | Las técnicas pasan a v0.4.0, E5 (#104) a v0.5.0 y E6 (#105) a v0.6.0 | Se pidieron como prioritarias |

## 3. Extensiones del núcleo

§3.4 se implementó con P1 (#147). §3.1–3.2 llegan con P2 (#148) y §3.3 con P3 (#149), donde tienen consumidor.

### 3.1 Estado auxiliar

- **Nombres.** Una clave `<algoritmo>/<clave del parámetro>`, por ejemplo `scaffold/trunk.0.weight`, pertenece al grupo de su parámetro: `group_of` quita el prefijo.
- **Viaje y combinación.** Cruza los enlaces y se guarda en los niveles que dicta la política de compartición de ese grupo. Los agregadores la combinan clave a clave, con el peso de muestras del edge que la envía.
- **Carga en el modelo.** `load_arrays` ya ignora las claves que el modelo no tiene, así que el estado global puede llevarlas sin romper nada.
- **Del edge hacia arriba.** `TrainResult` gana `aux: dict[str, ndarray]`, que el edge añade a su `Payload`.
- **Del coordinador hacia abajo.** El entrenador las recibe dentro de `received`. En la ronda 1 no existen, y cada técnica define su valor inicial (normalmente cero).
- **En el servidor.** Los optimizadores reemplazan las claves auxiliares sin aplicarles su paso: el momento de FedAdam no debe tocar una variable de control. Un optimizador que sí las usa, como `fednova`, las lee explícitamente.

### 3.2 Estadísticos de ronda

- `TrainResult` gana `steps`, los pasos locales de optimización, que el edge envía como métrica `train_steps`.
- `_train_metrics` reduce todas las métricas `train_*` por media ponderada por ejemplos. También suma `train_edges`, los edges que entrenaron, a medida que el informe sube por el árbol.
- El colector añade `staleness`, la antigüedad media en rondas de las contribuciones que combina (0 si todas son de la ronda), y el coordinador `edges_total`, los edges registrados.
- El optimizador de servidor pasa a ser `apply(global_state, aggregated, stats)`, donde `stats` son esas métricas de la ronda. Los optimizadores actuales lo ignoran.

### 3.3 Referencia para los agregadores

`aggregate(contributions, source, reference)` recibe además el estado que el nodo envió hacia abajo en esa ronda (`self.sent`). Los agregadores que trabajan con incrementos (recorte de norma, DP central) calculan `contribución − referencia`. Los demás la ignoran.

### 3.4 Modelos personales y evaluación con ajuste fino

- **Modelo personal.** Un entrenador puede exponer `personal() -> nn.Module | None` (Ditto, APFL). Si lo hace, el edge puntúa también el modelo `personal` sobre su `local_val`.
- **Ajuste fino.** `evaluation.edge.finetune` es un entrenador, por ejemplo `{name: standard, local_epochs: 2}`. Copia el modelo recibido, lo entrena con él sobre los datos locales y lo puntúa como `finetuned`; es el protocolo habitual de FedBABU y Per-FedAvg. `finetune` y `finetuned` en `models` van juntos: uno sin el otro es un error.
- **Flujos aleatorios.** El ajuste fino y el entrenamiento personal de Ditto y APFL usan un flujo hijo del generador del nodo (`child_rng`). Así, puntuar nunca cambia el entrenamiento, y el modelo global de Ditto y APFL coincide con el de FedAvg para la misma semilla.
- **Validación local estratificada.** `data.roles.local_val_split: class_tail` reserva las últimas filas de cada clase en lugar de las últimas del sujeto, que suelen ser de una sola condición.
- **Resultado en los informes.** Los agregadores combinan esas puntuaciones como las de `received` y `local`. El informe compara `global`, `personal` y `finetuned` por nivel.

### 3.5 Representación del modelo

`ModularMLP.features(x, dataset)` devuelve la salida del tronco, y `forward` pasa a usarla. MOON la necesita para su pérdida contrastiva.

### 3.6 Eje `attack`

- **Config.** `attack: {name, fraction, start_round, ...params}` va en la config del experimento.
- **Edges maliciosos.** Se eligen con la semilla de la ejecución entre los edges que entrenan, en una proporción `fraction` dentro de cada dataset. El evento `data.attack` los registra, y el edge lleva `malicious: true` en sus etiquetas.
- **Ganchos.** Un ataque implementa uno o los dos:
  - `on_data(data) -> data`, antes de entrenar;
  - `on_update(state, received) -> state`, antes de enviar.

| Ataque | Gancho | Efecto |
|---|---|---|
| `label_flip` | datos | `y → n_clases − 1 − y` |
| `sign_flip` | envío | Envía `x − s·(y_i − x)` |
| `gaussian` | envío | Envía `x + N(0, σ²)` |
| `scale` | envío | Envía `x + s·(y_i − x)` (sustitución de modelo) |

`label_flip` corrompe etiquetas reales para simular a un atacante. No son datos sintéticos de entrenamiento: no viola docs/RULES.md. La regla lo dirá de forma explícita.

**Métricas.**
- La precisión bajo ataque se obtiene barriendo `attack.fraction: [0, 0.1, 0.2, 0.3]`.
- Los agregadores que seleccionan (Krum, Multi-Krum, Bulyan) emiten `diagnostic.selection`, con las contribuciones conservadas y descartadas. Cruzando eso con `malicious` se calculan la precisión y la exhaustividad de la detección.

### 3.7 Eje `privacy`

- **En el edge (DP local).** `privacy: {name: local_dp, clip, sigma}` recorta la norma del incremento a `clip` y le suma `N(0, (sigma·clip)²)` antes de enviarlo. Es otro gancho `on_update`.
- **En el agregador (DP central).** El agregador `dp_fedavg` recorta el incremento de cada hijo respecto a la referencia, promedia y suma `N(0, (sigma·clip/m)²)`, donde `m` es el número de hijos (McMahan et al., 2018).
- **ε.** Cada ronda se emite `privacy.epsilon` acumulado para un `delta` configurable:
  - el mecanismo gaussiano tiene RDP `α/(2σ²)` por ronda y se compone sumando;
  - ε = min_α (T·α/(2σ²) + log(1/δ)/(α−1)), con T el número de rondas.

## 4. Catálogo por fases

Cada plugin lleva título, descripción y explicación en español, sus parámetros con descripción y su referencia. Así el tutorial del Studio lo muestra sin trabajo extra.

### P1 — Personalización

| Plugin | Eje | Qué hace | Referencia |
|---|---|---|---|
| `ditto` | entrenador | Entrena el global como `standard` y, aparte, un modelo personal con término proximal `λ/2·‖v − w‖²` hacia el global. Puntúa `personal` | Li et al., 2021 |
| `apfl` | entrenador | Modelo personal `v` mezclado con el global: `α·v + (1−α)·w`. `α` es fijo o se aprende | Deng et al., 2020 |
| `fedrep` | entrenador | Primero `head_epochs` sobre la cabeza con el cuerpo congelado, luego `local_epochs` sobre el cuerpo. Exige que la cabeza sea local (`fedper`, o una compartición `custom` con `head.*: local`); si no, `ConfigError` antes de lanzar el primer escenario | Collins et al., 2021 |
| `fedbabu` | entrenador | `standard` con la cabeza congelada en su inicialización. Se evalúa con ajuste fino (§3.4) | Oh et al., 2022 |
| `lg_fedavg` | compartición | Adaptadores y tronco locales; cabezas globales | Liang et al., 2020 |

### P2 — Deriva con datos no IID

| Plugin | Eje | Qué hace | Referencia |
|---|---|---|---|
| `scaffold` | entrenador | Corrige cada gradiente con `c − c_i`. Mantiene `c_i`, envía `c_i⁺` como `scaffold/<clave>` y recibe `c` en el estado global (el promedio de los `c_i⁺`). Con participación parcial, `c` es el promedio de los participantes; se documenta como aproximación | Karimireddy et al., 2020 |
| `fednova` | entrenador + optimizador | El edge envía `(y_i − x)/τ_i` como `fednova/<clave>`. El optimizador `fednova` aplica `x ← x + τ̄·d̄`, con `d̄` el agregado y `τ̄` la media de `train_steps` | Wang et al., 2020 |
| `feddyn` | entrenador + optimizador | Término lineal con el gradiente previo del edge más uno proximal (`α`). El servidor mantiene `h` y aplica `θ ← θ̄ − h/α`. La fracción de participantes es `train_edges / edges_total` | Acar et al., 2021 |
| `moon` | entrenador | Pérdida contrastiva sobre `features`: acerca la representación local a la del global y la aleja de la del modelo local anterior (`μ`, temperatura `τ`) | Li et al., 2021 |

`fednova` y `feddyn` necesitan su entrenador y su optimizador juntos. El plan valida el par y, si falta uno, da un error con la ruta.

### P3 — Robustez y privacidad

| Plugin | Eje | Qué hace | Referencia |
|---|---|---|---|
| `krum` | agregador | Elige la contribución con menor suma de distancias a sus `n − f − 2` vecinas | Blanchard et al., 2017 |
| `multi_krum` | agregador | Promedia (FedAvg) las `m` mejores según Krum | Blanchard et al., 2017 |
| `geometric_median` | agregador | Mediana geométrica ponderada por Weiszfeld (iteraciones y tolerancia) | Pillutla et al., 2022 (RFA) |
| `bulyan` | agregador | Multi-Krum seguido de media recortada coordenada a coordenada | El Mhamdi et al., 2018 |
| `norm_clip` | agregador | Recorta la norma del incremento de cada hijo a `bound` y aplica FedAvg | Sun et al., 2019 |
| `dp_fedavg` | agregador | DP central (§3.7) | McMahan et al., 2018 |
| `local_dp` | privacidad | DP local en el edge (§3.7) | — |
| `label_flip`, `sign_flip`, `gaussian`, `scale` | ataque | §3.6 | — |

Los agregadores que comparan contribuciones enteras (Krum, Multi-Krum, Bulyan, mediana geométrica) puntúan sobre las claves que tienen todos los hijos. Después combinan clave a clave las contribuciones elegidas, como FedAvg, para que las claves que solo traen algunos hijos sigan funcionando (adaptadores por dataset). Si los hijos no bastan para el `f` pedido, el agregador da un error claro al cerrar la ronda.

### P4 — Servidor y asincronía

| Plugin | Eje | Qué hace | Referencia |
|---|---|---|---|
| `fedyogi` | optimizador | Yogi sobre el pseudogradiente | Reddi et al., 2021 |
| `fedadagrad` | optimizador | Adagrad sobre el pseudogradiente | Reddi et al., 2021 |
| `fedasync` | optimizador | `x ← (1−α_s)·x + α_s·x̄`, con `α_s = α·(1 + staleness)^(−a)` | Xie et al., 2019 |
| FedBuff | combinación | `quorum: K` (entero) + `staleness: next_round` + `stale_weighting: polynomial`, con `experiments/fedbuff.yaml` como ejemplo | Nguyen et al., 2022 |

Cada fase tiene su propio plan de implementación, su issue y su PR.

## 5. Comparar técnicas

- **Entre escenarios.** Por ejemplo, `sweep: {learning.trainer: [standard, fedprox, scaffold, ditto]}` o `{attack.fraction: [0, 0.2]}`. El barrido admite valores que son plugins completos con parámetros. Una clave `a,b` barre varias rutas a la vez: `learning.sharing,learning.trainer.name: [[fedper, fedrep], [fedavg, ditto]]` da un escenario por pareja.
- **Entre niveles.** Cada nodo de la topología puede tener su `aggregator` y la raíz su `server_optimizer`, así que la técnica también se compara por nivel.
- **Informes.** `onion_fl report --by scenario` y la vista Comparar del Studio dan la media ± IC sobre las semillas por nivel y métrica, incluidas `personal`, `finetuned`, `privacy.epsilon` y las de selección.
- **Experimentos de ejemplo.** Uno por fase sobre SWELL + WESAD: `experiments/techniques_personalisation.yaml`, `techniques_drift.yaml`, `techniques_robustness.yaml` y `techniques_server.yaml`. Sus resultados se confirman en `results/` y se citan en el README.

## 6. Pruebas

- **Matemática sin aprendizaje,** con matrices escritas a mano:
  - Krum elige el grupo honesto y Multi-Krum promedia los `m` correctos;
  - la mediana geométrica de puntos conocidos coincide con la esperada;
  - Bulyan y `norm_clip` acotan la influencia de un valor extremo;
  - FedNova coincide con FedAvg cuando los `τ_i` son iguales;
  - FedYogi y FedAdagrad siguen sus fórmulas;
  - las claves auxiliares viajan según la compartición y el optimizador no las toca;
  - la cota de ε coincide con la forma cerrada.
- **Ataques y privacidad:** los ganchos transforman lo esperado sobre estados escritos a mano; la elección de edges maliciosos es determinista por semilla y respeta `fraction`.
- **Protocolo con el entrenador `stub`:** las claves auxiliares suben y bajan por un árbol de dos niveles; `train_steps` y `train_edges` llegan al optimizador; las puntuaciones `personal` y `finetuned` se agregan.
- **Entrenadores sobre los extractos reales** (`data/samples/`, se saltan sin datos):
  - `ditto` y `apfl` producen un modelo personal distinto del global;
  - `fedrep` y `fedbabu` no cambian lo que debe quedar fijo;
  - `scaffold` y `feddyn` actualizan su estado de cliente;
  - `moon` baja su pérdida contrastiva.
- **Configuración:** los pares obligatorios (`fednova`, `feddyn`) y `fedrep` sin cabeza local dan `ConfigError` con la ruta.

## 7. Fuera de alcance

- Agregación segura criptográfica.
- Ataques de puerta trasera con disparador.
- FedBN (el modelo no tiene normalización por lotes).
- pFedMe y Per-FedAvg con segundo orden.
- Las familias de modelos (subproyecto 9).
