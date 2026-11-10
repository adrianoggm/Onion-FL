# Familias de modelos más allá de `modular_mlp`: diseño (subproyecto 9)

- **Fecha:** 2026-11-10. **Issue:** #229 (sigue a #43).
- **Alcance:** qué familias de modelos admite el framework, cómo encajan en las claves con espacio de nombres (§7.1 de la spec del framework) y cuál se implementa primero. La spec de técnicas (2026-10-03) dejó este subproyecto pendiente con tres candidatas: lineales, 1D-CNN/LSTM sobre señal cruda y árboles federados.

## 1. Lo que una familia tiene que cumplir

Todo lo que viene después del modelo (compartición, agregación por clave, optimizadores de servidor, entrenadores, privacidad, diagnósticos) trabaja con un diccionario `clave → array`. Una familia encaja sin tocar el núcleo si cumple el siguiente contrato:

1. **Claves.** Cada clave del `state_dict` está en `adapter.<ds>.*` o `adapter.<n>.*` (adaptador común), en `trunk.<n>.*` o `trunk.<ds>.*`, o en `head.<tarea|ds>.*`. `group_of` y `SharingPolicy.scope_of` deciden con ellas qué viaja y hasta dónde, sin saber qué familia hay detrás.
2. **Construcción por nodo.** `build(shapes, seed)` recibe los `DataShape` (`n_features`, `n_classes`, tarea) y crea solo las partes de esos datasets. Cada parte se inicializa con (seed, ruta del módulo), así que empieza igual en el modelo de un edge y en el global del coordinador.
3. **Agregación por clave.** Los arrays se combinan con medias ponderadas (o sus variantes robustas) de forma independiente por clave, con los pesos que da el número de muestras. Eso exige parámetros de forma fija y que promediar dos modelos tenga sentido, es decir, que haya un espacio de parámetros común.
4. **Entrenadores.** Los entrenadores de serie son de PyTorch y usan `model(x)`, `named_parameters()`, `state_dict()` y, en MOON, `features(x)`. `random` usa `init_parameters(seed)`.
5. **Opciones de compartición.** Las políticas `independent` y `harmonized` exigen opciones del modelo (`trunk`, `heads`, `adapters`). Una familia que quiera admitirlas declara esas mismas opciones. Si no las tiene, `check_model` devuelve un error de compartición claro, no un `AttributeError`.

## 2. Candidatas

| Familia | ¿Encaja con `SubjectData`? | ¿Encaja con la agregación por clave? | Decisión |
|---|---|---|---|
| **Transformer sobre features** (FT-Transformer, Gorishniy et al., 2021) | Sí: lee la misma fila de features por ventana | Sí: todos sus parámetros son tensores de forma fija | **Se implementa ahora** (`feature_transformer`) |
| Lineal / regresión logística | Sí | Sí | No aporta una familia nueva. `modular_mlp` con `trunk_hidden: []` ya es adaptador + cabeza, y solo lo separa de un modelo lineal la ReLU del adaptador. Si se necesita como baseline, es una opción de activación, no un plugin nuevo |
| 1D-CNN / LSTM / TCN sobre señal cruda | **No.** `SubjectData.X` es `float32[n, f]`: una fila por ventana con features ya agregadas (HRV, EDA, interacción). No hay señal cruda, y el orden de las columnas no es temporal | Sí, una vez que hay datos | Aplazada. Necesita ampliar el contrato con una vista de secuencia (`X: [n, canales, longitud]` o ventanas consecutivas del mismo sujeto), con su ingesta y su caché. Ese cambio va en su propia issue |
| Modelo temporal sobre ventanas consecutivas (GRU sobre las k filas anteriores) | Parcial: existe `t` por fila, pero los entrenadores barajan filas sueltas y la evaluación es por fila | Sí | Aplazada por la misma razón: el contrato de muestras de los entrenadores (`xs[idx]`) tendría que cambiar |
| Árboles / GBDT federados | Sí con las features | **No.** Un árbol no tiene un espacio de parámetros común que se pueda promediar por clave. Hay que intercambiar histogramas o árboles enteros (SecureBoost, FedTree), con otro rol de agregador | Fuera de alcance: sería otro protocolo, no otra familia de modelo |

**Por qué el transformer sobre features.** Es la única familia nueva que cabe en el contrato de datos actual y que, a la vez, cambia de verdad el sesgo inductivo del modelo. El MLP mezcla todas las features desde la primera capa. El transformer convierte cada feature en un token y aprende interacciones entre pares de features con atención. Además, su forma encaja con las claves de manera natural:

- **`adapter.<ds>`** es el tokenizador: `token_i = x_i · W_i + b_i`, con `W, b ∈ R^{n_features × d}`. Es por dataset por construcción, porque cada dataset tiene sus columnas. Con `adapters: shared` (features armonizadas) hay un único tokenizador en `adapter.0.*`.
- **`trunk`** son el token de clase y los bloques de atención (pre-norm, atención multi-cabeza, feed-forward con GELU, residuales), más una `LayerNorm` final. **No depende de `n_features`**: datasets con 16 y 30 features comparten exactamente los mismos tensores del tronco. En el MLP también se comparte, pero solo porque el adaptador proyecta a un ancho común. Aquí el tronco opera sobre un conjunto de tokens de tamaño variable.
- **`head.<tarea>`** es una capa lineal sobre el token de clase normalizado. `features(x)` devuelve ese token, que es lo que MOON contrasta.

Compartición, agregación y optimizadores de servidor no cambian: FedAvg, los agregadores robustos, FedAdam/Yogi/Adagrad y el DP en el edge trabajan con los mismos diccionarios por clave. La agregación jerárquica sigue siendo igual a la plana, porque la media ponderada por clave es asociativa con los pesos de muestras. Hay un test que lo comprueba con las claves de esta familia.

## 3. `feature_transformer`

| Opción | Por defecto | Significado |
|---|---|---|
| `token_dim` | 16 | Anchura de cada token |
| `blocks` | 1 | Bloques de atención del tronco |
| `attention_heads` | 2 | Cabezas; deben dividir `token_dim` |
| `ffn_factor` | 2 | Anchura de la feed-forward, en múltiplos de `token_dim` |
| `adapters`, `trunk`, `heads` | `per_dataset`, `shared`, `per_task` | Las mismas opciones que `modular_mlp`, así que `independent` y `harmonized` le valen |
| `dropout` | 0.1 | En las ramas residuales y en la feed-forward |

- **Inicialización.** Las capas lineales, como en el MLP (Kaiming uniforme, con generador por ruta). Tokenizador y token de clase, uniformes en ±1/√d. `LayerNorm`, a 1 y 0.
- **Entrenadores.** Funcionan todos los de serie: `standard`, `fedprox`, `ditto`, `apfl`, `fedrep`, `fedbabu`, `scaffold`, `fednova`, `feddyn`, `moon` y `stub`. Hay un test por entrenador sobre el extracto real de SWELL. Las inicializaciones `random`, `checkpoint` y `run` también funcionan.
- **Lo que no admite.** No admite un tronco que dependa del orden de las columnas (no es convolucional). Su coste crece con n_features²: con FEMNIST (784 píxeles = 784 tokens) es caro y no tiene sentido, porque para imágenes haría falta una CNN 2D, que queda pendiente junto con la 1D-CNN. Los `baselines` centralizados (`onion_fl.baselines`) siguen siendo un registro aparte y no lo incluyen.
- **Código.** `ModularModule` reúne lo que comparten las dos familias: datasets únicos, cabezas compartidas con el mismo número de clases, el adaptador común y el enrutado por dataset. El MLP sigue dando exactamente los mismos pesos y las mismas salidas que antes.

## 4. Aceptación

`experiments/model_families.yaml` compara las dos familias con la misma configuración que FedAvg en `techniques_server`: SWELL + WESAD, cuatro fogs, reparto segregado, 20 rondas, SGD y 3 semillas. El lr de cada familia se elige antes sobre los sujetos de validación (`model_families_select.yaml`). Los resultados, con su identidad (`topology_id`, `config_id`, `run_id`, `run_hash` y el commit) y el IC del 95 %, están en `results/model_families/INDEX.md`.

## 5. Pendiente

- Vista de secuencia en el contrato de datos, para 1D-CNN/TCN/LSTM sobre señal cruda (WESAD tiene las señales de muñeca a 4–64 Hz).
- CNN 2D para FEMNIST.
- Árboles federados, como protocolo propio (histogramas en lugar de pesos).
