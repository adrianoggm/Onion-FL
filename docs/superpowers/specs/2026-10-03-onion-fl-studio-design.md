# Onion-FL Studio: diseño del front (subproyecto 7)

Especificación de la issue #103. Parte de los requisitos del núcleo para el front (§11 de la spec del framework) y de lo pedido: una biblioteca de topologías con su editor, el estado y el rendimiento de lo ejecutado, comparativas entre topologías y escenarios por nivel, y un modo tutorial que enseñe de forma visual qué implica cada opción.

## 1. Objetivo

Una aplicación local, `onion_fl serve`, que abre en el navegador:

| Área | Qué hace |
|---|---|
| **Topologías** | Biblioteca de `topologies/*.yaml` con su grafo, `topology_id` y YAML; editor por niveles con validación al momento; guardar como nueva o sobrescribir |
| **Experimentos** | Lista de `experiments/*.yaml`; validación con la ruta de cada error; escenarios del barrido con su `config_id`; plan en seco (composición por fog y grupos por enlace); lanzar en simulación o en real |
| **Ejecuciones** | Lista de `runs/` con estado (`running`, `finished`, `incomplete`, `failed`); detalle con identidad, resumen, métricas por nivel a lo largo de las rondas, tráfico y diagnósticos; los eventos llegan en vivo mientras corre |
| **Comparar** | Elegir experimento o ejecuciones, nivel, métrica y agrupación (`topology_id`, `scenario`, `dataset`…): media ± IC sobre las semillas por ronda, con la dispersión entre nodos del nivel |
| **Tutorial** | Cada eje de plugins con su explicación, sus parámetros y una previsualización en seco: qué grupos cruzan cada enlace con cada compartición, cómo queda cada fog con cada reparto y α, cuánto tarda un mensaje con cada perfil de red |

## 2. Decisiones

| Decisión | Elección | Por qué |
|---|---|---|
| Backend | FastAPI dentro del paquete (`onion_fl.studio`), servido con uvicorn | Es Python, como el resto; reutiliza las funciones puras del núcleo sin duplicar lógica |
| Frontend | Una SPA en JavaScript sin dependencias ni paso de compilación, con SVG propio para grafos y gráficas | Se instala con el paquete y funciona sin internet; el informe HTML ya dibuja con SVG |
| Datos | Los ficheros del repositorio (`topologies/`, `experiments/`, `runs/`, `datasets/`) | Una sola fuente de verdad, la misma que usa la línea de órdenes |
| Lanzar ejecuciones | Un proceso hijo `onion_fl run` por petición | El servidor no se bloquea y la ejecución es la misma que desde la consola |
| En vivo | Sondeo incremental de `events.jsonl` por número de línea | Simple y válido en simulación y en real; sin websockets |
| Seguridad | Escucha en `127.0.0.1` por defecto; nombres validados con `^[A-Za-z0-9_.-]+$`, sin rutas | Es una herramienta local de investigación; nada fuera de los directorios configurados |

Las dependencias van en un extra `studio` (`fastapi`, `uvicorn`), incluido en `dev` para los tests.

## 3. API

Todas las respuestas son JSON. Los errores de validación devuelven 422 con `{"errors": ["ruta: mensaje", ...]}`.

| Método y ruta | Devuelve |
|---|---|
| `GET /api/schema` | JSON Schema del experimento más el catálogo de plugins (`experiment_schema()`) |
| `GET /api/topologies` | `[{name, topology_id, levels, nodes, leaves}]` |
| `GET /api/topologies/{name}` | `{name, yaml, graph}` |
| `POST /api/topologies/validate` | Cuerpo: la topología (forma compacta o general). Devuelve `{graph}` o 422 |
| `PUT /api/topologies/{name}` | Valida y guarda `topologies/<name>.yaml`; devuelve `{graph}` |
| `GET /api/experiments` | `[{name, description, topology, scenarios, seeds}]` |
| `GET /api/experiments/{name}` | `{name, yaml, config, scenarios: [{name, seed, config_id}]}` |
| `POST /api/experiments/validate` | Cuerpo: la config. Devuelve escenarios o 422 |
| `POST /api/experiments/{name}/plan` | El plan en seco de cada escenario; 422 si faltan datos |
| `POST /api/experiments/{name}/run` | Cuerpo `{scenario?, mode, workers}`. Lanza el proceso y devuelve `{pid}` |
| `GET /api/runs` | `[{run_id, experiment, scenario, seed, status, started_at, finished_at, topology_id, config_id, rounds, final}]` |
| `GET /api/runs/{run_id}` | `{meta, summary, roles, composition}` |
| `GET /api/runs/{run_id}/events?after=N&limit=M` | Las líneas a partir de la `N`, con `next` para el siguiente sondeo |
| `GET /api/runs/{run_id}/series?level&name&model&dataset` | Serie por ronda (media, mínimo y máximo entre los nodos del nivel) |
| `GET /api/compare?experiment&level&metric&by` | La tabla de `Runs.compare` |
| `GET /api/tutorial` | Ejes con sus plugins: título, descripción, explicación y JSON Schema de parámetros |
| `POST /api/preview/sharing` | `{topology, sharing, model, datasets}` → grupos por enlace y qué guarda cada nivel |
| `POST /api/preview/placement` | `{topology, placement, datasets: {nombre: sujetos}}` → composición por fog |
| `POST /api/preview/link` | `{profile, bytes}` → latencia media y percentiles, tiempo de transmisión y pérdida |

Las previsualizaciones son funciones puras sin datos. El reparto trabaja con recuentos hipotéticos de sujetos (contabilidad, no aprendizaje); la compartición, con los nombres de los grupos de un modelo construido con formas de relleno.

## 4. Vistas

- **Barra lateral** con las cinco áreas; la URL guarda la vista (`#/topologies/four_fogs`).
- **Topologías:**
  - lista a la izquierda y, a la derecha, el grafo con los niveles en filas, nodos coloreados por nivel y enlaces etiquetados con transporte/codec/perfil;
  - pestañas Grafo, YAML y Editor;
  - el editor permite añadir y quitar niveles y nodos, elegir el padre y los perfiles por defecto de cada nivel; cada cambio revalida y redibuja.
- **Experimentos:** detalle con escenarios y `config_id`, botón Plan (composición por fog en barras apiladas por dataset y grupos que circulan por cada enlace) y botón Ejecutar (modo y trabajadores).
- **Ejecuciones:**
  - tabla filtrable por experimento y estado;
  - el detalle muestra la identidad (con copia), el resumen, las curvas de métricas por nivel (global, cada nivel de agregación y edge), el tráfico por ronda y los diagnósticos;
  - mientras el estado sea `running`, sondea los eventos cada dos segundos.
- **Comparar:** formulario (experimento, nivel, métrica, agrupar por) → gráfica de media ± IC por ronda con una línea por grupo, y tabla de la última ronda.
- **Tutorial:**
  - una tarjeta por eje (topología, compartición, reparto, red, agregación, entrenamiento, evaluación…) con la explicación de cada plugin y sus parámetros;
  - las tarjetas con previsualización tienen controles (deslizador de α, selector de preset…) que redibujan al momento.

## 5. Pruebas

- **API** con el `TestClient` de FastAPI sobre un directorio temporal con topologías, experimentos y ejecuciones de prueba: listar, validar (con errores que señalan la ruta), guardar, plan sin datos (422 claro), eventos incrementales, series, comparar y previsualizaciones.
- **Seguridad de rutas:** nombres con `..`, `/` o `\` devuelven 400 y nunca salen de su directorio.
- **Lanzar:** el proceso se crea con los argumentos esperados (sin ejecutar uno real en los tests de la API).
- **Front:** el índice y los recursos estáticos se sirven; una prueba de humo en navegador recorre las cinco vistas. Las ejecuciones de ejemplo usan el entrenador de prueba, que no aprende (docs/RULES.md).

## 6. Fuera de alcance

Autenticación y uso multiusuario, edición de descriptores de datasets desde el front, y despliegue remoto (E6).
