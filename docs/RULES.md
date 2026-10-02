# Reglas del proyecto

Estas reglas protegen la validez científica de los resultados. Cada una existe por un problema real que ya ocurrió en este repositorio.

## 1. Solo datos reales para aprender y evaluar

- **No se usan datos sintéticos** (por ejemplo, generados con `np.random`) para entrenar, validar o evaluar modelos. Tampoco para "rellenar" features que faltan: `process_swell_rri.py` lo hacía, y escribía los valores inventados con los nombres de los CSV reales de SWELL.
- **Datasets autorizados:** SWELL, SWEET y WESAD, obtenidos de su fuente original y colocados en `data/` (que no está en git). ECG5000 se eliminó: sus particiones tenían fuga entre train y test.
- **Tests:**
  - Los tests de **protocolo y runtime** pueden usar el *entrenador de prueba*, que no aprende y devuelve pesos deterministas, con edges sin datos. Así se ejecutan en el CI.
  - Los tests que entrenan o evalúan aprendizaje usan extractos reales (`data/samples/`) y se saltan cuando no están.
  - Mockear MQTT, Flower o el sistema de ficheros está permitido.

## 2. Evaluación sin fugas

- **El test global se hace sobre sujetos completos reservados**, y esos sujetos son los mismos en todos los escenarios que se comparan.
- **El split `per_subject` no separa sujetos.** Divide las muestras de cada sujeto entre train, val y test, así que solo sirve como validación local en el edge, nunca como test global.
- **Ninguna estadística de preprocesado se ajusta con datos de val o test.** Escalado, imputación y filtrado de varianza se calculan con la bolsa de entrenamiento.
- **Las columnas meta no son features.** Identificadores de bloque, sesión, tiempo o sujeto (`blok`, `timestamp`, `PP`, …) se excluyen de forma explícita. La columna `blok` infló los baselines de SWELL hasta ~0,99 antes del commit `002246f`. Antes de usar un dataset nuevo, revisa su ficha (`data inspect`), que avisa de las features con una correlación casi perfecta con la etiqueta.

## 3. Resultados citables

- **Solo se cita un número que esté respaldado por un fichero commiteado**, y ese fichero indica la topología y la ejecución que lo generaron (`topology_id`, `config_id`, `run_id`, `run_hash`).
- **Los resultados anteriores al refactor** están en `results/legacy/`; consulta las salvedades de su `INDEX.md` antes de citarlos.

## 4. Forma de trabajo

- **Ramas.** Cada issue de GitHub tiene su rama `task/#N`, creada desde `develop`, que entra en `develop` por PR con el CI en verde. `main` solo recibe versiones funcionales etiquetadas.
- **Commits.** Formato `type(scope): Imperative summary in English`. Son atómicos, uno por elemento, y van sin `Co-Authored-By`.
- **Calidad.** `ruff check .`, `ruff format --check .` y `pytest` deben pasar antes de abrir un PR.
