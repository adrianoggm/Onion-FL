---
name: tarea-github
description: Úsala en Onion-FL cuando se diga "vamos con la issue N" o "con la #N", se pida crear la rama de una issue, abrir o revisar su PR, mergear, cerrar una issue, dar de alta issues o milestones en GitHub, publicar una versión de develop a main, o explicar cómo se trabaja en este repositorio.
---

# Trabajar una issue de Onion-FL en GitHub

Es el mismo flujo que en GitLab con Innovasur, pero en GitHub.

**Una issue = una rama `task/#N` desde `develop` = un PR hacia `develop`.**

`main` solo recibe versiones que funcionan, etiquetadas, desde `develop`.

```
issue #N (milestone vX.Y.Z) → task/#N desde develop → commits atómicos → PR task/#N → develop (CI verde) → merge → cerrar #N
develop ──PR "release vX.Y.Z"──► main → tag vX.Y.Z + notas + milestone cerrado
```

Las etapas van **con confirmación entre cada una**: no las juntes ni te adelantes.

Los comandos están escritos para Git Bash. En Windows PowerShell 5.1 no existe `&&`.

## Herramienta

```bash
python .claude/skills/tarea-github/scripts/gh.py <comando>     # --help para ver todos
```

Habla con la API REST de GitHub usando la credencial que ya guarda git (Git Credential Manager). **No uses `gh` ni pidas instalarlo.**

| Comando | Para qué |
|---|---|
| `whoami` | Comprueba la credencial y los permisos sobre el repo |
| `issues [--milestone vX.Y.Z] [--state open\|closed\|all]` | Lista las issues |
| `issue N` | Muestra la issue y el estado de sus dependencias (`**Depende de:**`) |
| `backlog fichero.md [--dry-run]` | Crea etiquetas, milestones e issues desde un backlog; si un título ya existe, no lo duplica |
| `pr N [--draft]` | Abre o actualiza el PR de `task/#N` hacia `develop`, con las etiquetas, el milestone y el enlace de la issue |
| `pr-status N` | Estado del PR y de los checks del CI |
| `merge P` | Mergea con merge commit (nunca squash) |
| `close N --pr P` | Comenta, cierra la issue y borra la rama remota |
| `release-pr vX.Y.Z` / `release vX.Y.Z` | Abre el PR de `develop` a `main` / crea el tag y las notas y cierra el milestone |

## Reglas duras

1. **Sin `Co-Authored-By` en los commits ni pie "Generated with" en los PR.** Los commits son del autor humano.
2. **Nunca commitees en `develop` ni en `main`, ni hagas push a `main`.** Los PR de tarea van siempre a `develop`.
3. **El push y el merge solo se hacen con una orden explícita.** Commitear no da permiso para empujar.
4. **GitHub no cierra la issue al mergear en `develop`**, porque no es la rama por defecto. `Closes #N` no basta: cierra con `gh.py close N --pr P`.
5. **Verifica tú y reporta la evidencia.** Si algo no se ha podido comprobar, dilo con todas las letras.
6. **El CI de GitHub solo corre en los PR hacia `main`**, para no gastar minutos. En los PR de tarea hacia `develop`, la verificación local de la etapa 3 es la garantía: repórtala en el PR antes de pedir el merge.

## Etapas

**0. Arranque.** Ejecuta `gh.py issue N`.
- Si alguna dependencia sigue abierta, dilo y pregunta antes de seguir.
- Si hay trabajo sin commitear de otra issue en el árbol de trabajo, para y pregunta.
- Lee lo que la spec (`docs/superpowers/specs/`) dice de esa fase; busca la clave del título, por ejemplo `Fase 1` para `[F1.1]`.

**1. Rama.** Proponla y espera confirmación:

```bash
git fetch origin && git switch develop && git pull --ff-only origin develop
git switch -c "task/#N"          # en minúscula, con almohadilla y entre comillas
```

Si hay cambios sin commitear, no los arrastres a la rama nueva sin preguntar.

Si `git ls-remote --heads origin develop` no devuelve nada, `develop` todavía no existe en GitHub. Propón crearla una única vez, desde `main` y con permiso: `git push -u origin develop`.

**2. Plan.** Propón qué cambia, en qué ficheros y cómo, y espera confirmación. Si no es trivial, usa superpowers:writing-plans.

**3. Implementación y verificación**, en el entorno de Python 3.11 del repo. La primera línea solo hace falta si `.venv` no existe:

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu && uv pip install --python .venv -e ".[dev]"
.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check . && .venv/Scripts/python -m pytest
```

**4. Commits.** Propón el reparto (qué ficheros lleva cada commit) y espera confirmación.
- Formato: `type(scope): Imperative summary in English`. Tipos: `feat`, `fix`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`, `style`. No hace falta poner `#N` en el commit: lo lleva la rama.
- Un elemento es un cambio lógico que deja el repo funcionando por sí solo. Si dos cambios no funcionan por separado (por ejemplo, mover un paquete y actualizar sus imports), van en el mismo commit.
- El formateo masivo va en su propio commit `style:`.
- Antes de cada commit, revisa `git diff --cached --name-only`.

**5. Push.** Solo cuando se pida. Si la orden es "commitea y pushea", alterna: commit, push, siguiente commit, push.

```bash
git push origin "HEAD:refs/heads/task/#N"     # el primero crea la rama remota
git branch --set-upstream-to="origin/task/#N"
```

**6. PR.** Solo con confirmación. Ejecuta `gh.py pr N`, que crea el PR con:
- base `develop` y título igual al nombre de la rama, `task/#N`;
- las **mismas etiquetas y el mismo milestone que la issue**, asignado a quien lo abre;
- un cuerpo que empieza por `Refs #N — <título de la issue>` (o `Closes`, si `develop` es la rama por defecto), seguido de los asuntos de los commits sin formatear.

Si las etiquetas de la issue no son las adecuadas, corrígelas en la issue antes de abrir el PR. Si el PR ya existe, `gh.py pr N` lo actualiza en lugar de duplicarlo; úsalo también para arreglar un PR al que le falten metadatos.

Pasa el resultado de la verificación local con `--note`, por ejemplo `gh.py pr N --note "ruff ok; pytest 136 passed, 4 skipped (Python 3.11)"`, para que quede en el PR. Devuelve la URL. Los PR hacia `develop` no tienen checks de GitHub, por diseño.

Si algo falla, arréglalo con un commit nuevo en la rama, sin `--amend` ni force push, y vuelve a empujar cuando se pida.

**7. Merge y cierre.** Solo con la orden:

```bash
python .claude/skills/tarea-github/scripts/gh.py merge P
python .claude/skills/tarea-github/scripts/gh.py close N --pr P
git switch develop && git pull --ff-only origin develop && git fetch --prune origin && git branch -d "task/#N"
```

En los PR hacia `main`, `merge` se niega a mergear si el CI no está en verde; no uses `--force` sin una orden expresa. En los PR hacia `develop` no hay CI, y `merge` solo avisa.

Al terminar, informa de lo integrado y propón la siguiente issue del milestone que ya no tenga dependencias abiertas.

**8. Release.** Cuando el milestone esté completo y se pida:
1. `gh.py release-pr vX.Y.Z`;
2. esperar a que el CI esté en verde (`gh.py pr-status` sobre el PR de release); es el único momento en que corre el CI de GitHub;
3. pedir aprobación y ejecutar `gh.py merge P`;
4. `gh.py release vX.Y.Z`.

## Dar de alta issues

Primero escribe un backlog en Markdown. Cada `## [CLAVE] Título` es una issue:

```markdown
## [F2.1] Núcleo: Message y codecs
Meta: labels=area:core,enhancement · milestone=v0.2.0 · depende=F1.1,#77

- [ ] Tarea…
**Aceptación:** …
```

`depende` admite claves definidas antes en el mismo fichero o números `#N`. Las etiquetas y los milestones que falten se crean solos.

1. Ejecuta `gh.py backlog fichero.md --dry-run`, enseña el resultado y espera confirmación.
2. Ejecuta `gh.py backlog fichero.md`.
3. Borra el fichero, porque a partir de ahí GitHub es la fuente de verdad.

## Errores comunes

| Error | Corrección |
|---|---|
| Rama `task/fase-0` o `task/77` | La rama es `task/#N`, con el número real de la issue |
| PR con título descriptivo, o abierto contra `main` | Título `task/#N` y base `develop`; `gh.py pr` ya lo hace bien |
| Esperar que `Closes #N` cierre la issue | No se cierra al mergear en `develop`: usa `gh.py close N --pr P` |
| Pasar a `git credential` el texto con una tubería de PowerShell | PowerShell añade un BOM y git rechaza el campo `protocol`; usa `gh.py`, que le pasa bytes |
| Crear el entorno virtual en una ruta muy larga | Supera los 260 caracteres de Windows y la instalación de scikit-learn falla; usa `.venv` en el repo |
| Usar el `python` global | Puede importar `flower_basic` desde otra copia del repo; usa `.venv` o `PYTHONPATH=src` |
