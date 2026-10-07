from __future__ import annotations

"""Declarative ingestion: a YAML description turns raw files into SubjectData (spec §8.2).

::

    name: sweet
    root: data/SWEET/sample_subjects
    source: {reader: csv, path: "{subject}/{subject}_Features_Day*.csv"}
    options: {modality: all}            # declared options and their defaults
    steps:
      - join: {source: {reader: csv, path: "{subject}/current_stress.csv"},
               by: [subject], time: {left: "Unnamed: 0", right: TS, floor: min}}
      - label: {default: binary, strategies: {binary: {task: stress_binary,
                column: MAXIMUM_STRESS, threshold: 3}}}
      - features: {exclude: [TS, "Unnamed: 0"]}
        when: {modality: all}           # a step can depend on an option

Steps run in order on one table. ``{subject}`` in a path fills the ``subject``
column; the ``subject`` and ``label`` steps record their source columns so
``features`` never picks them, and ``features`` must be explicit: meta
columns are excluded on purpose, not by luck (docs/RULES.md).
"""

import pickle
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from onion_fl.core.registry import Registry
from onion_fl.data.contract import DataError, SubjectData, natural_key
from onion_fl.learning.model import NAME


class SourceSpec(BaseModel):
    """A reader and a path under the dataset root; other keys are reader params."""

    model_config = ConfigDict(extra="allow")

    reader: str
    path: str
    normalize_columns: bool = Field(
        False,
        description="Columnas sin espacios de borde, '_' por espacio y en minúscula",
    )

    def reader_params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME)
    description: str = ""
    root: str = "."
    source: SourceSpec
    options: dict[str, Any] = Field(default_factory=dict)
    steps: list[dict[str, Any]]
    min_samples_per_subject: PositiveInt = 1


def load_spec(path: str | Path) -> DatasetSpec:
    return DatasetSpec.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    )


@dataclass
class IngestContext:
    root: Path
    options: Mapping[str, Any]
    task: str | None = None
    n_classes: int | None = None
    meta_columns: set[str] = field(default_factory=set)


# --- readers --------------------------------------------------------------------

readers = Registry("reader")


class CsvParams(BaseModel):
    sep: str | None = Field(",", description="Separador; null lo detecta")
    encoding: str = "utf-8"
    decimal: str = "."
    na_values: list[str] = Field(default_factory=list)


@readers.register("csv", title="CSV", description="Texto delimitado.", params=CsvParams)
class CsvReader:
    def __init__(
        self, sep: str | None, encoding: str, decimal: str, na_values: list[str]
    ):
        self.sep, self.encoding, self.decimal = sep, encoding, decimal
        self.na_values = na_values or None

    def read(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep=self.sep,
            engine="python" if self.sep is None else "c",
            encoding=self.encoding,
            decimal=self.decimal,
            na_values=self.na_values,
        )


class ExcelParams(BaseModel):
    sheet: str | int = Field(0, description="Hoja por nombre o posición")


@readers.register(
    "excel",
    title="Excel",
    description="Hoja de un .xlsx (necesita openpyxl).",
    params=ExcelParams,
)
class ExcelReader:
    def __init__(self, sheet: str | int = 0) -> None:
        self.sheet = sheet

    def read(self, path: Path) -> pd.DataFrame:
        return pd.read_excel(path, sheet_name=self.sheet)


@readers.register(
    "parquet", title="Parquet", description="Tabla Parquet (necesita pyarrow)."
)
class ParquetReader:
    def read(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)


class PickleParams(BaseModel):
    key: str | None = Field(
        None, description="Ruta con puntos hasta la tabla dentro del objeto"
    )


@readers.register(
    "pickle",
    title="Pickle",
    description="Tabla de pandas guardada con pickle; solo ficheros de confianza.",
    params=PickleParams,
)
class PickleReader:
    def __init__(self, key: str | None = None) -> None:
        self.key = key

    def read(self, path: Path) -> pd.DataFrame:
        obj = pd.read_pickle(path)  # executes code from the file: trusted data only
        for part in self.key.split(".") if self.key else []:
            obj = obj[part]
        if not isinstance(obj, pd.DataFrame):
            raise DataError(
                f"{path}: {self.key!r} is a {type(obj).__name__}, not a table"
            )
        return obj


@readers.register(
    "npz",
    title="NumPy npz",
    description="Un array por columna; las matrices se expanden en name_0, name_1…",
)
class NpzReader:
    def read(self, path: Path) -> pd.DataFrame:
        columns: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                array = archive[name]
                if array.ndim == 1:
                    columns[name] = array
                else:
                    flat = array.reshape(len(array), -1)
                    columns |= {f"{name}_{i}": flat[:, i] for i in range(flat.shape[1])}
        return pd.DataFrame(columns)


def _glob_regex(pattern: str) -> re.Pattern[str]:
    parts, seen = [], False
    for token in re.split(r"(\{subject\}|\*|\?)", pattern):
        if token == "{subject}":
            parts.append("(?P=subject)" if seen else "(?P<subject>[^/]+)")
            seen = True
        elif token == "*":
            parts.append("[^/]*")
        elif token == "?":
            parts.append("[^/]")
        else:
            parts.append(re.escape(token))
    return re.compile("".join(parts))


WESAD_RATES = {
    "chest": {"ACC": 700, "ECG": 700, "EMG": 700, "EDA": 700, "TEMP": 700, "RESP": 700},
    "wrist": {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4},
}
WESAD_LABEL_RATE = 700


class WesadParams(BaseModel):
    location: Literal["chest", "wrist"] = "wrist"
    signals: list[str] | str | None = Field(
        None,
        description="Señales (lista o 'A,B'); por defecto todas las de la ubicación",
    )

    @model_validator(mode="after")
    def _known(self) -> WesadParams:
        available = WESAD_RATES[self.location]
        unknown = [s for s in _split(self.signals) or [] if s not in available]
        if unknown:
            raise ValueError(
                f"{self.location} has no signals {unknown}; available {list(available)}"
            )
        return self


def _split(value: list[str] | str | None) -> list[str] | None:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


@readers.register(
    "wesad_pickle",
    title="WESAD",
    description="Fichero S<n>.pkl de WESAD: señales de pecho o muñeca y etiqueta a 700 Hz.",
    params=WesadParams,
    explain=(
        "Las señales de muñeca se mantienen a la frecuencia de la etiqueta (muestra "
        "y retención); la columna condition trae la etiqueta original (0-7)."
    ),
)
class WesadPickleReader:
    def __init__(self, location: str = "wrist", signals=None) -> None:
        self.location = location
        self.signals = _split(signals) or list(WESAD_RATES[location])

    def read(self, path: Path) -> pd.DataFrame:
        with open(path, "rb") as handle:  # the dataset's own format; trusted files only
            data = pickle.load(handle, encoding="latin1")
        labels = np.asarray(data["label"])
        rates = WESAD_RATES[self.location]
        # The published chest files spell two signals "Resp" and "Temp".
        stored = {k.upper(): v for k, v in data["signal"][self.location].items()}
        arrays = {}
        for name in self.signals:
            if name not in stored:
                raise DataError(f"{path} has no {self.location} signal {name}")
            values = np.asarray(stored[name], dtype=np.float64)
            arrays[name] = values.reshape(len(values), -1)
        # Keep the label rows every signal covers, as the old windows did.
        n = min(
            [len(labels)]
            + [len(a) * WESAD_LABEL_RATE // rates[k] for k, a in arrays.items()]
        )
        held = np.arange(n)
        columns = {}
        for name, values in arrays.items():
            index = held * rates[name] // WESAD_LABEL_RATE  # sample and hold
            for i in range(values.shape[1]):
                columns[f"{name.lower()}_{i}"] = values[index, i]
        columns["condition"] = labels[:n]
        return pd.DataFrame(columns)


def source_files(source: SourceSpec, root: Path) -> list[tuple[Path, str | None]]:
    """Files the path matches, with the subject that ``{subject}`` captured."""
    regex = _glob_regex(source.path)
    found = []
    for path in sorted(root.glob(source.path.replace("{subject}", "*"))):
        match = regex.fullmatch(path.relative_to(root).as_posix())
        if match is not None:
            found.append((path, match.groupdict().get("subject")))
    if not found:
        raise DataError(f"no file matches {(root / source.path).as_posix()}")
    return found


def read_file(reader: Any, source: SourceSpec, path: Path, subject: str | None):
    frame = reader.read(path)
    if source.normalize_columns:
        frame.columns = [
            str(c).strip().replace(" ", "_").lower() for c in frame.columns
        ]
    if subject is not None:
        frame["subject"] = subject
    return frame


def read_source(source: SourceSpec, root: Path) -> pd.DataFrame:
    """Read every file the path matches; ``{subject}`` becomes the ``subject`` column."""
    reader = readers.create(source.reader, source.reader_params())
    frames = [read_file(reader, source, *found) for found in source_files(source, root)]
    return pd.concat(frames, ignore_index=True)


# --- steps -----------------------------------------------------------------------

steps = Registry("step")


class SubjectParams(BaseModel):
    column: str = "subject"
    regex: str | None = Field(
        None, description="Extrae el sujeto (primer grupo o coincidencia entera)"
    )


@steps.register(
    "subject",
    title="Sujeto",
    description="Columna (y regex opcional) de donde sale el identificador de sujeto.",
    params=SubjectParams,
)
class SubjectStep:
    def __init__(self, column: str = "subject", regex: str | None = None) -> None:
        self.column = column
        self.regex = re.compile(regex) if regex else None

    def _extract(self, value: str) -> str | None:
        match = self.regex.search(value)
        if match is None:
            return None
        return match.group(1) if self.regex.groups else match.group(0)

    def apply(self, df: pd.DataFrame, ctx: IngestContext | None) -> pd.DataFrame:
        _require(df, [self.column], "subject")
        values = df[self.column].astype(str).str.strip()
        if self.regex is not None:
            extracted = values.map(self._extract)
            bad = values[extracted.isna()].unique()[:3].tolist()
            if bad:
                raise DataError(f"subject regex does not match {bad}")
            values = extracted
        if ctx is not None and self.column != "subject":
            ctx.meta_columns.add(self.column)
        return df.assign(subject=values)


class ReplaceParams(BaseModel):
    values: dict[Any, Any] = Field(description="Valor -> sustituto; null es ausente")
    columns: list[str] | None = None


@steps.register(
    "replace",
    title="Sustituir",
    description="Cambia valores (por ejemplo códigos de error) por otros o por ausentes.",
    params=ReplaceParams,
)
class ReplaceStep:
    def __init__(
        self, values: dict[Any, Any], columns: list[str] | None = None
    ) -> None:
        self.values = {k: np.nan if v is None else v for k, v in values.items()}
        self.columns = columns

    def apply(self, df: pd.DataFrame, ctx: IngestContext | None) -> pd.DataFrame:
        columns = self.columns or list(df.columns)
        _require(df, columns, "replace")
        out = df.copy()
        for column in columns:
            # map, not DataFrame.replace: replace's silent downcasting is deprecated
            out[column] = (
                out[column].map(lambda v: self.values.get(v, v)).infer_objects()
            )
        return out


class SelectParams(BaseModel):
    query: str = Field(description="Expresión de pandas.DataFrame.query")


@steps.register(
    "select",
    title="Filtrar filas",
    description="Conserva las filas que cumplen una expresión.",
    params=SelectParams,
)
class SelectStep:
    def __init__(self, query: str) -> None:
        self.query = query

    def apply(self, df: pd.DataFrame, ctx: IngestContext | None) -> pd.DataFrame:
        return df.query(self.query).reset_index(drop=True)


class TimeJoin(BaseModel):
    left: str
    right: str
    floor: str = Field("min", description="Resolución común (min, s, h…)")


class JoinParams(BaseModel):
    source: SourceSpec
    by: list[str] = Field(
        default_factory=list,
        description="Columnas clave (no 'on': YAML lo lee como true)",
    )
    time: TimeJoin | None = None
    how: Literal["inner", "left"] = "inner"
    deduplicate: bool = Field(
        False, description="Una fila por clave en el otro fichero"
    )
    pad: dict[str, PositiveInt] = Field(
        default_factory=dict,
        description="Claves que se rellenan con ceros por la derecha hasta esa "
        "longitud en ambos lados antes de unir (hojas que perdieron los ceros finales)",
    )


@steps.register(
    "join",
    title="Unir",
    description="Une otro fichero por claves y, opcionalmente, por tiempo redondeado.",
    params=JoinParams,
)
class JoinStep:
    def __init__(self, source, by, time, how, deduplicate, pad) -> None:
        self.source = SourceSpec.model_validate(source)
        self.by, self.how, self.deduplicate = list(by), how, deduplicate
        self.pad = dict(pad)
        self.time = None if time is None else TimeJoin.model_validate(time)
        self._right: pd.DataFrame | None = None

    def apply(self, df: pd.DataFrame, ctx: IngestContext) -> pd.DataFrame:
        if self._right is None:  # read once, then joined with every source file
            self._right = read_source(self.source, ctx.root)
        left, right = df.copy(), self._right.copy()
        keys = list(self.by)
        _require(left, keys, "join")
        _require(right, keys, "join")
        for key in keys:
            left[key], right[key] = left[key].astype(str), right[key].astype(str)
            if key in self.pad:
                width = self.pad[key]
                left[key] = left[key].str.ljust(width, "0")
                right[key] = right[key].str.ljust(width, "0")
        if self.time is not None:
            t = self.time
            left["_time"] = pd.to_datetime(left[t.left], errors="coerce").dt.floor(
                t.floor
            )
            right["_time"] = pd.to_datetime(right[t.right], errors="coerce").dt.floor(
                t.floor
            )
            right = right.dropna(subset=["_time"])
            keys.append("_time")
        if not keys:
            raise DataError("join needs 'by' columns or 'time'")
        if self.deduplicate:
            right = right.drop_duplicates(subset=keys)
        out = left.merge(right, on=keys, how=self.how, suffixes=("", "_right"))
        return out.drop(columns=["_time"], errors="ignore")


TIME = "__time__"  # each row's seconds: metadata for streams, never a feature


class TimeParams(BaseModel):
    column: str | None = Field(
        None, description="Columna con la marca de tiempo de cada fila"
    )
    format: str | None = Field(
        None,
        description="Formato strptime de la columna; sin él, segundos o una fecha "
        "que pandas entienda",
    )
    rate: PositiveFloat | None = Field(
        None,
        description="Filas por segundo: el tiempo sale de la posición de la fila en "
        "su fichero",
    )
    pad_to: PositiveInt | None = Field(
        None,
        description="Rellena con ceros por la derecha hasta esta longitud antes de "
        "leer (hojas que perdieron los ceros finales)",
    )

    @model_validator(mode="after")
    def _one_source(self) -> TimeParams:
        if (self.column is None) == (self.rate is None):
            raise ValueError("time needs a column or rate, and only one of them")
        return self


@steps.register(
    "time",
    title="Tiempo",
    description="El tiempo de cada fila en segundos, de una columna o de su posición.",
    params=TimeParams,
    explain=(
        "Es un metadato para los streams: nunca es una feature, y su columna de "
        "origen tampoco. Cada ventana toma el tiempo de su última fila, y cada "
        "sujeto cuenta desde su primera observación."
    ),
)
class TimeStep:
    def __init__(
        self,
        column: str | None,
        format: str | None,
        rate: float | None,
        pad_to: int | None,
    ) -> None:
        self.column, self.format, self.rate, self.pad_to = column, format, rate, pad_to

    def apply(self, df: pd.DataFrame, ctx: IngestContext | None) -> pd.DataFrame:
        if self.rate is not None:
            return df.assign(**{TIME: np.arange(len(df)) / self.rate})
        _require(df, [self.column], "time")
        raw = df[self.column]
        if self.format is None and pd.api.types.is_numeric_dtype(raw):
            seconds = raw.astype(float)
        else:
            text = raw.astype(str)
            if self.pad_to is not None:
                text = text.str.ljust(self.pad_to, "0")
            stamps = pd.to_datetime(text, format=self.format, errors="coerce")
            seconds = (stamps - pd.Timestamp(0)).dt.total_seconds()
        bad = raw[seconds.isna()].unique()[:3].tolist()
        if bad:
            raise DataError(f"time: rows of {self.column!r} without a time: {bad}")
        if ctx is not None:
            ctx.meta_columns.add(self.column)
        return df.assign(**{TIME: seconds.to_numpy(np.float64)})


STATS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": lambda a: a.mean(axis=0),
    "std": lambda a: a.std(axis=0),
    "min": lambda a: a.min(axis=0),
    "max": lambda a: a.max(axis=0),
    "median": lambda a: np.median(a, axis=0),
}
REDUCE: dict[str, Callable[[np.ndarray], float]] = {
    "mean_round": lambda a: float(np.round(a.mean())),  # half to even, as before
    "mode": lambda a: float(np.bincount(a.astype(np.int64)).argmax()),
    "first": lambda a: float(a[0]),
    "last": lambda a: float(a[-1]),
}


class WindowParams(BaseModel):
    size: PositiveInt = Field(description="Filas por ventana")
    overlap: float = Field(0.0, ge=0, lt=1)
    stats: list[Literal["mean", "std", "min", "max", "median"]] = Field(
        default_factory=lambda: list(STATS)
    )
    columns: list[str] | None = Field(
        None, description="Canales; por defecto los numéricos"
    )
    label: Literal["mean_round", "mode", "first", "last"] = "mean_round"


@steps.register(
    "window",
    title="Ventanas",
    description="Ventanas deslizantes por sujeto con estadísticos por canal.",
    params=WindowParams,
    explain="Se descartan las ventanas con alguna fila sin clase; la etiqueta se reduce por ventana.",
)
class WindowStep:
    def __init__(self, size, overlap, stats, columns, label) -> None:
        self.size, self.stats, self.columns, self.label = size, stats, columns, label
        self.step = max(1, round(size * (1 - overlap)))

    def apply(self, df: pd.DataFrame, ctx: IngestContext | None) -> pd.DataFrame:
        if "subject" not in df.columns:
            raise DataError("window runs per subject: put a subject step before it")
        reserved = {"subject", "label", TIME} | (ctx.meta_columns if ctx else set())
        columns = self.columns or [
            c
            for c in df.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
        ]
        _require(df, columns, "window")
        names = [f"{c}_{s}" for c in columns for s in self.stats]
        has_label, timed = "label" in df.columns, TIME in df.columns
        rows = []
        for subject, group in df.groupby("subject", sort=False):
            values = group[columns].to_numpy(dtype=np.float64)
            labels = group["label"].to_numpy(dtype=np.float64) if has_label else None
            times = group[TIME].to_numpy(dtype=np.float64) if timed else None
            for start in range(0, len(group) - self.size + 1, self.step):
                end = start + self.size
                row: list[Any] = [subject]
                if labels is not None:
                    window_labels = labels[start:end]
                    if np.isnan(window_labels).any():
                        continue
                    row.append(REDUCE[self.label](window_labels))
                if times is not None:  # observable once the window is complete
                    row.append(float(times[end - 1]))
                stats = np.stack(
                    [STATS[s](values[start:end]) for s in self.stats], axis=1
                )
                rows.append(row + stats.ravel().tolist())  # channel-major, then stat
        head = ["subject", "label"] if has_label else ["subject"]
        head += [TIME] if timed else []
        return pd.DataFrame(rows, columns=head + names)


class LabelRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(pattern=NAME)
    column: str
    map: dict[Any, int] | None = Field(None, description="Valor -> clase")
    threshold: float | None = Field(None, description="Clase 1 si valor >= umbral")
    bins: list[float] | None = Field(None, description="Cortes crecientes entre clases")
    round: bool = Field(False, description="Redondear antes de aplicar la regla")
    n_classes: int | None = Field(None, ge=2)

    @model_validator(mode="after")
    def _one_rule(self) -> LabelRule:
        rules = [r for r in (self.map, self.threshold, self.bins) if r is not None]
        if len(rules) > 1:
            raise ValueError("use only one of map, threshold or bins")
        if not rules and self.n_classes is None:
            raise ValueError("needs map, threshold, bins or n_classes")
        if self.bins is not None and self.bins != sorted(self.bins):
            raise ValueError("bins must be increasing")
        return self

    def classes(self) -> int:
        if self.n_classes is not None:
            return self.n_classes
        if self.map is not None:
            return max(self.map.values()) + 1
        return 2 if self.threshold is not None else len(self.bins or []) + 1

    def apply(self, values: pd.Series) -> pd.Series:
        if self.round or self.threshold is not None or self.bins is not None:
            values = pd.to_numeric(values, errors="coerce")
        if self.round:
            values = values.round()
        if self.map is not None:
            text = {str(k): v for k, v in self.map.items()}
            exact = values.map(lambda v: self.map.get(v, text.get(str(v).strip())))
            return pd.to_numeric(exact, errors="coerce").astype(float)
        if self.threshold is not None:
            return (values >= self.threshold).astype(float).where(values.notna())
        if self.bins is not None:
            classes = pd.Series(np.digitize(values, self.bins), index=values.index)
            return classes.astype(float).where(values.notna())
        return pd.to_numeric(values, errors="coerce").astype(float)


class LabelParams(BaseModel):
    task: str | None = None
    column: str | None = None
    map: dict[Any, int] | None = None
    threshold: float | None = None
    bins: list[float] | None = None
    round: bool = False
    n_classes: int | None = None
    strategies: dict[str, LabelRule] = Field(
        default_factory=dict,
        description="Reglas con nombre, elegidas con la opción label",
    )
    default: str | None = None

    @model_validator(mode="after")
    def _check(self) -> LabelParams:
        if self.strategies:
            if self.default not in self.strategies:
                raise ValueError(f"default must be one of {sorted(self.strategies)}")
        else:
            LabelRule.model_validate(self.model_dump(exclude={"strategies", "default"}))
        return self


@steps.register(
    "label",
    title="Etiqueta",
    description="Columna de la etiqueta y su regla: mapa, umbral o intervalos.",
    params=LabelParams,
    explain="Las filas sin clase se descartan. Con strategies, el experimento elige la regla.",
)
class LabelStep:
    def __init__(self, strategies: dict, default: str | None, **rule: Any) -> None:
        if strategies:
            self.rules = {k: LabelRule.model_validate(v) for k, v in strategies.items()}
            self.default = default
        else:
            self.rules, self.default = (
                {"default": LabelRule.model_validate(rule)},
                "default",
            )

    def apply(self, df: pd.DataFrame, ctx: IngestContext | None) -> pd.DataFrame:
        name = (ctx.options.get("label") if ctx else None) or self.default
        if name not in self.rules:
            raise DataError(
                f"unknown label strategy {name!r}; available {sorted(self.rules)}"
            )
        rule = self.rules[name]
        _require(df, [rule.column], "label")
        if ctx is not None:
            ctx.task, ctx.n_classes = rule.task, rule.classes()
            ctx.meta_columns |= {r.column for r in self.rules.values()}
        return df.assign(label=rule.apply(df[rule.column]))


class FeaturesParams(BaseModel):
    include: list[str] | None = Field(None, description="Lista exacta de features")
    exclude: list[str] = Field(default_factory=list)
    regex: str | None = Field(None, description="Features cuyo nombre coincide")

    @model_validator(mode="after")
    def _one_selector(self) -> FeaturesParams:
        if self.include is not None and self.regex is not None:
            raise ValueError("use include or regex, not both")
        return self


@steps.register(
    "features",
    title="Features",
    description="Qué columnas son features; el resto se descarta.",
    params=FeaturesParams,
    explain="Las columnas de sujeto y etiqueta nunca son features; las meta deben excluirse a mano.",
)
class FeaturesStep:
    def __init__(self, include, exclude, regex) -> None:
        self.include, self.exclude = include, set(exclude)
        self.regex = re.compile(regex) if regex else None

    def apply(self, df: pd.DataFrame, ctx: IngestContext | None) -> pd.DataFrame:
        reserved = {"subject", "label", TIME} | (ctx.meta_columns if ctx else set())
        if self.include is not None:
            _require(df, self.include, "features")
            taken = sorted(reserved & set(self.include))
            if taken:
                raise DataError(
                    f"features cannot include subject or label columns {taken}"
                )
            columns = list(self.include)
        else:
            columns = [c for c in df.columns if c not in reserved]
            if self.regex is not None:
                columns = [c for c in columns if self.regex.search(str(c))]
        columns = [c for c in columns if c not in self.exclude]
        if not columns:
            raise DataError("no feature columns left")
        out = df[[c for c in ("subject", "label", TIME) if c in df.columns]].copy()
        for column in columns:
            out[column] = _numeric(df[column], column)
        return out


def _numeric(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        raise DataError(f"feature {name!r} is a timestamp; exclude it")
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    parsed = pd.to_numeric(series, errors="coerce")
    bad = series[parsed.isna() & series.notna()].unique()[:3].tolist()
    if bad:
        raise DataError(
            f"feature {name!r} has non-numeric values such as {bad}; "
            "replace them or exclude the column"
        )
    return parsed.astype(float)


def _require(df: pd.DataFrame, columns: list[str], step: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataError(f"{step}: missing columns {missing}")


# --- pipeline ----------------------------------------------------------------------


def resolve_options(spec: DatasetSpec, options: Mapping[str, Any] | None) -> dict:
    """Declared defaults plus the given options; ``label`` picks a label strategy."""
    if "subject" in spec.options:
        raise DataError("'subject' is reserved for {subject} in paths, not an option")
    given = dict(options or {})
    unknown = sorted(set(given) - set(spec.options) - {"label"})
    if unknown:
        raise DataError(f"unknown options {unknown}; declared {sorted(spec.options)}")
    resolved = {**spec.options, **given}
    for entry in spec.steps:
        if "label" in entry and "label" not in resolved:
            resolved["label"] = (entry["label"] or {}).get("default")
    return resolved


def _active(when: Mapping[str, Any], options: Mapping[str, Any]) -> bool:
    for option, expected in when.items():
        if option not in options:
            raise DataError(
                f"when: unknown option {option!r}; declared {sorted(options)}"
            )
        allowed = expected if isinstance(expected, list) else [expected]
        if options[option] not in allowed:
            return False
    return True


def _fill(value: Any, options: Mapping[str, Any]) -> Any:
    """Replace ``{option}`` in strings; a string that is only ``{option}`` takes its value."""
    if isinstance(value, dict):
        return {k: _fill(v, options) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, options) for v in value]
    if not isinstance(value, str):
        return value
    for name, option in options.items():
        if value == f"{{{name}}}":
            return option
        value = value.replace(f"{{{name}}}", str(option))
    return value


def _compile(spec: DatasetSpec, options: Mapping[str, Any] | None):
    """Resolve options and build the reader and the active steps, without reading data."""
    resolved = resolve_options(spec, options)
    source = SourceSpec.model_validate(_fill(spec.source.model_dump(), resolved))
    reader = readers.create(source.reader, source.reader_params())
    active = []
    for entry in spec.steps:
        names = [k for k in entry if k != "when"]
        if len(names) != 1:
            raise DataError(f"each steps entry needs exactly one step, got {names}")
        if _active(entry.get("when") or {}, resolved):
            params = _fill(entry[names[0]] or {}, resolved)
            active.append((names[0], steps.create(names[0], params)))
    return resolved, source, reader, active


def _since_first(seconds: pd.Series) -> np.ndarray:
    values = seconds.to_numpy(np.float64)
    return values - values.min()


def check_spec(spec: DatasetSpec, options: Mapping[str, Any] | None = None) -> None:
    """Validate a description and its options without touching the data."""
    _compile(spec, options)


def ingest(
    spec: DatasetSpec,
    options: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> list[SubjectData]:
    """Run the description and return one ``SubjectData`` per subject, in natural order.

    Each source file goes through the steps on its own (a subject's day, a WESAD
    recording), so memory stays bounded and windows never span two files.
    """
    resolved, source, reader, active = _compile(spec, options)
    ctx = IngestContext(
        root=Path(_fill(str(root or spec.root), resolved)), options=resolved
    )
    frames, first = [], None
    for path, subject in source_files(source, ctx.root):
        df = read_file(reader, source, path, subject)
        for _, step in active:
            df = step.apply(df, ctx)
        if first is None:
            first = (path, list(df.columns))
        elif list(df.columns) != first[1]:
            missing = sorted(set(first[1]) - set(df.columns))
            extra = sorted(set(df.columns) - set(first[1]))
            raise DataError(
                f"{path} ends with other columns than {first[0]}: "
                f"missing {missing}, extra {extra}"
            )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    ran = {name for name, _ in active}

    if "subject" not in df.columns:
        raise DataError(
            "no subject: add a subject step or {subject} to the source path"
        )
    if ctx.task is None or ctx.n_classes is None:
        raise DataError("no label step")
    if "features" not in ran:
        raise DataError("add a features step: meta columns are excluded explicitly")
    # A column a subject, label or time step read is meta, even when that step
    # came after features: it never becomes one.
    reserved = {"subject", "label", TIME} | ctx.meta_columns
    features = [c for c in df.columns if c not in reserved]
    df = df.dropna(subset=["label"]).assign(subject=lambda d: d["subject"].astype(str))
    return [
        SubjectData(
            X=part[features].to_numpy(dtype=np.float32),
            y=part["label"].to_numpy(),
            dataset=spec.name,
            subject=subject,
            task=ctx.task,
            n_classes=ctx.n_classes,
            feature_names=[str(c) for c in features],
            t=_since_first(part[TIME]) if TIME in part.columns else None,
        )
        for subject, part in sorted(
            df.groupby("subject"), key=lambda kv: natural_key(kv[0])
        )
        if len(part) >= spec.min_samples_per_subject
    ]
