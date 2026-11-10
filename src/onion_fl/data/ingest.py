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

import json
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
    na_values: list[str] = Field(
        default_factory=list, description="Textos que se leen como vacíos"
    )


@readers.register(
    "excel",
    title="Excel",
    description="Hoja de un .xlsx (necesita openpyxl: extra excel).",
    params=ExcelParams,
)
class ExcelReader:
    def __init__(
        self, sheet: str | int = 0, na_values: list[str] | None = None
    ) -> None:
        self.sheet, self.na_values = sheet, na_values or None

    def read(self, path: Path) -> pd.DataFrame:
        return pd.read_excel(path, sheet_name=self.sheet, na_values=self.na_values)


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


# --- Empatica E4 (WESAD wrist, PhysioNet wearables, Nurse Stress, Stress-Predict) --

# Each E4 signal and its channels, in the order every E4 reader emits them, so
# the window features have the same names and order in every E4 dataset.
E4_CHANNELS = {
    "ACC": ["acc_x", "acc_y", "acc_z"],  # raw, 1/64 g
    "BVP": ["bvp"],
    "EDA": ["eda"],  # μS
    "TEMP": ["temp"],  # °C
    "HR": ["hr"],  # beats per minute, computed by the device from BVP
    "IBI": ["ibi"],  # seconds between beats, at irregular times
}
E4_DEFAULT = "ACC,BVP,EDA,TEMP"


def _e4_signals(value: list[str] | str | None, available: Mapping[str, Any]) -> list:
    chosen = _split(value) or _split(E4_DEFAULT)
    unknown = [s for s in chosen if s not in available]
    if unknown:
        raise ValueError(f"unknown E4 signals {unknown}; available {list(available)}")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"signals are repeated: {chosen}")
    return [s for s in available if s in chosen]  # the canonical order


def _e4_table(blocks: list[tuple[np.ndarray, dict[str, np.ndarray]]]) -> pd.DataFrame:
    """Stack signals sampled at different times into one table sorted by ``time``.

    Each row is one sample of one signal: its channels have values and every
    other column is empty. The multi-rate window step reads each channel back
    at its own rate.
    """
    total = sum(len(times) for times, _ in blocks)
    names = [name for _, columns in blocks for name in columns]
    out = {"time": np.concatenate([t for t, _ in blocks] or [np.empty(0)])}
    out |= {name: np.full(total, np.nan) for name in names}
    offset = 0
    for times, columns in blocks:
        for name, values in columns.items():
            out[name][offset : offset + len(times)] = values
        offset += len(times)
    order = np.argsort(out["time"], kind="stable")
    return pd.DataFrame({name: values[order] for name, values in out.items()})


class E4CsvParams(BaseModel):
    signals: list[str] | str | None = Field(
        E4_DEFAULT,
        description="Señales E4 (lista o 'A,B'): ACC, BVP, EDA, TEMP, HR, IBI",
    )

    @model_validator(mode="after")
    def _known(self) -> E4CsvParams:
        _e4_signals(self.signals, E4_CHANNELS)
        return self


@readers.register(
    "empatica_e4_csv",
    title="Empatica E4 (CSV)",
    description="Carpeta de una sesión E4: un CSV por señal, cada una a su tasa nativa.",
    params=E4CsvParams,
    explain=(
        "Cada CSV trae la marca de tiempo de inicio en la fila 1 (Unix en segundos, o "
        "una fecha UTC como en PhysioNet) y la frecuencia en la fila 2; IBI.csv trae "
        "el inicio y luego pares (desfase, IBI). La ruta es la carpeta o un fichero "
        "suyo. La columna time es la marca de cada muestra en segundos Unix: es un "
        "metadato, así que el descriptor necesita un paso time."
    ),
)
class EmpaticaE4CsvReader:
    """The CSV export of an Empatica E4 session (E4 connect, one file per signal)."""

    def __init__(self, signals: list[str] | str | None = E4_DEFAULT) -> None:
        self.signals = _e4_signals(signals, E4_CHANNELS)

    def read(self, path: Path) -> pd.DataFrame:
        folder = path if path.is_dir() else path.parent
        blocks = []
        for signal in self.signals:
            file = folder / f"{signal}.csv"
            if not file.exists():
                raise DataError(f"{folder} has no {signal}.csv")
            reader = self._ibi if signal == "IBI" else self._regular
            blocks.append(reader(file, E4_CHANNELS[signal]))
        return _e4_table(blocks)

    @staticmethod
    def _regular(file: Path, names: list[str]):
        head = _e4_rows(file, 2)
        if head.shape != (2, len(names)):
            raise DataError(
                f"{file}: expected {len(names)} columns and the start and rate rows"
            )
        starts = {e4_seconds(v, file) for v in head[0]}
        rates = {e4_seconds(v, file) for v in head[1]}
        if len(starts) > 1 or len(rates) > 1 or min(rates) <= 0:
            raise DataError(f"{file}: columns with different starts or rates {head}")
        values = _e4_values(file, 2, len(names))
        times = starts.pop() + np.arange(len(values)) / rates.pop()
        return times, {name: values[:, i] for i, name in enumerate(names)}

    @staticmethod
    def _ibi(file: Path, names: list[str]):
        start = e4_seconds(_e4_rows(file, 1)[0, 0], file)
        values = _e4_values(file, 1, 2)  # E4 connect may export an empty IBI
        return start + values[:, 0], {names[0]: values[:, 1]}


def e4_seconds(value: Any, file: Path | str = "") -> float:
    """An E4 time as Unix seconds: a number, or a UTC date as PhysioNet writes it."""
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        stamp = pd.Timestamp(text)
    except ValueError:
        raise DataError(
            f"{file}: {text!r} is neither Unix seconds nor a date"
        ) from None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return (stamp - pd.Timestamp(0)).total_seconds()


def _e4_rows(file: Path, n: int) -> np.ndarray:
    """The first ``n`` rows as text: the start time (and the rate)."""
    try:
        rows = pd.read_csv(file, header=None, nrows=n, dtype=str)
    except pd.errors.EmptyDataError:
        raise DataError(f"{file} is empty") from None
    if len(rows) < n:
        raise DataError(f"{file}: expected {n} header rows, got {len(rows)}")
    return rows.to_numpy()


def _e4_values(file: Path, skip: int, width: int) -> np.ndarray:
    try:
        values = pd.read_csv(
            file, header=None, skiprows=skip, skipinitialspace=True
        ).to_numpy(np.float64)
    except pd.errors.EmptyDataError:
        return np.empty((0, width))
    if values.shape[1] != width:
        raise DataError(f"{file}: expected {width} columns, got {values.shape[1]}")
    return values


# Sessions of the first stage are S<nn> (protocol V1), of the second f<nn> (V2).
PHYSIONET_STAGES = {"S": "V1", "f": "V2"}


@readers.register(
    "physionet_wearable",
    title="PhysioNet Wearable Device (E4)",
    description="Sesión E4 de PhysioNet (estrés inducido y ejercicio) con su tramo de protocolo.",
    params=E4CsvParams,
    explain=(
        "Lee Wearable_Dataset/<ACTIVIDAD>/<sesión>/ como empatica_e4_csv y añade "
        "session (ACTIVIDAD/sesión) y segment (ACTIVIDAD_V<versión>_<k>), donde k es "
        "el número de tags pulsados hasta esa muestra: 0 antes del primero y k entre "
        "el tag k y el k+1. La versión sale del prefijo de la sesión (S = V1, f = V2). "
        "Ambas columnas son metadatos: la etiqueta se declara mapeando segment."
    ),
)
class PhysioNetWearableReader:
    """PhysioNet's Wearable Device Dataset: E4 sessions with protocol tags."""

    def __init__(self, signals: list[str] | str | None = E4_DEFAULT) -> None:
        self.e4 = EmpaticaE4CsvReader(signals)

    def read(self, path: Path) -> pd.DataFrame:
        folder = path if path.is_dir() else path.parent
        activity, session = folder.parent.name, folder.name
        version = PHYSIONET_STAGES.get(session[:1])
        if version is None:
            raise DataError(f"{folder}: a session starts with S or f, got {session!r}")
        df = self.e4.read(folder)
        tags = np.sort(_e4_tags(folder / "tags.csv"))
        k = np.searchsorted(tags, df["time"].to_numpy(np.float64), side="right")
        return df.assign(
            session=f"{activity}/{session}",
            segment=[f"{activity}_{version}_{i}" for i in k],
        )


def _e4_tags(file: Path) -> np.ndarray:
    """The times of the button presses (one per line); a session may have none."""
    if not file.exists():
        raise DataError(f"{file.parent} has no tags.csv")
    lines = [line.strip() for line in file.read_text(encoding="utf-8").splitlines()]
    return np.array([e4_seconds(line, file) for line in lines if line], np.float64)


class WesadE4Params(BaseModel):
    signals: list[str] | str | None = Field(
        E4_DEFAULT, description="Señales de muñeca (lista o 'A,B'): ACC, BVP, EDA, TEMP"
    )

    @model_validator(mode="after")
    def _known(self) -> WesadE4Params:
        _e4_signals(self.signals, WESAD_RATES["wrist"])
        return self


@readers.register(
    "wesad_e4",
    title="WESAD muñeca (E4)",
    description="Muñeca de un S<n>.pkl de WESAD a tasas nativas, como el lector E4.",
    params=WesadE4Params,
    explain=(
        "Da las mismas columnas que empatica_e4_csv (time en segundos desde el "
        "inicio de la grabación) más condition, la etiqueta original (0-7) a 700 Hz "
        "en filas propias."
    ),
)
class WesadE4Reader:
    """WESAD's wrist device is an Empatica E4: the same channels as its CSV export."""

    def __init__(self, signals: list[str] | str | None = E4_DEFAULT) -> None:
        self.signals = _e4_signals(signals, WESAD_RATES["wrist"])

    def read(self, path: Path) -> pd.DataFrame:
        with open(path, "rb") as handle:  # the dataset's own format; trusted files only
            data = pickle.load(handle, encoding="latin1")
        wrist = {k.upper(): v for k, v in data["signal"]["wrist"].items()}
        blocks = []
        for signal in self.signals:
            if signal not in wrist:
                raise DataError(f"{path} has no wrist signal {signal}")
            values = np.asarray(wrist[signal], dtype=np.float64)
            values = values.reshape(len(values), -1)
            times = np.arange(len(values)) / WESAD_RATES["wrist"][signal]
            names = E4_CHANNELS[signal]
            blocks.append((times, {n: values[:, i] for i, n in enumerate(names)}))
        labels = np.asarray(data["label"], dtype=np.float64)
        blocks.append(
            (np.arange(len(labels)) / WESAD_LABEL_RATE, {"condition": labels})
        )
        return _e4_table(blocks)


class LeafParams(BaseModel):
    writers: list[str] | None = Field(
        None,
        description="Usuarios que se leen (submuestra fija); por defecto todos",
    )

    @model_validator(mode="after")
    def _distinct(self) -> LeafParams:
        if self.writers is not None and len(set(self.writers)) != len(self.writers):
            raise ValueError("writers are repeated")
        return self


@readers.register(
    "leaf_json",
    title="LEAF JSON",
    description="Fichero de LEAF (users, user_data con x e y): una fila por muestra.",
    params=LeafParams,
    explain=(
        "Cada usuario es un sujeto (columna user); x se expande en x_0, x_1… y la "
        "etiqueta queda en la columna y. Con writers solo se leen esos usuarios."
    ),
)
class LeafJsonReader:
    """LEAF's federated format (Caldas et al., 2018), as FEMNIST's ``all_data``."""

    def __init__(self, writers: list[str] | None = None) -> None:
        self.writers = None if writers is None else set(writers)

    def read(self, path: Path) -> pd.DataFrame:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not data["users"]:
            raise DataError(f"{path} has no users")
        # The width of any user, so a file without the chosen writers still
        # gives the same columns as the others.
        first = data["user_data"][data["users"][0]]["x"]
        width = len(first[0]) if first else 0
        blocks = [np.empty((0, width), dtype=np.float32)]
        users: list[str] = []
        labels = [np.empty(0, dtype=np.int64)]
        for user in data["users"]:
            if self.writers is not None and user not in self.writers:
                continue
            samples = data["user_data"][user]
            if not samples["y"]:
                continue
            try:
                x = np.asarray(samples["x"], dtype=np.float32)
            except ValueError as error:  # rows of different widths
                raise DataError(f"{path}: user {user!r}: {error}") from None
            y = np.asarray(samples["y"], dtype=np.int64)
            if x.shape != (len(y), width):
                raise DataError(
                    f"{path}: user {user!r} has x of shape {x.shape} and "
                    f"{len(y)} labels; expected width {width}"
                )
            blocks.append(x)
            users += [user] * len(y)
            labels.append(y)
        frame = pd.DataFrame(
            np.concatenate(blocks), columns=[f"x_{i}" for i in range(width)]
        )
        return frame.assign(
            user=pd.Series(users, dtype=object), y=np.concatenate(labels)
        )


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
        seconds = _seconds(df[self.column], self.format, self.pad_to, "time")
        if ctx is not None:
            ctx.meta_columns.add(self.column)
        return df.assign(**{TIME: seconds})


def _seconds(
    raw: pd.Series,
    format: str | None,
    pad_to: int | None,
    step: str,
    timezone: str | None = None,
) -> np.ndarray:
    """Seconds since the epoch: numbers as they are, text parsed as dates.

    With ``timezone``, dates without a zone are local times there and the result
    is in Unix UTC seconds; without it, they are taken as they read.
    """
    if format is None and pd.api.types.is_numeric_dtype(raw):
        seconds = raw.astype(float)
    else:
        text = raw.astype(str)
        if pad_to is not None:
            text = text.str.ljust(pad_to, "0")
        stamps = pd.to_datetime(text, format=format, errors="coerce")
        epoch = pd.Timestamp(0)
        if timezone is not None:
            stamps = stamps.dt.tz_localize(
                timezone, ambiguous="raise", nonexistent="raise"
            ).dt.tz_convert("UTC")
            epoch = pd.Timestamp(0, tz="UTC")
        seconds = (stamps - epoch).dt.total_seconds()
    bad = raw[seconds.isna()].unique()[:3].tolist()
    if bad:
        raise DataError(f"{step}: rows of {raw.name!r} without a time: {bad}")
    return seconds.to_numpy(np.float64)


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
    # one class throughout, or no label: the window is dropped
    "unanimous": lambda a: float(a[0]) if (a == a[0]).all() else float("nan"),
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


EPS = 1e-6  # seconds: below any sample interval, above float error on Unix times


class MultirateWindowParams(BaseModel):
    seconds: PositiveFloat = Field(description="Duración de cada ventana")
    overlap: float = Field(0.0, ge=0, lt=1)
    stats: list[Literal["mean", "std", "min", "max", "median"]] = Field(
        default_factory=lambda: list(STATS)
    )
    columns: list[str] | None = Field(
        None, description="Canales; por defecto los numéricos"
    )
    label: Literal["mean_round", "mode", "first", "last", "unanimous"] = Field(
        "mean_round",
        description="Cómo resume la ventana sus muestras de etiqueta; unanimous "
        "descarta las ventanas que mezclan clases",
    )
    label_rows: str | None = Field(
        None,
        description="Columna cuyas filas con valor son las muestras de la etiqueta; "
        "por defecto, todas las filas",
    )


@steps.register(
    "multirate_window",
    title="Ventanas multi-tasa",
    description="Ventanas de N segundos por sujeto; cada canal, con sus muestras a su tasa.",
    params=MultirateWindowParams,
    explain=(
        "Cada canal usa solo sus filas con valor, así que señales a tasas distintas "
        "conviven en una tabla. Las ventanas van del último inicio al primer final de "
        "los canales; se descartan las que dejan un canal sin muestras o tienen una "
        "muestra de etiqueta sin clase. Cada ventana toma el tiempo de su final."
    ),
)
class MultirateWindowStep:
    def __init__(self, seconds, overlap, stats, columns, label, label_rows) -> None:
        self.size, self.stats, self.columns = float(seconds), stats, columns
        self.label, self.label_rows = label, label_rows
        self.step = self.size * (1 - overlap)

    def apply(self, df: pd.DataFrame, ctx: IngestContext | None) -> pd.DataFrame:
        if "subject" not in df.columns:
            raise DataError("window runs per subject: put a subject step before it")
        if TIME not in df.columns:
            raise DataError("multirate_window needs each row's time: add a time step")
        reserved = {"subject", "label", TIME, self.label_rows}
        reserved |= ctx.meta_columns if ctx else set()
        columns = self.columns or [
            c
            for c in df.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
        ]
        _require(df, columns + [c for c in [self.label_rows] if c], "multirate_window")
        names = [f"{c}_{s}" for c in columns for s in self.stats]
        has_label = "label" in df.columns
        rows = []
        for subject, group in df.groupby("subject", sort=False):
            times = group[TIME].to_numpy(dtype=np.float64)
            tracks = [_track(times, group[c].to_numpy(np.float64)) for c in columns]
            labels = None
            if has_label:
                held = (
                    group[self.label_rows].notna().to_numpy()
                    if self.label_rows
                    else np.ones(len(group), dtype=bool)
                )
                labels = _track(
                    times, group["label"].to_numpy(np.float64), present=held
                )
            span = tracks + ([labels] if labels is not None else [])
            if any(len(t) == 0 for t, _ in span):
                continue
            start = max(t[0] for t, _ in span)
            end = min(t[-1] + _interval(t) for t, _ in span)
            count = int(np.floor((end - start - self.size) / self.step + EPS)) + 1
            for k in range(max(count, 0)):
                lo = start + k * self.step
                hi = lo + self.size
                row: list[Any] = [subject]
                if labels is not None:
                    window = _within(labels, lo, hi)
                    if len(window) == 0 or np.isnan(window).any():
                        continue
                    row.append(REDUCE[self.label](window))
                values = [_within(track, lo, hi) for track in tracks]
                if any(len(v) == 0 for v in values):
                    continue
                row.append(hi)  # observable once the window is complete
                rows.append(row + [STATS[s](v) for v in values for s in self.stats])
        head = ["subject", "label"] if has_label else ["subject"]
        return pd.DataFrame(rows, columns=head + [TIME] + names)


def _track(times: np.ndarray, values: np.ndarray, present: np.ndarray | None = None):
    """One channel's samples (rows with a value, or ``present``), sorted by time."""
    keep = ~np.isnan(values) if present is None else present
    t, v = times[keep], values[keep]
    order = np.argsort(t, kind="stable")
    return t[order], v[order]


def _interval(times: np.ndarray) -> float:
    """The usual time between samples; signals sampled at one instant count once."""
    distinct = np.unique(times)
    return float(np.median(np.diff(distinct))) if len(distinct) > 1 else 0.0


def _within(track: tuple[np.ndarray, np.ndarray], lo: float, hi: float) -> np.ndarray:
    """Values sampled in ``[lo, hi)``."""
    times, values = track
    a, b = np.searchsorted(times, [lo - EPS, hi - EPS])
    return values[a:b]


class IntervalsParams(BaseModel):
    source: SourceSpec = Field(description="Tabla con un intervalo por fila")
    start: str = Field("start", description="Columna del inicio")
    end: str = Field("end", description="Columna del final (no incluido)")
    value: str = Field("class", description="Columna con la clase del intervalo")
    column: str = Field("interval", description="Columna nueva con la clase por fila")
    by: list[str] | dict[str, str] = Field(
        default_factory=list,
        description="Claves comunes, como subject, o {columna de la fila: columna "
        "de la tabla}",
    )
    format: str | None = Field(
        None,
        description="Formato strptime de inicio y final; sin él, segundos o una "
        "fecha que pandas entienda",
    )
    date: str | None = Field(
        None,
        description="Columna con el día, si inicio y final son horas del día; un "
        "final que no pasa del inicio es del día siguiente",
    )
    timezone: str | None = Field(
        None,
        description="Zona IANA de las fechas sin zona (p. ej. Etc/GMT+6); se pasan "
        "a segundos Unix UTC",
    )
    overlap: Literal["error", "max"] = Field(
        "error",
        description="Intervalos solapados: error, o la clase mayor (numérica) en "
        "las filas que caen en varios",
    )
    outside: float | str | None = Field(
        None,
        description="Clase de las filas fuera de todo intervalo, solo en los "
        "ficheros con algún intervalo de su clave",
    )

    @model_validator(mode="after")
    def _known_timezone(self) -> IntervalsParams:
        if self.timezone is not None:
            try:
                pd.Timestamp(0).tz_localize(self.timezone)
            except Exception as exc:
                raise ValueError(f"unknown timezone {self.timezone!r}") from exc
        return self


@steps.register(
    "intervals",
    title="Etiquetar por intervalos",
    description="Da a cada fila la clase del intervalo (inicio, fin) de otra tabla que la contiene.",
    params=IntervalsParams,
    explain=(
        "Para tags de protocolo y eventos de encuesta. Compara con el tiempo de cada "
        "fila, así que va después de un paso time y en sus unidades. Las filas fuera "
        "de todo intervalo quedan sin clase, o con la clase outside si su fichero "
        "tiene algún intervalo de su clave. Los intervalos de una clave no pueden "
        "solaparse salvo con overlap: max. Con timezone, las fechas locales pasan a "
        "Unix UTC. La columna nueva es meta: nunca es una feature."
    ),
)
class IntervalsStep:
    def __init__(
        self,
        source,
        start,
        end,
        value,
        column,
        by,
        format,
        date,
        timezone,
        overlap,
        outside,
    ) -> None:
        self.source = SourceSpec.model_validate(source)
        self.start, self.end, self.value, self.column = start, end, value, column
        self.by = dict(by) if isinstance(by, Mapping) else {k: k for k in by}
        self.format, self.date, self.timezone = format, date, timezone
        self.overlap, self.outside = overlap, outside
        self._table: pd.DataFrame | None = None

    def _read(self, root: Path) -> pd.DataFrame:
        table = read_source(self.source, root)
        dates = [self.date] if self.date else []
        needed = [self.start, self.end, self.value, *dates, *self.by.values()]
        _require(table, needed, "intervals")
        start, end = table[self.start], table[self.end]
        if self.date:
            day = table[self.date]
            if pd.api.types.is_datetime64_any_dtype(day):
                day = day.dt.strftime("%Y-%m-%d")
            day = day.astype(str) + " "
            start = (day + start.astype(str)).rename(self.start)
            end = (day + end.astype(str)).rename(self.end)
        begins = _seconds(start, self.format, None, "intervals", self.timezone)
        ends = _seconds(end, self.format, None, "intervals", self.timezone)
        if self.date:  # a time of day that does not pass the start is the next day
            ends = np.where(ends <= begins, ends + 86400.0, ends)
        if self.overlap == "max":
            numbers = pd.to_numeric(table[self.value], errors="coerce")
            if numbers.isna().ne(table[self.value].isna()).any():
                raise DataError(
                    f"intervals: overlap max needs numeric classes in {self.value!r}"
                )
        return table.assign(_start=begins, _end=ends).reset_index(drop=True)

    def apply(self, df: pd.DataFrame, ctx: IngestContext) -> pd.DataFrame:
        if TIME not in df.columns:
            raise DataError("intervals compares with each row's time: add a time step")
        if self._table is None:  # read once, then used with every source file
            self._table = self._read(ctx.root)
        table = self._table
        _require(df, list(self.by), "intervals")
        times = df[TIME].to_numpy(np.float64)
        hit = np.full(len(df), -1)
        left = _keys(df, list(self.by)) if self.by else np.zeros(len(df), str)
        if self.by:
            right = _keys(table, list(self.by.values()))
            groups = [(left == key, right == key) for key in np.unique(right)]
        else:
            groups = [(np.ones(len(df), bool), np.ones(len(table), bool))]
        for rows, intervals in groups:
            part = table[intervals].sort_values("_start", kind="stable")
            where = np.flatnonzero(rows)
            if part.empty or len(where) == 0:
                continue
            starts, ends = part["_start"].to_numpy(), part["_end"].to_numpy()
            if (ends <= starts).any():
                raise DataError(f"intervals: empty intervals in {self.source.path}")
            if self.overlap == "error" and (starts[1:] < ends[:-1]).any():
                raise DataError(
                    f"intervals: overlapping intervals in {self.source.path}"
                )
            # Later writes win: in increasing class order (missing classes first),
            # a row in several intervals keeps the largest class.
            order = np.arange(len(part))
            if self.overlap == "max":
                numbers = pd.to_numeric(part[self.value]).fillna(-np.inf)
                order = np.argsort(numbers.to_numpy(), kind="stable")
            t = times[where]
            for i in order:
                inside = (t >= starts[i]) & (t < ends[i])
                hit[where[inside]] = part.index[i]
        classes = table[self.value].reindex(hit).set_axis(df.index)
        if self.outside is not None:
            # Only in a file with an interval of its key: elsewhere, nothing says
            # whether the rows are outside every event or simply not surveyed.
            held = np.unique(left[hit >= 0])
            lone = (hit < 0) & np.isin(left, held)
            classes = classes.astype(object).where(~lone, self.outside)
            classes = classes.infer_objects()
        ctx.meta_columns.add(self.column)
        return df.assign(**{self.column: classes})


def _keys(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """One text key per row from ``columns``, so int and text ids compare equal."""
    keys = df[columns[0]].astype(str)
    for column in columns[1:]:  # by column, not by row: sessions have millions
        keys = keys + "\x1f" + df[column].astype(str)
    return keys.to_numpy()


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
            codes, uniques = pd.factorize(values)  # each distinct value once
            found = [self.map.get(v, text.get(str(v).strip())) for v in uniques]
            exact = pd.to_numeric(pd.Series(found, dtype=object), errors="coerce")
            exact = np.append(exact.to_numpy(np.float64), np.nan)  # code -1: missing
            return pd.Series(exact[codes], index=values.index)
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
        head = df[[c for c in ("subject", "label", TIME) if c in df.columns]]
        # One concat, not a column at a time: wide tables (FEMNIST's 784
        # pixels) would fragment the frame.
        values = pd.DataFrame(
            {column: _numeric(df[column], column) for column in columns},
            index=df.index,
        )
        return pd.concat([head, values], axis=1)


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
    # Files that end without rows (a session a select step excluded, an E4
    # session without labelled windows) are left out of the concatenation,
    # which would otherwise warn about them.
    df = pd.concat([f for f in frames if len(f)] or frames[:1], ignore_index=True)
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
