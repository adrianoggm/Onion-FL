from __future__ import annotations

"""Analysis API shared by the report, notebooks and the front (spec §10.6).

::

    runs = load_runs("runs/", experiment="mix_ab")
    runs.metrics(level="global", name="accuracy")            # one row per event
    runs.compare(level="fog", metric="accuracy", by=["topology_id", "dataset"])

``compare`` first combines the nodes of the level inside each run (mean, min,
max and spread), then the runs of each group over ``over`` (the seed): the
mean with a 95 % t interval, per round.
"""

import html
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from onion_fl.observability.events import read_events

BASE = ("t_virtual", "round", "level", "node", "role", "kind", "name", "value")
RUN_COLUMNS = ("run_id", "topology_id", "config_id", "seed", "scenario", "experiment")


@dataclass(frozen=True)
class RunRecord:
    path: Path
    meta: dict[str, Any]

    @property
    def experiment(self) -> str | None:
        config = self.meta.get("config") or {}
        return config.get("experiment", config.get("name"))


class Runs:
    def __init__(self, records: Sequence[RunRecord]) -> None:
        self.records = list(records)
        self._events: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self.records)

    def events(self) -> pd.DataFrame:
        """Every event of every run, tags as columns, with the run's identifiers."""
        if self._events is None:
            rows = []
            for record in self.records:
                ids = {k: record.meta.get(k) for k in RUN_COLUMNS[:-1]}
                ids["experiment"] = record.experiment
                for event in read_events(record.path / "events.jsonl"):
                    tags = {
                        k: v
                        for k, v in (event.get("tags") or {}).items()
                        if k not in BASE
                    }
                    rows.append({**ids, **tags, **{k: event.get(k) for k in BASE}})
            self._events = pd.DataFrame(rows)
        return self._events

    def metrics(
        self, level: str | None = None, name: str | None = None, **tags: Any
    ) -> pd.DataFrame:
        """Metric and diagnostic events; ``name`` is the full name or ``eval.``, ``round.`` or ``diagnostic.`` plus it."""
        df = self.events()
        if df.empty:
            return df
        keep = df["kind"].isin(["metric", "diagnostic"])
        if level is not None:
            keep &= df["level"] == level
        if name is not None:
            keep &= df["name"].isin(
                [name, *(f"{p}.{name}" for p in ("eval", "round", "diagnostic"))]
            )
        for key, value in tags.items():
            keep &= (df[key] == value) if key in df.columns else False
        return df[keep].reset_index(drop=True)

    def compare(
        self,
        level: str,
        metric: str,
        by: Sequence[str] = ("topology_id",),
        over: str = "seed",
        **tags: Any,
    ) -> pd.DataFrame:
        """Mean ± 95 % CI over ``over`` per group and round, plus the spread across nodes."""
        df = self.metrics(level=level, name=metric, **tags)
        columns = [
            *by,
            "round",
            "n",
            "mean",
            "ci_low",
            "ci_high",
            "node_min",
            "node_max",
            "node_spread",
        ]
        if df.empty:
            return pd.DataFrame(columns=columns)
        by = list(by)
        for key in by:
            if key not in df.columns:
                df[key] = None
        per_run = (
            df.groupby([*by, "run_id", over, "round"], dropna=False)["value"]
            .agg(
                node_mean="mean",
                node_min="min",
                node_max="max",
                node_spread=lambda v: float(np.std(v)),
            )
            .reset_index()
        )
        rows = []
        for key, group in per_run.groupby([*by, "round"], dropna=False):
            values = group["node_mean"].to_numpy(float)
            n, mean = len(values), float(values.mean())
            if n > 1:
                half = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
                low, high = mean - half, mean + half
            else:
                low = high = float("nan")
            rows.append(
                dict(zip([*by, "round"], key, strict=True))
                | {
                    "n": n,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "node_min": float(group["node_min"].mean()),
                    "node_max": float(group["node_max"].mean()),
                    "node_spread": float(group["node_spread"].mean()),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def report(
        self,
        path: str | Path,
        metrics: Sequence[str] = ("accuracy", "loss"),
        by: Sequence[str] = ("topology_id",),
    ) -> Path:
        """Static HTML comparing the groups at every level, built on ``compare``."""
        return write_report(self, Path(path), metrics, by)


def load_runs(path: str | Path, experiment: str | None = None) -> Runs:
    records = []
    for run_json in sorted(Path(path).glob("*/run.json")):
        record = RunRecord(
            run_json.parent, json.loads(run_json.read_text(encoding="utf-8"))
        )
        if experiment is None or record.experiment == experiment:
            records.append(record)
    return Runs(records)


# --- HTML report ------------------------------------------------------------------------------

COLOURS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


def _label(row: pd.Series, by: Sequence[str]) -> str:
    return " · ".join(f"{k}={row[k]}" for k in by)


def _chart(table: pd.DataFrame, by: Sequence[str]) -> str:
    """Mean per round of each group as SVG polylines; no JavaScript, no dependencies."""
    width, height, pad = 520, 220, 30
    rounds, values = table["round"].astype(float), table["mean"].astype(float)
    x0, x1 = rounds.min(), max(rounds.max(), rounds.min() + 1)
    y0, y1 = values.min(), max(values.max(), values.min() + 1e-9)

    def point(r: float, v: float) -> str:
        x = pad + (r - x0) / (x1 - x0) * (width - 2 * pad)
        y = height - pad - (v - y0) / (y1 - y0) * (height - 2 * pad)
        return f"{x:.1f},{y:.1f}"

    lines = []
    for i, (_, group) in enumerate(table.groupby(list(by), dropna=False)):
        group = group.sort_values("round")
        points = " ".join(
            point(r, v) for r, v in zip(group["round"], group["mean"], strict=True)
        )
        colour = COLOURS[i % len(COLOURS)]
        label = html.escape(_label(group.iloc[0], by))
        lines.append(
            f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{points}">'
            f"<title>{label}</title></polyline>"
        )
    axis = (
        f'<text x="{pad}" y="{height - 8}" font-size="11">round {x0:g}</text>'
        f'<text x="{width - pad - 50}" y="{height - 8}" font-size="11">round {x1:g}</text>'
        f'<text x="2" y="{pad}" font-size="11">{y1:.3g}</text>'
        f'<text x="2" y="{height - pad}" font-size="11">{y0:.3g}</text>'
    )
    return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">{"".join(lines)}{axis}</svg>'


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return "–" if np.isnan(value) else f"{value:.3f}"
    return html.escape(str(value))


def write_report(
    runs: Runs, path: Path, metrics: Sequence[str], by: Sequence[str]
) -> Path:
    events = runs.events()
    levels = [] if events.empty else sorted(events["level"].dropna().unique())
    sections = []
    for level in levels:
        for metric in metrics:
            table = runs.compare(level=level, metric=metric, by=by)
            if table.empty:
                continue
            last = table.sort_values("round").groupby(list(by), dropna=False).tail(1)
            head = "".join(f"<th>{html.escape(c)}</th>" for c in last.columns)
            body = "".join(
                "<tr>" + "".join(f"<td>{_cell(v)}</td>" for v in row) + "</tr>"
                for row in last.itertuples(index=False)
            )
            sections.append(
                f"<section><h2>{html.escape(level)} · {html.escape(metric)}</h2>"
                f"{_chart(table, by)}<h3>Last round</h3>"
                f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>"
            )
    run_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{_cell(r.meta.get(k))}</td>"
            for k in ("run_id", "topology_id", "config_id", "seed", "run_hash")
        )
        + "</tr>"
        for r in runs
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Onion-FL report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111; }}
table {{ border-collapse: collapse; margin: .5rem 0 1.5rem; font-size: .9rem; }}
th, td {{ border: 1px solid #ddd; padding: .25rem .5rem; text-align: left; }}
section {{ margin-bottom: 2rem; }}
</style></head><body>
<h1>Onion-FL report</h1>
<p>Mean over seeds with a 95 % t interval; node columns combine the nodes of each level inside a run.</p>
{"".join(sections) or "<p>No metrics to compare.</p>"}
<h2>Runs</h2>
<table><thead><tr><th>run_id</th><th>topology_id</th><th>config_id</th><th>seed</th><th>run_hash</th></tr></thead>
<tbody>{run_rows}</tbody></table>
</body></html>
"""
    path.write_text(page, encoding="utf-8")
    return path
