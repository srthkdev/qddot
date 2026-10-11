"""Compare two recorded episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._data import read_json, read_table


def diff(a: str | Path, b: str | Path, tolerance: float = 1e-6) -> dict[str, Any]:
    """Compare signals, events, verdict robustness, and run metadata."""
    left, right = Path(a), Path(b)
    _, left_rows = read_table(left / "signals.parquet")
    _, right_rows = read_table(right / "signals.parquet")
    left_names = set(left_rows[0]) if left_rows else set()
    right_names = set(right_rows[0]) if right_rows else set()
    signals = {}
    shared = sorted((left_names & right_names) - {"t"})
    times = np.asarray([float(row["t"]) for row in left_rows])
    for name in shared:
        count = min(len(left_rows), len(right_rows))
        x = np.asarray([left_rows[i][name] for i in range(count)], dtype=np.float64)
        y = np.asarray([right_rows[i][name] for i in range(count)], dtype=np.float64)
        indices = np.flatnonzero(np.abs(x - y) > tolerance)
        signals[name] = {"first_divergence_t": float(times[indices[0]]) if indices.size else None,
                         "max_abs_delta": float(np.max(np.abs(x - y))) if count else 0.0}
    _, left_events = read_table(left / "events.parquet")
    _, right_events = read_table(right / "events.parquet")
    identity = lambda event: (event.get("kind"), event.get("a"), event.get("b"))
    left_ids = {identity(event) for event in left_events}
    right_ids = {identity(event) for event in right_events}
    verdict_a = read_json(left / "verdict.json")
    verdict_b = read_json(right / "verdict.json")
    specs_a = {item["name"]: item for item in verdict_a.get("specs", [])}
    specs_b = {item["name"]: item for item in verdict_b.get("specs", [])}
    robustness = {name: {"a": specs_a.get(name, {}).get("robustness"),
                         "b": specs_b.get(name, {}).get("robustness"),
                         "delta": (float(specs_b[name]["robustness"]) - float(specs_a[name]["robustness"]))
                         if name in specs_a and name in specs_b and specs_a[name].get("robustness") is not None
                         and specs_b[name].get("robustness") is not None else None}
                  for name in specs_a.keys() | specs_b.keys()}
    meta_a, meta_b = read_json(left / "meta.json"), read_json(right / "meta.json")
    return {"episodes": [verdict_a.get("episode"), verdict_b.get("episode")],
            "signals": signals,
            "events_only_a": [event for event in left_events if identity(event) not in right_ids],
            "events_only_b": [event for event in right_events if identity(event) not in left_ids],
            "robustness": robustness,
            "metadata": {"controller_sha256": [meta_a.get("controller_sha256"), meta_b.get("controller_sha256")],
                         "perturb": [meta_a.get("perturb"), meta_b.get("perturb")],
                         "seed": [meta_a.get("seed"), meta_b.get("seed")]}}
