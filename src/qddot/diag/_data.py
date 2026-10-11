"""Small readers shared by diagnosis commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.is_file():
        return [], []
    connection = duckdb.connect(database=":memory:")
    try:
        cursor = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
        names = [item[0] for item in cursor.description or []]
        rows = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
        return names, rows
    finally:
        connection.close()


def parse_stl_signals(stl: str) -> list[str]:
    """Return signal tokens from STL while excluding operators and function names."""
    import re

    reserved = {"always", "eventually", "until", "not", "and", "or", "implies", "abs"}
    result: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", stl):
        if token.lower() not in reserved and token not in result:
            result.append(token)
    return result


def finite_values(rows: list[dict[str, Any]], signal: str) -> np.ndarray:
    return np.asarray([row[signal] for row in rows if signal in row], dtype=np.float64)
