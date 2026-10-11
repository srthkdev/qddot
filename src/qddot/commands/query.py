"""DuckDB query command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import typer
from rich.table import Table

from ._out import console, emit_json, exit_error


def _available_columns(connection: Any) -> str:
    parts = []
    for view in ("signals", "events", "verdicts"):
        try:
            rows = connection.execute(f"DESCRIBE {view}").fetchall()
            parts.append(f"{view}({', '.join(row[0] for row in rows)})")
        except duckdb.Error:
            parts.append(f"{view}(unavailable)")
    return "; ".join(parts)


def register(app: typer.Typer) -> None:
    @app.command()
    def query(
        sql: str = typer.Argument(...),
        json_output: bool = typer.Option(False, "--json"),
        limit: int = typer.Option(50, "--limit", min=1),
        root: Path = typer.Option(Path(".qddot"), "--root"),  # noqa: B008
    ) -> None:
        """Query recorded signals, events, and verdicts with DuckDB SQL."""
        from qddot.store import connect

        connection = connect(root)
        try:
            cursor = connection.execute(sql)
            names = [column[0] for column in cursor.description or []]
            rows = cursor.fetchmany(limit + 1)
        except duckdb.Error as exc:
            exit_error(RuntimeError(f"DuckDB: {exc}\nAvailable views and columns: {_available_columns(connection)}"), json_output)
        truncated = len(rows) > limit
        rows = rows[:limit]
        if json_output:
            emit_json({"columns": names, "rows": [dict(zip(names, row, strict=False)) for row in rows], "truncated": truncated})
            return
        table = Table(box=None, show_header=True, header_style="bold")
        for name in names:
            table.add_column(name, overflow="fold")
        for row in rows:
            table.add_row(*("NULL" if value is None else str(value) for value in row))
        console.print(table)
        if truncated:
            console.print(f"showing first {limit} rows; results truncated (use --limit to change)")
