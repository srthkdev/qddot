"""DuckDB views over recorded qddot episodes."""

from __future__ import annotations

from pathlib import Path

import duckdb


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _view_for_parquet(connection: duckdb.DuckDBPyConnection, name: str, path: Path, columns: str) -> None:
    matches = list(path.parent.parent.glob(f"{path.parent.name}/{path.name}"))
    if not matches:
        connection.execute(f"CREATE VIEW {name} AS SELECT {columns} WHERE FALSE")
        return
    glob = _sql_path(path)
    connection.execute(
        f"CREATE VIEW {name} AS SELECT *, regexp_extract(filename, '/runs/([^/]+)/', 1) AS episode "
        f"FROM read_parquet('{glob}', filename = true, union_by_name = true)"
    )


def connect(root: str | Path = Path(".qddot")) -> duckdb.DuckDBPyConnection:
    """Return an in-memory connection with views over available episode files."""
    root = Path(root)
    runs = root / "runs"
    connection = duckdb.connect(database=":memory:")
    _view_for_parquet(connection, "signals", runs / "*" / "signals.parquet", "CAST(NULL AS DOUBLE) AS t, CAST(NULL AS VARCHAR) AS episode")
    _view_for_parquet(connection, "events", runs / "*" / "events.parquet", "CAST(NULL AS DOUBLE) AS t, CAST(NULL AS VARCHAR) AS kind, CAST(NULL AS VARCHAR) AS a, CAST(NULL AS VARCHAR) AS b, CAST(NULL AS DOUBLE) AS value, CAST(NULL AS VARCHAR) AS detail, CAST(NULL AS VARCHAR) AS episode")
    verdict_files = list(runs.glob("*/verdict.json"))
    if verdict_files:
        glob = _sql_path(runs / "*" / "verdict.json")
        connection.execute(
            "CREATE VIEW verdicts AS SELECT "
            "regexp_extract(filename, '/runs/([^/]+)/', 1) AS episode, "
            "* EXCLUDE (filename, specs), to_json(specs) AS specs "
            f"FROM read_json_auto('{glob}', filename = true, union_by_name = true)"
        )
    else:
        connection.execute(
            "CREATE VIEW verdicts AS SELECT CAST(NULL AS VARCHAR) AS episode, "
            "CAST(NULL AS VARCHAR) AS task, CAST(NULL AS VARCHAR) AS controller, "
            "CAST(NULL AS BIGINT) AS seed, CAST(NULL AS BOOLEAN) AS passed, "
            "CAST(NULL AS DOUBLE) AS robustness, CAST(NULL AS JSON) AS specs WHERE FALSE"
        )
    return connection
