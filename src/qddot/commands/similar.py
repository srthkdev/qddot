"""Retrieve related failure cards for a recorded episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from qddot.memory.store import database_path, index_runs, search

from ._out import console, emit_json, exit_error


def _lines(result: dict[str, Any]) -> list[str]:
    lines = [f"similar failures for {result['episode']}"]
    if result.get("note"):
        lines.append(str(result["note"]))
    if not result["results"]:
        lines.append("no related failure cards found")
    for item in result["results"]:
        causes = ", ".join(item["causes"]) or "no likely cause recorded"
        lines.append(f"{item['score']:.3f} {item['task']} {item['episode']}")
        lines.append(f"  causes: {causes}")
        if item.get("fix_diff"):
            lines.extend(f"  {line}" for line in item["fix_diff"].splitlines())
    return lines[:30]


def register(app: typer.Typer) -> None:
    @app.command()
    def similar(
        episode: Path = typer.Argument(...),  # noqa: B008
        k: int = typer.Option(3, "-k", min=1, max=20),
        global_store: bool = typer.Option(False, "--global"),
        root: Path = typer.Option(Path(".qddot"), "--root"),  # noqa: B008
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Find prior failures with a similar task or likely cause."""
        db_path = database_path(root, global_store)
        try:
            index_runs(root / "runs", db_path)
            result = search(episode, db_path, k)
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, json_output)
        if json_output:
            emit_json(result)
        else:
            for line in _lines(result):
                console.print(line)
