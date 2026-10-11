"""Recent episode listing command."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from ._out import console, exit_error


def register(app: typer.Typer) -> None:
    @app.command()
    def runs(last: int = typer.Option(10, "--last", min=1), root: Path = typer.Option(Path(".qddot"), "--root")) -> None:  # noqa: B008
        """List recent episodes by verdict file modification time."""
        try:
            paths = list((root / "runs").glob("*/verdict.json"))
            paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            if not paths:
                console.print("no runs found")
                return
            table = Table(box=None, show_header=True, header_style="bold")
            for heading in ("episode", "result", "robustness", "task"):
                table.add_column(heading)
            for path in paths[:last]:
                verdict = json.loads(path.read_text(encoding="utf-8"))
                table.add_row(path.parent.name, "PASS" if verdict.get("passed") else "FAIL", str(verdict.get("robustness", "n/a")), str(verdict.get("task", "")))
            console.print(table)
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, False)
