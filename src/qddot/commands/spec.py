"""Task specification listing command."""

from __future__ import annotations

from pathlib import Path

import typer

from ._out import console, exit_error


def register(app: typer.Typer) -> None:
    @app.command()
    def spec(task: Path = typer.Argument(...)) -> None:  # noqa: B008
        """List a task's specs with English explanations."""
        try:
            from qddot.spec import to_english
            from qddot.task import load_task

            loaded = load_task(task)
            for item in loaded.specs:
                console.print(f"{item['name']}: {item['stl']}")
                console.print(f"  {to_english(item['stl'])}")
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, False)
