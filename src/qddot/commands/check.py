"""Static model check command."""

from __future__ import annotations

from pathlib import Path

import typer

from ._out import console, emit_json, exit_error


def register(app: typer.Typer) -> None:
    @app.command()
    def check(model: Path = typer.Argument(...), json_output: bool = typer.Option(False, "--json")) -> None:  # noqa: B008
        """Run static checks on a MuJoCo model."""
        try:
            from qddot.check.static import check_model

            results = check_model(model)
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, json_output)
        if json_output:
            emit_json(results)
        elif not results:
            console.print("PASS no issues found")
        else:
            for issue in results:
                console.print(f"{issue['severity'].upper():7} {issue['code']}: {issue['message']} [{issue['element']}]")
        if any(issue.get("severity") == "error" for issue in results):
            raise typer.Exit(1)
