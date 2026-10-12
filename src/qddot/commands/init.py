"""Initialize a runnable qddot project."""

from __future__ import annotations

from pathlib import Path

import typer

from ._out import console, exit_error


def register(app: typer.Typer) -> None:
    @app.command()
    def init(
        directory: Path = typer.Argument(Path(".")),  # noqa: B008
        robot: str = typer.Option(..., "--robot"),
        task: str | None = typer.Option(None, "--task"),
        assets: Path | None = typer.Option(None, "--assets"),  # noqa: B008
    ) -> None:
        """Scaffold a runnable project for a robot model."""
        try:
            from qddot.scaffold import scaffold_project

            task_path = scaffold_project(directory, robot, task, assets)
        except Exception as exc:  # noqa: BLE001 - CLI errors should be concise.
            exit_error(exc, False)
        console.print(f"Created {task_path.parent}")
        console.print(f"Next: qddot spec {task_path}")
        console.print(f"Next: qddot run {task_path} --controller {task_path.parent / 'controller.py'}")
