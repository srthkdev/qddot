"""Bring recorded robot data into qddot as an episode."""

from __future__ import annotations

from pathlib import Path

import typer

from ._out import console, emit_json, exit_error


def register(app: typer.Typer) -> None:
    @app.command()
    def ingest(
        source: Path = typer.Argument(..., help="LeRobot dataset dir, .mcap, .csv or .parquet"),  # noqa: B008
        task: Path | None = typer.Option(None, "--task", help="Evaluate this task's specs on the data"),  # noqa: B008
        episode: int = typer.Option(0, "--episode", help="LeRobot episode index"),
        rename: list[str] = typer.Option([], "--rename", help="new=old signal rename (repeatable)"),  # noqa: B008
        name: str | None = typer.Option(None, "--name", help="Episode id to write"),
        root: Path = typer.Option(Path(".qddot"), "--root"),  # noqa: B008
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Turn a real-robot log into an episode: specs, why, query and diff then work on it."""
        try:
            from qddot.ingest import ingest as run_ingest

            pairs = dict(item.split("=", 1) for item in rename)
            verdict = run_ingest(source, task, episode, pairs, root, name)
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, json_output)
        if json_output:
            emit_json(verdict)
        else:
            status = "PASS" if verdict["passed"] else "FAIL"
            console.print(f"{status} {verdict['episode']}  ({verdict['stats']['control_steps'] + 1} samples, recorded)")
            for spec in verdict["specs"]:
                console.print(f"  {spec['name']:<20} {'PASS' if spec['passed'] else 'FAIL'}  "
                              f"rob {spec['robustness']:.3f}  worst at t={spec['witness_t']:.2f}s")
            if verdict["events_summary"]:
                console.print("  events: " + ", ".join(f"{k} x{v}" for k, v in verdict["events_summary"].items()))
            console.print(f"  next: qddot why {root / 'runs' / verdict['episode']}")
        raise typer.Exit(0 if verdict["passed"] else 1)
