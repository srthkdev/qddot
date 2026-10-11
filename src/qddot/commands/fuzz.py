"""Perturbation fuzz command."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from ._out import console, emit_json, exit_error


def register(app: typer.Typer) -> None:
    @app.command()
    def fuzz(
        task: Path = typer.Argument(...),  # noqa: B008
        controller: Path = typer.Option(..., "--controller"),  # noqa: B008
        n: int = typer.Option(50, "-n", min=1),
        workers: int = typer.Option(os.cpu_count() or 1, "--workers", min=1),
        seed: int = typer.Option(0, "--seed"),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Run repeated perturbed simulations."""
        try:
            from qddot.sim.fuzz import fuzz as run_fuzz
            from qddot.task import load_task

            summary = run_fuzz(load_task(task), controller, n=n, workers=workers, seed=seed)
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, json_output)
        if json_output:
            emit_json(summary)
        else:
            console.print(f"fuzz: {summary.get('passed', 0)}/{summary.get('n', n)} passed  pass rate {summary.get('pass_rate', 0):.1%}")
            for item in summary.get("worst", [])[:5]:
                console.print(f"  {item.get('episode', 'episode')}  robustness {item.get('robustness', float('nan')):.3f}")
            correlations = summary.get("by_param", {}) or {}
            ranked = sorted(correlations.items(), key=lambda pair: abs(pair[1].get("pass_corr") or 0), reverse=True)
            if ranked:
                param, values = ranked[0]
                corr = values.get("pass_corr")
                console.print(f"  most correlated: {param} (pass_corr {'n/a' if corr is None else f'{corr:+.3f}'})")
        if int(summary.get("passed", 0)) < int(summary.get("n", n)):
            raise typer.Exit(1)
