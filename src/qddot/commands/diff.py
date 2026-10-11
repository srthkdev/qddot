"""Compare two episodes."""

from __future__ import annotations

from pathlib import Path

import typer

from ._out import console, emit_json, exit_error


def register(app: typer.Typer) -> None:
    @app.command()
    def diff(
        a: Path = typer.Argument(...),  # noqa: B008
        b: Path = typer.Argument(...),  # noqa: B008
        tolerance: float = typer.Option(1e-6, "--tolerance", min=0),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Compare two recorded runs."""
        try:
            from qddot.diag import diff as compare

            result = compare(a, b, tolerance)
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, json_output)
        if json_output:
            emit_json(result)
            return
        console.print(f"episodes: {result['episodes'][0]} vs {result['episodes'][1]}")
        changed = [(name, data) for name, data in result["signals"].items() if data["first_divergence_t"] is not None]
        console.print(f"signal divergences: {len(changed)}/{len(result['signals'])}")
        for name, data in changed[:12]:
            console.print(f"  {name} first differs at {data['first_divergence_t']:.3f}s (max {data['max_abs_delta']:.4g})")
        console.print(f"events only A/B: {len(result['events_only_a'])}/{len(result['events_only_b'])}")
        for name, values in sorted(result["robustness"].items()):
            console.print(f"{name}: {values['a']} -> {values['b']} (delta {values['delta']})")
        console.print(f"metadata: {result['metadata']}")
