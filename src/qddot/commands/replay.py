"""Replay an episode to a selected time."""

from __future__ import annotations

from pathlib import Path

import typer

from ._out import console, emit_json, exit_error


def register(app: typer.Typer) -> None:
    @app.command()
    def replay(
        episode: Path = typer.Argument(...),  # noqa: B008
        at: float = typer.Option(..., "--at", min=0),
        frame: bool = typer.Option(False, "--frame"),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Re-simulate an episode and save its integration state."""
        try:
            from qddot.diag import replay as replay_episode

            result = replay_episode(episode, at, frame)
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, json_output)
        if json_output:
            emit_json(result)
            return
        console.print(f"state at t={result['t']:.3f}s saved to {result['state']}")
        for joint in result["joints"]:
            console.print(f"{joint['name']}: q={joint['pos']:.5g} qd={joint['vel']:.5g}")
        console.print(f"active contacts: {result['active_contacts'] or 'none'}")
        if result.get("frame"):
            console.print(f"frame: {result['frame']}")
