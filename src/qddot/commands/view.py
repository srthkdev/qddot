"""Render an episode as an offline GIF and HTML report."""

from __future__ import annotations

from pathlib import Path

import typer

from ._out import console, exit_error


def register(app: typer.Typer) -> None:
    @app.command("view")
    def view_command(
        episode: Path = typer.Argument(...),  # noqa: B008
        open_browser: bool = typer.Option(False, "--open", help="Open the HTML report in a browser."),
        mp4: bool = typer.Option(False, "--mp4", help="Also encode replay.mp4 (requires ffmpeg)."),
    ) -> None:
        """Write a replay GIF and self-contained HTML report for an episode."""
        try:
            from qddot.view import create_view

            gif_path, html_path = create_view(episode, open_browser=open_browser, mp4=mp4)
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, False)
        console.print(f"wrote {gif_path}")
        console.print(f"wrote {html_path}")
        if mp4:
            console.print(f"wrote {episode / 'replay.mp4'}")
