"""Run one controller episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import typer

from ._out import console, emit_json, exit_error


def _parse_perturb(items: list[str]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"perturbation '{item}' must use key=value")
        key, raw = item.split("=", 1)
        if not key:
            raise ValueError(f"perturbation '{item}' has an empty key")
        if raw.lower() == "null":
            values[key] = None
        else:
            try:
                values[key] = float(raw)
            except ValueError as exc:
                raise ValueError(f"perturbation '{item}' must have a numeric or null value") from exc
    return values


def _first_event_by_kind(path: Path) -> dict[str, tuple[str, float]]:
    if not path.is_file():
        return {}
    try:
        rows = duckdb.connect(database=":memory:").execute(
            "SELECT kind, a, t FROM read_parquet(?) ORDER BY t", [str(path)]
        ).fetchall()
    except duckdb.Error:
        return {}
    first: dict[str, tuple[str, float]] = {}
    for kind, element, time in rows:
        first.setdefault(str(kind), (str(element), float(time)))
    return first


def _summary(verdict: dict[str, Any], first_events: dict[str, tuple[str, float]],
             top_cause: dict[str, Any] | None = None) -> list[str]:
    status = "PASS" if verdict.get("passed") else "FAIL"
    robustness = verdict.get("robustness")
    rob_text = "n/a" if robustness is None else f"{float(robustness):.3f}"
    lines = [f"{status} {verdict.get('task', 'task')}  robustness {rob_text}  episode {verdict.get('episode', 'unknown')}"]
    for spec in verdict.get("specs", []):
        witness = spec.get("witness_t")
        suffix = f"  worst at t={float(witness):.2f}s" if witness is not None else ""
        if witness is not None and spec.get("window"):
            start, end = spec["window"]
            suffix += f" (window {float(start):.2f}-{float(end):.2f}s)"
        value = spec.get("robustness")
        metric = f"  rob {float(value):.3f}" if value is not None else ""
        lines.append(f"  {spec.get('name', 'spec'):<20} {'PASS' if spec.get('passed') else 'FAIL'}{metric}{suffix}")
    counts = verdict.get("events_summary", {}) or {}
    if counts:
        parts = []
        for kind, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]:
            first = first_events.get(kind)
            extra = f" (first {first[0]} @{first[1]:.2f}s)" if first else ""
            parts.append(f"{kind} x{count}{extra}")
        lines.append("  events: " + ", ".join(parts))
    if not verdict.get("passed") and top_cause:
        lines.append(f"  why: {top_cause['cause']} [{top_cause['confidence']}]")
    episode_dir = verdict.get("episode_dir", f".qddot/runs/{verdict.get('episode', 'unknown')}")
    lines.append(f"  files: {episode_dir}/  (signals.parquet events.parquet verdict.json log.mcap)")
    lines.append('  next: qddot query "select ..." | qddot fuzz ...')
    return lines[:12]


def register(app: typer.Typer) -> None:
    @app.command()
    def run(
        task: Path = typer.Argument(...),  # noqa: B008
        controller: Path = typer.Option(..., "--controller"),  # noqa: B008
        seed: int = typer.Option(0, "--seed"),
        perturb: list[str] = typer.Option([], "--perturb"),  # noqa: B008
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Run a controller against a task."""
        try:
            perturb_values = _parse_perturb(perturb)
            from qddot.sim.runner import run_episode
            from qddot.task import load_task

            result = run_episode(load_task(task), controller, seed=seed, perturb=perturb_values or None)
            verdict = dict(result.verdict)
            verdict.setdefault("episode_dir", str(result.episode_dir))
        except Exception as exc:  # noqa: BLE001 - CLI errors must stay machine-readable.
            exit_error(exc, json_output)
        if json_output:
            emit_json(verdict)
        else:
            top_cause = None
            if not verdict.get("passed"):
                try:
                    from qddot.diag import why as diagnose

                    causes = diagnose(result.episode_dir).get("likely_causes", [])
                    top_cause = causes[0] if causes else None
                except Exception:  # noqa: BLE001 - A useful summary must not hide the verdict.
                    top_cause = None
            for line in _summary(verdict, _first_event_by_kind(result.episode_dir / "events.parquet"), top_cause):
                console.print(line)
        if not verdict.get("passed", False):
            raise typer.Exit(1)
