"""Command line interface for qddot."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import duckdb
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
console = Console(no_color=True, highlight=False)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _emit_json(value: Any) -> None:
    typer.echo(json.dumps(value, default=_json_default, sort_keys=True))


def _fail(message: str, code: int = 2) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


def _parse_perturb(items: list[str]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"perturbation '{item}' must use key=value")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError(f"perturbation '{item}' has an empty key")
        if raw_value.lower() == "null":
            values[key] = None
        else:
            try:
                values[key] = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"perturbation '{item}' must have a numeric or null value") from exc
    return values


def _run_summary(verdict: dict[str, Any], first_events: dict[str, tuple[str, float]] | None = None) -> list[str]:
    status = "PASS" if verdict.get("passed") else "FAIL"
    robustness = verdict.get("robustness")
    rob_text = "n/a" if robustness is None else f"{float(robustness):.3f}"
    lines = [f"{status} {verdict.get('task', 'task')}  robustness {rob_text}  episode {verdict.get('episode', 'unknown')}"]
    for spec in verdict.get("specs", []):
        witness = spec.get("witness_t")
        window = spec.get("window")
        suffix = ""
        if witness is not None:
            suffix = f"  worst at t={float(witness):.2f}s"
            if window and len(window) == 2:
                suffix += f" (window {float(window[0]):.2f}-{float(window[1]):.2f}s)"
        spec_rob = spec.get("robustness")
        lines.append(f"  {spec.get('name', 'spec'):<20} {'PASS' if spec.get('passed') else 'FAIL'}  rob {float(spec_rob):.3f}{suffix}" if spec_rob is not None else f"  {spec.get('name', 'spec'):<20} {'PASS' if spec.get('passed') else 'FAIL'}{suffix}")
    event_counts = verdict.get("events_summary", {}) or {}
    if event_counts:
        event_parts = []
        for kind, count in sorted(event_counts.items(), key=lambda item: (-item[1], item[0]))[:3]:
            first = (first_events or {}).get(kind)
            detail = f" (first {first[0]} @{first[1]:.2f}s)" if first else ""
            event_parts.append(f"{kind} x{count}{detail}")
        lines.append("  events: " + ", ".join(event_parts))
    episode_dir = verdict.get("episode_dir")
    if not episode_dir and verdict.get("episode"):
        episode_dir = f".qddot/runs/{verdict['episode']}"
    if episode_dir:
        lines.append(f"  files: {episode_dir}/  (signals.parquet events.parquet verdict.json log.mcap)")
    lines.append('  next: qddot query "select ..." | qddot fuzz ...')
    return lines[:12]


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


@app.command()
def run(
    task: Path = typer.Argument(..., exists=True, readable=True),  # noqa: B008
    controller: Path = typer.Option(..., "--controller", exists=True, readable=True),  # noqa: B008
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
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        _fail(str(exc))
        return
    if json_output:
        _emit_json(verdict)
    else:
        for line in _run_summary(verdict, _first_event_by_kind(result.episode_dir / "events.parquet")):
            console.print(line)
    if not verdict.get("passed", False):
        raise typer.Exit(1)


@app.command()
def fuzz(
    task: Path = typer.Argument(..., exists=True, readable=True),  # noqa: B008
    controller: Path = typer.Option(..., "--controller", exists=True, readable=True),  # noqa: B008
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
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        _fail(str(exc))
        return
    if json_output:
        _emit_json(summary)
    else:
        console.print(f"fuzz: {summary.get('passed', 0)}/{summary.get('n', n)} passed  pass rate {summary.get('pass_rate', 0):.1%}")
        for item in summary.get("worst", [])[:5]:
            console.print(f"  {item.get('episode', 'episode')}  robustness {item.get('robustness', float('nan')):.3f}")
        correlations = summary.get("by_param", {}) or {}
        ranked = sorted(
            correlations.items(),
            key=lambda pair: abs(pair[1].get("pass_corr") or 0),
            reverse=True,
        )
        if ranked:
            param, values = ranked[0]
            correlation = values.get("pass_corr")
            value_text = "n/a" if correlation is None else f"{correlation:+.3f}"
            console.print(f"  most correlated: {param} (pass_corr {value_text})")
    if int(summary.get("passed", 0)) < int(summary.get("n", n)):
        raise typer.Exit(1)


@app.command()
def check(
    model: Path = typer.Argument(..., exists=True, readable=True),  # noqa: B008
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run static checks on a MuJoCo model."""
    from qddot.check.static import check_model

    results = check_model(model)
    if json_output:
        _emit_json(results)
    else:
        if not results:
            console.print("PASS no issues found")
        for issue in results:
            console.print(f"{issue['severity'].upper():7} {issue['code']}: {issue['message']} [{issue['element']}]")
    if any(issue.get("severity") == "error" for issue in results):
        raise typer.Exit(1)


@app.command()
def spec(task: Path = typer.Argument(..., exists=True, readable=True)) -> None:  # noqa: B008
    """List a task's specs with English explanations."""
    try:
        from qddot.spec import to_english
        from qddot.task import load_task

        loaded = load_task(task)
        for item in loaded.specs:
            console.print(f"{item['name']}: {item['stl']}")
            console.print(f"  {to_english(item['stl'])}")
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        _fail(str(exc))


def _available_columns(connection: Any) -> str:
    parts: list[str] = []
    for view in ("signals", "events", "verdicts"):
        try:
            rows = connection.execute(f"DESCRIBE {view}").fetchall()
            parts.append(f"{view}({', '.join(row[0] for row in rows)})")
        except duckdb.Error:
            parts.append(f"{view}(unavailable)")
    return "; ".join(parts)


@app.command()
def query(
    sql: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json"),
    limit: int = typer.Option(50, "--limit", min=1),
    root: Path = typer.Option(Path(".qddot"), "--root"),  # noqa: B008
) -> None:
    """Query recorded signals, events, and verdicts with DuckDB SQL."""
    from qddot.store import connect

    connection = connect(root)
    try:
        cursor = connection.execute(sql)
        names = [column[0] for column in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)
    except duckdb.Error as exc:
        _fail(f"DuckDB: {exc}\nAvailable views and columns: {_available_columns(connection)}")
        return
    truncated = len(rows) > limit
    rows = rows[:limit]
    if json_output:
        _emit_json({"columns": names, "rows": [dict(zip(names, row, strict=False)) for row in rows], "truncated": truncated})
        return
    table = Table(box=None, show_header=True, header_style="bold")
    for name in names:
        table.add_column(name, overflow="fold")
    for row in rows:
        table.add_row(*("NULL" if value is None else str(value) for value in row))
    console.print(table)
    if truncated:
        console.print(f"showing first {limit} rows; results truncated (use --limit to change)")


@app.command()
def runs(
    last: int = typer.Option(10, "--last", min=1),
    root: Path = typer.Option(Path(".qddot"), "--root"),  # noqa: B008
) -> None:
    """List recent episodes by verdict file modification time."""
    paths = list((root / "runs").glob("*/verdict.json"))
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not paths:
        console.print("no runs found")
        return
    table = Table(box=None, show_header=True, header_style="bold")
    for heading in ("episode", "result", "robustness", "task"):
        table.add_column(heading)
    for path in paths[:last]:
        try:
            verdict = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        table.add_row(path.parent.name, "PASS" if verdict.get("passed") else "FAIL", str(verdict.get("robustness", "n/a")), str(verdict.get("task", "")))
    console.print(table)


if __name__ == "__main__":
    app()
