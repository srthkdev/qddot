"""List, lint, or evaluate task specifications."""

from __future__ import annotations

from pathlib import Path

import typer

from ._out import console, exit_error

_GROUPS = (
    ("Time", lambda name, _: name == "t"),
    ("Joints", lambda name, _: name.startswith("joint.")),
    ("Actuators", lambda name, _: name.startswith("act.")),
    ("Task signals", lambda name, declared: name in declared),
    ("Contacts", lambda name, _: name.startswith("col_")),
    ("Events", lambda name, _: name.startswith("ev.")),
)


def _print_signals(names, available: dict) -> None:
    for name in names:
        console.print(f"  {name}: {available[name]['kind']} ({available[name]['units']})")


def _check(task, stl: str) -> None:
    from qddot.spec import to_english
    from qddot.spec.lint import task_signals, validate

    available = task_signals(task)
    names = validate(stl, available)
    console.print(f"Valid: {to_english(stl)}")
    console.print("Signals:")
    _print_signals(names, available)


def _list_signals(task) -> None:
    from qddot.spec.lint import task_signals

    available = task_signals(task)
    declared = {item["name"] for item in task.signals}
    for title, belongs in _GROUPS:
        selected = [name for name in available if belongs(name, declared)]
        if selected:
            console.print(f"{title}:")
            _print_signals(selected, available)


def _evaluate(stl: str, episode: str, root: Path) -> None:
    from qddot.spec import evaluate, to_english
    from qddot.spec.lint import episode_signals, validate

    path, values = episode_signals(episode, root)
    validate(stl, {name: {"kind": "recorded signal", "units": "-"} for name in values})
    result = evaluate(stl, values["t"], values)
    witness = "n/a" if result.witness_t is None else f"{result.witness_t:.3f}s"
    console.print(f"{'PASS' if result.passed else 'FAIL'} robustness {result.robustness:.6g} "
                  f"witness {witness}")
    console.print(f"  {to_english(stl)}")
    console.print(f"  episode signals: {path}")


def register(app: typer.Typer) -> None:
    @app.command()
    def spec(
        task: Path = typer.Argument(...),  # noqa: B008
        check: str | None = typer.Option(None, "--check", help="Validate an STL spec against the task's signals."),
        signals: bool = typer.Option(False, "--signals", help="List every signal a spec can use."),
        evaluate: str | None = typer.Option(None, "--eval", help="Evaluate an STL spec on a recorded episode."),
        episode: str | None = typer.Option(None, "--episode"),
        root: Path = typer.Option(Path(".qddot"), "--root"),  # noqa: B008
    ) -> None:
        """List, lint, or evaluate task specifications."""
        try:
            from qddot.task import load_task

            if int(check is not None) + int(signals) + int(evaluate is not None) > 1:
                raise ValueError("choose only one of --check, --signals, or --eval")
            if (episode is None) != (evaluate is None):
                raise ValueError("--eval and --episode must be used together")
            loaded = load_task(task)
            if check is not None:
                return _check(loaded, check)
            if signals:
                return _list_signals(loaded)
            if evaluate is not None:
                return _evaluate(evaluate, episode, root)
            from qddot.spec import to_english

            for item in loaded.specs:
                console.print(f"{item['name']}: {item['stl']}")
                console.print(f"  {to_english(item['stl'])}")
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, False)
