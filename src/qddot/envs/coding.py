"""Grade controller source code against a task: the reward for agentic-coding RL."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from qddot.sim.fuzz import fuzz
from qddot.sim.runner import run_episode
from qddot.task import Task, load_task

FUZZ_BAR = 0.9


@dataclass(frozen=True)
class Feedback:
    passed: bool          # nominal run satisfies every spec
    robustness: float     # min spec robustness of the nominal run (<0 means violated)
    fuzz_pass_rate: float | None
    reward: float         # in [0, 1]
    text: str             # what the agent sees: verdict, failing spec, diagnosis


def evaluate_source(task: Task | str | Path, source: str, n_fuzz: int = 10, seed: int = 0,
                    root: Path | None = None) -> Feedback:
    """Run `source` (a Python file defining `Controller`) nominally, then under fuzz if it passes.

    Reward: 0 for errors or failure (with a small robustness-shaped term), 0.5 for a
    nominal pass, up to 1.0 as the fuzz pass rate reaches the bar.
    """
    task = task if isinstance(task, Task) else load_task(task)
    root = Path(root or tempfile.mkdtemp(prefix="qddot-env-"))
    root.mkdir(parents=True, exist_ok=True)
    controller = root / "controller.py"
    controller.write_text(source)
    verdict = run_episode(task, controller, seed=seed, root=root, record_mcap=False).verdict
    robustness = float(verdict["robustness"] if verdict["robustness"] is not None else -1.0)
    lines = [_verdict_line(verdict)]
    if not verdict["passed"]:
        from qddot.diag import why

        lines += _why_lines(why(root / "runs" / verdict["episode"]))
        shaped = 0.25 * max(0.0, 1.0 + robustness) if not verdict["errors"] else 0.0
        return Feedback(False, robustness, None, min(shaped, 0.25), "\n".join(lines))
    summary = fuzz(task, controller, n=n_fuzz, seed=seed + 1, root=root)
    rate = float(summary["pass_rate"])
    lines.append(f"fuzz: {summary['passed']}/{summary['n']} passed under perturbation "
                 f"({'meets' if rate >= FUZZ_BAR else 'below'} the {FUZZ_BAR:.0%} bar)")
    return Feedback(True, robustness, rate, 0.5 + 0.5 * min(1.0, rate / FUZZ_BAR), "\n".join(lines))


def _verdict_line(verdict: dict) -> str:
    if verdict["errors"]:
        return "ERROR: " + verdict["errors"][0].strip().splitlines()[-1]
    specs = ", ".join(f"{s['name']} {'PASS' if s['passed'] else 'FAIL'} ({s['robustness']:.3g})"
                      for s in verdict["specs"])
    return f"{'PASS' if verdict['passed'] else 'FAIL'}: {specs}"


def _why_lines(result: dict) -> list[str]:
    lines = []
    failed = result.get("first_failing_spec")
    if failed:
        lines.append(f"first failing spec: {failed['name']} at t={failed['witness_t']:.2f}s: "
                     f"{failed.get('english', failed.get('stl'))}")
    lines += [f"{s['signal']}: {s['value']:.4g} (range {s['min']:.4g}..{s['max']:.4g}), {s['trend']}"
              for s in result.get("signals", [])[:3]]
    first = result.get("first_events", {})
    if first:
        lines.append("first events: " + "; ".join(f"{k} {e.get('a')} @{float(e['t']):.2f}s"
                                                  for k, e in first.items()))
    return lines
