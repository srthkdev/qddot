"""Integration check for the installed Newton backend and cartpole task."""

from pathlib import Path

import pytest

from qddot.sim.runner import run_episode
from qddot.task import load_task


def test_cartpole_good_controller_runs_on_newton(tmp_path: Path) -> None:
    pytest.importorskip("newton")
    task = load_task(Path(__file__).parents[2] / "tasks/cartpole_balance/task.yaml")
    task.backend = "newton"
    result = run_episode(task, task.path.parent / "controllers/good.py", root=tmp_path,
                         record_mcap=False)
    assert result.episode_dir.is_dir()
    assert not result.verdict["errors"]
    print(f"Newton cartpole verdict: {'PASS' if result.verdict['passed'] else 'FAIL'}")
    assert result.verdict["passed"]
