"""Integration check for the installed Newton backend and cartpole task."""

from pathlib import Path

import numpy as np
import pytest

from qddot.sim.backend import load_backend
from qddot.sim.runner import run_episode
from qddot.task import load_task


def test_cartpole_good_controller_runs_on_newton(tmp_path: Path) -> None:
    pytest.importorskip("newton")
    task = load_task(Path(__file__).parents[2] / "tasks/cartpole_balance/task.yaml")
    task.backend = "newton"
    result = run_episode(task, Path(__file__).parent / "fixtures" / "cartpole_lqr.py", root=tmp_path,
                         record_mcap=False)
    assert result.episode_dir.is_dir()
    assert not result.verdict["errors"]
    print(f"Newton cartpole verdict: {'PASS' if result.verdict['passed'] else 'FAIL'}")
    assert result.verdict["passed"]


def test_contact_force_is_unavailable_without_mujoco_warp() -> None:
    pytest.importorskip("newton")
    fixture = Path(__file__).parent / "fixtures" / "box_drop.xml"
    sim = load_backend("newton", str(fixture))
    sim.reset({}, 1.0, 1.0)
    for _ in range(round(1.0 / sim.timestep)):
        sim.step()

    assert sim.measures_contact_force is False
    assert sim.contacts()
    assert all(np.isnan(contact.normal_force) for contact in sim.contacts())
