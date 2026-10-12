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
    print(
        f"Newton cartpole verdict: {'PASS' if result.verdict['passed'] else 'FAIL'}; "
        f"wall_s={result.verdict['stats']['wall_s']:.3f}"
    )
    assert result.verdict["passed"]


def test_contact_force_is_measured_from_mujoco_warp() -> None:
    pytest.importorskip("newton")
    fixture = Path(__file__).parent / "fixtures" / "box_drop.xml"
    sim = load_backend("newton", str(fixture))
    sim.reset({}, 1.0, 1.0)
    for _ in range(round(1.0 / sim.timestep)):
        sim.step()

    contacts = sim.contacts()
    assert sim.measures_contact_force is True
    assert contacts
    assert all(np.isfinite(contact.normal_force) and contact.normal_force > 0 for contact in contacts)
    rest_force = sum(contact.normal_force for contact in contacts)
    print(f"Newton box rest force: {rest_force:.3f} N")
    assert rest_force == pytest.approx(2.0 * 9.81, rel=0.03)


def test_stacked_boxes_report_measured_pair_forces() -> None:
    pytest.importorskip("newton")
    fixture = Path(__file__).parent / "fixtures" / "stack.xml"
    sim = load_backend("newton", str(fixture))
    sim.reset({}, 1.0, 1.0)
    for _ in range(round(1.0 / sim.timestep)):
        sim.step()

    forces = {frozenset((contact.a, contact.b)): contact.normal_force for contact in sim.contacts()}
    floor_bottom = forces[frozenset(("floor", "bottom"))]
    bottom_top = forces[frozenset(("bottom", "top"))]
    print(f"Newton stacked-box forces: floor-bottom={floor_bottom:.3f} N, bottom-top={bottom_top:.3f} N")
    assert np.isclose(floor_bottom, 5 * 9.81, rtol=0.03)
    assert np.isclose(bottom_top, 3 * 9.81, rtol=0.03)
