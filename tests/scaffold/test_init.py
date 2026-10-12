from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from qddot.scaffold import scaffold_project
from qddot.sim.runner import run_episode
from qddot.task import load_task


@pytest.mark.parametrize("robot", ["so101", "ur5e"])
def test_menagerie_scaffold_runs_and_passes(robot: str, tmp_path: Path) -> None:
    task_path = scaffold_project(tmp_path / robot, robot)
    task = load_task(task_path)
    result = run_episode(task, task_path.parent / "controller.py",
                         root=tmp_path / ".qddot", record_mcap=False)

    assert result.verdict["passed"] is True
    assert [item["stl"] for item in task.specs] == [
        "always(col_robot_floor < 0.5)", "always(ev.joint_limit < 0.5)"
    ]
    assert (task_path.parent / ".gitignore").read_text().strip() == ".qddot/"


def test_urdf_scaffold_converts_and_runs(tmp_path: Path) -> None:
    urdf = tmp_path / "two_link.urdf"
    urdf.write_text(_two_link_urdf(), encoding="utf-8")
    project = tmp_path / "project"

    task_path = scaffold_project(project, str(urdf))
    task = load_task(task_path)
    model = mujoco.MjModel.from_xml_path(str(task.model_path))
    result = run_episode(task, project / "controller.py", root=tmp_path / ".qddot", record_mcap=False)

    assert task.model_path.suffix == ".xml"
    assert model.nu == 2
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    assert result.verdict["passed"] is True


def _two_link_urdf() -> str:
    return '''<?xml version="1.0"?>
<robot name="two_link">
  <link name="base">
    <inertial><origin xyz="0 0 0.1"/><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><origin xyz="0 0 0.1"/><geometry><box size="0.1 0.1 0.2"/></geometry></visual>
    <collision><origin xyz="0 0 0.1"/><geometry><box size="0.1 0.1 0.2"/></geometry></collision>
  </link>
  <link name="link1">
    <inertial><origin xyz="0.1 0 0"/><mass value="0.5"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
    <visual><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.04 0.04"/></geometry></visual>
    <collision><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.04 0.04"/></geometry></collision>
  </link>
  <link name="link2">
    <inertial><origin xyz="0.1 0 0"/><mass value="0.3"/><inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/></inertial>
    <visual><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.03 0.03"/></geometry></visual>
    <collision><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.03 0.03"/></geometry></collision>
  </link>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="link1"/><origin xyz="0 0 0.2"/><axis xyz="0 0 1"/><limit lower="-1.5" upper="1.5" effort="10" velocity="2"/></joint>
  <joint name="elbow" type="revolute"><parent link="link1"/><child link="link2"/><origin xyz="0.2 0 0"/><axis xyz="0 0 1"/><limit lower="-1.5" upper="1.5" effort="10" velocity="2"/></joint>
</robot>'''
