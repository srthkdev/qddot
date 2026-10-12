from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from qddot.cli import app
from qddot.sim.runner import run_episode
from qddot.task import load_task

runner = CliRunner()


def _task(tmp_path: Path) -> Path:
    model = tmp_path / "model.xml"
    model.write_text('''<mujoco><option timestep="0.002"/><worldbody>
      <geom name="floor" type="plane" size="0 0 0.1"/>
      <body name="arm" pos="0 0 0.2"><joint name="shoulder" type="hinge" range="-1 1"/>
        <geom type="sphere" size="0.05" mass="1"/><site name="tip" pos="0.1 0 0"/>
      </body></worldbody><actuator><position name="motor" joint="shoulder" kp="10" ctrllimited="true" ctrlrange="-1 1"/></actuator></mujoco>''', encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text('''name: sample
model: model.xml
duration: 0.1
control_dt: 0.02
signals:
  - {name: tip_z, kind: site_pos, ref: tip, axis: z}
contact_pairs:
  - {name: arm_floor, a: [arm], b: [floor]}
specs: []
''', encoding="utf-8")
    return task


def test_check_valid_and_invalid_signal_names(tmp_path: Path) -> None:
    task = _task(tmp_path)
    valid = runner.invoke(app, ["spec", str(task), "--check", "always(tip_z < 1)"])
    assert valid.exit_code == 0, valid.output
    assert "Valid:" in valid.output
    assert "tip_z: site pos (m)" in valid.output

    invalid = runner.invoke(app, ["spec", str(task), "--check", "always(tip_zz < 1)"])
    assert invalid.exit_code == 2
    assert "^" in invalid.output
    assert "Close matches: tip_z" in invalid.output


def test_signals_are_grouped(tmp_path: Path) -> None:
    result = runner.invoke(app, ["spec", str(_task(tmp_path)), "--signals"])

    assert result.exit_code == 0, result.output
    assert "Joints:" in result.output
    assert "Actuators:" in result.output
    assert "Task signals:" in result.output
    assert "Contacts:" in result.output
    assert "Events:" in result.output


def test_eval_uses_existing_episode_signals(tmp_path: Path) -> None:
    task_path = _task(tmp_path)
    task = load_task(task_path)
    controller = tmp_path / "controller.py"
    controller.write_text("class Controller:\n    def step(self, obs, t): return {'motor': 0.0}\n", encoding="utf-8")
    episode = run_episode(task, controller, root=tmp_path / ".qddot", record_mcap=False)

    result = runner.invoke(app, ["spec", str(task_path), "--eval", "always(tip_z < 1)",
                                 "--episode", episode.verdict["episode"], "--root", str(tmp_path / ".qddot")])

    assert result.exit_code == 0, result.output
    assert "PASS robustness" in result.output
    assert "witness 0.000s" in result.output
