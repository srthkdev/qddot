from __future__ import annotations

from pathlib import Path

import mujoco
import pytest
import yaml
from PIL import Image

from qddot.task import TaskError, load_task
from qddot.view import _configured_camera, _set_camera_for_model, _write_mp4


def _task_file(tmp_path: Path, render: dict) -> Path:
    repo = Path(__file__).resolve().parents[2]
    model = repo / "tasks/cartpole_balance/model.xml"
    task = {
        "name": "render_test", "model": str(model), "duration": 1.0, "control_dt": 0.02,
        "render": render,
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task), encoding="utf-8")
    return path


def test_render_config_loads_and_configures_free_camera(tmp_path: Path) -> None:
    config = {"camera": "free", "width": 1280, "height": 720, "fps": 30,
              "lookat": [0.0, 0.1, 0.2], "distance": 1.5, "azimuth": 130, "elevation": -22}
    task = load_task(_task_file(tmp_path, config))
    model = mujoco.MjModel.from_xml_path(str(task.model_path))
    camera = _configured_camera(mujoco, task.render)
    _set_camera_for_model(camera, model, task.render)
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert list(camera.lookat) == config["lookat"]
    assert camera.distance == config["distance"]
    assert camera.azimuth == config["azimuth"]
    assert camera.elevation == config["elevation"]


def test_render_config_rejects_invalid_dimensions(tmp_path: Path) -> None:
    path = _task_file(tmp_path, {"camera": "free", "width": 0})
    with pytest.raises(TaskError, match="render.width"):
        load_task(path)


def test_named_camera_selects_model_camera(tmp_path: Path) -> None:
    task = load_task(_task_file(tmp_path, {"camera": "tracking"}))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="tracking" pos="0 0 1"/></worldbody></mujoco>'
    )
    camera = _configured_camera(mujoco, task.render)
    _set_camera_for_model(camera, model, task.render)
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FIXED
    assert camera.fixedcamid == 0


def test_mp4_writer_uses_ffmpeg_when_available(tmp_path: Path) -> None:
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path / "frame.mp4"
    _write_mp4(path, [Image.new("RGB", (32, 32), "red")], 30)
    assert path.is_file() and path.stat().st_size > 0
