from __future__ import annotations

import json
from pathlib import Path

import yaml

from qddot.sim.runner import run_episode
from qddot.task import load_task
from qddot.view import create_view


def test_view_creates_valid_gif_and_self_contained_html(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    source_task = repo / "tasks/cartpole_balance/task.yaml"
    raw = yaml.safe_load(source_task.read_text(encoding="utf-8"))
    raw["duration"] = 0.2
    raw["model"] = str((source_task.parent / raw["model"]).resolve())
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    controller = repo / "tasks/cartpole_balance/controllers/good.py"
    result = run_episode(load_task(task_path), controller, root=tmp_path / "data", record_mcap=False)
    gif, report = create_view(result.episode_dir)
    assert gif.read_bytes().startswith((b"GIF87a", b"GIF89a"))
    content = report.read_text(encoding="utf-8")
    assert "data:image/gif;base64," in content
    assert "<svg" in content
    assert "Episode replay" in content
    saved_verdict = json.loads((result.episode_dir / "verdict.json").read_text(encoding="utf-8"))
    assert saved_verdict["episode"] in content
