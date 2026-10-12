from __future__ import annotations

import json
from pathlib import Path

import yaml

from qddot.sim.runner import run_episode
from qddot.task import load_task
from qddot.view import _build_html, _timeline, create_view


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
    assert "Overall robustness: <b>" in content
    saved_verdict = json.loads((result.episode_dir / "verdict.json").read_text(encoding="utf-8"))
    assert saved_verdict["episode"] in content


def test_timeline_has_legend_and_collapses_repeated_events() -> None:
    events = [{"t": index / 100, "kind": "saturation", "a": "motor", "b": None}
              for index in range(12)]
    verdict = {"specs": [{"name": "a deliberately long spec label", "window": [0.1, 0.8],
                           "witness_t": 0.4}]}
    svg = _timeline(events, verdict, 1.0)
    assert "saturation" in svg
    assert "0.000–0.110s (12 events)" in svg
    assert "<rect" in svg
    assert 'at 0.000s' not in svg


def test_failed_html_shows_rounded_robustness_and_likely_cause(tmp_path: Path, monkeypatch) -> None:
    import qddot.diag

    monkeypatch.setattr(qddot.diag, "why", lambda _path: {"likely_causes": [{
        "cause": "wrong control sign", "confidence": "high", "evidence": ["pole angle diverges"]
    }]})
    gif = tmp_path / "empty.gif"
    gif.write_bytes(b"GIF89a")
    verdict = {"passed": False, "episode": "example", "task": "cartpole", "seed": 0,
               "robustness": -1.23456, "stats": {}, "specs": [{"name": "upright",
                 "passed": False, "robustness": -1.23456}]}
    page = _build_html(tmp_path, gif, verdict, [])
    assert "-1.235" in page
    assert "Likely causes" in page
    assert "wrong control sign" in page
