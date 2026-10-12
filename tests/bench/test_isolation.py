"""The benchmark must never hand an agent a path or file that leads to the answers."""
import os
import tempfile
from pathlib import Path

import pytest

from bench.isolation import audit, secret_files, secrets_locked
from bench.workdir import build_workdir, is_secret
from qddot.sim.runner import run_episode
from qddot.task import load_task

ROOT = Path(__file__).resolve().parents[2]
TASKS = sorted(p.parent for p in (ROOT / "tasks").glob("*/task.yaml"))


@pytest.mark.parametrize("task_dir", TASKS, ids=[t.name for t in TASKS])
def test_workdir_is_self_contained_and_secret_free(task_dir: Path, tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    workdir = build_workdir(task_dir, "qddot", repo_root=ROOT, destination=tmp_path / "w", tools=tools)
    for path in workdir.rglob("*"):
        if path.is_file():
            assert not is_secret(path), path
            if path.suffix in {".md", ".yaml", ".xml", ".py", ".json"}:
                assert str(ROOT) not in path.read_text(errors="ignore"), f"repo path leaks via {path}"
    task = load_task(workdir / "task.yaml")
    assert workdir in task.model_path.resolve().parents
    if not task.editable_model:
        result = run_episode(task, task_dir / "controllers" / "good.py", root=Path(tempfile.mkdtemp()))
        assert result.verdict["passed"], "the isolated copy must behave exactly like the original"


def test_secrets_are_unreadable_while_locked_and_restored_after() -> None:
    secrets = secret_files(ROOT)
    assert any(p.name == "good.py" for p in secrets)
    with secrets_locked(ROOT):
        assert not os.access(secrets[0], os.R_OK)
    assert all(os.access(p, os.R_OK) for p in secrets)


def test_audit_flags_reaching_for_answers() -> None:
    assert audit(f"cat {ROOT}/tasks/x/controllers/good.py", ROOT)
    assert audit("ls controllers/buggy_sign.py", ROOT)
    assert audit("qddot run task.yaml --controller controller.py", ROOT) == []
