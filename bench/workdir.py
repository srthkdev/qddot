"""Build isolated agent workdirs from task inputs."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

CONTROLLER_STUB = '''class Controller:
    def reset(self, info: dict) -> None:
        pass

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        return {}
'''
FORBIDDEN = {"good.py", "KNOWN_FAILURES.md", "expected.yaml", "model_fixed.xml"}


def is_secret(path: Path) -> bool:
    """Files an agent must never see: reference solutions, seeded bugs and their notes."""
    return path.name in FORBIDDEN or path.name.startswith("buggy_")


def _copy_task(task_dir: Path, dest: Path) -> None:
    """Copy a task's public files (models, scenes, prompt), never controllers or secrets."""
    def ignore(directory: str, names: list[str]) -> set[str]:
        here = Path(directory)
        return {n for n in names if is_secret(here / n) or (here == task_dir and n == "controllers")}

    shutil.copytree(task_dir, dest, ignore=ignore)


def _asset_dirs(task_dir: Path, model: str) -> set[str]:
    """Robot asset folders (under repo `assets/`) the task's model files reference."""
    text = model + "".join(p.read_text(errors="ignore") for p in task_dir.rglob("*.xml") if not is_secret(p))
    return set(re.findall(r"assets/([A-Za-z0-9_.-]+)", text))


def _condition_files(dest: Path, condition: str, tools: Path, skill_dir: Path) -> None:
    if condition == "none":
        (dest / "CONDITION.md").write_text(
            "No simulator is available in this condition. Implement carefully using the task description.\n"
        )
    elif condition == "stdout":
        (dest / "sim.py").write_text(_stdout_runner())
    elif condition in {"qddot", "qddot_memory"}:
        (dest / "CONDITION.md").write_text(
            f"Use qddot to run, check, and debug the controller. CLI: `{tools / 'qddot'}`.\n"
            "Run the task using its task.yaml and controller.py. Do not claim success without a PASS verdict.\n"
        )
        if (skill_dir / "SKILL.md").is_file():
            shutil.copytree(skill_dir, dest / "skills" / "qddot", dirs_exist_ok=True)
            # Claude Code discovers project skills natively under .claude/skills.
            shutil.copytree(skill_dir, dest / ".claude" / "skills" / "qddot", dirs_exist_ok=True)
    else:
        raise ValueError(f"Unknown condition: {condition}")


def _stdout_runner() -> str:
    return '''from pathlib import Path
import sys

from qddot.task import load_task
from qddot.sim.runner import run_episode

root = Path(__file__).resolve().parent
task = load_task(root / "task.yaml")
result = run_episode(task, root / "controller.py", seed=0, root=root / ".qddot")
signals_path = result.episode_dir / "signals.parquet"
try:
    import pyarrow.parquet as pq
    table = pq.read_table(signals_path)
    names = [item["name"] for item in task.signals[:2]]
    data = table.to_pydict()
    for index in range(2):
        if index < len(names):
            name = names[index]
            print(f"{name}: {data[name][-1]}")
        else:
            print(f"signal_{index + 1}: unavailable")
except Exception:
    print("signal_1: unavailable")
    print("signal_2: unavailable")
print("done")
'''


def build_workdir(
    task_dir: Path,
    condition: str,
    *,
    repo_root: Path,
    destination: Path | None = None,
    tools: Path | None = None,
) -> Path:
    """Create a self-contained workdir with only the public task inputs.

    Nothing in it points back into the repository: models and robot assets are
    copied (mirroring `tasks/<name>/` and `assets/<robot>/`), and qddot comes
    from `tools`, an install outside the repo (see `bench.isolation`).
    """
    import os
    import tempfile

    task_dir, repo_root = task_dir.resolve(), repo_root.resolve()
    if destination is None:
        destination = Path(tempfile.mkdtemp(prefix="qddot-bench-"))
    else:
        destination.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((task_dir / "task.yaml").read_text())
    local = Path("tasks") / task_dir.name
    _copy_task(task_dir, destination / local)
    for robot in _asset_dirs(task_dir, cfg["model"]):
        source = repo_root / "assets" / robot
        if source.is_dir():
            shutil.copytree(source, destination / "assets" / robot, dirs_exist_ok=True)
    cfg["model"] = os.path.normpath(local / cfg["model"])
    (destination / "task.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    prompt = task_dir / "PROMPT.md"
    (destination / "PROMPT.md").write_text(prompt.read_text() if prompt.is_file() else
                                           f"# {cfg.get('name', task_dir.name)}\n\n{cfg.get('description', '')}\n")
    starter = task_dir / "controllers" / "starter.py"
    (destination / "controller.py").write_text(starter.read_text() if starter.is_file() else CONTROLLER_STUB)
    tools = tools or repo_root / ".venv" / "bin"
    _condition_files(destination, condition, tools, repo_root / "src" / "qddot" / "skill_data" / "qddot")
    leaked = [str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file() and is_secret(p)]
    if leaked:
        raise AssertionError(f"Forbidden task files leaked: {leaked}")
    return destination
