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
FORBIDDEN = {"good.py", "KNOWN_FAILURES.md", "expected.yaml"}


def _copy_model_tree(source: Path, dest: Path) -> None:
    """Copy an XML model and recursively referenced local resources."""
    visited: set[Path] = set()

    def copy_file(path: Path, target: Path) -> None:
        resolved = path.resolve()
        if resolved in visited or not resolved.is_file():
            return
        visited.add(resolved)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, target)
        if resolved.suffix.lower() != ".xml":
            return
        text = resolved.read_text(errors="ignore")
        for reference in re.findall(r'(?:file|mesh|texture)="([^"]+)"', text):
            if reference.startswith(("/", "http")):
                continue
            dependency = resolved.parent / reference
            if dependency.is_file():
                copy_file(dependency, target.parent / reference)
            elif dependency.is_dir():
                shutil.copytree(dependency, target.parent / reference, dirs_exist_ok=True)

    if source.is_dir():
        shutil.copytree(source, dest, dirs_exist_ok=True)
    else:
        copy_file(source, dest)


def _condition_files(dest: Path, condition: str, repo_root: Path) -> None:
    if condition == "none":
        (dest / "CONDITION.md").write_text(
            "No simulator is available in this condition. Implement carefully using the task description.\n"
        )
    elif condition == "stdout":
        (dest / "sim.py").write_text(_stdout_runner())
    elif condition in {"qddot", "qddot_memory"}:
        cli = repo_root / ".venv" / "bin" / "qddot"
        (dest / "CONDITION.md").write_text(
            f"Use qddot to run, check, and debug the controller. CLI: `{cli}`.\n"
            "Run the task using its task.yaml and controller.py. Do not claim success without a PASS verdict.\n"
        )
        skill = repo_root / "skills" / "qddot" / "SKILL.md"
        if skill.is_file():
            skill_dest = dest / "skills" / "qddot"
            skill_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skill, skill_dest / "SKILL.md")
            # Claude Code discovers project skills natively under .claude/skills.
            native = dest / ".claude" / "skills" / "qddot"
            shutil.copytree(skill.parent, native, dirs_exist_ok=True)
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
) -> Path:
    """Create a fresh workdir containing only the public task inputs."""
    task_dir = task_dir.resolve()
    if destination is None:
        import tempfile

        destination = Path(tempfile.mkdtemp(prefix="qddot-bench-"))
    else:
        destination.mkdir(parents=True, exist_ok=False)
    cfg_path = task_dir / "task.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    prompt = task_dir / "PROMPT.md"
    if prompt.is_file():
        shutil.copy2(prompt, destination / "PROMPT.md")
    else:
        (destination / "PROMPT.md").write_text(
            f"# {cfg.get('name', task_dir.name)}\n\n{cfg.get('description', '')}\n\n"
            "Implement controller.py with a Controller class. Optional reset(info: dict) returns None; "
            "step(obs: dict[str, float], t: float) returns {actuator_name: control_value}. "
            "Missing actuators keep their prior control.\n"
        )
    model_source = (task_dir / cfg["model"]).resolve()
    model_dest = destination / "model"
    _copy_model_tree(model_source, model_dest if model_source.is_dir() else model_dest / model_source.name)
    cfg["model"] = str((model_dest if model_source.is_dir() else model_dest / model_source.name).relative_to(destination))
    (destination / "task.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    starter = task_dir / "controllers" / "starter.py"
    shutil.copy2(starter, destination / "controller.py") if starter.is_file() else (destination / "controller.py").write_text(CONTROLLER_STUB)
    _condition_files(destination, condition, repo_root.resolve())
    leaked = [path.name for path in destination.rglob("*") if path.is_file() and (path.name in FORBIDDEN or path.name.startswith("buggy_"))]
    if leaked:
        raise AssertionError(f"Forbidden task files leaked: {leaked}")
    return destination
