from pathlib import Path

from bench.workdir import build_workdir

ROOT = Path(__file__).resolve().parents[2]


def test_workdir_excludes_private_files_and_has_condition(tmp_path):
    workdir = build_workdir(ROOT / "tasks/cartpole_balance", "none", repo_root=ROOT, destination=tmp_path / "wd")
    assert (workdir / "PROMPT.md").is_file()
    assert (workdir / "task.yaml").is_file()
    assert (workdir / "controller.py").is_file()
    assert (workdir / "CONDITION.md").is_file()
    names = {path.name for path in workdir.rglob("*") if path.is_file()}
    assert not any(name in {"good.py", "KNOWN_FAILURES.md", "expected.yaml"} or name.startswith("buggy_") for name in names)
    assert "model" in (workdir / "task.yaml").read_text()


def test_stdout_runner_has_three_prints(tmp_path):
    workdir = build_workdir(ROOT / "tasks/so101_reach", "stdout", repo_root=ROOT, destination=tmp_path / "stdout")
    try:
        source = (workdir / "sim.py").read_text()
        assert source.count("print(") == 4  # two declared-signal prints plus fallback and done
    finally:
        import shutil

        shutil.rmtree(workdir)
