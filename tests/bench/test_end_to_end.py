import json
import shutil
from pathlib import Path

import yaml

from bench.config import AgentSpec, Condition, RunPlan
from bench.report import report
from bench.run import run_plan

ROOT = Path(__file__).resolve().parents[2]


def test_fake_agent_grid_grades_and_reports(tmp_path):
    source = ROOT / "tasks/cartpole_balance"
    task = tmp_path / "cartpole_balance"
    (task / "controllers").mkdir(parents=True)
    for filename in ("task.yaml", "model.xml"):
        shutil.copy2(source / filename, task / filename)
    for filename in ("good.py", "buggy_slow.py"):
        shutil.copy2(source / "controllers" / filename, task / "controllers" / filename)
    task_config = yaml.safe_load((task / "task.yaml").read_text())
    task_config["duration"] = 0.1
    (task / "task.yaml").write_text(yaml.safe_dump(task_config, sort_keys=False))
    run_id = "bench-test"
    plan = RunPlan([task], [AgentSpec("fake", str(task / "controllers/good.py")), AgentSpec("fake", str(task / "controllers/buggy_slow.py"))], [Condition("none")], [0], run_id=run_id, jobs=1)
    output = run_plan(plan, tmp_path, yes=True, dry_run=False, budget={"n_fuzz": 4, "final_message": "Done"},
                      tools=ROOT / ".venv" / "bin")
    try:
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(rows) == 2
        assert {row["config"]["model"].split("/")[-1] for row in rows} == {"good.py", "buggy_slow.py"}
        assert all("fuzz_pass_rate" in row and row["config"]["adapter"] == "fake" for row in rows)
        markdown, chart = report([output], tmp_path / "report")
        assert markdown.is_file() and chart.is_file()
        assert "Success" in markdown.read_text()
    finally:
        output.unlink(missing_ok=True)
