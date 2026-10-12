"""Grid expansion, resumability, and JSONL persistence."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bench.config import AgentSpec, Condition, RunPlan
from bench.grade import add_false_done, grade
from bench.workdir import build_workdir

ADAPTERS = {"claude_code", "codex", "opencode", "fake"}


def discover_tasks(repo_root: Path, selection: str, private: bool = False) -> list[Path]:
    roots = [repo_root / "tasks"]
    if private and (repo_root / "tasks_private").is_dir():
        roots.append(repo_root / "tasks_private")
    found = sorted(path.parent for root in roots for path in root.glob("*/task.yaml"))
    if selection == "all":
        return found
    requested = {name.strip() for name in selection.split(",") if name.strip()}
    selected = [path for path in found if path.name in requested]
    missing = requested - {path.name for path in selected}
    if missing:
        raise ValueError(f"Unknown task(s): {', '.join(sorted(missing))}")
    return selected


def _row_key(row: dict) -> tuple:
    cfg = row["config"]
    return cfg["task"], cfg["agent"], cfg["condition"], cfg["seed"]


_CONDITION_LINES = {
    "none": "You cannot run a simulator in this environment. Reason carefully about the code.",
    "stdout": "You can simulate once with `{python} sim.py`; it prints only final values.",
    "qddot": ("Use the qddot CLI (`{qddot}`) to run, check, debug and fuzz your controller. "
              "See CONDITION.md and skills/qddot/SKILL.md."),
}


def compose_prompt(task_prompt: str, condition: str, tools: Path) -> str:
    """Wrap a task's PROMPT.md with the shared setup instructions for a condition."""
    line = _CONDITION_LINES.get(condition, _CONDITION_LINES["qddot"])
    line = line.format(python=tools / "python", qddot=tools / "qddot")
    return (
        f"{task_prompt.strip()}\n\n"
        "## Setup\n"
        "- Implement `class Controller` in `controller.py` in this directory. Its docstring lists "
        "the observation keys, actuators and `info` contents.\n"
        "- `task.yaml` describes the robot model, duration, control rate and acceptance specs.\n"
        f"- {line}\n"
        "- When you finish, end with a short summary that states clearly whether your controller "
        "satisfies the task.\n"
    )


def _run_trial(task_dir: Path, agent: AgentSpec, condition: Condition, seed: int, repo_root: Path,
               budget: dict[str, Any], tools: Path) -> dict:
    import importlib

    from bench.isolation import audit

    adapter = importlib.import_module(f"bench.agents.{agent.adapter}")
    workdir = build_workdir(task_dir, condition.name, repo_root=repo_root, tools=tools)
    prompt = compose_prompt((workdir / "PROMPT.md").read_text(), condition.name, tools)
    trial_budget = dict(budget)
    if condition.name in {"qddot", "qddot_memory"}:
        server = {"command": str(tools / "qddot"), "args": ["mcp"]}
        mcp_config = workdir / ".mcp.qddot.json"
        mcp_config.write_text(json.dumps({"mcpServers": {"qddot": server}}))
        trial_budget.update(mcp_config=str(mcp_config), mcp_server=server)
    result = adapter.run_agent(workdir, prompt, agent.model, trial_budget)
    graded = grade(task_dir, workdir / "controller.py", seed + 100_000, int(budget.get("n_fuzz", 50)), workdir)
    add_false_done(graded, result.final_message)
    transcript = Path(result.transcript_path)
    contamination = audit(transcript.read_text(errors="ignore") if transcript.is_file() else "", repo_root)
    return {**graded, "contamination": contamination,
            "agent_result": {"final_message": result.final_message, "transcript_path": str(result.transcript_path),
                             "wall_s": result.wall_s, "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
                             "cost_usd": result.cost_usd, "exit_code": result.exit_code},
            "config": {"task": task_dir.name, "agent": agent.key, "adapter": agent.adapter, "model": agent.model,
                       "extra_args": list(agent.extra_args), "condition": condition.name, "seed": seed,
                       "budget": budget}}


def run_plan(plan: RunPlan, repo_root: Path, *, yes: bool = False, dry_run: bool = True, budget: dict | None = None,
             tools: Path | None = None) -> Path | None:
    """Print a plan or execute trials and append results."""
    budget = budget or {}
    trials = [(task, agent, condition, seed) for task in plan.tasks for agent in plan.agents for condition in plan.conditions for seed in plan.seeds]
    if dry_run or not yes:
        print(f"run_id: {plan.run_id}\ntrials: {len(trials)}\njobs: {plan.jobs}")
        print("agent | trials | rough budget")
        for agent in plan.agents:
            count = sum(1 for _, candidate, _, _ in trials if candidate == agent)
            print(f"{agent.key} | {count} | estimate unavailable")
        return None
    output_dir = repo_root / "bench" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{plan.run_id}.jsonl"
    existing = set()
    if output.exists():
        for line in output.read_text().splitlines():
            try:
                existing.add(_row_key(json.loads(line)))
            except (ValueError, KeyError, TypeError):
                continue
    pending = [trial for trial in trials if (trial[0].name, trial[1].key, trial[2].name, trial[3]) not in existing]
    from bench.isolation import secrets_locked, tools_dir

    tools = tools or tools_dir(Path(__file__).resolve().parents[1])
    with secrets_locked(repo_root), ThreadPoolExecutor(max_workers=max(1, plan.jobs)) as pool, \
            output.open("a") as stream:
        futures = [pool.submit(_run_trial, *trial, repo_root, budget, tools) for trial in pending]
        for future in as_completed(futures):
            stream.write(json.dumps(future.result(), sort_keys=True) + "\n")
            stream.flush()
    return output


def parse_agents(value: str) -> list[AgentSpec]:
    agents = []
    for item in value.split(","):
        adapter, _, model = item.strip().partition(":")
        if adapter not in ADAPTERS:
            raise ValueError(f"Unknown agent adapter: {adapter}")
        agents.append(AgentSpec(adapter, model))
    return agents
