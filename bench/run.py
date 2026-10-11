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


def _run_trial(task_dir: Path, agent: AgentSpec, condition: Condition, seed: int, repo_root: Path, budget: dict[str, Any]) -> dict:
    import importlib

    adapter = importlib.import_module(f"bench.agents.{agent.adapter}")
    workdir = build_workdir(task_dir, condition.name, repo_root=repo_root)
    prompt = (workdir / "PROMPT.md").read_text()
    result = adapter.run_agent(workdir, prompt, agent.model, budget)
    graded = grade(task_dir, workdir / "controller.py", seed + 100_000, int(budget.get("n_fuzz", 50)))
    add_false_done(graded, result.final_message)
    return {**graded, "agent_result": {"final_message": result.final_message, "transcript_path": str(result.transcript_path), "wall_s": result.wall_s, "tokens_in": result.tokens_in, "tokens_out": result.tokens_out, "cost_usd": result.cost_usd, "exit_code": result.exit_code}, "config": {"task": task_dir.name, "agent": agent.key, "adapter": agent.adapter, "model": agent.model, "extra_args": list(agent.extra_args), "condition": condition.name, "seed": seed}}


def run_plan(plan: RunPlan, repo_root: Path, *, yes: bool = False, dry_run: bool = True, budget: dict | None = None) -> Path | None:
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
    with ThreadPoolExecutor(max_workers=max(1, plan.jobs)) as pool, output.open("a") as stream:
        futures = [pool.submit(_run_trial, *trial, repo_root, budget) for trial in pending]
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
