"""Claude Code adapter."""

from pathlib import Path

from bench.agents import AgentResult
from bench.agents._common import claude_parse, result_for


def build_command(model: str, budget: dict) -> list[str]:
    command = [
        "claude", "-p", "", "--output-format", "json", "--model", model,
        "--permission-mode", "bypassPermissions",
    ]
    if budget.get("max_budget_usd") is not None:
        command.extend(["--max-budget-usd", str(budget["max_budget_usd"])])
    command.extend(budget.get("extra_args", []))
    return command


def run_agent(workdir: Path, prompt_text: str, model: str, budget: dict) -> AgentResult:
    command = build_command(model, budget)
    command[2] = prompt_text
    return result_for(command, workdir, "", budget, claude_parse, workdir / "transcript.txt")
