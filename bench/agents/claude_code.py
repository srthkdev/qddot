"""Claude Code adapter."""

from pathlib import Path

from bench.agents import AgentResult
from bench.agents._common import claude_parse, result_for


def build_command(model: str, budget: dict) -> list[str]:
    return [
        "claude", "-p", "", "--output-format", "json", "--model", model,
        "--max-turns", str(budget.get("max_turns", 20)),
        "--permission-mode", "bypassPermissions",
    ]


def run_agent(workdir: Path, prompt_text: str, model: str, budget: dict) -> AgentResult:
    command = build_command(model, budget)
    command[2] = prompt_text
    return result_for(command, workdir, "", budget, claude_parse, workdir / "transcript.txt")
