"""Codex CLI adapter."""

from pathlib import Path

from bench.agents import AgentResult
from bench.agents._common import codex_parse, result_for


def build_command(workdir: Path, model: str, budget: dict) -> list[str]:
    return [
        "codex", "exec", "--json", "-m", model,
        "-c", f"model_reasoning_effort={budget.get('effort', 'medium')}",
        "-s", "workspace-write", "-C", str(workdir),
        *budget.get("extra_args", []), "-",
    ]


def run_agent(workdir: Path, prompt_text: str, model: str, budget: dict) -> AgentResult:
    return result_for(build_command(workdir, model, budget), workdir, prompt_text, budget, codex_parse, workdir / "transcript.jsonl")
