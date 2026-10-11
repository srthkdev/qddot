"""OpenCode adapter."""

from pathlib import Path

from bench.agents import AgentResult
from bench.agents._common import opencode_parse, result_for


def build_command(model: str) -> list[str]:
    return ["opencode", "run", "--model", model]


def run_agent(workdir: Path, prompt_text: str, model: str, budget: dict) -> AgentResult:
    return result_for(build_command(model), workdir, prompt_text, budget, opencode_parse, workdir / "transcript.jsonl")
