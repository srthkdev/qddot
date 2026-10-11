"""OpenCode adapter."""

from pathlib import Path

from bench.agents import AgentResult
from bench.agents._common import opencode_parse, result_for


def build_command(model: str, budget: dict | None = None) -> list[str]:
    return ["opencode", "run", "--format", "json", "--auto", "--model", model, *(budget or {}).get("extra_args", [])]


def run_agent(workdir: Path, prompt_text: str, model: str, budget: dict) -> AgentResult:
    return result_for(build_command(model, budget), workdir, prompt_text, budget, opencode_parse, workdir / "transcript.jsonl")
