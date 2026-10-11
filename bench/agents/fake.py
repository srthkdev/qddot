"""Deterministic fake agent used by tests and smoke runs."""

import shutil
from pathlib import Path

from bench.agents import AgentResult


def run_agent(workdir: Path, prompt_text: str, model: str, budget: dict) -> AgentResult:
    source = Path(budget.get("controller_source", model))
    if source.is_file():
        shutil.copy2(source, workdir / "controller.py")
    message = str(budget.get("final_message", "Controller complete."))
    transcript = workdir / "transcript.txt"
    transcript.write_text(message)
    return AgentResult(message, transcript, 0.0, 0, 0, None, 0)
