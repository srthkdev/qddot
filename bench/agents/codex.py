"""Codex CLI adapter.

Isolation: the user's config.toml is ignored (auth still works), so the model,
effort and MCP servers are exactly what the benchmark passes.
"""

import json
from pathlib import Path

from bench.agents import AgentResult
from bench.agents._common import codex_parse, result_for


def build_command(workdir: Path, model: str, budget: dict) -> list[str]:
    command = [
        "codex", "exec", "--json", "--skip-git-repo-check", "--ignore-user-config",
        "-m", model,
        "-c", f"model_reasoning_effort={budget.get('effort', 'high')}",
        "-s", "workspace-write", "-C", str(workdir),
    ]
    server = budget.get("mcp_server")
    if server:
        command.extend([
            "-c", f"mcp_servers.qddot.command={json.dumps(server['command'])}",
            "-c", f"mcp_servers.qddot.args={json.dumps(server['args'])}",
        ])
    command.extend([*budget.get("extra_args", []), "-"])
    return command


def run_agent(workdir: Path, prompt_text: str, model: str, budget: dict) -> AgentResult:
    return result_for(build_command(workdir, model, budget), workdir, prompt_text, budget,
                      codex_parse, workdir / "transcript.jsonl")
