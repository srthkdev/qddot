"""Agent adapters used by the benchmark runner."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentResult:
    final_message: str
    transcript_path: Path
    wall_s: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None
    exit_code: int = 0
