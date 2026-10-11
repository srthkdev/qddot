"""Configuration models for benchmark grids."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class AgentSpec:
    adapter: str
    model: str = ""
    extra_args: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.adapter}:{self.model}" if self.model else self.adapter


@dataclass(frozen=True)
class Condition:
    name: str


@dataclass
class RunPlan:
    tasks: list[Path]
    agents: list[AgentSpec]
    conditions: list[Condition]
    seeds: list[int]
    run_id: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    jobs: int = 1
    private: bool = False
