"""Task configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class TaskError(ValueError):
    """Raised when a task file is invalid."""


@dataclass
class Task:
    name: str
    description: str
    model: str
    duration: float
    control_dt: float
    path: Path
    model_path: Path
    init: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    signals: list[dict[str, Any]] = field(default_factory=list)
    contact_pairs: list[dict[str, Any]] = field(default_factory=list)
    perturb: dict[str, Any] = field(default_factory=dict)
    specs: list[dict[str, Any]] = field(default_factory=list)


def load_task(path: str | Path) -> Task:
    """Load and validate a task YAML file."""
    task_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TaskError(f"cannot read task {task_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TaskError("task file must contain a YAML mapping")

    required = ("name", "model", "duration", "control_dt")
    missing = [key for key in required if key not in raw]
    if missing:
        raise TaskError(f"missing required field(s): {', '.join(missing)}")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise TaskError("name must be a non-empty string")
    if not isinstance(raw["model"], str) or not raw["model"].strip():
        raise TaskError("model must be a non-empty path")
    duration = _positive_number(raw["duration"], "duration")
    control_dt = _positive_number(raw["control_dt"], "control_dt")
    model_path = (task_path.parent / raw["model"]).resolve()
    if not model_path.is_file():
        raise TaskError(f"model file does not exist: {model_path}")
    mappings = ("init", "target", "perturb")
    for key in mappings:
        if key in raw and not isinstance(raw[key], dict):
            raise TaskError(f"{key} must be a mapping")
    for key in ("signals", "contact_pairs", "specs"):
        if key in raw and (not isinstance(raw[key], list) or not all(isinstance(v, dict) for v in raw[key])):
            raise TaskError(f"{key} must be a list of mappings")
    for i, signal in enumerate(raw.get("signals", [])):
        if not isinstance(signal.get("name"), str) or not signal["name"]:
            raise TaskError(f"signals[{i}].name must be a non-empty string")
        if signal.get("kind") not in {"site_pos", "body_pos", "distance"}:
            raise TaskError(f"signals[{i}].kind must be site_pos, body_pos, or distance")
    for i, pair in enumerate(raw.get("contact_pairs", [])):
        if not pair.get("name") or not isinstance(pair.get("a"), list) or not isinstance(pair.get("b"), list):
            raise TaskError(f"contact_pairs[{i}] requires name and body-name lists a and b")
    for i, spec in enumerate(raw.get("specs", [])):
        if not isinstance(spec.get("name"), str) or not isinstance(spec.get("stl"), str):
            raise TaskError(f"specs[{i}] requires string name and stl")

    return Task(
        name=raw["name"].strip(), description=str(raw.get("description", "")),
        model=raw["model"], duration=duration, control_dt=control_dt, path=task_path,
        model_path=model_path, init=raw.get("init", {}), target=raw.get("target", {}),
        signals=raw.get("signals", []), contact_pairs=raw.get("contact_pairs", []),
        perturb=raw.get("perturb", {}), specs=raw.get("specs", []),
    )


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise TaskError(f"{name} must be a positive number")
    return float(value)
