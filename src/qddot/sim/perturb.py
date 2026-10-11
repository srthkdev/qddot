"""Deterministic task perturbation sampling and application helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

PARAM_DEFAULTS: dict[str, float | None] = {
    "friction_scale": 1.0,
    "mass_scale": 1.0,
    "obs_noise_std": 0.0,
    "latency_ms": 0.0,
    "jitter_ms": 0.0,
    "controller_budget_ms": None,
}


def sample(task: Any, rng: np.random.Generator) -> dict[str, float | None]:
    """Sample concrete perturbations from the task's configured ranges."""
    result = dict(PARAM_DEFAULTS)
    for name, value in task.perturb.items():
        if name not in result:
            continue
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"perturb.{name} must have two range endpoints")
            result[name] = float(rng.uniform(float(value[0]), float(value[1])))
        elif value is None:
            result[name] = None
        else:
            result[name] = float(value)
    return result


def resolve(values: dict[str, Any] | None) -> dict[str, float | None]:
    """Overlay explicit perturbations on nominal settings."""
    result = dict(PARAM_DEFAULTS)
    if values:
        unknown = set(values) - set(result)
        if unknown:
            raise ValueError(f"unknown perturb parameter(s): {', '.join(sorted(unknown))}")
        result.update(values)
    for name, value in result.items():
        if value is not None:
            result[name] = float(value)
    return result
