"""Parallel deterministic perturbation sweeps."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from qddot.sim.perturb import sample
from qddot.sim.runner import run_episode
from qddot.task import Task


def fuzz(task: Task, controller_path: Path, n: int, workers: int | None = os.cpu_count(),
         seed: int = 0, root: Path = Path(".qddot")) -> dict[str, Any]:
    """Run n seeded perturbation samples in a process pool."""
    if n < 1:
        raise ValueError("n must be at least 1")
    count_workers = max(1, min(int(workers or 1), n))
    arguments = []
    sampled: list[dict[str, float | None]] = []
    for index in range(n):
        item_seed = seed + index
        params = sample(task, np.random.default_rng(item_seed))
        sampled.append(params)
        arguments.append((task, Path(controller_path), item_seed, params, Path(root)))
    if count_workers == 1:
        results = [_run_one(args) for args in arguments]
    else:
        with ProcessPoolExecutor(max_workers=count_workers) as pool:
            results = list(pool.map(_run_one, arguments))
    passes = np.asarray([bool(result["passed"]) for result in results], dtype=np.float64)
    by_param: dict[str, dict[str, float | None]] = {}
    for name in sorted({key for params in sampled for key in params}):
        vals = np.asarray([params.get(name) for params in sampled], dtype=object)
        numeric = [(float(value), float(passed)) for value, passed in zip(vals, passes, strict=True)
                   if value is not None and isinstance(value, (int, float))]
        correlation = None
        if len(numeric) >= 2:
            x, y = np.asarray(numeric, dtype=np.float64).T
            if np.std(x) > 0 and np.std(y) > 0:
                value = float(np.corrcoef(x, y)[0, 1])
                correlation = value if np.isfinite(value) else None
        by_param[name] = {"pass_corr": correlation}
    ordered = sorted(results, key=lambda item: (
        float("inf") if item["robustness"] is None else float(item["robustness"]), item["episode"]))
    return {"n": n, "passed": int(np.sum(passes)), "pass_rate": float(np.mean(passes)),
            "worst": [{"episode": item["episode"], "robustness": item["robustness"],
                       "perturb": item["perturb"]} for item in ordered[:5]],
            "by_param": by_param}


def _run_one(args: tuple[Task, Path, int, dict[str, float | None], Path]) -> dict[str, Any]:
    task, controller_path, seed, params, root = args
    result = run_episode(task, controller_path, seed=seed, perturb=params, root=root)
    return {"episode": result.verdict["episode"], "passed": result.verdict["passed"],
            "robustness": result.verdict["robustness"], "perturb": params}
