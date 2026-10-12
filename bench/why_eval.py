"""Measure qddot why's cause categories against documented task failures."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from qddot.diag import why
from qddot.sim.fuzz import fuzz
from qddot.sim.runner import run_episode
from qddot.task import load_task

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TASKS = ROOT / "tasks"
PRIVATE_TASKS = Path(os.environ.get("QDDOT_TASKS_DIR", ROOT / "tasks_private"))
DEFAULT_CACHE = ROOT / ".qddot" / "why_eval"


def _cached_episode(root: Path, expected_id: str | None = None) -> Path | None:
    if expected_id:
        exact = list(root.rglob(expected_id))
        for path in exact:
            if (path / "verdict.json").is_file():
                return path
    else:
        cached = next(root.rglob("verdict.json"), None)
        if cached is not None:
            return cached.parent
    return None


def _run_nominal(task: Any, controller: Path, cache: Path, run_missing: bool) -> Path:
    directory = cache / task.name / controller.stem / "nominal"
    directory.mkdir(parents=True, exist_ok=True)
    cached = _cached_episode(directory)
    if cached:
        return cached
    if not run_missing:
        raise FileNotFoundError(f"no cached nominal run for {task.name}/{controller.stem}")
    result = run_episode(task, controller, seed=0, root=directory, record_mcap=False)
    return result.episode_dir


def _run_fuzz(task: Any, controller: Path, cache: Path, run_missing: bool
              ) -> tuple[Path, dict[str, float | None]]:
    directory = cache / task.name / controller.stem / "fuzz"
    directory.mkdir(parents=True, exist_ok=True)
    cached: list[tuple[float, Path]] = []
    for verdict_path in directory.rglob("verdict.json"):
        try:
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not verdict.get("passed"):
            cached.append((float(verdict.get("robustness", float("inf"))), verdict_path.parent))
    if cached:
        selected = min(cached, key=lambda item: item[0])[1]
        return selected, _failure_correlations(directory)
    if not run_missing:
        raise FileNotFoundError(f"no cached failing fuzz run for {task.name}/{controller.stem}")
    fuzz(task, controller, n=20, workers=min(4, os.cpu_count() or 1), seed=1000,
         root=directory)
    failed = []
    for verdict_path in directory.rglob("verdict.json"):
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        if not verdict.get("passed"):
            failed.append((float(verdict.get("robustness", float("inf"))), verdict_path.parent))
    if not failed:
        raise RuntimeError(f"fuzz found no failing episode for {task.name}/{controller.stem}")
    selected = min(failed, key=lambda item: item[0])[1]
    return selected, _failure_correlations(directory)


def _failure_correlations(directory: Path) -> dict[str, float | None]:
    samples = []
    for verdict_path in directory.rglob("verdict.json"):
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        meta_path = verdict_path.parent / "meta.json"
        if meta_path.is_file():
            perturb = json.loads(meta_path.read_text(encoding="utf-8")).get("perturb", {})
            samples.append((perturb, float(bool(verdict.get("passed")))))
    result: dict[str, float | None] = {}
    for name in sorted({key for perturb, _passed in samples for key in perturb}):
        pairs = [(float(perturb[name]), passed) for perturb, passed in samples
                 if isinstance(perturb.get(name), (float, int))]
        if len(pairs) < 2:
            result[name] = None
            continue
        values = np.asarray(pairs, dtype=np.float64)
        if np.std(values[:, 0]) == 0 or np.std(values[:, 1]) == 0:
            result[name] = None
        else:
            result[name] = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
    return result


def _report(task_root: Path, cache: Path, run_missing: bool) -> dict[str, Any]:
    confusion: Counter[tuple[str, str]] = Counter()
    task_rows: list[dict[str, Any]] = []
    good_failures: list[str] = []
    total = 0
    top1 = 0
    top3 = 0
    for task_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
        expected_path = task_dir / "expected.yaml"
        if not expected_path.is_file():
            continue
        expected = yaml.safe_load(expected_path.read_text(encoding="utf-8")) or {}
        controllers = [(name, info) for name, info in expected.items()
                       if name.startswith("buggy_") and isinstance(info, dict) and "cause" in info]
        if not controllers:
            continue
        task = load_task(task_dir / "task.yaml")
        task_n = task_top1 = task_top3 = 0
        for name, info in controllers:
            controller = task_dir / "controllers" / f"{name}.py"
            nominal = _run_nominal(task, controller, cache, run_missing)
            perturb_correlations = None
            if info.get("first_failing_spec") is None:
                episode, perturb_correlations = _run_fuzz(task, controller, cache, run_missing)
            else:
                episode = nominal
            diagnosis = why(episode, nominal_episode_dir=nominal,
                            perturb_correlations=perturb_correlations)
            categories = [cause.get("category") for cause in diagnosis.get("likely_causes", [])]
            expected_category = str(info["cause"])
            first = categories[0] if categories else "<none>"
            task_n += 1
            task_top1 += first == expected_category
            task_top3 += expected_category in categories[:3]
            total += 1
            top1 += first == expected_category
            top3 += expected_category in categories[:3]
            confusion[(expected_category, first)] += 1
        task_rows.append({"task": task_dir.name, "n": task_n,
                          "top1": task_top1 / task_n, "top3": task_top3 / task_n})
        good = task_dir / "controllers" / "good.py"
        if good.is_file():
            good_task = load_task(task_dir / "task.yaml")
            fixed_model = expected.get("fixed_model")
            good_key = "good_fixed" if fixed_model else "good"
            if fixed_model:
                good_task.model_path = (task_dir / fixed_model).resolve()
            good_cache = cache / task.name / good_key / "nominal"
            good_cache.mkdir(parents=True, exist_ok=True)
            cached_good = _cached_episode(good_cache)
            if cached_good:
                good_nominal = cached_good
            elif run_missing:
                good_nominal = run_episode(good_task, good, seed=0, root=good_cache,
                                           record_mcap=False).episode_dir
            else:
                raise FileNotFoundError(f"no cached nominal run for {task.name}/{good_key}")
            good_diagnosis = why(good_nominal)
            if good_diagnosis["verdict"] != "PASS" or good_diagnosis["likely_causes"]:
                good_failures.append(f"{task_dir.name}: PASS/cause check failed")
    return {"tasks": task_rows, "n": total, "top1": top1 / total if total else 0.0,
            "top3": top3 / total if total else 0.0,
            "confusion": confusion, "good_failures": good_failures}


def evaluate(task_root: Path = PUBLIC_TASKS, cache: Path = DEFAULT_CACHE,
             run_missing: bool = True) -> dict[str, Any]:
    """Evaluate all buggy controllers for a task tree, caching generated episodes."""
    return _report(task_root, cache, run_missing)


def _print_report(label: str, report: dict[str, Any]) -> None:
    print(f"\n{label}")
    print(f"{'task':30} {'n':>4} {'top-1':>8} {'top-3':>8}")
    for row in report["tasks"]:
        print(f"{row['task']:30} {row['n']:4d} {row['top1']:7.1%} {row['top3']:7.1%}")
    print(f"{'OVERALL':30} {report['n']:4d} {report['top1']:7.1%} {report['top3']:7.1%}")
    print("Confusion (expected -> predicted):")
    for (expected, predicted), count in sorted(report["confusion"].items()):
        print(f"  {expected:24} -> {predicted:24} {count}")
    if report["good_failures"]:
        print("Good controller failures: " + "; ".join(report["good_failures"]))
    else:
        print("Good controller checks: all nominal runs PASS with no causes")


def _json_report(report: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in report.items() if key != "confusion"}
    result["confusion"] = {f"{expected} -> {predicted}": count
                           for (expected, predicted), count in report["confusion"].items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", action="store_true", help="evaluate QDDOT_TASKS_DIR")
    parser.add_argument("--cached-only", action="store_true", help="do not generate missing episodes")
    parser.add_argument("--json", action="store_true", help="write the report as JSON")
    args = parser.parse_args()
    if args.private:
        report = evaluate(PRIVATE_TASKS, DEFAULT_CACHE, not args.cached_only)
        _print_report("PRIVATE TASKS", report)
        reports = {"private": report}
    else:
        public_report = evaluate(PUBLIC_TASKS, DEFAULT_CACHE, not args.cached_only)
        _print_report("PUBLIC TASKS", public_report)
        private_report = evaluate(PRIVATE_TASKS, DEFAULT_CACHE, not args.cached_only)
        _print_report("PRIVATE HELD-OUT CHECK", private_report)
        reports = {"public": public_report, "private": private_report}
    if args.json:
        print(json.dumps({key: _json_report(value) for key, value in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
