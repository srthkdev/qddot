"""Rule based diagnosis for recorded episodes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from qddot.spec import SpecError, to_english

from ._data import finite_values, parse_stl_signals, read_json, read_table


def _trend(times: np.ndarray, values: np.ndarray, witness: float) -> str:
    mask = (times >= max(float(times[0]), witness - 1.0)) & (times <= witness)
    sample = values[mask]
    if sample.size < 3:
        return "flat"
    delta = np.diff(sample)
    scale = max(float(np.ptp(sample)), 1e-9)
    significant = delta[np.abs(delta) > scale * 0.03]
    if significant.size < 2:
        return "flat"
    signs = np.sign(significant)
    changes = int(np.count_nonzero(signs[1:] != signs[:-1]))
    if changes >= max(2, int(signs.size * 0.35)):
        return "oscillating"
    net = float(sample[-1] - sample[0])
    return "rising" if net > scale * 0.05 else "falling" if net < -scale * 0.05 else "flat"


def _spec_details(spec: dict[str, Any], signals: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    stl = str(spec.get("stl", ""))
    names = parse_stl_signals(stl)
    if not signals:
        return [], names
    times = np.asarray([float(row["t"]) for row in signals])
    witness = float(spec.get("witness_t") or 0.0)
    window = spec.get("window") or [witness, witness]
    low, high = map(float, window)
    result = []
    for name in names:
        values = finite_values(signals, name)
        if values.size != len(times):
            continue
        index = int(np.argmin(np.abs(times - witness)))
        selected = values[(times >= low) & (times <= high)]
        if not selected.size:
            selected = values[index:index + 1]
        result.append({"signal": name, "value": float(values[index]),
                       "min": float(np.min(selected)), "max": float(np.max(selected)),
                       "trend": _trend(times, values, witness)})
    return result, names


def _target_constants(stl: str) -> dict[str, float]:
    """Read simple signal-vs-constant targets from atomic comparisons."""
    pattern = re.compile(r"(joint\.[\w.]+\.pos|[A-Za-z_]\w*)\s*-\s*\(?(\-?\d+(?:\.\d+)?)\)?")
    return {match.group(1): float(match.group(2)) for match in pattern.finditer(stl)}


def _causes(spec: dict[str, Any], signal_rows: list[dict[str, Any]],
            events: list[dict[str, Any]], perturb: dict[str, Any], witness: float,
            task_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not signal_rows:
        return []
    all_times = np.asarray([float(row["t"]) for row in signal_rows])
    stl_signals = parse_stl_signals(str(spec.get("stl", "")))
    window_end = float((spec.get("window") or [witness, witness])[1])
    evidence_end = max(witness, window_end)
    before = [event for event in events if float(event.get("t", 0)) <= evidence_end]
    at_or_before = lambda kind: [e for e in before if e.get("kind") == kind]
    causes: list[dict[str, Any]] = []

    def add(rule: str, title: str, evidence: list[str], confidence: str) -> None:
        if evidence:
            causes.append({"rule": rule, "cause": title, "evidence": evidence, "confidence": confidence})

    deadline = at_or_before("deadline_miss")
    saturation = at_or_before("saturation")
    limit = at_or_before("joint_limit")
    collisions = [e for e in before if e.get("kind") in {"collision", "contact_start"}]
    if deadline:
        add("deadline", "controller compute time exceeded its budget",
            [f"deadline_miss first occurred at {float(deadline[0]['t']):.3f}s ({float(deadline[0].get('value') or 0):.2f} ms)"], "high")
    if limit:
        add("joint_limit", "joint range was reached", [f"joint_limit on {limit[0].get('a')} at {float(limit[0]['t']):.3f}s"], "high")

    # Compare referenced signals against constants in the failing atomic predicates.
    constants = {}
    for item in task_specs:
        constants.update(_target_constants(str(item.get("stl", ""))))
    tracking: list[tuple[str, float, float, float, float]] = []
    for name, target in constants.items():
        if name not in signal_rows[0]:
            continue
        values = finite_values(signal_rows, name)
        if values.size != len(all_times):
            continue
        mask = (all_times >= max(float(all_times[0]), witness - 1.0)) & (all_times <= witness)
        if not np.any(mask):
            continue
        current = float(values[np.argmin(np.abs(all_times - witness))])
        error = abs(current - target)
        movement = abs(float(values[np.where(mask)[0][-1]]) - float(values[np.where(mask)[0][0]]))
        tracking.append((name, target, current, error, movement))
    tracked_bad = [item for item in tracking if item[3] > 0.25]
    if len(tracked_bad) >= 2 and collisions:
        add("mapping", "joint commands appear cross-mapped, with contact disturbance",
            [f"{len(tracked_bad)} target joint signals remain more than 0.25 rad from target",
             f"contact/collision began at {float(collisions[0]['t']):.3f}s"], "high")

    if constants and tracking:
        near_zero = [item for item in tracking if abs(item[1]) > 0.5 and abs(item[2]) < 0.15]
        # Current position close to zero while requested target is far from zero is
        # a scale/unit symptom when no event explains the poor tracking.
        if len(near_zero) >= min(2, len(constants)) and not collisions:
            add("unit_scale", "command units or scale do not match the joint signals",
                [f"{len(near_zero)} joints remain near zero despite nonzero targets"], "high")

    # Estimate monotonic divergence in every referenced non-event signal.
    divergent: list[str] = []
    oscillating: list[str] = []
    for name in stl_signals:
        if name not in signal_rows[0] or name.startswith(("ev.", "col_")):
            continue
        values = finite_values(signal_rows, name)
        if values.size < 4:
            continue
        end = int(np.argmin(np.abs(all_times - evidence_end)))
        sample = values[:end + 1]
        if sample.size < 4:
            continue
        magnitude = np.abs(sample - sample[0])
        tail = magnitude[-max(2, sample.size // 5):]
        head = magnitude[:max(2, sample.size // 5)]
        if float(np.median(tail)) > float(np.median(head)) + 0.03:
            steps = np.diff(magnitude)
            if np.mean(steps >= -max(float(np.ptp(magnitude)) * 0.03, 1e-5)) > 0.55:
                divergent.append(name)
        if _trend(all_times, values, witness) == "oscillating" and float(np.max(tail)) > float(np.max(head)) + 0.03:
            oscillating.append(name)
    if saturation:
        target_match = re.search(r"(?:abs\()?([A-Za-z_][\w.]*)\)?\s*<\s*(\d+(?:\.\d+)?)", str(spec.get("stl", "")))
        if target_match:
            signal_name, bound = target_match.group(1), float(target_match.group(2))
            if signal_name in signal_rows[0]:
                values = finite_values(signal_rows, signal_name)
                early = values[all_times <= evidence_end]
                if early.size and float(np.mean(np.abs(early) >= bound)) >= 0.8:
                    divergent.append(signal_name)
    limit_is_target = "joint_limit" in str(spec.get("stl", ""))
    if divergent and not deadline and not (limit and limit_is_target):
        title = "tracking error grows from the start; check sign or coordinate frame"
        evidence = [f"diverging signal(s): {', '.join(divergent[:3])}"]
        if saturation:
            evidence.append(f"saturation began at {float(saturation[0]['t']):.3f}s")
        add("divergence", title, evidence, "high" if len(divergent) >= 2 else "med")
    if oscillating:
        add("oscillation", "growing oscillation suggests excessive gain or instability",
            [f"growing oscillation in {', '.join(oscillating[:3])}"], "med")

    causes = list({item["rule"]: item for item in causes}.values())

    special = []
    for kind, title in (("slip", "object contact slipped before failure"),
                        ("contact_end", "object contact was lost before failure")):
        relevant = [e for e in at_or_before(kind) if "floor" not in str(e.get("a", "")) and "floor" not in str(e.get("b", ""))]
        if relevant:
            special.append((kind, title, relevant))
    for kind, title, rows in special:
        add(kind, title, [f"{kind} involving {rows[0].get('a')} at {float(rows[0]['t']):.3f}s"], "med")
    if any(e.get("kind") == "nan_state" for e in before):
        add("nan_state", "simulation state became non-finite", ["nan_state event recorded"], "high")
    if float(perturb.get("latency_ms") or 0) > 0:
        add("latency", "failure may be sensitive to delayed observations",
            [f"observed latency perturbation: {float(perturb['latency_ms']):g} ms"], "low")

    priorities = {"deadline": 0, "unit_scale": 1, "mapping": 2, "divergence": 3,
                  "joint_limit": 4, "oscillation": 5, "slip": 6, "contact_end": 7,
                  "nan_state": 0, "latency": 8}
    if "joint_limit" in str(spec.get("stl", "")):
        priorities["joint_limit"] = 0
    return sorted(causes, key=lambda item: priorities.get(item["rule"], 9))


def why(episode_dir: str | Path) -> dict[str, Any]:
    """Explain the first failed spec using recorded signals and events."""
    directory = Path(episode_dir)
    verdict = read_json(directory / "verdict.json")
    meta = read_json(directory / "meta.json")
    _, signal_rows = read_table(directory / "signals.parquet")
    _, events = read_table(directory / "events.parquet")
    failed = next((spec for spec in verdict.get("specs", []) if not spec.get("passed")), None)
    result: dict[str, Any] = {"episode": verdict.get("episode"), "passed": bool(verdict.get("passed")),
                              "verdict": "PASS" if verdict.get("passed") else "FAIL",
                              "first_failing_spec": None, "signals": [], "nearby_events": [],
                              "first_events": {}, "likely_causes": [], "next_commands": []}
    if failed is None:
        return result
    witness = float(failed.get("witness_t") or 0.0)
    details, _ = _spec_details(failed, signal_rows)
    stl = str(failed.get("stl", ""))
    try:
        english = to_english(stl)
    except SpecError:
        english = stl
    result["first_failing_spec"] = {"name": failed.get("name"), "stl": stl,
                                    "english": english, "witness_t": witness,
                                    "window": failed.get("window")}
    result["signals"] = details
    nearby = [event for event in events if abs(float(event.get("t", 0)) - witness) <= 0.5]
    nearby.sort(key=lambda event: abs(float(event.get("t", 0)) - witness))
    result["nearby_events"] = nearby[:8]
    first_events: dict[str, dict[str, Any]] = {}
    for event in events:
        kind = str(event.get("kind"))
        if kind in {"saturation", "joint_limit", "slip", "contact_end", "deadline_miss"} and float(event.get("t", 0)) <= witness:
            first_events.setdefault(kind, event)
    result["first_events"] = first_events
    task_data = yaml.safe_load(Path(meta["task_path"]).read_text(encoding="utf-8"))
    result["likely_causes"] = _causes(failed, signal_rows, events, meta.get("perturb", {}), witness,
                                      task_data.get("specs", []))
    signal_column = details[0]["signal"] if details else "t"
    signal_sql = '\\"' + signal_column.replace('"', '""') + '\\"'
    result["next_commands"] = [
        f"qddot query \"select t, {signal_sql} from signals where episode = '{verdict.get('episode')}' order by t limit 20\"",
        f"qddot replay {directory} --at {witness:.3f}",
    ]
    return result
