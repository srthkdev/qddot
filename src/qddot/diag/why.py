"""Rule based diagnosis for recorded episodes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from qddot.spec import SpecError, to_english

from ._data import finite_values, parse_stl_signals, read_json, read_table

_RULE_CATEGORY = {
    "deadline": "deadline", "joint_limit": "joint_limit", "divergence": "sign_or_frame",
    "oscillation": "gain_instability", "unit_scale": "units", "units": "units",
    "mapping": "wrong_mapping", "wrong_mapping": "wrong_mapping", "latency": "latency_sensitivity",
    "latency_sensitivity": "latency_sensitivity", "slip": "lost_contact", "contact_end": "lost_contact",
    "lost_contact": "lost_contact", "saturation": "saturation", "windup": "windup",
    "missing_compensation": "missing_compensation", "wrong_equilibrium": "wrong_equilibrium",
    "model_error": "model_error",
}


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
    targets = {match.group(1): float(match.group(2)) for match in pattern.finditer(stl)}
    for match in re.finditer(r"abs\(\s*(joint\.[\w.]+\.pos)\s*\)", stl):
        targets.setdefault(match.group(1), 0.0)
    for match in re.finditer(r"(joint\.[\w.]+\.pos)\s*(?:<|<=|>|>=)\s*(-?\d+(?:\.\d+)?)", stl):
        targets.setdefault(match.group(1), float(match.group(2)))
    return targets


def _causes(spec: dict[str, Any], signal_rows: list[dict[str, Any]],
            events: list[dict[str, Any]], perturb: dict[str, Any], witness: float,
            task_specs: list[dict[str, Any]], nominal_rows: list[dict[str, Any]] | None = None,
            nominal_verdict: dict[str, Any] | None = None,
            perturb_ranges: dict[str, Any] | None = None,
            perturb_correlations: dict[str, float | None] | None = None) -> list[dict[str, Any]]:
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
            causes.append({"rule": rule, "category": _RULE_CATEGORY[rule], "cause": title,
                           "evidence": evidence, "confidence": confidence})

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
    if constants and tracking:
        near_zero = [item for item in tracking if abs(item[1]) > 0.5 and abs(item[2]) < 0.15]
        # Current position close to zero while requested target is far from zero is
        # a scale/unit symptom when no event explains the poor tracking.
        if len(near_zero) >= min(2, len(constants)) and not collisions:
            add("units", "command units or scale do not match the joint signals",
                [f"{len(near_zero)} joints remain near zero despite nonzero targets"], "high")

        # A persistent position residual with little motion indicates a load
        # bias or missing feedforward term rather than an unstable transient.
        steady = [row for row in signal_rows if float(row["t"]) >= float(all_times[-1]) * 0.75]
        for name, target, current, error, _movement in tracking:
            velocity_name = name[:-4] + ".vel" if name.endswith(".pos") else ""
            if len(steady) < 4 or not velocity_name or velocity_name not in steady[0]:
                continue
            positions = np.asarray([float(row[name]) for row in steady])
            velocities = np.asarray([float(row[velocity_name]) for row in steady])
            offset = float(np.mean(positions) - target)
            if abs(offset) > 0.08 and float(np.sqrt(np.mean(velocities ** 2))) < 0.03:
                add("missing_compensation", "steady position offset with low velocity suggests an uncompensated load",
                    [f"{name} holds {offset:+.3f} from target with low steady velocity"], "med")
                break

    # Estimate monotonic divergence in every referenced non-event signal.
    divergent: list[str] = []
    oscillating: list[str] = []
    dynamics_names = list(dict.fromkeys(
        stl_signals
        + [name for name in signal_rows[0]
           if (name.startswith("joint.") and name.endswith(".vel"))
           or (name.startswith("act.") and name.endswith(".ctrl"))]
    ))
    for name in dynamics_names:
        if name not in signal_rows[0] or name.startswith(("ev.", "col_")):
            continue
        values = finite_values(signal_rows, name)
        if values.size < 4:
            continue
        end = int(np.argmin(np.abs(all_times - evidence_end)))
        sample = values[:end + 1]
        if sample.size < 4:
            continue
        target = constants.get(name) if name.startswith("joint.") else None
        magnitude = np.abs(sample - target) if target is not None else np.abs(sample - sample[0])
        tail = magnitude[-max(2, sample.size // 5):]
        head = magnitude[:max(2, sample.size // 5)]
        if name in stl_signals and float(np.median(tail)) > float(np.median(head)) + 0.03:
            steps = np.diff(magnitude)
            if np.mean(steps >= -max(float(np.ptp(magnitude)) * 0.03, 1e-5)) > 0.55:
                divergent.append(name)
        if _trend(all_times, values, witness) == "oscillating" and float(np.max(tail)) > float(np.max(head)) + 0.03:
            oscillating.append(name)
    if not divergent and collisions:
        predicate = re.search(r"([A-Za-z_][\w.]*)\s*>=?\s*(-?\d+(?:\.\d+)?)", str(spec.get("stl", "")))
        if predicate and predicate.group(1) in signal_rows[0]:
            name, bound = predicate.group(1), float(predicate.group(2))
            values = finite_values(signal_rows, name)
            at_witness = float(values[np.argmin(np.abs(all_times - witness))])
            joint_motion = 0.0
            for column in signal_rows[0]:
                if column.startswith("joint.") and column.endswith(".pos"):
                    positions = finite_values(signal_rows, column)
                    joint_motion = max(joint_motion, float(np.ptp(positions)))
            if values.size and float(np.ptp(values)) < 0.03 and at_witness < bound and joint_motion > 0.2:
                add("divergence", "the controlled mechanism moves while the task object stays short of its goal",
                    [f"{name} remains near {at_witness:.3f} below the required {bound:g} despite joint motion"], "med")
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
    zero_equilibrium_signals = [name for name in stl_signals if name in constants
                                and constants[name] == 0.0 and name.startswith("joint.")]
    if (zero_equilibrium_signals and witness >= 0.8 and not saturation
            and all(abs(float(signal_rows[0][name])) <= 0.2
                    and abs(float(signal_rows[-1][name])) >= 0.2
                    for name in zero_equilibrium_signals)):
        add("wrong_equilibrium", "the controller starts near the specified equilibrium but drifts away over time",
            [f"zero-centered joint signal(s) drift beyond 0.2 rad: {', '.join(zero_equilibrium_signals[:3])}"], "med")
    if divergent and not deadline and not (limit and limit_is_target):
        title = "tracking error grows from the start; check sign or coordinate frame"
        divergent = list(dict.fromkeys(divergent))
        evidence = [f"diverging signal(s): {', '.join(divergent[:3])}"]
        if saturation:
            evidence.append(f"saturation began at {float(saturation[0]['t']):.3f}s")
        add("divergence", title, evidence, "high" if len(divergent) >= 2 else "med")
    if oscillating:
        add("oscillation", "growing oscillation suggests excessive gain or instability",
            [f"growing oscillation in {', '.join(oscillating[:3])}"], "med")

    causes = list({item["rule"]: item for item in causes}.values())
    seen_evidence: set[str] = set()
    for cause in causes:
        cause["evidence"] = [line for line in cause["evidence"]
                             if not (line in seen_evidence or seen_evidence.add(line))]

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

    actuator_names = [name[4:-5] for name in signal_rows[0]
                      if name.startswith("act.") and name.endswith(".ctrl")]
    joint_names = [name[6:-4] for name in signal_rows[0]
                   if name.startswith("joint.") and name.endswith(".vel")]
    cross_matches = []
    for actuator in actuator_names:
        ctrl = finite_values(signal_rows, f"act.{actuator}.ctrl")
        if ctrl.size < 8 or np.std(ctrl) < 1e-9:
            continue
        correlations = []
        for joint in joint_names:
            velocity = finite_values(signal_rows, f"joint.{joint}.vel")
            if velocity.size != ctrl.size or np.std(velocity) < 1e-9:
                continue
            corr = float(np.corrcoef(ctrl, velocity)[0, 1])
            if np.isfinite(corr):
                correlations.append((joint, abs(corr)))
        normalize = lambda name: re.sub(r"(?:_?joint|_?motor|_?drive|_?actuator)$", "", name.lower())
        own_joint = normalize(actuator)
        own = next((score for joint, score in correlations if normalize(joint) == own_joint), None)
        if own is None:
            continue
        other = max((score for joint, score in correlations if joint != actuator), default=0.0)
        if other > 0.72 and other > own + 0.2:
            cross_matches.append((actuator, max(correlations, key=lambda item: item[1])[0]))
    if len(cross_matches) >= 2:
        add("wrong_mapping", "actuator commands correlate more strongly with other joint motion",
            [f"off-diagonal command/velocity correlation for {', '.join(a for a, _ in cross_matches[:3])}"], "high")
    for actuator in actuator_names:
        target = constants.get(actuator)
        if target is None or actuator not in joint_names:
            continue
        ctrl = finite_values(signal_rows, f"act.{actuator}.ctrl")
        if not ctrl.size:
            continue
        ratio = abs(target) / max(float(np.median(np.abs(ctrl))), 1e-9)
        if any(abs(ratio / scale - 1.0) < 0.12 for scale in (57.2958, 1000.0)):
            add("units", "command magnitude suggests a degree/radian or millisecond/second mismatch",
                [f"{actuator} target-to-command ratio is about {ratio:.1f}"], "high")
            break

    sat_times = [float(item.get("t", 0.0)) for item in saturation]
    if len(saturation) == 1 and sat_times and constants:
        after_sat = [row for row in signal_rows if float(row["t"]) > max(sat_times)]
        for name, target in constants.items():
            tail_start = int(len(after_sat) * 0.75)
            tail = after_sat[tail_start:]
            if len(tail) < 4 or name not in tail[0]:
                continue
            values = np.asarray([float(row[name]) for row in tail])
            signed = (values - target) * (values[-1] - target)
            if abs(float(values[-1] - target)) > 0.05 and np.mean(signed > 0) > 0.8:
                add("windup", "position remains driven past its target after actuator saturation",
                    [f"{name} remains {values[-1] - target:+.3f} from target after saturation"], "med")
                break

    if len(saturation) >= 2 and not divergent:
        add("saturation", "actuator commands reached their limits",
            [f"actuator saturation re-entered at {float(saturation[1]['t']):.3f}s"], "high")

    if nominal_rows and nominal_verdict and nominal_verdict.get("passed"):
        latency = float(perturb.get("latency_ms") or 0.0)
        latency_range = (perturb_ranges or {}).get("latency_ms", [0, 0])
        max_latency = float(latency_range[-1]) if isinstance(latency_range, list) else 0.0
        latency_corr = (perturb_correlations or {}).get("latency_ms")
        mass_corr = (perturb_correlations or {}).get("mass_scale")
        if ((latency_corr is not None and latency_corr < -0.5 and max_latency > 0
             and latency >= max_latency * 0.75)
                or (max_latency > 0 and latency >= max_latency * 0.85)):
            add("latency_sensitivity", "failure occurs with delayed observations after a passing nominal run",
                [f"nominal passed; failing run latency is {latency:g} ms ({latency / max_latency:.0%} of configured maximum)"], "high")
        elif ((mass_corr is not None and mass_corr < -0.35)
              or abs(float(perturb.get("mass_scale", 1.0) or 1.0) - 1.0) > 0.08):
            add("oscillation", "failure under mass scaling suggests insufficient stability margin",
                [f"nominal passed; failing run mass scale is {float(perturb['mass_scale']):g}"], "med")
            add("missing_compensation", "steady tracking offset under changed load suggests missing load compensation",
                [f"nominal passed; failing run mass scale is {float(perturb['mass_scale']):g}"], "low")

    priorities = {"deadline": 0, "latency_sensitivity": 0, "unit_scale": 1, "units": 1,
                  "mapping": 2, "wrong_mapping": 2, "wrong_equilibrium": 2,
                  "missing_compensation": 3, "saturation": 2, "divergence": 3, "windup": 2,
                  "joint_limit": 7, "oscillation": 2, "slip": 6, "contact_end": 6,
                  "nan_state": 0, "latency": 8}
    if "joint_limit" in str(spec.get("stl", "")):
        priorities["joint_limit"] = 0
    return sorted(causes, key=lambda item: priorities.get(item["rule"], 9))


def why(episode_dir: str | Path, nominal_episode_dir: str | Path | None = None,
        perturb_correlations: dict[str, float | None] | None = None) -> dict[str, Any]:
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
    nominal_rows = None
    nominal_verdict = None
    if nominal_episode_dir is not None:
        nominal_dir = Path(nominal_episode_dir)
        _, nominal_rows = read_table(nominal_dir / "signals.parquet")
        nominal_verdict = read_json(nominal_dir / "verdict.json")
    result["likely_causes"] = _causes(failed, signal_rows, events, meta.get("perturb", {}), witness,
                                      task_data.get("specs", []), nominal_rows, nominal_verdict,
                                      task_data.get("perturb", {}), perturb_correlations)
    signal_column = details[0]["signal"] if details else "t"
    signal_sql = '\\"' + signal_column.replace('"', '""') + '\\"'
    result["next_commands"] = [
        f"qddot query \"select t, {signal_sql} from signals where episode = '{verdict.get('episode')}' order by t limit 20\"",
    ]
    if not meta.get("recorded"):  # recorded logs have no simulator state to replay
        result["next_commands"].append(f"qddot replay {directory} --at {witness:.3f}")
    return result
