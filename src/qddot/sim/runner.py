"""Run a controller against a MuJoCo task and persist the episode."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from qddot import __version__
from qddot.record.writer import write_episode
from qddot.sim.events import EventDetector
from qddot.sim.perturb import resolve
from qddot.task import Task


@dataclass
class EpisodeResult:
    episode_dir: Path
    verdict: dict[str, Any]


def run_episode(task: Task, controller_path: Path, seed: int = 0,
                perturb: dict[str, Any] | None = None, root: Path = Path(".qddot"),
                record_mcap: bool = True) -> EpisodeResult:
    """Run one deterministic episode and write its artifacts."""
    started = time.perf_counter()
    controller_path = Path(controller_path).resolve()
    root = Path(root)
    params = resolve(perturb)
    params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
    param_hash = hashlib.sha256(params_json.encode()).hexdigest()[:6]
    episode_id = f"{task.name}-{controller_path.stem}-s{seed}-{param_hash}"
    episode_dir = root / "runs" / episode_id
    rng = np.random.default_rng(seed)
    errors: list[str] = []
    detector = EventDetector()
    model = mujoco.MjModel.from_xml_path(str(task.model_path))
    data = mujoco.MjData(model)
    _apply_initial_state(model, data, task)
    friction_scale = float(params["friction_scale"] or 1.0)
    mass_scale = float(params["mass_scale"] or 1.0)
    model.geom_friction[:, 0] *= friction_scale
    model.body_mass[:] *= mass_scale
    model.body_inertia[:] *= mass_scale
    mujoco.mj_forward(model, data)
    physics_dt = float(model.opt.timestep)
    substeps = round(task.control_dt / physics_dt)
    if substeps < 1 or not np.isclose(substeps * physics_dt, task.control_dt, rtol=0, atol=1e-10):
        raise ValueError(f"control_dt {task.control_dt} must be a multiple of model timestep {physics_dt}")
    control_steps = max(1, round(task.duration / task.control_dt))
    joint_types = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    joint_ids = [i for i in range(model.njnt) if int(model.jnt_type[i]) in joint_types]
    joint_names = [_name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in joint_ids]
    actuator_names = [_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    initial_ctrl = np.array(data.ctrl, dtype=np.float64, copy=True)
    controller = _load_controller(controller_path)
    info = {"actuators": actuator_names, "joints": joint_names, "control_dt": task.control_dt,
            "target": task.target, "model_path": str(task.model_path)}
    if hasattr(controller, "reset"):
        controller.reset(info)

    signal_rows: list[dict[str, float]] = []
    obs_history: list[dict[str, float]] = []
    _record_state(signal_rows, obs_history, model, data, task, detector, 0.0, initial_ctrl)
    latency_steps = max(0, round(float(params["latency_ms"] or 0.0) / (task.control_dt * 1000)))
    noise_std = float(params["obs_noise_std"] or 0.0)
    jitter_std_steps = float(params["jitter_ms"] or 0.0) / (physics_dt * 1000)
    budget = params["controller_budget_ms"]
    last_ctrl = initial_ctrl.copy()
    action_queue: list[tuple[int, np.ndarray]] = []
    physics_count = 0

    for control_index in range(control_steps):
        detector.begin_control_step()
        sample_index = max(0, len(obs_history) - 1 - latency_steps)
        obs = dict(obs_history[sample_index])
        if noise_std:
            for name in obs:
                obs[name] += float(rng.normal(0.0, noise_std))
        call_started = time.perf_counter()
        try:
            output = controller.step(obs, control_index * task.control_dt)
            if not isinstance(output, dict):
                raise TypeError("Controller.step must return a mapping of actuator names to values")
        except Exception:  # noqa: BLE001 - controller failures are episode results.
            error = traceback.format_exc()
            errors.append("\n".join(error.splitlines()[-20:]))
            detector.add(control_index * task.control_dt, "controller_error", controller_path.name,
                         detail={"traceback": errors[-1]})
            break
        compute_ms = (time.perf_counter() - call_started) * 1000.0
        if budget is not None and compute_ms > float(budget):
            detector.add(control_index * task.control_dt, "deadline_miss", controller_path.name,
                         value=compute_ms)

        proposed = last_ctrl.copy()
        for name, value in output.items():
            if name not in actuator_names:
                continue
            aid = actuator_names.index(name)
            raw = float(value)
            low, high = model.actuator_ctrlrange[aid]
            if model.actuator_ctrllimited[aid]:
                clipped = float(np.clip(raw, low, high))
            else:
                clipped = raw
            detector.saturation(control_index * task.control_dt, aid, name, raw, clipped)
            proposed[aid] = clipped
        last_ctrl = proposed
        jitter = round(float(rng.normal(0.0, jitter_std_steps))) if jitter_std_steps else 0
        action_queue.append((physics_count + max(0, jitter), proposed.copy()))
        action_queue.sort(key=lambda item: item[0])

        for substep in range(substeps):
            while action_queue and action_queue[0][0] <= physics_count:
                _, applied = action_queue.pop(0)
                data.ctrl[:] = applied
            mujoco.mj_step(model, data)
            physics_count += 1
            now = physics_count * physics_dt
            detector.detect_physics(model, data, now, task.contact_pairs, control_index)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                detector.add(now, "nan_state", "state")
                errors.append("MuJoCo state contains non-finite qpos or qvel")
                break
        if "nan_state" in detector.step_kinds:
            break
        _record_state(signal_rows, obs_history, model, data, task, detector,
                      (control_index + 1) * task.control_dt, data.ctrl)

    times = np.asarray([row["t"] for row in signal_rows], dtype=np.float64)
    signal_arrays = {name: np.asarray([row[name] for row in signal_rows], dtype=np.float64)
                     for name in signal_rows[0] if name != "t"}
    spec_results: list[dict[str, Any]] = []
    robustness: float | None = None
    if task.specs:
        try:
            from qddot.spec import evaluate_specs
            spec_results = evaluate_specs(task.specs, times, signal_arrays)
            vals = [float(spec["robustness"]) for spec in spec_results if spec.get("robustness") is not None]
            robustness = min(vals) if vals else None
        except ImportError:
            errors.append("spec module unavailable")
    violations = [spec for spec in spec_results if not spec.get("passed", True)]
    first_violation = None
    if violations:
        critical = min(violations, key=lambda item: float(item.get("witness_t") or 0.0))
        first_violation = {"spec": critical.get("name"), "t": critical.get("witness_t")}
    event_summary: dict[str, int] = {}
    for event in detector.emitted:
        event_summary[event["kind"]] = event_summary.get(event["kind"], 0) + 1
    passed = not errors and all(item.get("passed", False) for item in spec_results)
    if not task.specs and not errors:
        passed = True
    elapsed = time.perf_counter() - started
    verdict = {
        "episode": episode_id, "task": task.name, "controller": str(controller_path), "seed": seed,
        "passed": passed, "robustness": robustness, "specs": spec_results,
        "first_violation": first_violation, "events_summary": event_summary, "errors": errors,
        "perturb": params, "env": _environment(),
        "stats": {"control_steps": len(signal_rows) - 1, "physics_steps": physics_count,
                  "wall_s": elapsed},
    }
    meta = {"task_path": str(task.path), "controller_path": str(controller_path),
            "controller_sha256": _sha256(controller_path), "seed": seed,
            "perturb": params, "env": verdict["env"]}
    write_episode(episode_dir, signal_rows, detector.emitted, verdict, meta, record_mcap)
    return EpisodeResult(episode_dir=episode_dir, verdict=verdict)


def _apply_initial_state(model: Any, data: Any, task: Task) -> None:
    if task.init.get("keyframe"):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, task.init["keyframe"])
        if key_id < 0:
            raise ValueError(f"unknown keyframe: {task.init['keyframe']}")
        data.qpos[:] = model.key_qpos[key_id]
        if model.nv and model.nkey and model.key_qvel.shape[1]:
            data.qvel[:] = model.key_qvel[key_id]
        if model.nu and model.nkey and model.key_ctrl.shape[1]:
            data.ctrl[:] = model.key_ctrl[key_id]
        if model.na and model.nkey and model.key_act.shape[1]:
            data.act[:] = model.key_act[key_id]
    for joint_name, value in task.init.get("qpos", {}).items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"unknown joint in init.qpos: {joint_name}")
        joint_types = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        if int(model.jnt_type[jid]) not in joint_types:
            raise ValueError(f"init.qpos supports hinge and slide joints only: {joint_name}")
        data.qpos[model.jnt_qposadr[jid]] = float(value)
    mujoco.mj_forward(model, data)


def _load_controller(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"qddot_controller_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load controller: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Controller()


def _record_state(rows: list[dict[str, float]], history: list[dict[str, float]], model: Any,
                  data: Any, task: Task, detector: EventDetector, t: float,
                  controls: np.ndarray) -> None:
    row: dict[str, float] = {"t": float(t)}
    obs: dict[str, float] = {"t": float(t)}
    for jid in range(model.njnt):
        joint_types = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        if int(model.jnt_type[jid]) not in joint_types:
            continue
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        pos, vel = f"joint.{name}.pos", f"joint.{name}.vel"
        row[pos] = obs[pos] = float(data.qpos[qadr])
        row[vel] = obs[vel] = float(data.qvel[vadr])
    for aid in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        ctrl, force = f"act.{name}.ctrl", f"act.{name}.force"
        row[ctrl] = obs[ctrl] = float(controls[aid])
        row[force] = obs[force] = float(data.actuator_force[aid])
    for signal in task.signals:
        value = _derived_signal(model, data, signal, task.target)
        row[signal["name"]] = obs[signal["name"]] = value
    for pair in task.contact_pairs:
        names_a, names_b = set(pair["a"]), set(pair["b"])
        active = any((item["a"] in names_a and item["b"] in names_b)
                     or (item["b"] in names_a and item["a"] in names_b)
                     for item in detector.active_contacts.values())
        row[f"col_{pair['name']}"] = float(active)
    for kind in ("collision", "joint_limit", "saturation", "deadline_miss"):
        row[f"ev.{kind}"] = float(kind in detector.step_kinds)
    rows.append(row)
    history.append(obs)


def _derived_signal(model: Any, data: Any, signal: dict[str, Any], target: dict[str, Any]) -> float:
    kind, ref = signal["kind"], signal.get("ref")
    if kind == "site_pos":
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ref)
        if index < 0:
            raise ValueError(f"unknown site signal reference: {ref}")
        return float(data.site_xpos[index, "xyz".index(signal.get("axis", "x"))])
    if kind == "body_pos":
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ref)
        if index < 0:
            raise ValueError(f"unknown body signal reference: {ref}")
        return float(data.xpos[index, "xyz".index(signal.get("axis", "x"))])
    a = _position(model, data, ref)
    dest = signal.get("to")
    b = np.asarray([target.get(axis, 0.0) for axis in ("x", "y", "z")], dtype=np.float64) if dest == "target" else _position(model, data, dest)
    return float(np.linalg.norm(a - b))


def _position(model: Any, data: Any, name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id >= 0:
        return data.site_xpos[site_id]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id >= 0:
        return data.xpos[body_id]
    raise ValueError(f"unknown site or body reference: {name}")


def _name(model: Any, objtype: Any, index: int) -> str:
    if objtype == mujoco.mjtObj.mjOBJ_BODY and index == 0:
        return "world"
    return mujoco.mj_id2name(model, objtype, index) or f"element_{index}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment() -> dict[str, str]:
    return {"mujoco": mujoco.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "qddot": __version__}
