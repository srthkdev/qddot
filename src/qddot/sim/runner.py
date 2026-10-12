"""Run a controller against a task on a physics backend and persist the episode."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qddot import __version__
from qddot.record.writer import write_episode
from qddot.sim.backend import Backend, load_backend
from qddot.sim.events import EventDetector
from qddot.sim.perturb import resolve
from qddot.task import Task

EVENT_INDICATORS = ("collision", "joint_limit", "saturation", "deadline_miss")


@dataclass
class EpisodeResult:
    episode_dir: Path
    verdict: dict[str, Any]


class Episode:
    """One steppable run of a task: ``observe()`` then ``act()`` each control step, then ``finish()``.

    ``run_episode`` drives it with a controller file; RL environment wrappers
    (Gymnasium, OpenEnv, verifiers) drive it with actions from a policy.
    """

    def __init__(self, task: Task, seed: int = 0, perturb: dict[str, Any] | None = None,
                 name: str = "policy") -> None:
        self.task, self.seed, self.started = task, seed, time.perf_counter()
        self.params = _resolve_params(task, perturb)
        digest = hashlib.sha256(json.dumps(self.params, sort_keys=True, separators=(",", ":")).encode())
        self.id = f"{task.name}-{name}-s{seed}-{digest.hexdigest()[:6]}"
        self.rng = np.random.default_rng(seed)
        self.sim = sim = load_backend(getattr(task, "backend", "mujoco"), str(task.model_path))
        sim.reset(task.init, float(self.params["friction_scale"] or 1.0), float(self.params["mass_scale"] or 1.0))
        self.substeps = round(task.control_dt / sim.timestep)
        if self.substeps < 1 or not np.isclose(self.substeps * sim.timestep, task.control_dt, rtol=0, atol=1e-10):
            raise ValueError(f"control_dt {task.control_dt} must be a multiple of model timestep {sim.timestep}")
        self.steps = max(1, round(task.duration / task.control_dt))
        self.detector = EventDetector()
        self.rows: list[dict[str, float]] = []
        self.history: list[dict[str, float]] = []
        self.errors: list[str] = []
        self.compute_log: list[float] = []
        self.latency_steps = max(0, round(float(self.params["latency_ms"] or 0.0) / (task.control_dt * 1000)))
        self.noise_std = float(self.params["obs_noise_std"] or 0.0)
        self.jitter_std_steps = float(self.params["jitter_ms"] or 0.0) / (sim.timestep * 1000)
        self.budget = self.params["controller_budget_ms"]
        self.last_ctrl = sim.ctrl()
        self.queue: list[tuple[int, np.ndarray]] = []
        self.index = self.physics_count = 0
        self.done = False
        _record(self.rows, self.history, sim, task, self.detector, 0.0, sim.ctrl())

    @property
    def t(self) -> float:
        return self.index * self.task.control_dt

    def info(self) -> dict[str, Any]:
        """What a controller's ``reset(info)`` receives."""
        return {"actuators": [a.name for a in self.sim.actuators], "joints": [j.name for j in self.sim.joints],
                "control_dt": self.task.control_dt, "target": self.task.target,
                "model_path": str(self.task.model_path), "backend": self.sim.name}

    def observe(self) -> dict[str, float]:
        """Start a control step and return the (delayed, noisy) observation."""
        self.detector.begin_control_step()
        obs = dict(self.history[max(0, len(self.history) - 1 - self.latency_steps)])
        if self.noise_std:
            for name in obs:
                obs[name] += float(self.rng.normal(0.0, self.noise_std))
        return obs

    def act(self, command: dict[str, Any], compute_ms: float = 0.0, who: str = "policy") -> None:
        """Apply a command (actuator name -> value) and advance one control step."""
        t, task, sim, detector = self.t, self.task, self.sim, self.detector
        self.compute_log.append(compute_ms)
        missed = self.budget is not None and compute_ms > float(self.budget)
        if missed:
            detector.deadline_miss(t, who, compute_ms)
        else:
            detector.deadline_met()
        self.last_ctrl = _apply_limits(command, self.last_ctrl, sim, detector, t)
        # A late command lands ceil(compute/control_dt) control steps later; jitter shifts it further.
        jitter = round(float(self.rng.normal(0.0, self.jitter_std_steps))) if self.jitter_std_steps else 0
        late = int(np.ceil(compute_ms / (task.control_dt * 1000.0))) if missed else 0
        self.queue.append((self.physics_count + late * self.substeps + max(0, jitter), self.last_ctrl.copy()))
        self.queue.sort(key=lambda item: item[0])
        for _ in range(self.substeps):
            while self.queue and self.queue[0][0] <= self.physics_count:
                sim.set_ctrl(self.queue.pop(0)[1])
            sim.step()
            self.physics_count += 1
            now = self.physics_count * sim.timestep
            detector.detect_physics(sim.contacts(), sim.joints, sim.joint_state()[0], now,
                                    task.contact_pairs, self.index)
            if not sim.finite():
                self.fail("simulation state contains non-finite positions or velocities", "nan_state", "state", now)
                return
        self.index += 1
        _record(self.rows, self.history, sim, task, detector, self.t, sim.ctrl())
        self.done = self.index >= self.steps

    def fail(self, message: str, kind: str = "controller_error", who: str = "", t: float | None = None,
             detail: dict[str, Any] | None = None) -> None:
        """End the episode with an error (controller exception, numerical blow-up)."""
        self.errors.append(message)
        self.detector.add(self.t if t is None else t, kind, who, detail=detail)
        self.done = True

    def verdict(self) -> dict[str, Any]:
        """Evaluate specs on everything recorded so far."""
        verdict = _verdict(self.task, self.rows, self.detector, self.errors, self.params, self.sim)
        verdict.update(episode=self.id, seed=self.seed)
        verdict["stats"] = {"control_steps": len(self.rows) - 1, "physics_steps": self.physics_count,
                            "wall_s": time.perf_counter() - self.started}
        return verdict

    def finish(self, root: Path = Path(".qddot"), record_mcap: bool = True,
               controller_path: Path | None = None) -> EpisodeResult:
        """Write the episode's artifacts and return its verdict."""
        verdict = self.verdict()
        meta = {"task_path": str(self.task.path), "seed": self.seed, "perturb": self.params,
                "env": verdict["env"], "compute_ms": [round(v, 6) for v in self.compute_log]}
        if controller_path is not None:
            verdict["controller"] = str(controller_path)
            meta.update(controller_path=str(controller_path), controller_sha256=_sha256(controller_path))
        episode_dir = Path(root) / "runs" / self.id
        write_episode(episode_dir, self.rows, self.detector.emitted, verdict, meta, record_mcap)
        return EpisodeResult(episode_dir=episode_dir, verdict=verdict)


def run_episode(task: Task, controller_path: Path, seed: int = 0,
                perturb: dict[str, Any] | None = None, root: Path = Path(".qddot"),
                record_mcap: bool = True,
                on_step: Callable[[int, float, Any, Any], None] | None = None,
                compute_ms_override: Sequence[float] | None = None) -> EpisodeResult:
    """Run a controller file through one deterministic episode and write its artifacts.

    ``on_step(control_index, t, *backend.native())`` is called after the initial
    state (index -1) and after every control step, for re-simulation tools.
    ``compute_ms_override`` replays recorded controller compute times so deadline
    effects are reproduced exactly instead of re-measured.
    """
    controller_path = Path(controller_path).resolve()
    episode = Episode(task, seed, perturb, name=controller_path.stem)
    controller = _load_controller(controller_path)
    if hasattr(controller, "reset"):
        controller.reset(episode.info())
    if on_step is not None:
        on_step(-1, 0.0, *episode.sim.native())
    while not episode.done:
        index, obs = episode.index, episode.observe()
        started = time.perf_counter()
        try:
            command = controller.step(obs, episode.t)
            if not isinstance(command, dict):
                raise TypeError("Controller.step must return a mapping of actuator names to values")
        except Exception:  # noqa: BLE001 - controller failures are episode results.
            trace = "\n".join(traceback.format_exc().splitlines()[-20:])
            episode.fail(trace, who=controller_path.name, detail={"traceback": trace})
            break
        compute_ms = (time.perf_counter() - started) * 1000.0
        if compute_ms_override is not None and index < len(compute_ms_override):
            compute_ms = float(compute_ms_override[index])
        episode.act(command, compute_ms, who=controller_path.name)
        if on_step is not None and episode.index > index:
            on_step(index, episode.t, *episode.sim.native())
    return episode.finish(root, record_mcap, controller_path)


def _resolve_params(task: Task, perturb: dict[str, Any] | None) -> dict[str, Any]:
    params = resolve(perturb)
    # A scalar task budget is an execution constraint, not a sampled perturbation,
    # so it stays active for nominal and partially specified runs.
    if params["controller_budget_ms"] is None and (perturb is None or "controller_budget_ms" not in perturb):
        task_budget = task.perturb.get("controller_budget_ms")
        if task_budget is not None and not isinstance(task_budget, (list, tuple)):
            params["controller_budget_ms"] = float(task_budget)
    return params


def _apply_limits(output: dict[str, Any], last: np.ndarray, sim: Backend,
                  detector: EventDetector, t: float) -> np.ndarray:
    """Merge a controller's output into the held command, clipping to actuator ranges."""
    proposed = last.copy()
    index = {act.name: i for i, act in enumerate(sim.actuators)}
    for name, value in output.items():
        if name not in index:
            continue
        aid, act, raw = index[name], sim.actuators[index[name]], float(value)
        clipped = float(np.clip(raw, act.low, act.high)) if act.limited else raw
        detector.saturation(t, aid, name, raw, clipped)
        proposed[aid] = clipped
    return proposed


def _record(rows: list[dict[str, float]], history: list[dict[str, float]], sim: Backend, task: Task,
            detector: EventDetector, t: float, controls: np.ndarray) -> None:
    """Append one signal row (all signals) and one observation (what the controller may see)."""
    obs: dict[str, float] = {"t": float(t)}
    positions, velocities = sim.joint_state()
    for joint, pos, vel in zip(sim.joints, positions, velocities, strict=True):
        obs[f"joint.{joint.name}.pos"], obs[f"joint.{joint.name}.vel"] = float(pos), float(vel)
    forces = sim.actuator_force()
    for aid, act in enumerate(sim.actuators):
        obs[f"act.{act.name}.ctrl"], obs[f"act.{act.name}.force"] = float(controls[aid]), float(forces[aid])
    for signal in task.signals:
        obs[signal["name"]] = _derived_signal(sim, signal, task.target)
    row = dict(obs)
    active = list(detector.active_contacts.values())
    for pair in task.contact_pairs:
        names_a, names_b = set(pair["a"]), set(pair["b"])
        row[f"col_{pair['name']}"] = float(any(
            (c.a in names_a and c.b in names_b) or (c.b in names_a and c.a in names_b) for c in active))
    for kind in EVENT_INDICATORS:
        row[f"ev.{kind}"] = float(kind in detector.step_kinds)
    rows.append(row)
    history.append(obs)


def _derived_signal(sim: Backend, signal: dict[str, Any], target: dict[str, Any]) -> float:
    kind, ref = signal["kind"], signal.get("ref")
    if kind in ("site_pos", "body_pos"):
        return float(sim.position(ref, kind.removesuffix("_pos"))["xyz".index(signal.get("axis", "x"))])
    a = sim.position(ref)
    to = signal.get("to")
    b = (np.asarray([target.get(axis, 0.0) for axis in "xyz"], dtype=np.float64)
         if to == "target" else sim.position(to))
    return float(np.linalg.norm(a - b))


def _verdict(task: Task, rows: list[dict[str, float]], detector: EventDetector, errors: list[str],
             params: dict[str, Any], sim: Backend) -> dict[str, Any]:
    times = np.asarray([row["t"] for row in rows], dtype=np.float64)
    signals = {name: np.asarray([row[name] for row in rows], dtype=np.float64)
               for name in rows[0] if name != "t"}
    specs: list[dict[str, Any]] = []
    if task.specs:
        from qddot.spec import evaluate_specs

        specs = evaluate_specs(task.specs, times, signals)
    scores = [float(s["robustness"]) for s in specs if s.get("robustness") is not None]
    failing = [s for s in specs if not s.get("passed", True)]
    summary: dict[str, int] = {}
    for event in detector.emitted:
        summary[event["kind"]] = summary.get(event["kind"], 0) + 1
    return {
        "task": task.name,
        "passed": not errors and all(s.get("passed", False) for s in specs),
        "robustness": min(scores) if scores else None,
        "specs": specs,
        # Specs are declared in diagnostic priority order, so the first failing one leads.
        "first_violation": {"spec": failing[0].get("name"), "t": failing[0].get("witness_t")} if failing else None,
        "events_summary": summary, "errors": errors, "perturb": params,
        "env": {sim.name: sim.version, "python": platform.python_version(),
                "platform": platform.platform(), "qddot": __version__},
    }


def _load_controller(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"qddot_controller_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load controller: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Controller()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
