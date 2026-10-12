"""Deterministic replay and keyframe rendering."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import mujoco
import numpy as np

from qddot.sim.runner import run_episode
from qddot.task import load_task

from ._data import read_json


def _resimulate(episode_dir: Path, times: list[float]) -> tuple[Path, dict[float, dict[str, Any]]]:
    """Replay recorded applied controls without importing or timing the controller.

    Applied controls reproduce deadline and observation delays at control-step
    boundaries. Physics is near-bitwise when controls only change on those
    boundaries; sub-step actuation jitter is absent from the sampled log and can
    leave small state differences.
    """
    meta = read_json(episode_dir / "meta.json")
    task = load_task(meta["task_path"])
    signal_file = episode_dir / "signals.parquet"
    connection = duckdb.connect(database=":memory:")
    try:
        cursor = connection.execute("SELECT * FROM read_parquet(?) ORDER BY t", [str(signal_file)])
        columns = [column[0] for column in cursor.description or []]
        rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()
    ctrl_names = [name for name in columns if name.startswith("act.") and name.endswith(".ctrl")]
    control_rows = [{name: float(row[name]) for name in ctrl_names} for row in rows]
    temporary_controller = '''\
import json
from pathlib import Path

class Controller:
    def reset(self, info):
        self.rows = json.loads(Path(__file__).with_suffix(".json").read_text())
        self.index = 0
        self.names = info["actuators"]

    def step(self, obs, t):
        self.index += 1
        row = self.rows[min(self.index, len(self.rows) - 1)]
        return {name: row["act." + name + ".ctrl"] for name in self.names}
'''
    captures: dict[float, dict[str, Any]] = {}
    targets = {max(0, round(max(0.0, float(value)) / task.control_dt)) for value in times}
    body_points: list[np.ndarray] = []

    def capture(_index: int, current: float, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        body_points.append(np.array(data.xpos[1:model.nbody], copy=True))
        sample = round(current / task.control_dt)
        if sample in targets and sample * task.control_dt not in captures:
            positions = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "world":
                         np.array(data.xpos[i], copy=True) for i in range(1, model.nbody)}
            rotations = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "world":
                         np.array(data.xmat[i].reshape(3, 3), copy=True) for i in range(1, model.nbody)}
            contacts = []
            for index in range(data.ncon):
                contact = data.contact[index]
                first = int(model.geom_bodyid[contact.geom1])
                second = int(model.geom_bodyid[contact.geom2])
                contacts.append([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, first) or "world",
                                 mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, second) or "world"])
            captures[sample * task.control_dt] = {"t": current, "positions": positions, "rotations": rotations,
                                        "contacts": contacts,
                                        "state": np.array(data.qpos, copy=True),
                                        "qvel": np.array(data.qvel, copy=True),
                                        "integration": _integration_state(model, data),
                                        "model": model, "data": data}

    with tempfile.TemporaryDirectory(prefix="qddot-resim-") as tmp:
        tmp_path = Path(tmp)
        controller_path = tmp_path / "recorded_controls.py"
        controller_path.write_text(temporary_controller, encoding="utf-8")
        controller_path.with_suffix(".json").write_text(json.dumps(control_rows), encoding="utf-8")
        recorded_perturb = meta.get("perturb") or {}
        playback_perturb = {
            "friction_scale": recorded_perturb.get("friction_scale", 1.0),
            "mass_scale": recorded_perturb.get("mass_scale", 1.0),
            "obs_noise_std": 0.0,
            "latency_ms": 0.0,
            "jitter_ms": 0.0,
            "controller_budget_ms": None,
        }
        result = run_episode(task, controller_path, seed=int(meta["seed"]),
                             perturb=playback_perturb, root=tmp_path / "replay", record_mcap=False,
                             on_step=capture)
    if body_points:
        points = np.concatenate(body_points, axis=0)
        bounds = (points.min(axis=0), points.max(axis=0))
        for capture_data in captures.values():
            capture_data["bounds"] = bounds
    # Compare recorded joint state at captured times to the control-driven replay.
    for capture_data in captures.values():
        nearest = min(rows, key=lambda row: abs(float(row["t"]) - capture_data["t"]))
        errors = []
        model = capture_data["model"]
        qpos_values = capture_data["state"]
        qvel_values = capture_data["qvel"]
        for joint_id in range(model.njnt):
            if int(model.jnt_type[joint_id]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qpos, qvel = int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
            errors.extend((abs(float(qpos_values[qpos]) - float(nearest[f"joint.{name}.pos"])),
                           abs(float(qvel_values[qvel]) - float(nearest[f"joint.{name}.vel"]))))
        capture_data["max_state_error"] = max(errors, default=0.0)
    return result.episode_dir, captures


def _integration_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION), dtype=np.float64)
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return state


def _restore_capture(capture: dict[str, Any]) -> None:
    model, data = capture["model"], capture["data"]
    mujoco.mj_setState(model, data, capture["integration"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)


def _render(model: mujoco.MjModel, data: mujoco.MjData, output: Path,
            width: int = 320, height: int = 240,
            bounds: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    points = data.xpos[1:model.nbody]
    low, high = bounds if bounds is not None else (points.min(axis=0), points.max(axis=0))
    camera.lookat[:] = (low + high) * 0.5
    span = max(float(np.max(high - low)), 0.2)
    camera.distance = max(0.4, span * 1.6)
    camera.azimuth = 135
    camera.elevation = -20
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        from PIL import Image

        Image.fromarray(renderer.render()).save(output)


def keyframes(episode_dir: str | Path, times: list[float]) -> list[Path]:
    """Replay an episode and render requested times to its keyframes folder."""
    directory = Path(episode_dir)
    if not times:
        return []
    _, captures = _resimulate(directory, times)
    output_dir = directory / "keyframes"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for requested, capture in sorted(captures.items()):
        path = output_dir / f"t_{round(requested * 1000)}.png"
        _restore_capture(capture)
        try:
            _render(capture["model"], capture["data"], path, bounds=capture.get("bounds"))
        except Exception:  # noqa: BLE001 - Headless CoreGraphics/OpenGL needs a portable fallback.
            from qddot.view import _skeleton_frame

            _skeleton_frame(capture["model"], capture["data"], capture["t"]).save(path)
        outputs.append(path)
    return outputs


def visual_diff(episode_dir: str | Path, times: list[float]) -> list[dict[str, Any]]:
    """Compare each replay frame with the first requested frame and t=0."""
    directory = Path(episode_dir)
    _, captures = _resimulate(directory, times)
    if 0.0 not in captures:
        _, zero = _resimulate(Path(episode_dir), [0.0])
        captures.update(zero)
    ordered = [captures[key] for key in sorted(captures)]
    requested_start = None
    if len(times) > 1:
        meta = read_json(directory / "meta.json")
        control_dt = load_task(meta["task_path"]).control_dt
        requested_start = round(max(0.0, float(times[1])) / control_dt) * control_dt
    baseline = captures.get(requested_start) if requested_start is not None else None
    if baseline is None:
        baseline = next((capture for capture in ordered if capture["t"] > 1e-9), None)
    zero = captures.get(0.0)
    result = []
    for after in ordered:
        if baseline is None or after is baseline or after is zero:
            continue
        comparisons = [("window_start", baseline)]
        if zero is not baseline:
            comparisons.append(("t=0", zero))
        for label, before in comparisons:
            if before is None or before is after:
                continue
            moved = []
            rotations = []
            for name in before["positions"].keys() & after["positions"].keys():
                distance = float(np.linalg.norm(after["positions"][name] - before["positions"][name]))
                if distance > 0.01:
                    moved.append({"body": name, "distance_m": distance})
                first, second = before["rotations"][name], after["rotations"][name]
                cosine = float(np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0))
                angle = float(np.degrees(np.arccos(cosine)))
                if angle > 5.0:
                    rotations.append({"body": name, "angle_deg": angle})
            result.append({"from_t": before["t"], "to_t": after["t"], "baseline": label,
                           "moved_bodies": sorted(moved, key=lambda item: -item["distance_m"]),
                           "rotated_bodies": sorted(rotations, key=lambda item: -item["angle_deg"]),
                           "contacts_before": before["contacts"], "contacts_after": after["contacts"],
                           "max_state_error": after.get("max_state_error", 0.0)})
    return result


def replay(episode_dir: str | Path, at: float, frame: bool = False) -> dict[str, Any]:
    """Replay to a time, save MuJoCo integration state, and summarize joints/contacts."""
    directory = Path(episode_dir)
    _, captures = _resimulate(directory, [at])
    if not captures:
        raise ValueError(f"no replay state captured near t={at:g}")
    capture = next(iter(captures.values()))
    model = capture["model"]
    data = capture["data"]
    _restore_capture(capture)
    state_path = directory / f"state_{round(at * 1000)}.npy"
    np.save(state_path, capture["integration"])
    joints = []
    for index in range(model.njnt):
        if int(model.jnt_type[index]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                                int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or f"joint_{index}"
        qpos = int(model.jnt_qposadr[index])
        qvel = int(model.jnt_dofadr[index])
        joints.append({"name": name, "pos": float(data.qpos[qpos]), "vel": float(data.qvel[qvel])})
    result = {"t": float(capture["t"]), "state": str(state_path), "joints": joints,
              "active_contacts": capture["contacts"]}
    if frame:
        frame_path = directory / f"frame_{round(at * 1000)}.png"
        try:
            _render(model, data, frame_path, bounds=capture.get("bounds"))
        except Exception:  # noqa: BLE001 - Headless CoreGraphics/OpenGL needs a portable fallback.
            from qddot.view import _skeleton_frame

            _skeleton_frame(model, data, capture["t"]).save(frame_path)
        result["frame"] = str(frame_path)
    return result
