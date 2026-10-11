"""Deterministic replay and keyframe rendering."""

from __future__ import annotations

import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from qddot.sim.runner import run_episode
from qddot.task import load_task

from ._data import read_json


def _resimulate(episode_dir: Path, times: list[float]) -> tuple[Path, dict[float, dict[str, Any]]]:
    meta = read_json(episode_dir / "meta.json")
    task = load_task(meta["task_path"])
    captures: dict[float, dict[str, Any]] = {}
    targets = sorted({max(0.0, float(value)) for value in times})

    def capture(_index: int, current: float, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if targets and abs(current - targets[0]) <= task.control_dt / 2 + 1e-9:
            positions = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "world":
                         np.array(data.xpos[i], copy=True) for i in range(1, model.nbody)}
            contacts = []
            for index in range(data.ncon):
                contact = data.contact[index]
                first = int(model.geom_bodyid[contact.geom1])
                second = int(model.geom_bodyid[contact.geom2])
                contacts.append([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, first) or "world",
                                 mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, second) or "world"])
            captures[targets.pop(0)] = {"t": current, "positions": positions,
                                        "contacts": contacts,
                                        "state": np.array(data.qpos, copy=True),
                                        "qvel": np.array(data.qvel, copy=True),
                                        "integration": _integration_state(model, data),
                                        "model": model, "data": data}

    with tempfile.TemporaryDirectory(prefix="qddot-resim-") as tmp:
        result = run_episode(task, Path(meta["controller_path"]), seed=int(meta["seed"]),
                             perturb=meta.get("perturb"), root=Path(tmp), record_mcap=False,
                             on_step=capture, compute_ms_override=meta.get("compute_ms"))
        return result.episode_dir, captures


def _integration_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION), dtype=np.float64)
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return state


def _render(model: mujoco.MjModel, data: mujoco.MjData, output: Path,
            width: int = 320, height: int = 240) -> None:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = model.stat.center
    camera.lookat[2] += 0.15
    camera.distance = max(1.0, float(model.stat.extent) * 1.8)
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
        _render(capture["model"], capture["data"], path)
        outputs.append(path)
    return outputs


def visual_diff(episode_dir: str | Path, times: list[float]) -> list[dict[str, Any]]:
    """Return body motion and contact changes between consecutive replay frames."""
    _, captures = _resimulate(Path(episode_dir), times)
    ordered = [captures[key] for key in sorted(captures)]
    result = []
    for before, after in pairwise(ordered):
        moved = []
        for name in before["positions"].keys() & after["positions"].keys():
            distance = float(np.linalg.norm(after["positions"][name] - before["positions"][name]))
            if distance > 0.01:
                moved.append({"body": name, "distance_m": distance})
        result.append({"from_t": before["t"], "to_t": after["t"],
                       "moved_bodies": sorted(moved, key=lambda item: -item["distance_m"]),
                       "contacts_before": before["contacts"], "contacts_after": after["contacts"]})
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
        _render(model, data, frame_path)
        result["frame"] = str(frame_path)
    return result
