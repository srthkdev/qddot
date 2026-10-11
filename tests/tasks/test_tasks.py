"""Standalone task/controller checks using MuJoCo, without qddot.sim imports."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "tasks"


def load_task(name: str) -> tuple[dict, Path]:
    task_path = TASKS / name / "task.yaml"
    task = yaml.safe_load(task_path.read_text())
    model_path = (task_path.parent / task["model"]).resolve()
    task["model_path"] = str(model_path)
    return task, model_path


def load_controller(path: Path):
    spec = importlib.util.spec_from_file_location(f"controller_{path.stem}_{path.parent.parent.name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Controller()


def simulate(task_name: str, controller_name: str) -> dict:
    task, model_path = load_task(task_name)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if task.get("init", {}).get("keyframe"):
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, task["init"]["keyframe"])
        mujoco.mj_resetDataKeyframe(model, data, key)
    for name, value in task.get("init", {}).get("qpos", {}).items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    controller = load_controller(TASKS / task_name / "controllers" / controller_name)
    controller.reset({
        "actuators": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)],
        "joints": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)],
        "control_dt": task["control_dt"], "target": task.get("target", {}), "model_path": str(model_path),
    })
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    site_ids = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i): i for i in range(model.nsite)}
    t_values, states, distances, saturated, deadline_misses, floor_contacts = [], [], [], [], [], []
    nsub = round(task["control_dt"] / model.opt.timestep)
    assert nsub >= 1 and np.isclose(nsub * model.opt.timestep, task["control_dt"])
    steps = round(task["duration"] / task["control_dt"])
    target = task.get("target", {})
    target_xyz = np.array([target.get(axis, 0.0) for axis in "xyz"])
    for step in range(steps + 1):
        mujoco.mj_forward(model, data)
        obs = {}
        for i, name in enumerate(joint_names):
            qadr, dadr = model.jnt_qposadr[i], model.jnt_dofadr[i]
            obs[f"joint.{name}.pos"] = float(data.qpos[qadr])
            obs[f"joint.{name}.vel"] = float(data.qvel[dadr])
        for i, name in enumerate(actuator_names):
            obs[f"act.{name}.ctrl"] = float(data.ctrl[i])
            obs[f"act.{name}.force"] = float(data.actuator_force[i])
        if "gripperframe" in site_ids:
            pos = data.site_xpos[site_ids["gripperframe"]].copy()
            obs.update(ee_x=float(pos[0]), ee_y=float(pos[1]), ee_z=float(pos[2]), dist_ee_target=float(np.linalg.norm(pos - target_xyz)))
            distances.append(float(np.linalg.norm(pos - target_xyz)))
        t_values.append(float(data.time))
        states.append((data.qpos.copy(), data.qvel.copy()))
        ctrl_start = time.perf_counter()
        output = controller.step(obs, float(data.time)) if step < steps else {}
        elapsed_ms = 1000 * (time.perf_counter() - ctrl_start)
        clipped = False
        for i, name in enumerate(actuator_names):
            if name in output:
                value = float(output[name])
                if model.actuator_ctrllimited[i]:
                    bounded = float(np.clip(value, *model.actuator_ctrlrange[i]))
                    clipped |= bounded != value
                    value = bounded
                data.ctrl[i] = value
        saturated.append(clipped)
        deadline_misses.append(elapsed_ms > task.get("perturb", {}).get("controller_budget_ms", float("inf")))
        for _ in range(nsub):
            mujoco.mj_step(model, data)
        has_floor_contact = False
        for ci in range(data.ncon):
            names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) for gid in (data.contact[ci].geom1, data.contact[ci].geom2))
            has_floor_contact |= "floor" in names
        floor_contacts.append(has_floor_contact)
    return {
        "task": task, "model": model, "states": states, "times": np.array(t_values),
        "distances": np.array(distances), "saturated": np.array(saturated),
        "deadline_misses": np.array(deadline_misses), "floor_contacts": np.array(floor_contacts),
    }


def within_one_second(times: np.ndarray, values: np.ndarray, threshold: float, start_limit: float) -> bool:
    for index, start in enumerate(times):
        if start > start_limit:
            break
        end = np.searchsorted(times, start + 1.0, side="left")
        if end < len(times) and np.max(values[index : end + 1]) < threshold:
            return True
    return False


def test_so101_controllers() -> None:
    good = simulate("so101_reach", "good.py")
    assert within_one_second(good["times"], good["distances"], 0.025, 2.5)
    assert not good["floor_contacts"].any()
    bug_jac = simulate("so101_reach", "buggy_jacobian_sign.py")
    assert np.min(bug_jac["distances"]) > 0.025
    assert bug_jac["floor_contacts"].any() and bug_jac["saturated"].any()
    bug_limit = simulate("so101_reach", "buggy_limit_avoidance.py")
    joint_id = mujoco.mj_name2id(bug_limit["model"], mujoco.mjtObj.mjOBJ_JOINT, "shoulder_lift")
    qid = bug_limit["model"].jnt_qposadr[joint_id]
    limit = bug_limit["model"].jnt_range[joint_id, 0]
    assert min(state[0][qid] for state in bug_limit["states"]) <= limit + 0.02


def test_cartpole_controllers() -> None:
    good = simulate("cartpole_balance", "good.py")
    angles = np.array([state[0][1] for state in good["states"]])
    carts = np.array([state[0][0] for state in good["states"]])
    assert within_one_second(good["times"], np.abs(angles), 0.05, 2.0)
    assert np.max(np.abs(carts)) < 0.8
    assert not good["saturated"].any()
    bug_sign = simulate("cartpole_balance", "buggy_velocity_sign.py")
    bug_angles = np.array([state[0][1] for state in bug_sign["states"]])
    assert abs(bug_angles[-1]) > 1.0 and bug_sign["saturated"].any()
    bug_slow = simulate("cartpole_balance", "buggy_slow.py")
    assert bug_slow["deadline_misses"].any()


def test_ur5e_controllers() -> None:
    names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"]
    targets = [-1.20, -1.35, 1.25]
    good = simulate("ur5e_track", "good.py")
    for name, goal in zip(names, targets):
        joint = mujoco.mj_name2id(good["model"], mujoco.mjtObj.mjOBJ_JOINT, name)
        qid = good["model"].jnt_qposadr[joint]
        trace = np.array([state[0][qid] for state in good["states"]])
        assert np.max(np.abs(trace[good["times"] >= 2.0] - goal)) < 0.04
    bug_order = simulate("ur5e_track", "buggy_joint_order.py")
    joint = mujoco.mj_name2id(bug_order["model"], mujoco.mjtObj.mjOBJ_JOINT, "shoulder_lift_joint")
    assert abs(bug_order["states"][-1][0][bug_order["model"].jnt_qposadr[joint]] - targets[1]) > 0.04
    bug_degrees = simulate("ur5e_track", "buggy_degrees.py")
    joint = mujoco.mj_name2id(bug_degrees["model"], mujoco.mjtObj.mjOBJ_JOINT, "elbow_joint")
    assert abs(bug_degrees["states"][-1][0][bug_degrees["model"].jnt_qposadr[joint]] - targets[2]) > 0.04
