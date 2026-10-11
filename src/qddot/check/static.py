"""Static diagnostics for MuJoCo models."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


def _name(model: mujoco.MjModel, kind: str, index: int) -> str:
    value = mujoco.mj_id2name(model, getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}"), index)
    return value or f"{kind}_{index}"


def _issue(severity: str, code: str, message: str, element: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message, "element": element}


def _geom_lowest_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        extent = float(size[0])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        extent = float(np.abs(rotation[2]) @ size[:3])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        extent = float(np.linalg.norm(rotation[2] * size[:3]))
    elif geom_type in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        radial = float(size[0]) * float(np.linalg.norm(rotation[2, :2]))
        extent = radial + float(size[1]) * abs(float(rotation[2, 2]))
    else:
        extent = float(np.abs(rotation[2]) @ size[:3])
    return float(data.geom_xpos[geom_id, 2]) - extent


def _model_issues(model: mujoco.MjModel) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    for body_id in range(1, model.nbody):
        body = _name(model, "body", body_id)
        mass = float(model.body_mass[body_id])
        inertia = np.asarray(model.body_inertia[body_id], dtype=float)
        if mass <= 1e-6:
            issues.append(_issue("warning", "body_mass", f"body mass is {mass:g} kg; assign mass above 1e-6 kg", body))
        if np.any(inertia < 0) or max(inertia) > sum(inertia) - max(inertia) + 1e-12:
            issues.append(_issue("error", "invalid_inertia", f"principal inertia {inertia.tolist()} violates the triangle inequality", body))

    position_joints: list[tuple[int, int]] = []
    for actuator_id in range(model.nu):
        trn_type = int(model.actuator_trntype[actuator_id])
        if trn_type != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        if not model.jnt_limited[joint_id] and int(model.jnt_type[joint_id]) in (
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        ):
            issues.append(_issue("warning", "unlimited_actuated_joint", f"actuated joint '{_name(model, 'joint', joint_id)}' has no range; add a limited joint range", _name(model, "joint", joint_id)))
        bias_type = int(model.actuator_biastype[actuator_id])
        kp = float(model.actuator_gainprm[actuator_id, 0])
        bias = float(model.actuator_biasprm[actuator_id, 1])
        is_position = bias_type == int(mujoco.mjtBias.mjBIAS_AFFINE) and np.isclose(kp, -bias) and kp != 0
        if is_position:
            position_joints.append((actuator_id, joint_id))

    for actuator_id, joint_id in position_joints:
        if not model.actuator_ctrllimited[actuator_id] or not model.jnt_limited[joint_id]:
            continue
        ctrl_low, ctrl_high = model.actuator_ctrlrange[actuator_id]
        joint_low, joint_high = model.jnt_range[joint_id]
        if ctrl_low < joint_low - 1e-5 or ctrl_high > joint_high + 1e-5:
            actuator = _name(model, "actuator", actuator_id)
            joint = _name(model, "joint", joint_id)
            issues.append(_issue("warning", "ctrlrange_outside_joint", f"position actuator ctrlrange [{ctrl_low:g}, {ctrl_high:g}] exceeds joint '{joint}' range [{joint_low:g}, {joint_high:g}]; align the ranges", actuator))

    if model.opt.timestep > 0.01:
        issues.append(_issue("warning", "large_timestep", f"physics timestep is {model.opt.timestep:g} s; use 0.01 s or less for stable contact simulation", "option.timestep"))

    geom_dims = {
        int(mujoco.mjtGeom.mjGEOM_SPHERE): 1,
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): 2,
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): 2,
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): 3,
        int(mujoco.mjtGeom.mjGEOM_BOX): 3,
        int(mujoco.mjtGeom.mjGEOM_MESH): 3,
    }
    for geom_id in range(model.ngeom):
        dims = geom_dims.get(int(model.geom_type[geom_id]), 0)
        if dims and np.any(model.geom_size[geom_id, :dims] <= 0):
            geom = _name(model, "geom", geom_id)
            issues.append(_issue("warning", "zero_geom_size", f"geom size {model.geom_size[geom_id, :dims].tolist()} has a zero or negative dimension; assign positive dimensions", geom))

    data = mujoco.MjData(model)
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    penetrating = [
        geom_id
        for i in range(data.ncon)
        if data.contact[i].dist < -0.001
        for geom_id in (int(data.contact[i].geom1), int(data.contact[i].geom2))
    ]
    for geom_id in sorted(set(penetrating)):
        geom = _name(model, "geom", geom_id)
        deepest = min(
            float(data.contact[i].dist)
            for i in range(data.ncon)
            if geom_id in (int(data.contact[i].geom1), int(data.contact[i].geom2))
            and data.contact[i].dist < -0.001
        )
        issues.append(_issue("warning", "initial_penetration", f"initial contact penetrates {abs(deepest) * 1000:.2f} mm; adjust the default pose or geometry", geom))

    plane_heights = [float(data.geom_xpos[i, 2]) for i in range(model.ngeom) if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE)]
    ground_z = max(plane_heights, default=0.0)
    contacted_bodies = {int(model.geom_bodyid[data.contact[i].geom1]) for i in range(data.ncon)} | {int(model.geom_bodyid[data.contact[i].geom2]) for i in range(data.ncon)}
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        if body_id in contacted_bodies:
            continue
        geom_ids = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == body_id]
        if not geom_ids:
            continue
        lowest = min(_geom_lowest_z(model, data, i) for i in geom_ids)
        if lowest > ground_z + 0.01:
            body = _name(model, "body", body_id)
            issues.append(_issue("info", "free_body_in_air", f"free body lowest geom is {lowest - ground_z:.3f} m above the ground with no contact; check its initial pose", body))

    return issues


def check_model(path: str | Path) -> list[dict[str, str]]:
    """Load a MuJoCo model and return actionable static diagnostics."""
    model_path = Path(path)
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    except (ValueError, RuntimeError, OSError) as exc:
        return [_issue("error", "load_error", f"could not load model '{model_path}': {exc}", str(model_path))]
    return _model_issues(model)
