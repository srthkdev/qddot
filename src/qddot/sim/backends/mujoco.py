"""MuJoCo reference backend."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from qddot.sim.backend import Contact, Limit

_ONE_DOF = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


class MujocoBackend:
    name = "mujoco"
    version = mujoco.__version__

    def __init__(self, model_path: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.timestep = float(self.model.opt.timestep)
        m = self.model
        self._joint_ids = [i for i in range(m.njnt) if int(m.jnt_type[i]) in _ONE_DOF]
        self._qadr = np.array([m.jnt_qposadr[i] for i in self._joint_ids], dtype=int)
        self._vadr = np.array([m.jnt_dofadr[i] for i in self._joint_ids], dtype=int)
        self.joints = [Limit(_name(m, mujoco.mjtObj.mjOBJ_JOINT, i), bool(m.jnt_limited[i]),
                             float(m.jnt_range[i][0]), float(m.jnt_range[i][1]))
                       for i in self._joint_ids]
        self.actuators = [Limit(_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                                bool(m.actuator_ctrllimited[i]),
                                float(m.actuator_ctrlrange[i][0]), float(m.actuator_ctrlrange[i][1]))
                          for i in range(m.nu)]

    def reset(self, init: dict[str, Any], friction_scale: float, mass_scale: float) -> None:
        m, d = self.model, self.data
        if init.get("keyframe"):
            key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, init["keyframe"])
            if key < 0:
                raise ValueError(f"unknown keyframe: {init['keyframe']}")
            d.qpos[:] = m.key_qpos[key]
            if m.nv and m.key_qvel.shape[1]:
                d.qvel[:] = m.key_qvel[key]
            if m.nu and m.key_ctrl.shape[1]:
                d.ctrl[:] = m.key_ctrl[key]
            if m.na and m.key_act.shape[1]:
                d.act[:] = m.key_act[key]
        for joint, value in init.get("qpos", {}).items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if jid < 0:
                raise ValueError(f"unknown joint in init.qpos: {joint}")
            if int(m.jnt_type[jid]) not in _ONE_DOF:
                raise ValueError(f"init.qpos supports hinge and slide joints only: {joint}")
            d.qpos[m.jnt_qposadr[jid]] = float(value)
        mujoco.mj_forward(m, d)
        m.geom_friction[:, 0] *= friction_scale
        m.body_mass[:] *= mass_scale
        m.body_inertia[:] *= mass_scale
        mujoco.mj_forward(m, d)

    def ctrl(self) -> np.ndarray:
        return np.array(self.data.ctrl, dtype=np.float64, copy=True)

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[:] = ctrl

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[self._qadr], self.data.qvel[self._vadr]

    def actuator_force(self) -> np.ndarray:
        return self.data.actuator_force

    def position(self, name: str, kind: str | None = None) -> np.ndarray:
        if kind in (None, "site"):
            site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site >= 0:
                return self.data.site_xpos[site]
            if kind == "site":
                raise ValueError(f"unknown site signal reference: {name}")
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body >= 0:
            return self.data.xpos[body]
        what = "body signal" if kind == "body" else "site or body"
        raise ValueError(f"unknown {what} reference: {name}")

    def contacts(self) -> list[Contact]:
        m, d = self.model, self.data
        found = []
        force = np.zeros(6, dtype=np.float64)
        for index in range(d.ncon):
            contact = d.contact[index]
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            body_a, body_b = int(m.geom_bodyid[geom_a]), int(m.geom_bodyid[geom_b])
            mujoco.mj_contactForce(m, d, index, force)
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            relative = self._body_velocity(body_a) - self._body_velocity(body_b)
            tangential = relative - np.dot(relative, normal) * normal
            found.append(Contact(
                key=tuple(sorted((body_a, body_b))),
                a=self._contact_name(body_a, geom_a), b=self._contact_name(body_b, geom_b),
                geom_a=_name(m, mujoco.mjtObj.mjOBJ_GEOM, geom_a),
                geom_b=_name(m, mujoco.mjtObj.mjOBJ_GEOM, geom_b),
                normal_force=abs(float(force[0])), slip_speed=float(np.linalg.norm(tangential)),
                penetration=max(0.0, -float(contact.dist))))
        return found

    def finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    def native(self) -> tuple[Any, Any]:
        return self.model, self.data

    def _body_velocity(self, body_id: int) -> np.ndarray:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
        return velocity[3:]

    def _contact_name(self, body_id: int, geom_id: int) -> str:
        # World-attached geoms (floor, walls, obstacles) are more useful by geom name.
        if body_id == 0:
            return _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        return _name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def _name(model: Any, objtype: Any, index: int) -> str:
    if objtype == mujoco.mjtObj.mjOBJ_BODY and index == 0:
        return "world"
    return mujoco.mj_id2name(model, objtype, index) or f"{objtype.name.lower()}_{index}"
