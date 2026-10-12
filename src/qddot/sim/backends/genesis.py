"""Genesis rigid-body backend for the MJCF subset supported by qddot.

Genesis handles the rigid-body integration and collision solve. MuJoCo is used
only to read MJCF names and metadata, and to preserve the source file's joint,
site, actuator, and keyframe ordering.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from qddot.sim.backend import Contact, Limit

_INIT_LOCK = threading.Lock()
_INITIALIZED = False
_ONE_DOF = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


def _initialize_genesis() -> Any:
    """Initialize Genesis once; CPU mode is required for reproducible runs."""
    global _INITIALIZED
    import genesis as gs

    with _INIT_LOCK:
        if not _INITIALIZED and not getattr(gs, "_initialized", False):
            # cpuinfo 9.0 omits all fields Genesis 1.4.4 probes on some ARM hosts.
            import genesis.utils.misc as genesis_misc

            info = genesis_misc.cpuinfo.get_cpu_info()
            if not any(info.get(key) for key in ("brand_raw", "hardware_raw", "vendor_id_raw")):
                info["hardware_raw"] = "CPU"
            genesis_misc.cpuinfo.get_cpu_info = lambda: info
            gs.init(backend=gs.cpu, logging_level="error", use_deterministic_algorithms=True)
        _INITIALIZED = True
    return gs


class GenesisBackend:
    """Physics backend backed by Genesis' CPU rigid solver."""

    name = "genesis"

    def __init__(self, model_path: str) -> None:
        self._gs = _initialize_genesis()
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.timestep = float(self.model.opt.timestep)
        self.version = self._gs.__version__
        self._scene = None
        self._entity = None
        self.data = None
        self._build_scene()
        self._ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self._forces = np.zeros(self.model.nu, dtype=np.float64)
        self._joint_ids = [i for i in range(self.model.njnt) if int(self.model.jnt_type[i]) in _ONE_DOF]
        self._qadr = np.array([self.model.jnt_qposadr[i] for i in self._joint_ids], dtype=int)
        self._vadr = np.array([self.model.jnt_dofadr[i] for i in self._joint_ids], dtype=int)
        genesis_joints = {joint.name: joint for joint in self._entity.joints}
        self._joint_dofs = np.array([
            int(genesis_joints[_name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)].dof_start)
            for joint_id in self._joint_ids
        ], dtype=int)
        self.joints = [
            Limit(_name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i), bool(self.model.jnt_limited[i]),
                  float(self.model.jnt_range[i, 0]), float(self.model.jnt_range[i, 1]))
            for i in self._joint_ids
        ]
        self.actuators = [
            Limit(_name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i), bool(self.model.actuator_ctrllimited[i]),
                  float(self.model.actuator_ctrlrange[i, 0]), float(self.model.actuator_ctrlrange[i, 1]))
            for i in range(self.model.nu)
        ]
        self._act_to_dof, self._act_gain, self._act_bias = self._actuator_map()
        self._link_ids = {link.name: link.idx_local for link in self._entity.links}
        self._link_names_by_global = {
            link.idx: link.name for link in self._scene.sim.rigid_solver.links
        }
        self._geom_names_by_global = {
            geom.idx: geom.metadata.get("name", str(geom.idx))
            for geom in self._scene.sim.rigid_solver.geoms
        }
        self._base_masses = np.asarray(self._entity.get_links_mass(), dtype=np.float64).reshape(-1).copy()
        self._base_inertias = np.asarray(self._entity.get_links_inertia(), dtype=np.float64).copy()
        self._base_friction = {
            geom.idx: float(self.model.geom_friction[i, 0])
            for i, geom in enumerate(self._scene.sim.rigid_solver.geoms)
        }
        self._sites = self._read_sites()
        self.reset({}, 1.0, 1.0)

    def _build_scene(self) -> None:
        gs = self._gs
        self._scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=self.timestep, gravity=tuple(self.model.opt.gravity)),
        )
        self._entity = self._scene.add_entity(gs.morphs.MJCF(file=str(self.model_path)))
        # Genesis currently initializes an OpenGL rasterizer even for physics-only
        # scenes. This backend is intentionally headless and never calls render APIs.
        self._scene._visualizer._rasterizer.build = lambda: None
        self._scene.build()

    def _actuator_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dofs, gains, biases = [], [], []
        for aid in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[aid, 0])
            if joint_id not in self._joint_ids:
                raise ValueError(f"Genesis backend supports actuators on hinge/slide joints only: {aid}")
            dofs.append(int(self._joint_dofs[self._joint_ids.index(joint_id)]))
            gear = float(self.model.actuator_gear[aid, 0])
            gains.append(gear * float(self.model.actuator_gainprm[aid, 0]))
            biases.append(gear * np.asarray(self.model.actuator_biasprm[aid, :3], dtype=np.float64))
        return (np.asarray(dofs, dtype=int), np.asarray(gains, dtype=np.float64),
                np.asarray(biases, dtype=np.float64).reshape((-1, 3)))

    def reset(self, init: dict[str, Any], friction_scale: float, mass_scale: float) -> None:
        key_name = init.get("keyframe")
        qpos = np.array(self.model.qpos0, dtype=np.float64, copy=True)
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        if key_name:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
            if key_id < 0:
                raise ValueError(f"unknown keyframe: {key_name}")
            qpos[:] = self.model.key_qpos[key_id]
            if self.model.key_qvel.shape[1]:
                qvel[:] = self.model.key_qvel[key_id]
        for name, value in init.get("qpos", {}).items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0 or int(self.model.jnt_type[joint_id]) not in _ONE_DOF:
                raise ValueError(f"unknown or unsupported joint in init.qpos: {name}")
            qpos[self.model.jnt_qposadr[joint_id]] = float(value)
        self._entity.set_dofs_position(qpos[self._qadr], dofs_idx_local=self._joint_dofs)
        self._entity.set_dofs_velocity(qvel[self._vadr], dofs_idx_local=self._joint_dofs)
        self._entity.set_links_mass(self._base_masses * float(mass_scale))
        self._entity.set_links_inertia(self._base_inertias * float(mass_scale))
        for geom in self._scene.sim.rigid_solver.geoms:
            base = self._base_friction[geom.idx]
            geom.set_friction(base * float(friction_scale))
        self._ctrl[:] = self.model.key_ctrl[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)] if key_name and self.model.key_ctrl.shape[1] else 0.0
        self._forces[:] = 0.0
        for aid, joint_dof in enumerate(self._act_to_dof):
            self._apply_actuator(aid, joint_dof, self._ctrl[aid])

    def ctrl(self) -> np.ndarray:
        return self._ctrl.copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self._ctrl[:] = np.asarray(ctrl, dtype=np.float64)
        for aid, joint_dof in enumerate(self._act_to_dof):
            self._apply_actuator(aid, int(joint_dof), self._ctrl[aid])

    def _apply_actuator(self, aid: int, joint_dof: int, ctrl: float) -> None:
        positions, velocities = self.joint_state()
        joint_idx = int(np.flatnonzero(self._joint_dofs == joint_dof)[0])
        force = (float(ctrl) * self._act_gain[aid] + self._act_bias[aid, 0]
                 + self._act_bias[aid, 1] * positions[joint_idx]
                 + self._act_bias[aid, 2] * velocities[joint_idx])
        self._entity.control_dofs_force(np.array([force], dtype=np.float32), dofs_idx_local=[int(joint_dof)])
        self._forces[aid] = force

    def step(self) -> None:
        self._scene.step()

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(self._entity.get_dofs_position(), dtype=np.float64).reshape(-1)
        vel = np.asarray(self._entity.get_dofs_velocity(), dtype=np.float64).reshape(-1)
        return pos.copy(), vel.copy()

    def actuator_force(self) -> np.ndarray:
        return self._forces.copy()

    def position(self, name: str, kind: str | None = None) -> np.ndarray:
        if kind not in (None, "site", "body"):
            raise ValueError(f"unknown position kind: {kind}")
        if kind != "body" and name in self._sites:
            body_idx, offset = self._sites[name]
            pos, quat = self._link_pose(body_idx)
            return pos + _rotate(quat, offset)
        if name not in self._link_ids:
            what = "body" if kind == "body" else "site or body"
            raise ValueError(f"unknown {what} reference: {name}")
        return self._link_pose(self._link_ids[name])[0]

    def contacts(self) -> list[Contact]:
        raw = self._entity.get_contacts()
        values = {key: _numpy(value) for key, value in raw.items()}
        result: dict[tuple[int, int], Contact] = {}
        for i, (geom_a, geom_b) in enumerate(zip(values["geom_a"], values["geom_b"], strict=True)):
            geom_a, geom_b = int(geom_a), int(geom_b)
            link_a, link_b = int(values["link_a"][i]), int(values["link_b"][i])
            name_a = self._contact_name(link_a, geom_a)
            name_b = self._contact_name(link_b, geom_b)
            key = tuple(sorted((link_a, link_b)))
            force = np.asarray(values["force_a"][i], dtype=np.float64)
            normal = np.asarray(values["normal"][i], dtype=np.float64)
            force_n = abs(float(np.dot(force, normal)))
            pos = np.asarray(values["position"][i], dtype=np.float64)
            va = self._link_velocity(link_a, pos)
            vb = self._link_velocity(link_b, pos)
            relative = va - vb
            slip = float(np.linalg.norm(relative - np.dot(relative, normal) * normal))
            contact = Contact(key, name_a, name_b, self._geom_names_by_global.get(geom_a, str(geom_a)),
                              self._geom_names_by_global.get(geom_b, str(geom_b)), force_n, slip,
                              max(0.0, float(values["penetration"][i])))
            prev = result.get(key)
            if prev is None:
                result[key] = contact
            else:
                result[key] = Contact(key, prev.a, prev.b, prev.geom_a, prev.geom_b,
                                      prev.normal_force + contact.normal_force,
                                      max(prev.slip_speed, contact.slip_speed),
                                      max(prev.penetration, contact.penetration))
        return list(result.values())

    def finite(self) -> bool:
        pos, vel = self.joint_state()
        return bool(np.isfinite(pos).all() and np.isfinite(vel).all())

    def native(self) -> tuple[Any, Any]:
        return self._scene, self._entity

    def _link_pose(self, link_idx: int) -> tuple[np.ndarray, np.ndarray]:
        link = self._entity.links[link_idx]
        pos = _numpy(link.get_pos()).reshape(3)
        quat = _numpy(link.get_quat()).reshape(4)
        return pos, quat

    def _link_velocity(self, link_idx: int, position: np.ndarray) -> np.ndarray:
        link = self._scene.sim.rigid_solver.links[link_idx]
        vel = _numpy(link.get_vel()).reshape(-1)
        if vel.size >= 6:
            linear, angular = vel[:3], vel[3:6]
            origin = self._link_pose(link_idx)[0]
            return linear + np.cross(angular, position - origin)
        return vel[:3]

    def _contact_name(self, link_idx: int, geom_idx: int) -> str:
        name = self._link_names_by_global.get(link_idx, "world")
        return self._geom_names_by_global.get(geom_idx, str(geom_idx)) if name == "world" else name

    def _read_sites(self) -> dict[str, tuple[int, np.ndarray]]:
        sites = {}
        for i in range(self.model.nsite):
            body_id = int(self.model.site_bodyid[i])
            body_name = _name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            sites[_name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)] = (
                self._link_ids[body_name], np.asarray(self.model.site_pos[i], dtype=np.float64).copy()
            )
        return sites


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    qvec = np.array([x, y, z])
    return vector + 2.0 * np.cross(qvec, np.cross(qvec, vector) + w * vector)


def _name(model: Any, objtype: Any, index: int) -> str:
    if objtype == mujoco.mjtObj.mjOBJ_BODY and index == 0:
        return "world"
    return mujoco.mj_id2name(model, objtype, index) or f"{objtype.name.lower()}_{index}"
