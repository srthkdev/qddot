"""Newton physics backend using its MJCF importer and MuJoCo-Warp solver."""

from __future__ import annotations

import inspect
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco_warp
import newton
import numpy as np
import warp as wp

from qddot.sim.backend import Contact, Limit

_WP_VEC5 = wp.types.vector(length=5, dtype=wp.float32)


@wp.func
def _legacy_contact_force_fn(
    opt_cone: int,
    contact_frame: wp.array2d[wp.mat33],
    contact_friction: wp.array[_WP_VEC5],
    contact_dim: wp.array[int],
    contact_efc_address: wp.array2d[int],
    efc_force: wp.array2d[float],
    njmax: int,
    nacon: wp.array[int],
    world_id: int,
    contact_id: int,
    to_world_frame: bool,
) -> wp.spatial_vector:
    """Bridge Newton 1.6's helper call to MuJoCo-Warp 3.15's data layout."""
    force = wp.spatial_vector()
    address = contact_efc_address[contact_id, 0]
    if contact_id >= 0 and contact_id < nacon[world_id] and address >= 0 and address < njmax:
        # Newton only consumes this vector for its contact report buffer. The
        # backend's public force values come from the MuJoCo-Warp contact data.
        force[0] = efc_force[world_id, address]
        if opt_cone == 0 and address + 1 < njmax:
            force[0] += efc_force[world_id, address + 1]
    if to_world_frame:
        top = wp.spatial_top(force) @ contact_frame[world_id, contact_id]
        bottom = wp.spatial_bottom(force) @ contact_frame[world_id, contact_id]
        force = wp.spatial_vector(top, bottom)
    return force


def _prepare_newton_mujoco_warp_compatibility() -> None:
    """Adapt Newton 1.6's legacy contact helper call for MuJoCo-Warp 3.15."""
    from mujoco_warp._src import support as mjw_support
    from newton._src.solvers.mujoco import kernels as newton_mj_kernels

    if len(inspect.signature(mjw_support.contact_force_fn).parameters) == 11:
        return
    newton_mj_kernels._import_contact_force_fn = lambda: _legacy_contact_force_fn


class NewtonBackend:
    """Run MJCF models in Newton's MuJoCo-Warp solver on CPU."""

    name = "newton"
    version = newton.__version__
    measures_contact_force = True

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path).resolve()
        cache_root = wp.config.kernel_cache_dir
        cache_parent = Path(cache_root).expanduser().parent if cache_root else None
        if cache_parent is None or not os.access(cache_parent, os.W_OK):
            wp.config.kernel_cache_dir = str(Path(tempfile.gettempdir()) / "qddot-warp-cache")
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(self.model_path), ctrl_direct=True, parse_sites=True)
        self.model = builder.finalize(device="cpu")
        self.timestep = float(self.model.mujoco.timestep) if hasattr(self.model.mujoco, "timestep") else self._xml_timestep()
        self.state = self.model.state()
        self._next_state = self.model.state()
        self.control = self.model.control()
        self._contacts = newton.CollisionPipeline(
            self.model, contact_matching="sticky", contact_report=True, deterministic=True
        )
        _prepare_newton_mujoco_warp_compatibility()
        self._solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_cpu=False, deterministic=wp.DeterministicMode.RUN_TO_RUN
        )
        q_starts = self.model.joint_q_start.numpy().astype(int)
        qd_starts = self.model.joint_qd_start.numpy().astype(int)
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        self._joint_ids = [i for i in range(self.model.joint_count)
                           if qd_starts[i + 1] - qd_starts[i] == 1]
        self._all_qadr, self._all_vadr = q_starts, qd_starts
        # Newton stores coordinates by joint; these starts preserve its imported MJCF order.
        self._qadr = [int(q_starts[i]) for i in self._joint_ids]
        self._vadr = [int(qd_starts[i]) for i in self._joint_ids]
        self.joints = [
            Limit(_short(self.model.joint_label[i]),
                  bool(np.isfinite(lower[self._vadr[k]]) and np.isfinite(upper[self._vadr[k]])
                       and abs(lower[self._vadr[k]]) < 1e9 and abs(upper[self._vadr[k]]) < 1e9),
                  float(lower[self._vadr[k]]), float(upper[self._vadr[k]]))
            for k, i in enumerate(self._joint_ids)
        ]
        namespace = self.model.mujoco
        self._actuator_names = list(getattr(namespace, "actuator_label", []))
        self._actuator_joint = _host(getattr(namespace, "actuator_trnid", None), (0, 2))[:, 0].astype(int)
        self._gear = _host(getattr(namespace, "actuator_gear", None), (0, 6))[:, 0].astype(np.float64)
        self._gain = _host(getattr(namespace, "actuator_gainprm", None), (0, 10))[:, 0].astype(np.float64)
        self._bias = _host(getattr(namespace, "actuator_biasprm", None), (0, 10)).astype(np.float64)
        limited = _host(getattr(namespace, "actuator_ctrllimited", None), (0,)).astype(bool)
        ranges = _host(getattr(namespace, "actuator_ctrlrange", None), (0, 2)).astype(np.float64)
        self.actuators = [Limit(n, bool(limited[i]), *ranges[i]) for i, n in enumerate(self._actuator_names)]
        self._ctrl = _host(getattr(namespace, "ctrl", None), (0,)).astype(np.float64)
        self._base_mass = self.model.body_mass.numpy().copy()
        self._base_inertia = self.model.body_inertia.numpy().copy()
        self._mj_body_map = self._solver.mjc_body_to_newton.numpy()[0].astype(int)
        self._base_friction = self.model.shape_material_mu.numpy().copy()
        self._shape_labels = list(self.model.shape_label)
        self._body_labels = list(self.model.body_label)
        self._shape_body = self.model.shape_body.numpy().astype(int)
        self._joint_by_name = {joint.name: k for k, joint in enumerate(self.joints)}
        self._shape_by_name = {_short(label): i for i, label in enumerate(self._shape_labels)}
        self._body_by_name = {_short(label): i for i, label in enumerate(self._body_labels)}
        self._site_shapes = {name: idx for name, idx in self._shape_by_name.items()
                             if name not in self._geom_names()}

    def reset(self, init: dict[str, Any], friction_scale: float, mass_scale: float) -> None:
        self.model.body_mass.assign(self._base_mass * mass_scale)
        self.model.body_inertia.assign(self._base_inertia * mass_scale)
        self.model.shape_material_mu.assign(self._base_friction * friction_scale)
        self._solver.notify_model_changed(
            newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.SHAPE_PROPERTIES
        )
        q = self.model.joint_q.numpy().copy()
        qd = np.zeros_like(self.model.joint_qd.numpy())
        ctrl = np.zeros_like(self._ctrl)
        if init.get("keyframe"):
            ctrl = self._load_keyframe(str(init["keyframe"]), q, qd, ctrl)
        for name, value in init.get("qpos", {}).items():
            if name not in self._joint_by_name:
                raise ValueError(f"unknown joint in init.qpos: {name}")
            q[self._qadr[self._joint_by_name[name]]] = float(value)
        self.state.joint_q.assign(q)
        self.state.joint_qd.assign(qd)
        self._ctrl = ctrl
        if self._ctrl.size:
            self.model.mujoco.ctrl.assign(self._ctrl.astype(np.float32))
        self.newton_fk()
        self.set_ctrl(self._ctrl)
        self._next_state.joint_q.assign(q)
        self._next_state.joint_qd.assign(qd)
        self.newton_fk(self._next_state)

    def ctrl(self) -> np.ndarray:
        return self._ctrl.copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self._ctrl = np.asarray(ctrl, dtype=np.float64).copy()
        if self._ctrl.size:
            self.model.mujoco.ctrl.assign(self._ctrl.astype(np.float32))
        forces = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        q = self.state.joint_q.numpy()
        qd = self.state.joint_qd.numpy()
        for i, joint_id in enumerate(self._actuator_joint):
            if joint_id < 0 or joint_id >= self.model.joint_count:
                continue
            dof = int(self._all_vadr[joint_id])
            coord = int(self._all_qadr[joint_id])
            # MJCF motor is gear*ctrl. Position actuators use their imported
            # gain and affine bias (position and velocity terms).
            force = self._gear[i] * (self._gain[i] * self._ctrl[i] + self._bias[i, 0]
                                      + self._bias[i, 1] * q[coord] + self._bias[i, 2] * qd[dof])
            forces[dof] += force
        self.control.joint_f.assign(forces)

    def step(self) -> None:
        contacts = self._contacts.contacts()
        self._contacts.collide(self.state, contacts)
        self._solver.step(self.state, self._next_state, self.control, contacts, self.timestep)
        self.state, self._next_state = self._next_state, self.state
        # Keep the reported control application in sync with current state for
        # position actuators, whose force depends on position and velocity.
        self.set_ctrl(self._ctrl)

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        q, qd = self.state.joint_q.numpy(), self.state.joint_qd.numpy()
        return (np.asarray([q[i] for i in self._qadr], dtype=np.float64),
                np.asarray([qd[i] for i in self._vadr], dtype=np.float64))

    def actuator_force(self) -> np.ndarray:
        values = self.control.joint_f.numpy()
        return np.asarray([values[int(self._all_vadr[j])] for j in self._actuator_joint], dtype=np.float64)

    def position(self, name: str, kind: str | None = None) -> np.ndarray:
        if kind in (None, "site") and name in self._site_shapes:
            idx = self._site_shapes[name]
            return self._shape_position(idx)
        if kind == "site":
            raise ValueError(f"unknown site signal reference: {name}")
        if kind in (None, "body") and name in self._body_by_name:
            return self.state.body_q.numpy()[self._body_by_name[name], :3].astype(np.float64)
        what = "body signal" if kind == "body" else "site or body"
        raise ValueError(f"unknown {what} reference: {name}")

    def contacts(self) -> list[Contact]:
        found: dict[tuple[int, int], Contact] = {}
        data = self._solver.mjw_data
        count = int(data.nacon.numpy()[0])
        if count == 0:
            return []
        contact_ids = wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=self.model.device)
        force_buffer = wp.empty(count, dtype=wp.spatial_vector, device=self.model.device)
        mujoco_warp.contact_force(self._solver.mjw_model, data, contact_ids, False, force_buffer)
        forces = force_buffer.numpy()
        geoms = data.contact.geom.numpy()[:count]
        positions = data.contact.pos.numpy()[:count]
        frames = data.contact.frame.numpy()[:count]
        distances = data.contact.dist.numpy()[:count]
        geom_body = self._solver.mjw_model.geom_bodyid.numpy()
        body_map = self._mj_body_map
        for i in range(count):
            ga, gb = (int(v) for v in geoms[i])
            ba, bb = int(body_map[int(geom_body[ga])]), int(body_map[int(geom_body[gb])])
            sa, sb = self._shape_for_geom(ga), self._shape_for_geom(gb)
            key = tuple(sorted((ba, bb)))
            normal_force = max(0.0, float(forces[i, 0]))
            normal = np.asarray(frames[i, :3], dtype=np.float64)
            point = np.asarray(positions[i], dtype=np.float64)
            penetration = max(0.0, -float(distances[i]))
            velocity_a = self._contact_point_velocity(ba, point)
            velocity_b = self._contact_point_velocity(bb, point)
            relative = velocity_a - velocity_b
            tangent = relative - np.dot(relative, normal) * normal
            contact = Contact(key, self._contact_name(ba, sa), self._contact_name(bb, sb),
                              _short(self._shape_labels[sa]), _short(self._shape_labels[sb]),
                              normal_force, float(np.linalg.norm(tangent)), penetration)
            if key in found:
                previous = found[key]
                found[key] = Contact(previous.key, previous.a, previous.b, previous.geom_a, previous.geom_b,
                                     previous.normal_force + normal_force,
                                     max(previous.slip_speed, contact.slip_speed),
                                     max(previous.penetration, penetration))
            else:
                found[key] = contact
        return list(found.values())

    def _shape_for_geom(self, geom_id: int) -> int:
        """Resolve a MuJoCo geom through the solver's explicit shape mapping."""
        shape_id = int(self._solver.mjc_geom_to_newton_shape.numpy()[0, geom_id])
        if shape_id < 0:
            raise ValueError(f"MuJoCo geom {geom_id} has no Newton shape mapping")
        return shape_id

    def finite(self) -> bool:
        return bool(np.isfinite(self.state.joint_q.numpy()).all()
                    and np.isfinite(self.state.joint_qd.numpy()).all())

    def native(self) -> tuple[Any, Any]:
        return self.model, self.state

    def newton_fk(self, state: Any | None = None) -> None:
        state = state or self.state
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _load_keyframe(self, name: str, q: np.ndarray, qd: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
        root = ET.parse(self.model_path).getroot()
        key = next((item for item in root.findall(".//keyframe/key") if item.get("name") == name), None)
        if key is None:
            raise ValueError(f"unknown keyframe: {name}")
        if key.get("qpos"):
            values = np.fromstring(key.get("qpos", ""), sep=" ")
            q[:min(values.size, q.size)] = values[:q.size]
        if key.get("qvel"):
            values = np.fromstring(key.get("qvel", ""), sep=" ")
            qd[:min(values.size, qd.size)] = values[:qd.size]
        if key.get("ctrl"):
            values = np.fromstring(key.get("ctrl", ""), sep=" ")
            ctrl[:min(values.size, ctrl.size)] = values[:ctrl.size]
        return ctrl

    def _xml_timestep(self) -> float:
        root = ET.parse(self.model_path).getroot()
        option = root.find("option")
        return float(option.get("timestep", "0.002")) if option is not None else 0.002

    def _geom_names(self) -> set[str]:
        root = ET.parse(self.model_path).getroot()
        return {geom.get("name", "") for geom in root.findall(".//geom")}

    def _shape_position(self, shape_id: int) -> np.ndarray:
        body = int(self._shape_body[shape_id])
        local = self.model.shape_transform.numpy()[shape_id]
        if body < 0:
            return np.asarray(local[:3], dtype=np.float64)
        world = self.state.body_q.numpy()[body]
        body_xform = wp.transformf(wp.vec3f(*world[:3]), wp.quatf(*world[3:7]))
        local_xform = wp.transformf(wp.vec3f(*local[:3]), wp.quatf(*local[3:7]))
        return np.asarray(wp.transform_point(body_xform, local_xform.p), dtype=np.float64)

    def _contact_point_velocity(self, body_id: int, point: np.ndarray) -> np.ndarray:
        if body_id < 0:
            return np.zeros(3, dtype=np.float64)
        twist = self.state.body_qd.numpy()[body_id]
        pose = self.state.body_q.numpy()[body_id]
        com_local = self.model.body_com.numpy()[body_id]
        rotation = wp.quatf(*pose[3:7])
        com_world = np.asarray(pose[:3], dtype=np.float64) + np.asarray(
            wp.quat_rotate(rotation, wp.vec3f(*com_local)), dtype=np.float64
        )
        return (np.asarray(twist[:3], dtype=np.float64)
                + np.cross(np.asarray(twist[3:6], dtype=np.float64), np.asarray(point, dtype=np.float64) - com_world))

    def _contact_name(self, body_id: int, shape_id: int) -> str:
        return _short(self._shape_labels[shape_id]) if body_id < 0 else _short(self._body_labels[body_id])


def _short(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _host(value: Any, empty_shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.empty(empty_shape, dtype=np.float32)
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)
