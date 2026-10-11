"""Stateful MuJoCo event detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np


@dataclass
class EventDetector:
    """Track contact and limit state so events are emitted on entry/exit."""

    active_contacts: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    active_limits: set[int] = field(default_factory=set)
    active_saturation: set[int] = field(default_factory=set)
    deadline_active: bool = False
    slip_steps: dict[tuple[int, int], int] = field(default_factory=dict)
    slip_last_control: dict[tuple[int, int], int] = field(default_factory=dict)
    absent_contact_steps: dict[tuple[int, int], int] = field(default_factory=dict)
    emitted: list[dict[str, Any]] = field(default_factory=list)
    step_kinds: set[str] = field(default_factory=set)

    def begin_control_step(self) -> None:
        self.step_kinds.clear()

    def detect_physics(self, model: Any, data: Any, t: float, pairs: list[dict[str, Any]],
                       control_index: int) -> None:
        contacts: dict[tuple[int, int], dict[str, Any]] = {}
        for index in range(data.ncon):
            contact = data.contact[index]
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            body_a, body_b = int(model.geom_bodyid[geom_a]), int(model.geom_bodyid[geom_b])
            key = tuple(sorted((body_a, body_b)))
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, index, force)
            normal_force = abs(float(force[0]))
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            velocity_a = _body_velocity(model, data, body_a)
            velocity_b = _body_velocity(model, data, body_b)
            relative = velocity_a - velocity_b
            tangential_speed = float(np.linalg.norm(relative - np.dot(relative, normal) * normal))
            name_a = _contact_name(model, body_a, geom_a)
            name_b = _contact_name(model, body_b, geom_b)
            geom_name_a = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
            geom_name_b = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)
            contacts[key] = {
                "a": name_a, "b": name_b, "force": normal_force,
                "geom_a": geom_name_a, "geom_b": geom_name_b,
                "slip_speed": tangential_speed, "penetration": max(0.0, -float(contact.dist)),
            }

        for key, item in contacts.items():
            if key not in self.active_contacts:
                detail = {"geom_a": item["geom_a"], "geom_b": item["geom_b"]}
                self._emit(t, "contact_start", item["a"], item["b"], item["force"], detail)
                if _declared_pair(item["a"], item["b"], pairs):
                    self._emit(t, "collision", item["a"], item["b"], item["force"], detail)
            if item["penetration"] > 0.005 and self.active_contacts.get(key, {}).get("penetration", 0) <= 0.005:
                self._emit(t, "collision", item["a"], item["b"], item["penetration"],
                           {"source": "penetration", "geom_a": item["geom_a"], "geom_b": item["geom_b"]})
            self.active_contacts[key] = item
            self.absent_contact_steps.pop(key, None)
        for key in self.active_contacts.keys() - contacts.keys():
            self.absent_contact_steps[key] = self.absent_contact_steps.get(key, 0) + 1
            if self.absent_contact_steps[key] >= 2:
                old = self.active_contacts.pop(key)
                self._emit(t, "contact_end", old["a"], old["b"], 0.0,
                           {"geom_a": old["geom_a"], "geom_b": old["geom_b"]})
                self.absent_contact_steps.pop(key, None)
                self.slip_steps.pop(key, None)
                self.slip_last_control.pop(key, None)

        for key, item in contacts.items():
            if self.slip_last_control.get(key) != control_index:
                self.slip_last_control[key] = control_index
                if item["slip_speed"] > 0.01:
                    count = self.slip_steps.get(key, 0) + 1
                    self.slip_steps[key] = count
                else:
                    self.slip_steps[key] = 0
                if self.slip_steps.get(key) == 2:
                    self._emit(t, "slip", item["a"], item["b"], item["slip_speed"])

        limited: set[int] = set()
        for jid in range(model.njnt):
            if not model.jnt_limited[jid]:
                continue
            jtype = model.jnt_type[jid]
            if int(jtype) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                continue
            qadr = int(model.jnt_qposadr[jid])
            low, high = model.jnt_range[jid]
            margin = 0.01 * (high - low)
            value = float(data.qpos[qadr])
            if value <= low + margin or value >= high - margin:
                limited.add(jid)
                if jid not in self.active_limits:
                    overshoot = max(low - value, value - high, 0.0)
                    self._emit(t, "joint_limit", _name(model, mujoco.mjtObj.mjOBJ_JOINT, jid), None, float(overshoot))
        self.active_limits = limited

    def saturation(self, t: float, actuator_id: int, name: str, unclipped: float, clipped: float) -> None:
        if unclipped != clipped:
            if actuator_id not in self.active_saturation:
                self._emit(t, "saturation", name, None, abs(unclipped - clipped))
            self.active_saturation.add(actuator_id)
        else:
            self.active_saturation.discard(actuator_id)

    def deadline_miss(self, t: float, controller: str, compute_ms: float) -> None:
        if not self.deadline_active:
            self._emit(t, "deadline_miss", controller, None, compute_ms)
        else:
            self.step_kinds.add("deadline_miss")
        self.deadline_active = True

    def deadline_met(self) -> None:
        self.deadline_active = False

    def add(self, t: float, kind: str, a: str = "", b: str | None = None,
            value: float = 0.0, detail: dict[str, Any] | None = None) -> None:
        self._emit(t, kind, a, b, value, detail)

    def _emit(self, t: float, kind: str, a: str, b: str | None, value: float,
              detail: dict[str, Any] | None = None) -> None:
        self.emitted.append({"t": float(t), "kind": kind, "a": a, "b": b,
                             "value": float(value), "detail": json.dumps(detail or {}, sort_keys=True)})
        self.step_kinds.add(kind)


def _name(model: Any, objtype: Any, index: int) -> str:
    if objtype == mujoco.mjtObj.mjOBJ_BODY and index == 0:
        return "world"
    return mujoco.mj_id2name(model, objtype, index) or f"{objtype.name.lower()}_{index}"


def _contact_name(model: Any, body_id: int, geom_id: int) -> str:
    if body_id == 0:
        return _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def _body_velocity(model: Any, data: Any, body_id: int) -> np.ndarray:
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
    return velocity[3:]


def _declared_pair(name_a: str, name_b: str, pairs: list[dict[str, Any]]) -> bool:
    return any((name_a in pair["a"] and name_b in pair["b"])
               or (name_b in pair["a"] and name_a in pair["b"]) for pair in pairs)
