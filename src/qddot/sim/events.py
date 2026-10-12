"""Engine-agnostic event detection over backend contacts, joint limits and controls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qddot.sim.backend import Contact, Limit

SLIP_SPEED = 0.01        # m/s tangential speed that counts as slipping
PENETRATION = 0.005      # m of penetration that counts as a collision
LIMIT_MARGIN = 0.01      # fraction of a joint's range treated as "at the limit"


@dataclass
class EventDetector:
    """Track contact and limit state so events are emitted on entry/exit."""

    active_contacts: dict[tuple[Any, ...], Contact] = field(default_factory=dict)
    active_limits: set[int] = field(default_factory=set)
    active_saturation: set[int] = field(default_factory=set)
    deadline_active: bool = False
    slip_steps: dict[tuple[Any, ...], int] = field(default_factory=dict)
    slip_last_control: dict[tuple[Any, ...], int] = field(default_factory=dict)
    absent_contact_steps: dict[tuple[Any, ...], int] = field(default_factory=dict)
    emitted: list[dict[str, Any]] = field(default_factory=list)
    step_kinds: set[str] = field(default_factory=set)

    def begin_control_step(self) -> None:
        self.step_kinds.clear()

    def detect_physics(self, contacts: list[Contact], joints: list[Limit], positions: np.ndarray,
                       t: float, pairs: list[dict[str, Any]], control_index: int) -> None:
        current = {contact.key: contact for contact in contacts}
        self._contacts(current, t, pairs)
        self._slips(current, t, control_index)
        self._limits(joints, positions, t)

    def _contacts(self, current: dict, t: float, pairs: list[dict[str, Any]]) -> None:
        for key, item in current.items():
            detail = {"geom_a": item.geom_a, "geom_b": item.geom_b}
            previous = self.active_contacts.get(key)
            if previous is None:
                self._emit(t, "contact_start", item.a, item.b, item.normal_force, detail)
                if _declared_pair(item.a, item.b, pairs):
                    self._emit(t, "collision", item.a, item.b, item.normal_force, detail)
            if item.penetration > PENETRATION and (previous is None or previous.penetration <= PENETRATION):
                self._emit(t, "collision", item.a, item.b, item.penetration,
                           {"source": "penetration", **detail})
            self.active_contacts[key] = item
            self.absent_contact_steps.pop(key, None)
        # Debounce: a contact must be absent for two physics steps before it ends.
        for key in self.active_contacts.keys() - current.keys():
            self.absent_contact_steps[key] = self.absent_contact_steps.get(key, 0) + 1
            if self.absent_contact_steps[key] >= 2:
                old = self.active_contacts.pop(key)
                self._emit(t, "contact_end", old.a, old.b, 0.0,
                           {"geom_a": old.geom_a, "geom_b": old.geom_b})
                for table in (self.absent_contact_steps, self.slip_steps, self.slip_last_control):
                    table.pop(key, None)

    def _slips(self, current: dict, t: float, control_index: int) -> None:
        # Slip is sampled once per control step and must persist for two of them.
        for key, item in current.items():
            if self.slip_last_control.get(key) == control_index:
                continue
            self.slip_last_control[key] = control_index
            self.slip_steps[key] = self.slip_steps.get(key, 0) + 1 if item.slip_speed > SLIP_SPEED else 0
            if self.slip_steps[key] == 2:
                self._emit(t, "slip", item.a, item.b, item.slip_speed)

    def _limits(self, joints: list[Limit], positions: np.ndarray, t: float) -> None:
        limited: set[int] = set()
        for index, joint in enumerate(joints):
            if not joint.limited:
                continue
            margin = LIMIT_MARGIN * (joint.high - joint.low)
            value = float(positions[index])
            if value <= joint.low + margin or value >= joint.high - margin:
                limited.add(index)
                if index not in self.active_limits:
                    overshoot = max(joint.low - value, value - joint.high, 0.0)
                    self._emit(t, "joint_limit", joint.name, None, float(overshoot))
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


def _declared_pair(name_a: str, name_b: str, pairs: list[dict[str, Any]]) -> bool:
    return any((name_a in pair["a"] and name_b in pair["b"])
               or (name_b in pair["a"] and name_a in pair["b"]) for pair in pairs)
