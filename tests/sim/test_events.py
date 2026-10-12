from __future__ import annotations

import numpy as np

from qddot.sim.backend import Contact, Limit
from qddot.sim.events import EventDetector

FLOOR = Contact(key=(0, 1), a="floor", b="robot", geom_a="floor", geom_b="foot",
                normal_force=3.0, slip_speed=0.0, penetration=0.001)


def _detect(detector: EventDetector, contacts: list[Contact], t: float, control: int = 0,
            joints: list[Limit] | None = None, q: list[float] | None = None) -> None:
    detector.detect_physics(contacts, joints or [], np.asarray(q or []), t, [], control)


def test_contact_end_requires_two_absent_physics_steps_and_names_geoms() -> None:
    detector = EventDetector()
    _detect(detector, [FLOOR], 0.0)
    start = detector.emitted[0]
    assert (start["kind"], start["a"], start["b"]) == ("contact_start", "floor", "robot")
    assert '"geom_a": "floor"' in start["detail"]

    _detect(detector, [], 0.01)
    assert not any(event["kind"] == "contact_end" for event in detector.emitted)
    _detect(detector, [FLOOR], 0.02)
    _detect(detector, [], 0.03)
    _detect(detector, [], 0.04)
    ends = [event for event in detector.emitted if event["kind"] == "contact_end"]
    assert [event["t"] for event in ends] == [0.04]


def test_slip_needs_two_control_steps_and_limits_emit_on_entry() -> None:
    detector = EventDetector()
    slipping = Contact(**{**FLOOR.__dict__, "slip_speed": 0.05})
    for control in range(3):
        _detect(detector, [slipping], 0.01 * control, control)
    assert [e["kind"] for e in detector.emitted].count("slip") == 1

    joint = [Limit("elbow", True, -1.0, 1.0)]
    for t, q in ((0.1, 0.0), (0.2, 0.995), (0.3, 1.0), (0.4, 0.0), (0.5, -1.0)):
        _detect(detector, [slipping], t, 9, joint, [q])
    limits = [e for e in detector.emitted if e["kind"] == "joint_limit"]
    assert [e["t"] for e in limits] == [0.2, 0.5]
