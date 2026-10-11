from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np

from qddot.sim.events import EventDetector


def test_contact_end_requires_two_absent_physics_steps_and_names_geoms(monkeypatch) -> None:
    model = SimpleNamespace(geom_bodyid=np.array([0, 1]), njnt=0)
    contact = SimpleNamespace(geom1=0, geom2=1, frame=np.array([1.0, 0.0, 0.0]), dist=-0.001)
    data = SimpleNamespace(ncon=1, contact=[contact])
    monkeypatch.setattr(mujoco, "mj_contactForce", lambda model, data, index, force: force.__setitem__(0, 3.0))
    monkeypatch.setattr(mujoco, "mj_id2name", lambda model, obj, index: {0: "floor", 1: "robot"}.get(index, "geom"))
    monkeypatch.setattr("qddot.sim.events._body_velocity", lambda model, data, body: np.zeros(3))

    detector = EventDetector()
    detector.detect_physics(model, data, 0.0, [], 0)
    start = detector.emitted[0]
    assert (start["a"], start["b"]) == ("floor", "robot")
    assert '"geom_a": "floor"' in start["detail"]

    data.ncon = 0
    detector.detect_physics(model, data, 0.01, [], 0)
    assert not any(event["kind"] == "contact_end" for event in detector.emitted)
    data.ncon = 1
    detector.detect_physics(model, data, 0.02, [], 0)
    data.ncon = 0
    detector.detect_physics(model, data, 0.03, [], 0)
    detector.detect_physics(model, data, 0.04, [], 0)
    ends = [event for event in detector.emitted if event["kind"] == "contact_end"]
    assert len(ends) == 1
    assert ends[0]["t"] == 0.04
