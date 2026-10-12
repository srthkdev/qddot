"""Behavioural contract every physics backend must satisfy.

Backends are compared on physics, not bits: tolerances allow engines with
different solvers. Each installed backend in `BACKENDS` is exercised.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from qddot.sim.backend import BACKENDS, load_backend

FIXTURES = Path(__file__).parent / "fixtures"


def _available() -> list[str]:
    names = []
    for name, target in BACKENDS.items():
        try:
            importlib.import_module(target.split(":")[0])
        except ImportError:
            continue
        names.append(name)
    return names


pytestmark = pytest.mark.parametrize("backend", _available())


def _pendulum(backend: str, q0: float = 0.0):
    sim = load_backend(backend, str(FIXTURES / "pendulum.xml"))
    sim.reset({"qpos": {"hinge": q0}}, 1.0, 1.0)
    return sim


def _needs_forces(sim) -> None:
    if not getattr(sim, "measures_contact_force", True):
        pytest.xfail(f"{sim.name} declares it cannot measure contact forces")


def _settle(sim, seconds: float) -> None:
    for _ in range(round(seconds / sim.timestep)):
        sim.step()


def test_describes_joints_and_actuators(backend: str) -> None:
    sim = _pendulum(backend)
    assert sim.timestep == pytest.approx(0.002)
    assert [(j.name, j.limited) for j in sim.joints] == [("hinge", True)]
    assert (sim.joints[0].low, sim.joints[0].high) == pytest.approx((-1.5, 1.5))
    actuator = sim.actuators[0]
    assert (actuator.name, actuator.limited, actuator.low, actuator.high) == ("torque", True, -2.0, 2.0)


def test_initial_state_and_positions(backend: str) -> None:
    sim = _pendulum(backend, q0=0.3)
    pos, vel = sim.joint_state()
    assert pos[0] == pytest.approx(0.3, abs=1e-6) and vel[0] == pytest.approx(0.0, abs=1e-6)
    # Hinge about +y rotates the +x tip towards -z for positive angles.
    tip = sim.position("tip", "site")
    assert tip == pytest.approx([0.5 * np.cos(0.3), 0.0, -0.5 * np.sin(0.3)], abs=1e-4)
    assert sim.position("link", "body") == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    with pytest.raises(ValueError):
        sim.position("missing")


def test_gravity_and_actuation(backend: str) -> None:
    sim = _pendulum(backend)
    _settle(sim, 0.2)
    falling = sim.joint_state()[0][0]
    assert falling > 0.05, "a horizontal pendulum must fall under gravity"
    pushed = _pendulum(backend)
    pushed.set_ctrl(np.array([-2.0]))
    _settle(pushed, 0.2)
    assert pushed.joint_state()[0][0] < falling, "negative torque must slow the fall"
    assert pushed.actuator_force()[0] == pytest.approx(-2.0, abs=0.05)
    assert pushed.ctrl()[0] == pytest.approx(-2.0)
    assert pushed.finite()


def test_resting_contact_force_and_names(backend: str) -> None:
    sim = load_backend(backend, str(FIXTURES / "box_drop.xml"))
    _needs_forces(sim)
    sim.reset({}, 1.0, 1.0)
    assert not sim.contacts()
    _settle(sim, 1.0)
    contacts = sim.contacts()
    assert contacts, "a dropped box must rest on the floor"
    names = {contacts[0].a, contacts[0].b}
    assert names == {"floor", "box"}
    total = sum(c.normal_force for c in contacts)
    assert total == pytest.approx(2.0 * 9.81, rel=0.2), "resting normal force must balance weight"
    assert all(c.slip_speed < 0.01 for c in contacts)
    assert len({c.key for c in contacts}) == 1, "contacts are deduplicated per body pair"


def test_mass_scaling(backend: str) -> None:
    sim = load_backend(backend, str(FIXTURES / "box_drop.xml"))
    _needs_forces(sim)
    sim.reset({}, 1.0, 1.5)
    _settle(sim, 1.0)
    assert sum(c.normal_force for c in sim.contacts()) == pytest.approx(1.5 * 2.0 * 9.81, rel=0.2)


def test_deterministic(backend: str) -> None:
    runs = []
    for _ in range(2):
        sim = _pendulum(backend, q0=0.2)
        sim.set_ctrl(np.array([0.7]))
        _settle(sim, 0.5)
        runs.append(np.array(sim.joint_state()[0]))
    assert np.array_equal(runs[0], runs[1])


def test_contact_force_is_measured_not_inferred(backend: str) -> None:
    """The floor carries both stacked boxes; a per-body weight estimate would report only one."""
    sim = load_backend(backend, str(FIXTURES / "stack.xml"))
    _needs_forces(sim)
    sim.reset({}, 1.0, 1.0)
    _settle(sim, 1.0)
    by_pair = {frozenset((c.a, c.b)): c.normal_force for c in sim.contacts()}
    assert by_pair[frozenset(("floor", "bottom"))] == pytest.approx(5.0 * 9.81, rel=0.2)
    assert by_pair[frozenset(("bottom", "top"))] == pytest.approx(3.0 * 9.81, rel=0.2)
