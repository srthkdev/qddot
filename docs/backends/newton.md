# Newton backend

## Install

Install Newton 1.6.1 and Warp 1.18 in the project's Python 3.12 environment:

```sh
uv pip install newton==1.6.1 warp-lang==1.18.0
```

The backend runs on Newton's CPU device on macOS; it does not require CUDA.
Newton's MJCF importer and `SolverFeatherstone` are used directly.
`SolverMuJoCo` would expose MuJoCo's constraint forces, but Newton 1.6.1
imports `mujoco_warp` unconditionally when constructing that solver. This
environment does not have `mujoco_warp` installed, and network access is
unavailable, so this backend cannot use that solver here.

## Support and gaps

The backend imports MJCF articulations, 1-DoF hinge and slide joints, body and
site positions, joint limits, primitive collision shapes, motor actuators, and
position actuators with their imported affine gain/bias. It supports initial
joint overrides, keyframe qpos/qvel/ctrl, friction scaling, and proportional
mass and inertia scaling. Actuator output is reported as generalized force or
torque at its target joint.

The supported path is reduced-coordinate articulated models. Free and ball
joints are not exposed in the protocol's 1-DoF joint list. Exotic actuator
transmissions, actuator activation dynamics, equality constraints, and MJCF
features that Newton does not import are not reproduced. Newton's Featherstone
solver does not expose per-contact impulses to this adapter. Consequently,
`measures_contact_force` is `False` and each contact's `normal_force` is NaN;
the backend does not infer force from body weight. Contact slip uses relative
velocity at each point, including the body's angular contribution. Contact
penetration is computed from the importer contact points and normal; it can
differ from MuJoCo's signed geom distance.

## Speed

Measured on macOS arm64 with Newton 1.6.1 / Warp 1.18 and MuJoCo 3.15, running
the 4 s `cartpole_balance` good-controller episode in this workspace, after
each backend's first run warmed its kernels: MuJoCo took **0.088 s** and Newton
took **3.123 s** of runner wall time. Both runs passed. Newton is about **35x
slower** for this small model. The first Newton run also includes Warp kernel
compilation and is slower still.

## Determinism

The CPU solver is configured with Warp's `RUN_TO_RUN` deterministic mode and
the collision pipeline uses deterministic contact ordering. Repeated runs in
one process on the same machine and Newton/Warp versions are deterministic.
Bitwise identity across Newton versions, platforms, or solver implementations
is not promised.

## Contact force conformance

Contact forces are unavailable with the installed Featherstone solver. The
Newton contact-force conformance checks therefore cannot pass in this
environment: force values are NaN rather than estimated from gravity. Once
`mujoco_warp` is available, the backend can use `SolverMuJoCo` and read its
MuJoCo constraint forces per contact point, summing those by body pair.
