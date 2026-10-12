# Newton backend

## Install

Install Newton 1.6.1, Warp 1.18, and the matching MuJoCo-Warp package in the
project's Python 3.12 environment. The backend uses Newton's `SolverMuJoCo`
with MuJoCo-Warp on the CPU device; it does not require CUDA.

## Support and gaps

The backend imports MJCF articulations, 1-DoF hinge and slide joints, body and
site positions, joint limits, primitive collision shapes, motor actuators, and
position actuators with their imported affine gain/bias. It supports initial
joint overrides, keyframe qpos/qvel/ctrl, friction scaling, and proportional
mass and inertia scaling. Actuator output is reported as generalized force or
torque at its target joint.

Contact forces come from the solver's MuJoCo-Warp constraints. The adapter
calls `mujoco_warp.contact_force` on `solver.mjw_data` after each step, then
sums each contact's normal component by body pair. This is a measured solver
force; `measures_contact_force` is true, and no force is inferred from mass or
gravity. Newton 1.6.1's report-conversion kernel uses the MuJoCo-Warp 3.12
contact-force signature, while MuJoCo-Warp 3.15.0 adds an adhesion argument
and stores frame data per world. A small compatibility function adapts that
internal report call; the physical solve and public force readings use the
installed MuJoCo-Warp implementation. Contact slip uses relative velocity at
each point, including the body's angular contribution. Penetration is read
from MuJoCo-Warp's signed contact distance. The supported path is reduced-coordinate articulated models;
free and ball joints are not exposed in the protocol's 1-DoF joint list.
Exotic actuator transmissions, actuator activation dynamics, equality
constraints, and MJCF features that Newton does not import are not reproduced.

## Speed

The focused backend test measured **4.813 s** of runner wall time for the
4-second cartpole episode on macOS arm64 with Newton 1.6.1, Warp 1.18, and
MuJoCo-Warp 3.15. The MuJoCo-Warp solver has a one-time Warp kernel
compilation cost, so first process timing can exceed warmed runs.

## Determinism

The CPU solver is configured with Warp's `RUN_TO_RUN` deterministic mode and
the collision pipeline uses deterministic contact ordering. Repeated runs in
one process on the same machine and Newton/Warp versions are deterministic.
Bitwise identity across Newton versions, platforms, or solver implementations
is not promised.
