# Known controller failures

Nominal values below come from `qddot.sim.runner.run_episode` using the vendored UR5e scene and `home` keyframe.

| Controller | Root cause and first failing spec | Approximate time and observed events | Minimal fix |
|---|---|---|---|
| `buggy_joint_order.py` | Swaps shoulder-lift and elbow commands. In the real runner, the resulting contact disturbance also keeps shoulder-pan outside its settling band at the required start of the final one-second hold, so `pan_settle` is the first failed spec. Lift ends near 0.258 rad and elbow near -1.347 rad. | Contact and collision events occur as the incorrectly configured arm moves. | Map each target joint to its matching actuator name. |
| `buggy_degrees.py` | Converts every radian-valued target as if it were degrees. `pan_settle` is the first failed spec; all three joints head toward values near zero rather than their requested poses. | No event is required; the failure is visible in the joint signals. | Use task target values directly in radians; remove `deg2rad`. |

The good controller's maximum error after t=2 s is 0.0123 rad across the three target joints and ends within 0.0123 rad of each target.

`buggy_phase_margin.py` adds a fixed joint-space lead whenever feedback is delayed, independent of the trajectory speed. The lead causes overshoot under latency perturbation; scale prediction by desired joint velocity and measured age.
