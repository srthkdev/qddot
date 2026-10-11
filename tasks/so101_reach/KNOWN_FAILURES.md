# Known controller failures

Nominal values below come from the standalone loop in `tests/tasks/test_tasks.py` using MuJoCo and the bundled scene.

| Controller | Root cause and first failing spec | Approximate time and observed events | Minimal fix |
|---|---|---|---|
| `buggy_jacobian_sign.py` | Negates the DLS correction. `reach_and_hold` never gets close: target distance starts at 0.237 m and grows, with minimum still 0.237 m. | Arm-floor contact starts around 0.30 s; actuator commands first clip around 0.08 s. Expect `saturation` and declared `collision`/contact events. | Use `q + dq` so the Jacobian update follows target-minus-current error. |
| `buggy_limit_avoidance.py` | Omits the joint-range guard and adds a persistent negative shoulder-lift bias. `no_joint_limit` fails first; the reach spec still passes. | Shoulder-lift enters the final 1% of its lower range near 0.66 s; actuator clipping starts around 0.26 s. Expect `joint_limit` and `saturation`. | Clamp joint targets inside `jnt_range` with a margin and remove the persistent bias. |

The good controller reaches 2.5 cm at about 0.42 s, ends at 0.00011 m error and has no nominal floor contact or clipping.
