# Known controller failures

Nominal values below come from `qddot.sim.runner.run_episode` using MuJoCo and the bundled scene.

| Controller | Root cause and first failing spec | Approximate time and observed events | Minimal fix |
|---|---|---|---|
| `buggy_jacobian_sign.py` | Negates the DLS correction. `reach_and_hold` is the first failed spec; target distance grows instead of converging. | Arm-floor contacts and declared collisions occur repeatedly, and commands saturate. | Use `q + dq` so the Jacobian update follows target-minus-current error. |
| `buggy_limit_avoidance.py` | Omits the joint-range guard and adds a persistent negative shoulder-lift bias. `no_joint_limit` fails first; the reach spec still passes. | The wrist first enters the limit warning band near 0.38 s, then shoulder-lift reaches its lower warning band. | Clamp joint targets inside `jnt_range` with a margin and remove the persistent bias. |

The good controller reaches and holds the target while preserving clearance from the runner's joint-limit warning band; it has no nominal floor contact or clipping.
