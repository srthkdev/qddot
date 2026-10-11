# Known controller failures

Nominal values below come from the standalone loop in `tests/tasks/test_tasks.py` using the vendored UR5e scene and `home` keyframe.

| Controller | Root cause and first failing spec | Approximate time and observed events | Minimal fix |
|---|---|---|---|
| `buggy_joint_order.py` | Swaps shoulder-lift and elbow commands. `lift_settle` is the first failed spec; at 4 s lift is 0.258 rad rather than -1.35 rad, and elbow is -1.347 rad rather than 1.25 rad. | Tracking error is evident during the 2 s move. Both swapped targets remain inside their actuator ranges, so this nominal failure is expected to have no saturation event. | Map each target joint to its matching actuator name. |
| `buggy_degrees.py` | Converts radian-valued task targets as if they were degrees. `elbow_settle` fails; elbow ends at 0.030 rad rather than 1.25 rad. | Error is evident by the 2 s trajectory endpoint. The degree-scaled endpoints remain inside actuator ranges in this nominal run, so no event is expected; the failure is visible in the joint signals. | Use task target values directly in radians; remove `deg2rad`. |

The good controller's maximum error after t=2 s is 0.0123 rad across the three target joints and ends within 0.0123 rad of each target.
