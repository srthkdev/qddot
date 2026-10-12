# Known controller and model failures

The challenge model carries an excessive forearm rotational inertia. The shape, mass and actuator limits remain plausible; the controller's inverse dynamics therefore predicts far less acceleration than the simulated arm produces. The fixed model restores a capsule-consistent inertia. Static model checks do not diagnose this discrepancy, which is deliberate: the other provided model variants isolate an incorrect elbow axis and an omitted shoulder range as inspection distractors.

| Variant | Expected behavior | Minimal repair |
|---|---|---|
| `buggy_stale_observation.py` | The controller reuses a five-cycle-old joint sample. It reaches the target nominally, but sensor latency and mass variation consume the settling margin; `target_reached` fails on perturbed trials. | Use the current observation for the inverse-dynamics state. |
| `buggy_joint_order.py` | Joint accelerations are paired in reverse order, producing a coupled but plausible wrong command; `target_reached` fails near 1 s with saturation. | Keep each acceleration aligned with its corresponding joint row. |
| `models/buggy_inertia.xml` | Good controller fails `target_reached` as the high inertia prevents settling by the required window. | Restore a realistic forearm inertia tensor. |
| `models/buggy_axis.xml` | Good controller fails `target_reached`; the elbow moves out of the task plane. | Restore the elbow hinge axis to the shared planar axis. |
| `models/buggy_unlimited.xml` | The good motion still passes, but static inspection flags the missing range as a warning. | Restore the shoulder range. |
