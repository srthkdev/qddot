# Known controller failures: so101_box_push

### `buggy_wrong_frame.py`

The Cartesian error sign is reversed, so the gripper misses the goal-directed push. `box_in_goal` fails and collision events are recorded. Restore the measured-minus-target convention used by the good controller.

### `buggy_stale_push_clock.py`

The pusher trajectory is retarded by 200 times sensor age. It passes with fresh nominal observations but delays contact under latency perturbations until the box misses `box_in_goal`. Use the controller clock or a bounded age correction.
