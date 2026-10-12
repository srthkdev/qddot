# Known controller failures

The original pick-and-lift setup did not provide a reachable grasp pose: the measured SO-101 gripper pose was displaced from the block, and moving the block into the reachable pose caused initial jaw/table penetration. The benchmark is therefore a contact-rich open-jaw polymer block push, which is reliable in this model.

- `buggy_early_close`: drives the jaw beyond its closing stop before engaging the block; the impact tips the block. First failing spec: `box_not_tipped`; events include joint-limit and saturation. `qddot why` reports the stop event and contact loss. Fix: keep the jaw open during the pusher approach.
- `buggy_weak_grip`: raises the pusher approach by 4 cm, leaving a shallow corner contact and insufficient travel. First failing spec: `box_in_goal`; events include contact end and slip. `qddot why` reports lost contact. Fix: restore the nominal pusher height.
- `buggy_latency`: converts a sensor age in seconds as if it were milliseconds, creating a large state-estimation error only under delayed observations.
