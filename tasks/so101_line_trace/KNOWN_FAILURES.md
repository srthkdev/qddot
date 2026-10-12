# Known controller failures: so101_line_trace

### buggy_units

Root cause: The phase is normalized by 1 s instead of the 2 s trace duration, so the halfway waypoint is passed early.

First failing spec: midline_waypoint. Approximate first failure time: 0.9 s. Events: none required. Minimal fix: restore the nominal trajectory scaling, frame convention, gravity model, or force margin described above.

### buggy_wrong_frame

Root cause: The Cartesian residual sign is reversed; the gripper moves away from the requested world-frame line.

First failing spec: start_point. Approximate first failure time: 0.9 s. Events: collision. Minimal fix: restore the nominal trajectory scaling, frame convention, gravity model, or force margin described above.

`buggy_noise_amplification.py` extrapolates delayed joint samples backward along their measured velocities before solving IK. The wrong direction correction degrades the narrow line tolerance only under delayed/noisy observations; use current-state feedback or a correctly signed bounded predictor.
