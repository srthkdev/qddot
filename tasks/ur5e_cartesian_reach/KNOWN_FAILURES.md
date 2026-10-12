# Known controller failures: ur5e_cartesian_reach

### `buggy_frame.py`

The Cartesian residual is negated before the damped least-squares solve, driving away from the target. `reach_hold` fails and collision events may occur. Restore the world-frame target-minus-measured residual.

### `buggy_feedback_gain.py`

The controller extrapolates delayed joint feedback backward with an excessive age multiplier before the DLS solve. It passes fresh feedback but loses target tolerance under latency. Use a bounded predictor with the correct velocity sign.
