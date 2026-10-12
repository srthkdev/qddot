# Known controller failures: ur5e_cartesian_reach

### buggy_frame

Root cause: The world-frame Cartesian residual is negated before the DLS solve, driving away from the target.

First failing spec: reach_hold. Approximate first failure time: 2.5 s. Events: collision. Minimal fix: restore the nominal trajectory scaling, frame convention, gravity model, or force margin described above.

### buggy_units

Root cause: The resolved-rate update uses a tenfold gain; it passes nominally but loses margin under sampled mass and observation perturbations.

First failing spec: none nominally; perturbation-sensitive. Approximate first failure time: 2.5 s. Events: none required. Minimal fix: restore the nominal trajectory scaling, frame convention, gravity model, or force margin described above.
