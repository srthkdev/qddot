# Known controller failures: so101_box_push

### buggy_wrong_frame

Root cause: The DLS Cartesian error sign is reversed, so the gripper does not sustain the intended push into the goal region.

First failing spec: box_in_goal. Approximate first failure time: 2.5 s. Events: collision. Minimal fix: restore the nominal trajectory scaling, frame convention, gravity model, or force margin described above.

### buggy_units

Root cause: Push travel uses the approach duration as its time base, producing a premature high-speed contact that passes nominally but has low perturbation margin.

First failing spec: none nominally; perturbation-sensitive. Approximate first failure time: 2.5 s. Events: none required. Minimal fix: restore the nominal trajectory scaling, frame convention, gravity model, or force margin described above.
