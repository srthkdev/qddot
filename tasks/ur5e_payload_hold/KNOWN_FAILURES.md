# Known controller failures: ur5e_payload_hold

### buggy_missing_compensation

Root cause: The controller computes gravity bias from an otherwise identical model without the attached 2 kg payload; shoulder compensation is insufficient and the pose sags.

First failing spec: payload_pose. Approximate first failure time: 2.5 s. Events: none required. Minimal fix: restore the nominal trajectory scaling, frame convention, gravity model, or force margin described above.

### buggy_fragile_gain

Root cause: Joint velocity damping is reduced below the loaded arm tuning; it passes nominally but is sensitive to payload mass scaling.

First failing spec: none nominally; perturbation-sensitive. Approximate first failure time: 2.5 s. Events: none required. Minimal fix: restore the nominal trajectory scaling, frame convention, gravity model, or force margin described above.
