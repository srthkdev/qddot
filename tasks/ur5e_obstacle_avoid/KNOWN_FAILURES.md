# Known controller failures

- `buggy_direct_path`: smooth direct joint interpolation crosses the box at about 1.58 s; `collision_free` fails with collision events. Check the swept links and route through the clear via pose.
- `buggy_age_scale`: the controller extrapolates delayed joint samples in the wrong direction and magnifies the resulting route error. It passes nominally but fails the reach or collision margin under latency fuzz; use bounded velocity prediction.
