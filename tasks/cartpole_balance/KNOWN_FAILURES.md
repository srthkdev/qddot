# Known controller failures

Nominal values below come from `qddot.sim.runner.run_episode` using the bundled cart-pole model. `controller_budget_ms: 10` is active on nominal runs; the deliberately slow bug's 12 ms sleep exceeds it on every call and each command is delayed by two control steps.

| Controller | Root cause and first failing spec | Approximate time and observed events | Minimal fix |
|---|---|---|---|
| `buggy_velocity_sign.py` | Reverses pole angular velocity in the LQR state. `upright_and_hold` fails because the pole diverges instead of settling. | Pole exceeds the ±0.05 rad band immediately from the 0.1 rad start, actuator clipping begins near 0.14 s, and the cart crosses the 0.8 m rail bound near 0.48 s. Expect `saturation`, `joint_limit` at the cart stop and pole contact events as it falls. | Feed `joint.pole.vel` with its measured sign. |
| `buggy_slow.py` | Sleeps 12 ms in a controller with a 10 ms budget. `upright_and_hold` fails as the unbalanced pole falls. | One `deadline_miss` entry event is emitted at t=0 while misses continue; clipping begins as the pole diverges, followed by rail and contact events. | Remove the sleep and keep the per-step computation below 10 ms. |

The good LQR first enters the 0.05 rad band near 0.19 s, ends at 0.00265 rad, keeps the cart within 0.206 m and has no nominal saturation.
