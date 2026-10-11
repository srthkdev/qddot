# Known controller failures

Nominal values below come from the standalone loop in `tests/tasks/test_tasks.py` using the bundled cart-pole model. `controller_budget_ms: 10` makes the deliberately slow bug reproducible; its 12 ms sleep exceeds that task default on every call.

| Controller | Root cause and first failing spec | Approximate time and observed events | Minimal fix |
|---|---|---|---|
| `buggy_velocity_sign.py` | Reverses pole angular velocity in the LQR state. `upright_and_hold` fails because the pole diverges instead of settling. | Pole exceeds the ±0.05 rad band immediately from the 0.1 rad start, actuator clipping begins near 0.14 s, and the cart crosses the 0.8 m rail bound near 0.48 s. Expect `saturation`, `joint_limit` at the cart stop and pole contact events as it falls. | Feed `joint.pole.vel` with its measured sign. |
| `buggy_slow.py` | Sleeps 12 ms in a controller with a 10 ms budget. `upright_and_hold` fails as the unbalanced pole falls. | `deadline_miss` begins at t=0 and persists; clipping begins near 0.09 s, rail bound crossing near 0.47 s. Expect `deadline_miss`, `saturation`, cart `joint_limit` and pole contact events. | Remove the sleep and keep the per-step computation below 10 ms. |

The good LQR first enters the 0.05 rad band near 0.19 s, ends at 0.00265 rad, keeps the cart within 0.206 m and has no nominal saturation.
