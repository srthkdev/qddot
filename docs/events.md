# Signals and events

Signals are wide-format float64 values sampled once per controller step, including `t=0`.

| Group | Names |
|---|---|
| Time | `t` |
| Joint state | `joint.<joint>.pos`, `joint.<joint>.vel` |
| Actuators | `act.<actuator>.ctrl` (applied), `act.<actuator>.force` |
| Task-derived | The configured signal name |
| Contact groups | `col_<pair>` (1 when any grouped contact exists) |
| Event indicators | `ev.collision`, `ev.joint_limit`, `ev.saturation`, `ev.deadline_miss` |

Event indicators are step-level values. Controller observations omit `ev.*` and `col_*` indicators.

## Event records

`events.parquet` has `t`, `kind`, `a`, `b`, `value`, and `detail` columns. Contact start/end pairs are deduplicated by body pair. Contact absence must persist for two physics steps before `contact_end` is emitted. A persistent contact with tangential relative speed over 0.01 m/s for two consecutive control steps emits `slip`.

| Kind | Meaning |
|---|---|
| `contact_start`, `contact_end` | Body-pair transitions; start value is normal force |
| `slip` | Sustained tangential contact motion |
| `collision` | Declared contact-pair collision or penetration over 5 mm |
| `joint_limit` | Limited joint enters the final 1% of its range or moves beyond it |
| `saturation` | Controller command is clipped to actuator `ctrlrange` |
| `deadline_miss` | Controller computation exceeds its configured budget; value is milliseconds |
| `nan_state` | Non-finite position or velocity; the episode stops |
| `controller_error` | Controller raised; traceback is recorded in verdict errors |

Transition events are emitted on state change to keep logs compact. When a command misses its deadline, the measured new command is queued for `ceil(compute_time / control_dt)` steps later; the preceding command stays active until then. Jitter adds a physics-step actuation delay.

Episode artifacts are `.qddot/runs/<episode>/signals.parquet`, `events.parquet`, `verdict.json`, `meta.json`, and `log.mcap`. See [Architecture](architecture.md) for the data layout and determinism details.
