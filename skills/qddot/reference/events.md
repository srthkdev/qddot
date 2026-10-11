# Signals, events, and query recipes

Each control step (including time zero) records float64 signals: `t`; `joint.<name>.pos` and `.vel`; `act.<name>.ctrl` (applied) and `.force`; task-declared signals; `col_<pair>` contact indicators; and event indicators `ev.collision`, `ev.joint_limit`, `ev.saturation`, `ev.deadline_miss`.

Events are stored with time, kind, primary/secondary party, value, and small JSON detail. Contact start/end are deduplicated by body pair. A contact must be absent for two physics steps before `contact_end`. Slip requires tangential speed above 0.01 m/s for two control steps. Joint limits trigger within 1% of the range or beyond. Saturation and joint-limit events emit on entry. Deadline-miss value is elapsed milliseconds. Non-finite state stops the run.

Useful SQL:

```sql
select t, "joint.elbow.pos", "joint.elbow.vel"
from signals where episode = 'EPISODE' order by t;

select t, kind, a, b, value from events
where episode = 'EPISODE' order by t;

select episode, passed, robustness from verdicts order by robustness;
```

Use quoted identifiers for signal names containing dots. `qddot replay EPISODE --at T` inspects state at a specific time; `qddot view EPISODE` creates an offline animation and report.
