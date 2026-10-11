# qddot

Run, check and debug robot code in physics before your coding agent claims it's done.

qddot gives coding agents (Claude Code, Codex, opencode) a fast verification loop for robotics code:

- **run:** run a controller headless and reproducibly in MuJoCo.
- **check:** check the run against temporal-logic specs, which yields a robustness score and the moment of violation.
- **fuzz:** randomize physics and timing (friction, mass, sensor noise, latency, jitter).
- **query:** query contacts, slips, joint-limit hits, saturation and missed deadlines with SQL.

It ships as one Python package that works as a library and a CLI. There are no servers: MuJoCo and DuckDB both run in-process.

> Status: early development (phase 1).

```bash
uv sync
uv run qddot run tasks/so101_reach/task.yaml --controller tasks/so101_reach/controllers/good.py
uv run qddot fuzz tasks/so101_reach/task.yaml --controller tasks/so101_reach/controllers/good.py -n 50
uv run qddot query "select kind, count(*) from events group by kind"
```

Robot models in `assets/` are vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) under their original licenses.
