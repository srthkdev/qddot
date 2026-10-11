# qddot benchmark harness

Run `python -m bench run --dry-run` to inspect a grid. Real agent runs require `--yes`; use `--agents`, `--conditions`, `--tasks`, `--seeds`, `--jobs`, and `--private` to choose the grid. Each run appends JSONL trials to `bench/results/<run_id>.jsonl`. Grade a controller with `python -m bench grade TASK_DIR CONTROLLER.py`; aggregate with `python -m bench report FILE.jsonl`.

Conditions are `none` (code only), `stdout` (three final signal values only), `qddot` (CLI and workflow skill), and `qddot_memory` (same inputs, with future memory integration reserved). The task workdir contains only prompt, task description, model inputs, controller starter, and condition tools. Grading runs outside that workdir with hidden seeds.

Fairness rules: every grid cell receives a fresh workdir; hidden task files and reference controllers are excluded; agent execution is opt-in; the fake adapter is for deterministic harness tests. The qddot CLI path is documented only in qddot conditions.

Limitations: workdir isolation is filesystem-level by construction but not a security sandbox. Control energy maps actuator names to joint velocity names by exact name first, then substring matching; it sums absolute control times velocity times `control_dt`. Overshoot and settling time are best-effort and are computed only when task target keys match recorded signal names (or their `_x`/`_z` variants). No price table is assumed, so dry-run cost is reported as unavailable. OpenCode token/cost reporting depends on its emitted event format.
