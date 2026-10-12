# Benchmark harness

The `bench/` package evaluates coding agents on robot-controller tasks. The public suite is listed in [`tasks/README.md`](../tasks/README.md); held-out fixtures live in gitignored `tasks_private/`.

## Conditions

| Condition | Agent receives |
|---|---|
| `none` | Task prompt and inputs; no simulation execution feedback |
| `stdout` | A nominal simulator helper that prints only final task numbers and “done” |
| `qddot` | qddot CLI and its workflow skill |
| `qddot_memory` | Currently the same tools and skill as `qddot`; reserved for a future memory condition |

Each trial receives a fresh temporary workdir containing the prompt, task, model inputs, and starter controller. Reference controllers, hidden failure descriptions, and grading files are excluded. The benchmark is designed to compare the value of feedback; it does not guarantee that workdir isolation is a security sandbox.

## Grading

The grader runs outside the agent workdir. It measures a nominal run, fuzz robustness over 50 hidden seeds, and task-specific control metrics including overshoot, settling time, and energy (`sum(abs(ctrl * qvel) * control_dt)`). It also records a **false-done** when the agent's final message claims success but the hidden grader reports failure.

Metrics are task dependent; some tracking metrics are best-effort and require target keys to match recorded signal names. See [`bench/README.md`](../bench/README.md) for fairness notes and implementation limitations.

## Run it

Inspect the planned grid first:

```sh
python -m bench run --dry-run
```

Example full command from the harness README (real agent calls require explicit `--yes`):

```sh
python -m bench run --agents claude_code:sonnet-5.5,codex:gpt-6-luna \
  --conditions none,qddot --tasks all --seeds 1 --dry-run
```

Then run with the desired grid and `--yes`. Results append to `bench/results/<run_id>.jsonl`; use `python -m bench grade TASK_DIR CONTROLLER.py` to grade one controller and `python -m bench report FILE.jsonl` to aggregate. Public benchmark results are coming soon; no result numbers are claimed here.
