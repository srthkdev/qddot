# qddot — engineering contract (read fully before coding)

qddot lets a coding agent **run, check and debug robot code in physics** before claiming it is done.

- **Delivery:** one Python package that is a library and a CLI (`qddot ...`).
- **No servers:**
  - MuJoCo runs in-process.
  - DuckDB is embedded.
  - All data lives in `.qddot/` inside the user's project.

## Rules for every contributor

- **Python environment:**
  - Python 3.12, managed by `uv`.
  - Run things with `uv run ...`.
  - Run tests with `uv run pytest -q`.
- **Dependencies:**
  - Already installed: mujoco 3.15, mcap, pyarrow, duckdb, numpy, typer, pyyaml, rtamt, rich, pillow, pytest, ruff.
  - **Do not add or upgrade dependencies, and do not edit `pyproject.toml`.** The network is unavailable.
  - If you truly need a new dependency, say so in your final message.
- **Ownership:** only create or edit the files your task assigns to you. Other agents are working in parallel on other modules.
  - If you need something from another module that doesn't exist yet, code against the contract below.
  - In tests, use a small local fake.
- **Code quality:**
  - Write production-quality code: type hints, small functions, clear names.
  - Docstrings only where they help.
  - No dead code.
  - `uv run ruff check` must be clean for your files.
- **Testing:**
  - Every module ships with pytest tests under `tests/` (file names given per task).
  - Tests must be fast (< 20 s total per module) and deterministic.
- **Output:** be concise and agent-friendly.
  - Short text by default.
  - `--json` for machine output.
  - Never dump raw arrays.
- **Determinism:** same task + controller + seed + perturb ⇒ bitwise-identical signals on the same machine and MuJoCo version.
  - Seed every RNG explicitly; use `numpy.random.default_rng(seed)`.
- **Repository:** do not commit. The orchestrator commits.

## Layout

```
src/qddot/
  __init__.py          (orchestrator)
  task.py              Task model + YAML loader                [runner agent]
  sim/runner.py        run_episode()                           [runner agent]
  sim/events.py        per-step event detection                [runner agent]
  sim/perturb.py       physics + timing perturbations          [runner agent]
  sim/fuzz.py          fuzz()                                  [runner agent]
  record/writer.py     Parquet + MCAP + verdict writers         [runner agent]
  spec/                STL parser, evaluator, robustness, english  [spec agent]
  check/static.py      check_model()                           [cli agent]
  store.py             DuckDB views over .qddot/runs         [cli agent]
  cli.py               Typer app, entry point `qddot`        [cli agent]
tasks/<task_name>/     built-in benchmark tasks                 [tasks agent]
assets/                vendored MuJoCo Menagerie models (SO-101, UR5e). Read-only.
tests/                 tests/<module>/test_*.py
```

## Contracts

### 1. Task file (`tasks/<name>/task.yaml`)

Paths are relative to the task file.

```yaml
name: so101_reach
description: Move the gripper to the target and hold.
model: ../../assets/robotstudio_so101/scene.xml   # or a model.xml next to the task
duration: 4.0            # simulated seconds
control_dt: 0.02         # controller period in seconds (must be a multiple of model timestep)
init:                    # optional
  keyframe: home         # optional keyframe name
  qpos: {shoulder_lift: -0.5}   # optional per-joint overrides (joint name -> value)
target: {x: 0.2, y: 0.0, z: 0.15}   # optional free-form dict passed to the controller as info["target"]
signals:                 # extra derived signals (built-in signals are always recorded)
  - {name: ee_x, kind: site_pos, ref: gripperframe, axis: x}
  - {name: dist_ee_target, kind: distance, ref: gripperframe, to: target}   # `to` = a site/body name or "target"
  - {name: cube_z, kind: body_pos, ref: box, axis: z}
contact_pairs:           # optional named pairs that produce collision indicator signals
  - {name: arm_table, a: [upper_arm, lower_arm, wrist, gripper], b: [floor]}   # body names
perturb:                 # default ranges used by `fuzz` (all optional)
  friction_scale: [0.7, 1.3]
  mass_scale: [0.9, 1.1]
  obs_noise_std: 0.002
  latency_ms: [0, 40]          # sensor-to-controller delay, rounded to control steps
  jitter_ms: 5                 # std of extra actuation delay, rounded to physics steps
  controller_budget_ms: 15     # wall-clock compute budget per control step (deadline)
specs:
  - name: reach
    stl: "eventually[0,2.5](always[0,1](dist_ee_target < 0.02))"
  - name: no_table_collision
    stl: "always(col_arm_table < 0.5)"
```

`task.py` exposes:

- `@dataclass Task` holding those fields, plus `path: Path` and `model_path: Path`.
- `load_task(path: str | Path) -> Task`.
  - Validates the fields.
  - Raises `TaskError` with a clear message on a bad file.

### 2. Signal naming

Signals are wide-format columns. They are all float64 and sampled once per **control step**, and include t=0.

| Signal | Name |
|---|---|
| Time | `t` |
| Joints (each named hinge/slide joint) | `joint.<name>.pos`, `joint.<name>.vel` |
| Actuators | `act.<name>.ctrl` (applied), `act.<name>.force` |
| Task-declared signals | their `name` |
| Contact pairs | `col_<pair_name>`: 1.0 if any contact between the groups at that step, else 0.0 |
| Event indicators | `ev.collision`, `ev.joint_limit`, `ev.saturation`, `ev.deadline_miss`: 1.0 if that event kind occurred during that control step |

### 3. Controller API (user code)

A Python file that defines `class Controller`:

```python
class Controller:
    def reset(self, info: dict) -> None: ...      # optional
    def step(self, obs: dict[str, float], t: float) -> dict[str, float]: ...
```

`info` contains:

- `actuators`: list of names
- `joints`: list of names
- `control_dt`
- `target`: the task's target dict, or `{}`
- `model_path`: str

`obs` maps every signal name (except `ev.*` and `col_*`) to its current value. Latency and noise apply when perturbed.

`step` returns `{actuator_name: ctrl}`. Missing actuators hold their previous ctrl. Values are clipped to `ctrlrange` (emitting a `saturation` event when clipped).

Controller exceptions are caught: the episode stops, `verdict.errors` records the traceback (last 20 lines), and `passed=False`.

### 4. Episode outputs: `.qddot/runs/<episode_id>/`

`episode_id` = `<task>-<controller_stem>-s<seed>-<6 hex of params hash>`.

| File | Contents |
|---|---|
| `signals.parquet` | wide table (section 2) |
| `events.parquet` | see the schema below |
| `verdict.json` | see the schema below |
| `log.mcap` | JSON-encoded channels `/signals` (one message per control step) and `/events` |
| `meta.json` | task path, controller path + sha256, seed, perturb params actually sampled, env info |

`events.parquet` columns:

| Column | Type | Meaning |
|---|---|---|
| `t` | float64 | time of the event |
| `kind` | str | one of the kinds below |
| `a` | str | primary body/geom/joint/actuator |
| `b` | str or null | second party, if any |
| `value` | float64 | normal force N / slip speed m/s / penetration m / overshoot rad / ms |
| `detail` | str | small JSON |

Event kinds:

- **`contact_start` / `contact_end`:** body pair, deduplicated per body pair; `value` = normal force at start.
- **`slip`:** a persisting contact whose tangential relative speed > 0.01 m/s for at least 2 consecutive control steps.
- **`collision`:** contact_start between bodies in a declared `contact_pairs` group, or `penetration` > 5 mm.
- **`joint_limit`:** a limited joint within 1% of its range of a limit, or beyond it; emitted on entry.
- **`saturation`:** controller output clipped to ctrlrange; emitted on entry per actuator.
- **`deadline_miss`:** controller wall time > `controller_budget_ms`; `value` = ms.
- **`nan_state`:** non-finite qpos/qvel. The episode stops.
- **`controller_error`:** the controller raised.

Events are emitted on **state change**, not every step, to keep tables small.
Contact absence is debounced: a contact must be absent for two consecutive physics
steps before `contact_end` is emitted. When controller computation exceeds its
budget, the new command is queued for `ceil(compute_time / control_dt)` control
steps later; the previous command remains applied until then. This gives a late
command a physical effect while preserving the measured controller output. A
scalar `controller_budget_ms` in a task is active on nominal runs; an explicit
`controller_budget_ms=null` disables it.

`verdict.json`:

```json
{
  "episode": "...", "task": "...", "controller": "path", "seed": 0,
  "passed": false,
  "robustness": -0.013,
  "specs": [{"name": "reach", "stl": "...", "english": "...", "passed": false,
             "robustness": -0.013, "witness_t": 1.84, "window": [1.5, 2.5]}],
  "first_violation": {"spec": "reach", "t": 1.84},
  "events_summary": {"contact_start": 3, "slip": 1},
  "errors": [],
  "perturb": {...},
  "env": {"mujoco": "3.15.0", "python": "3.12.x", "platform": "macOS-arm64", "qddot": "0.1.0"},
  "stats": {"control_steps": 200, "physics_steps": 1000, "wall_s": 0.21}
}
```

`passed` = all specs passed and no errors. `robustness` = min over specs.
`first_violation.spec` is the first failing spec in task declaration order;
its `t` is that spec's witness time.

### 5. Spec package API (`qddot.spec`)

- `parse(stl: str) -> Node`: raises `SpecError` with position info.
- `to_english(stl: str) -> str`: deterministic, template-based back-translation.
- `evaluate(stl: str, t: np.ndarray, signals: Mapping[str, np.ndarray]) -> SpecResult`. The dataclass has:
  - `passed`
  - `robustness: float` (the robustness at time 0)
  - `witness_t: float | None`: the time that determined the robustness (the critical sample)
  - `window: tuple[float, float] | None`: the time interval the critical operator ranged over
- `evaluate_specs(specs: list[dict], t, signals) -> list[dict]`: dicts shaped like `verdict.specs` entries.

Grammar (discrete time, samples at control steps, bounds in seconds):

- Atoms: `sig < c`, `sig <= c`, `sig > c`, `sig >= c`, `sig == c` (with an `abs(...)` helper), and simple arithmetic `a - b`, `a + b`, `k * a` over signals and constants.
- Boolean: `not`, `and`, `or`, `implies`, and parentheses.
- Temporal: `always`, `eventually`, `until`, each with an optional `[a,b]` bound. With no bound they cover the rest of the trace.
- Unknown signal names raise `SpecError`, listing the available names closest to the unknown one.

Quantitative semantics are the standard STL robustness (Donzé–Maler); bounded windows are truncated at the trace end.

### 6. Other APIs

- `qddot.sim.runner.run_episode(task: Task, controller_path: Path, seed: int = 0, perturb: dict | None = None, root: Path = Path(".qddot"), record_mcap: bool = True) -> EpisodeResult`, where `EpisodeResult` has `episode_dir: Path` and `verdict: dict`.
  - `perturb=None` means nominal (no perturbation).
  - A dict holds explicit sampled values, e.g. `{"friction_scale": 0.8, "mass_scale": 1.0, "obs_noise_std": 0.0, "latency_ms": 20, "jitter_ms": 0, "controller_budget_ms": null}`.
- `qddot.sim.perturb.sample(task: Task, rng) -> dict`: samples concrete values from `task.perturb` ranges.
- `qddot.sim.fuzz.fuzz(task, controller_path, n: int, workers: int = os.cpu_count(), seed: int = 0, root=...) -> dict`.
  - Uses a process pool; episode i uses seed `seed + i` and `sample(task, default_rng(seed + i))`.
  - Returns `{"n", "passed", "pass_rate", "worst": [top 5 by robustness with episode ids + perturb], "by_param": {param: {"pass_corr": float}}}`.
- `qddot.check.static.check_model(path) -> list[dict]`. Each dict is `{severity: error|warning|info, code, message, element}`. Checks:
  - load errors
  - bodies with mass ≤ 1e-6 or invalid inertia (triangle inequality)
  - hinge/slide joints that are actuated but unlimited
  - actuator ctrlrange outside the joint range for position actuators
  - initial penetration > 1 mm at the default/first keyframe
  - timestep > 0.01
  - geoms with zero size
  - free bodies starting in the air (info)
- `qddot.store.connect(root=Path(".qddot")) -> duckdb.DuckDBPyConnection` with views:
  - `signals`: all episodes, plus an `episode` column
  - `events`: all episodes, plus an `episode` column
  - `verdicts`: one row per episode, with flattened top-level fields and `specs` as JSON

## CLI (Typer, `qddot`)

| Command | Does |
|---|---|
| `run TASK --controller FILE [--seed N] [--perturb k=v ...] [--json]` | Prints a ≤ 12-line verdict: PASS/FAIL, robustness, per-spec line with witness time, top events, episode id |
| `fuzz TASK --controller FILE [-n 50] [--workers W] [--seed 0] [--json]` | Prints pass rate, worst 5, and the param most correlated with failure |
| `check MODEL [--json]` | Static model checks |
| `spec TASK` | Lists specs with their English back-translations |
| `query "SQL" [--json] [--limit 50]` | DuckDB over the views |
| `runs [--last 10]` | Lists recent episodes |

Phase 2 commands (`why`, `diff`, `replay`, `view`, `similar`) are not in scope yet. Leave clean extension points.

## Phase 2 contract

### CLI structure (after the fix-up refactor)

- `src/qddot/cli.py` only builds the Typer app and registers commands from `src/qddot/commands/<name>.py`.
- Each command module exposes `register(app: typer.Typer) -> None`.
- To add a command, create its module and add one import line to the `COMMANDS` list in `cli.py`; nothing else changes.
- Shared output helpers live in `src/qddot/commands/_out.py`:
  - plain, non-wrapping output: `rich` with `soft_wrap=True`, `markup=False`, `highlight=False`;
  - `emit_json`;
  - exit codes (0 pass, 1 fail, 2 error).

### Diagnosis (`src/qddot/diag/`)

#### `why EPISODE`

`why(episode_dir) -> dict`, printed in ≤ 25 lines.

Contents:

- **Verdict line.**
- **First failing spec:** its English form, plus the witness time and window.
- **Spec signals:** for each signal referenced by the failing spec, the value at the witness, the min and max in the window, and the trend over the 1 s before the witness (rising / falling / oscillating / flat).
- **Nearby events:** events within ±0.5 s of the witness, ranked by proximity, plus first occurrences of saturation / joint_limit / slip / contact_end / deadline_miss before the witness.
- **"likely causes":** rule-based hypotheses, each with evidence and a confidence of low / med / high. The rules:
  - saturation preceding the error growth → actuator limit or gains too high;
  - joint_limit → limit not respected;
  - error growing monotonically from the start → sign or frame error;
  - oscillation with growing amplitude → gain too high or instability;
  - contact_end / slip on the object before failure → lost grip;
  - deadline_miss → controller too slow;
  - nan_state → numerical blow-up;
  - obs latency perturbed and failing only under perturbation → latency sensitivity.
- **2–3 suggested next commands:** concrete `qddot query` SQL and `replay` lines.

#### Keyframes

`keyframes(episode_dir, times) -> list[Path]` re-simulates deterministically from `meta.json` (same seed and perturb) and renders PNGs with `mujoco.Renderer` (default 320×240, with a fixed camera that looks at the robot). Renders go to `<episode>/keyframes/t_<ms>.png`.

`why --frames` renders frames at witness−0.5 s, the witness, and the end. Each frame comes with a **text visual-diff**:

- body positions that moved more than 1 cm between frames;
- contacts present or absent.

This gives a text diff, not pixels. An optional `--vlm` hook calls a pluggable function, `qddot.diag.vlm.describe(images) -> str`. It is disabled unless a provider is configured, and is never required.

#### `diff A B`

Reports:

- per-signal first divergence time beyond a tolerance (`1e-6` absolute by default);
- events only in A / only in B (compared by kind and parties);
- spec robustness deltas;
- differences in controller sha256, perturb and seed.

#### `replay EPISODE --at T`

Deterministically re-simulates to T and saves `mj_getState(mjSTATE_INTEGRATION)` to `<episode>/state_<ms>.npy`. It prints the joint positions and velocities and the active contacts at T. `--frame` renders a PNG.

### Integrations

#### Skills

`skills/qddot/SKILL.md` is an agent-facing workflow:

1. Write or confirm the specs; read the English back-translation.
2. `qddot run`.
3. On FAIL: `qddot why`, then query and replay, then fix.
4. `qddot fuzz` must pass at ≥ 90% before claiming done.
5. Never claim done without a PASS verdict id.

`qddot skill install [--claude|--codex] [--project]` copies the skill to `~/.claude/skills/qddot/`, `~/.codex/skills/qddot/`, or `./.claude/skills/qddot/`.

#### `qddot mcp`

An MCP server (stdio) built with `mcp.server.fastmcp.FastMCP`. Its tools are `run`, `fuzz`, `why`, `diff`, `query`, `spec`, `check`, `replay`. They are thin wrappers around the same library functions and return compact dicts. They never return raw arrays.

#### `qddot view EPISODE`

Writes `<episode>/replay.gif`, rendered with Pillow at ~20 fps, plus `replay.html`. The HTML is a self-contained page showing the GIF, the verdict and an events timeline.

### Benchmark (`bench/`, a separate top-level package that imports qddot)

**Task sets:**

- Public tasks live in `tasks/`.
- Held-out tasks live in `tasks_private/`, which is gitignored.
- Each task has a `PROMPT.md`: the user-facing request given to the agent. It contains no hints from `KNOWN_FAILURES.md`.
- Each task has a `controllers/starter.py` stub.

**Agent adapters** (`bench/agents/*.py`) run a headless CLI inside a fresh temporary workdir with a time and turn budget:

- `claude_code` (`claude -p ... --output-format json`)
- `codex` (`codex exec --json -m <model>`)
- `opencode` (`opencode run`)
- `fake` (a scripted agent for tests)

**The workdir** contains `PROMPT.md`, `task.yaml`, the model files and `controller.py` (copied from the starter). Under no circumstances does it contain `good.py`, the buggy controllers, or `KNOWN_FAILURES.md`.

**Conditions:**

| Condition | The agent gets |
|---|---|
| `none` | Cannot execute simulations; a code-only instruction |
| `stdout` | A `sim.py` that runs nominally and prints only the final distance/angle numbers and "done" |
| `qddot` | The full CLI + skill |
| `qddot_memory` | The full CLI + skill, plus `similar` once it exists (optional) |

**Grading** (`bench/grade.py`) runs on the agent's final `controller.py` with qddot outside the agent:

- the nominal run;
- `fuzz` with hidden seeds (n = 50);
- control metrics: overshoot, settling time, energy (sum |ctrl·qvel|·dt);
- **false-done**: the agent claimed success (parsed from its final message), but the hidden grader says it failed.

**Results** are appended to `bench/results/<run_id>.jsonl`. `bench/report.py` aggregates them into markdown tables and matplotlib charts.

**CLI:**

- `python -m bench run --agents claude_code:sonnet-5.5,codex:gpt-6-luna --conditions none,qddot --tasks all --seeds 1 --dry-run`
- `--dry-run` prints the planned grid and a cost estimate placeholder. Nothing calls real agents unless `--yes` is given.

### Runner hooks for re-simulation (added by the orchestrator)

`run_episode(..., on_step=None, compute_ms_override=None)`:

- **`on_step(control_index, t, model, data)`:** called once with index -1 after the initial state, then after each control step.
- **`meta.json["compute_ms"]`:** stores the measured controller compute time per control step.
- **`compute_ms_override`:** for exact re-simulation (`replay`, `view`, keyframes), pass `compute_ms_override=meta["compute_ms"]`. It reproduces the recorded deadline effects instead of re-measuring wall time.
- **Output location:** re-simulation should write to a temporary root, so the original episode is not overwritten.
