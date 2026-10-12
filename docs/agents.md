# Coding agents

qddot's agent workflow gives an agent concrete physics, spec, and execution feedback in its project directory.

## Install the skill

```sh
qddot skill install --codex
qddot skill install --claude
# Project-local Claude Code installation:
qddot skill install --project
```

The skill directs the agent to inspect the task/model, read the STL English back-translation, run the controller, diagnose failures with `why`/query/replay, fix, fuzz at least 50 perturbations, and report a PASS episode ID with its fuzz pass rate. `--force` replaces a different existing installation.

Current packaging note: the installer locates `skills/qddot/SKILL.md` relative to the source tree, but the package build does not include that directory. Skill installation therefore works from a source checkout; a packaged installation needs the skill assets bundled before this command can work there.

## MCP

Run `qddot mcp` as an MCP stdio process. Available tools wrap `run`, `fuzz`, `why`, `diff`, `query`, `spec`, `check`, `replay`, and `similar`. Responses are compact structured results, not raw signal arrays. Configure the MCP client to launch the installed `qddot` executable with command `qddot` and argument `mcp`; client configuration varies by agent.

## Prompting pattern

Give the agent a measurable goal and point it at the task and controller. Ask it to keep the task spec aligned with the requested behavior, then require evidence from qddot:

```text
Implement the controller in controller.py for tasks/my_task/task.yaml.
First inspect the model and read the task specs with qddot spec. Update specs only
if the stated goal requires it. Run qddot, diagnose any failure with why/query/replay,
and iterate. Then run at least 50 fuzz samples. Report the PASS episode ID and
the fuzz pass rate; do not claim completion without both.
```

For controller work, keep the task, model, and output directory accessible in the agent's working project. Episode data stays under `.qddot/`, so the agent can inspect and query it locally. Treat a pass rate as evidence over sampled configured perturbations, not proof of physical-robot safety.
