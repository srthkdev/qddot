# Specifications (STL)

Task specs use discrete-time Signal Temporal Logic evaluated at control samples. Bounds are seconds and windows are truncated at the end of the recorded trace.

## Grammar

- Comparisons: `sig < c`, `sig <= c`, `sig > c`, `sig >= c`, `sig == c`.
- Arithmetic: signals, constants, parentheses, unary `+` / `-`, `abs(expr)`, addition, subtraction, multiplication, division.
- Boolean: `not`, `and`, `or`, `implies`, and parentheses.
- Temporal: `always`, `eventually`, `until`, optionally followed by `[lower,upper]`.

Examples:

```text
always(abs(joint.elbow.pos) < 1.2)
eventually[0,2.5](always[0,1](dist_ee_target < 0.02))
joint.gripper.pos > 0.2 until[0,3](cube_z > 0.1)
```

Unbounded temporal operators range to the end of the trace. `until` means the left condition holds through the point where the right condition becomes true. Comparisons are strict syntactically, but zero robustness is considered passing.

## Robustness and witnesses

The evaluator follows quantitative STL robustness: comparisons produce a signed margin (for example `x < 2` has robustness `2 - x`); conjunction takes the minimum, disjunction the maximum, `not` negates, `always` takes a minimum over its window, and `eventually` takes a maximum. A result passes when robustness at time zero is nonnegative.

Each spec verdict reports its robustness at time zero, a `witness_t` sample that determines that result, and a `window` for the outermost temporal operator. A witness is useful for inspecting signals, but does not by itself imply a causal explanation. For nested formulas, the reported window belongs to the first temporal operator from the root.

The actual CLI's cart-pole back-translation is:

```text
upright_and_hold: eventually[0,2](always[0,2](abs(joint.pole.pos) < 0.05))
  within 2 s, |joint.pole.pos| drops below 0.05 and stays there for 2 s
```

## Read-back and errors

Run `qddot spec TASK` to print each formula and its deterministic English rendering. Read it before trusting the spec: back-translation is a template aid, not a proof that the requirement matches your intent. Unknown signals raise `SpecError` with suggestions and available names.

## Pitfalls

- Signal names must match recorded columns exactly; joint signals are `joint.<name>.pos` and `.vel`.
- Bounds are seconds, not sample counts. The trace is sampled at the task's `control_dt`.
- A bounded window near the end is truncated; it does not require samples beyond the trace.
- `==` robustness is `-|lhs-rhs|`; exact equality is fragile for noisy or floating-point quantities. Prefer a tolerance band such as `abs(x - target) < epsilon`.
- A spec can pass with a small positive margin. Inspect robustness, witness, and signals instead of treating PASS as a broad safety guarantee.
