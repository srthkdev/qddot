# Debugging and controller design

1. Read the verdict's first failing spec, witness time, and operator window. A negative robustness indicates a violation; identify which signal constrained it.
2. Use `qddot why` for trends and nearby events. Query the full trace around the witness, then replay at the witness and immediately before it.
3. Make one evidence-based change at a time. Preserve the spec unless the requirement itself was wrong; update it explicitly when the user requirement changes.
4. Verify a nominal PASS, then run fuzzing with the task's perturbations. Review the worst cases and parameter correlations rather than relying only on average pass rate.
5. Report episode ids, pass rate, and any remaining failure modes.

Typical control fixes: correct joint ordering and radians/degrees; use the correct world/local frame; apply damping to oscillating position control; ramp targets instead of stepping them; respect `ctrlrange` and joint limits; and keep the controller's per-step work below its deadline.
