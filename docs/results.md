# Benchmark results (October 2026)

Every number here comes from `bench/results/*.jsonl`, regenerated with `python -m bench report`.

**Isolation:**
- Agents worked in self-contained workdirs, with qddot installed outside the repo.
- Reference controllers and seeded bugs were unreadable during the runs.
- Transcript audits found **0 contaminated trials** in the runs reported here.
- An earlier pilot was **discarded** because the audit caught an agent reading a reference controller (it is kept in `bench/results/contaminated/` for transparency).

## Hero task: ALOHA 2 bimanual cube handover

Codex CLI with GPT-6 Luna (high reasoning), 3 seeds per condition, graded on hidden seeds with 20 perturbed runs.

| Condition | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Print-only simulator (one run, final values only) | ❌, **claimed success** (false done) | ❌ | ❌ |
| qddot (CLI + MCP + skill) | ✅ 100% fuzz | ✅ 100% fuzz | ✅ 100% fuzz |

The seed-0 failure looks correct on screen: the cube is lifted and handed over. But the two grippers collide at t = 2.88 s, and the agent reported success. qddot's `no_arm_arm_collision` spec catches it. See `docs/media/hero/aloha.mp4`, rendered from the agents' actual controllers with `scripts/make_hero_video.py`.

## Pilot: 15 smaller tasks

One seed per cell. The tasks are arm reach and tracking, PID, LQR balance, a wheeled balancer, differential drive, a latency-sensitive servo, box pushing, obstacle avoidance and model repair.

| Agent | No simulator | Print-only sim | qddot |
|---|---:|---:|---:|
| Claude Code, Haiku 5.5 | 10/15 (67%) | **15/15 (100%)** | 14/15 (93%) |
| Codex, GPT-6 Luna high | 8/15 (53%) | 13/15 (87%) | **15/15 (100%)** |
| False "done" claims | 3 | 0 | 1 |

## What this does and does not show

- **Running the physics is the biggest single lever:** +33 to +47 points over writing robot code blind.
- **On these smaller tasks, qddot vs a bare simulator is mixed:** +13 points for Luna, −7 for Haiku (one task, `latency_servo`). With one seed per cell, this is within noise. Most of these tasks are solvable from final values alone.
- **On the hard, contact-rich bimanual task, structured verification decided it:** 3/3 vs 0/3, including a confident false claim of success.
- **Not yet measured:**
  - multiple seeds on the pilot;
  - more hard tasks;
  - frontier models (Opus 5.5, GPT-6 Astra);
  - open models;
  - the `qddot_memory` condition.
