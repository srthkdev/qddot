"""Command line interface for qddot benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path

from bench.config import Condition, RunPlan
from bench.report import report
from bench.run import discover_tasks, parse_agents, run_plan


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bench")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--agents", default="claude_code:sonnet-5.5,codex:gpt-6-luna")
    run_parser.add_argument("--conditions", default="none,qddot")
    run_parser.add_argument("--tasks", default="all")
    run_parser.add_argument("--seeds", type=int, default=1)
    run_parser.add_argument("--private", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true", default=True)
    run_parser.add_argument("--yes", action="store_true")
    run_parser.add_argument("--jobs", type=int, default=1)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--n-fuzz", type=int, default=50)
    run_parser.add_argument("--timeout-s", type=float, default=900)
    run_parser.add_argument("--max-usd", type=float, default=None,
                            help="Per-trial spend cap where the agent CLI supports it")
    grade_parser = sub.add_parser("grade")
    grade_parser.add_argument("task", type=Path)
    grade_parser.add_argument("controller", type=Path)
    grade_parser.add_argument("--seed", type=int, default=900000)
    grade_parser.add_argument("--n-fuzz", type=int, default=50)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("files", nargs="+", type=Path)
    report_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if args.command == "run":
        tasks = discover_tasks(repo_root, args.tasks, args.private)
        plan = RunPlan(tasks, parse_agents(args.agents), [Condition(name) for name in args.conditions.split(",")], list(range(args.seeds)), jobs=args.jobs, private=args.private)
        if args.run_id:
            plan.run_id = args.run_id
        run_plan(plan, repo_root, yes=args.yes, dry_run=not args.yes, budget={"n_fuzz": args.n_fuzz, "timeout_s": args.timeout_s,
                                                                   "max_budget_usd": args.max_usd})
    elif args.command == "grade":
        import json

        from bench.grade import grade

        print(json.dumps(grade(args.task, args.controller, args.seed, args.n_fuzz), indent=2))
    else:
        out = args.output or repo_root / "bench" / "results" / args.files[0].stem
        print(*report(args.files, out), sep="\n")


if __name__ == "__main__":
    main()
