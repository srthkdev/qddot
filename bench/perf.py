"""Measure qddot's speed: simulation throughput per task and agent-facing command latency.

    python -m bench.perf [--out docs/performance.md]
"""
from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import mujoco

from qddot.sim.runner import run_episode
from qddot.task import load_task

REPO = Path(__file__).resolve().parent.parent
QDDOT = str(REPO / ".venv" / "bin" / "qddot")


def _median_s(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def task_throughput(root: Path) -> list[dict]:
    """Nominal episode wall time per built-in task, using its good controller."""
    rows = []
    for task_dir in sorted(p for p in (REPO / "tasks").iterdir() if (p / "task.yaml").is_file()):
        task = load_task(task_dir / "task.yaml")
        controller = task_dir / "controllers" / "good.py"
        model = mujoco.MjModel.from_xml_path(str(task.model_path))
        wall = _median_s(lambda: run_episode(task, controller, root=root))
        steps = round(task.duration / model.opt.timestep)
        rows.append({"task": task_dir.name, "dof": model.nv, "sim_s": task.duration,
                     "wall_s": wall, "realtime_x": task.duration / wall,
                     "steps_per_s": steps / wall})
    return rows


def _cli(*args: str, cwd: Path) -> float:
    start = time.perf_counter()
    subprocess.run([QDDOT, *args], cwd=cwd, capture_output=True, check=False)
    return time.perf_counter() - start


def command_latency(work: Path) -> list[dict]:
    """End-to-end CLI latency as an agent sees it (process start included)."""
    task = REPO / "tasks" / "cartpole_balance" / "task.yaml"
    buggy = task.parent / "controllers" / "buggy_velocity_sign.py"
    good = task.parent / "controllers" / "good.py"
    _cli("run", str(task), "--controller", str(buggy), cwd=work)
    _cli("run", str(task), "--controller", str(good), cwd=work)
    episode = next((work / ".qddot" / "runs").glob("*buggy_velocity_sign*"))
    sql = "select kind, count(*) from events group by kind"
    commands = {
        "qddot --help (startup)": ["--help"],
        "run (4 s cart-pole episode)": ["run", str(task), "--controller", str(buggy)],
        "why": ["why", str(episode)],
        "why --frames": ["why", str(episode), "--frames"],
        "query (events by kind)": ["query", sql],
        "diff": ["diff", str(episode), str(next((work / ".qddot/runs").glob("*good*")))],
        "fuzz -n 20 (all cores)": ["fuzz", str(task), "--controller", str(good), "-n", "20"],
        "similar (warm memory)": ["similar", str(episode)],
        "view (GIF + HTML)": ["view", str(episode)],
    }
    _cli("similar", str(episode), cwd=work)  # build the memory index once
    return [{"command": name, "latency_s": statistics.median(_cli(*args, cwd=work) for _ in range(3))}
            for name, args in commands.items()]


def render(throughput: list[dict], latency: list[dict]) -> str:
    machine = f"{platform.machine()} {platform.mac_ver()[0] or platform.release()}, {os.cpu_count()} cores"
    lines = [f"Measured with `python -m bench.perf` on {machine}, MuJoCo {mujoco.__version__}, "
             "median of 3 runs.", "",
             "| Task | DoF | Sim time | Wall time | Real-time factor | Physics steps/s |",
             "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['task']} | {r['dof']} | {r['sim_s']:.1f} s | {r['wall_s']:.2f} s | "
              f"{r['realtime_x']:.0f}x | {r['steps_per_s']:,.0f} |" for r in throughput]
    lines += ["", "| Command | Latency |", "|---|---:|"]
    lines += [f"| `{r['command']}` | {r['latency_s']:.2f} s |" for r in latency]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bench.perf")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="qddot-perf-") as tmp:
        work = Path(tmp)
        table = render(task_throughput(work / ".qddot"), command_latency(work))
    print(table)
    if args.out:
        args.out.write_text("# Performance\n\n" + table)


if __name__ == "__main__":
    main()
