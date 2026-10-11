"""Aggregate benchmark JSONL files into Markdown and a chart."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def report(files: list[Path], output_dir: Path) -> tuple[Path, Path]:
    rows = [json.loads(line) for file in files for line in file.read_text().splitlines() if line.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        cfg = row["config"]
        groups[(cfg["agent"], cfg["condition"])].append(row)
    lines = ["# qddot benchmark report", "", "| Agent | Condition | Success | Fuzz pass rate | False-done rate | Tokens in/out | Cost USD | Wall seconds |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (agent, condition), items in sorted(groups.items()):
        n = len(items)
        success = sum(bool(item.get("success")) for item in items) / n
        fuzz_rate = sum(float(item.get("fuzz_pass_rate", 0)) for item in items) / n
        false_done = sum(bool(item.get("false_done")) for item in items) / n
        tin = sum(int(item.get("agent_result", {}).get("tokens_in", 0)) for item in items)
        tout = sum(int(item.get("agent_result", {}).get("tokens_out", 0)) for item in items)
        costs = [item.get("agent_result", {}).get("cost_usd") for item in items]
        cost = sum(float(value) for value in costs if value is not None)
        wall = sum(float(item.get("agent_result", {}).get("wall_s", 0)) for item in items)
        lines.append(f"| {agent} | {condition} | {success:.1%} | {fuzz_rate:.1%} | {false_done:.1%} | {tin}/{tout} | {cost:.4f} | {wall:.2f} |")
    markdown = output_dir / "report.md"
    markdown.write_text("\n".join(lines) + "\n")
    chart = output_dir / "success.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agents = sorted({agent for agent, _ in groups})
    conditions = sorted({condition for _, condition in groups})
    width = 0.8 / max(len(conditions), 1)
    fig, axis = plt.subplots(figsize=(max(5, len(agents) * 1.6), 3.5))
    for index, condition in enumerate(conditions):
        values = []
        for agent in agents:
            items = groups.get((agent, condition), [])
            values.append(sum(bool(item.get("success")) for item in items) / len(items) if items else 0)
        positions = [i + (index - (len(conditions) - 1) / 2) * width for i in range(len(agents))]
        axis.bar(positions, values, width=width, label=condition)
    axis.set_xticks(range(len(agents)), agents, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Success rate")
    axis.legend()
    fig.tight_layout()
    fig.savefig(chart)
    plt.close(fig)
    return markdown, chart
