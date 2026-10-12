"""Signal discovery and validation for task specifications."""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
from typing import Any

import mujoco

from . import parse
from .ast import Expr, Node
from .parser import SpecError


def task_signals(task: Any) -> dict[str, dict[str, str]]:
    """Return available signal metadata without running a simulation."""
    model = mujoco.MjModel.from_xml_path(str(task.model_path))
    signals: dict[str, dict[str, str]] = {"t": {"kind": "time", "units": "s"}}
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
        units = "rad" if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE) else "m"
        signals[f"joint.{name}.pos"] = {"kind": "joint position", "units": units}
        signals[f"joint.{name}.vel"] = {"kind": "joint velocity", "units": f"{units}/s"}
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or f"actuator_{aid}"
        signals[f"act.{name}.ctrl"] = {"kind": "actuator control", "units": "model units"}
        signals[f"act.{name}.force"] = {"kind": "actuator force", "units": "N"}
    for item in task.signals:
        kind = item["kind"]
        signals[item["name"]] = {"kind": kind.replace("_", " "), "units": "m"}
    for pair in task.contact_pairs:
        signals[f"col_{pair['name']}"] = {"kind": "contact indicator", "units": "boolean"}
    for event in ("collision", "joint_limit", "saturation", "deadline_miss"):
        signals[f"ev.{event}"] = {"kind": "event indicator", "units": "boolean"}
    return signals


def referenced_signals(stl: str) -> list[str]:
    """Return signal names in formula order, without duplicates."""
    tree = parse(stl)
    found: list[str] = []

    def visit_expr(expr: Expr | None) -> None:
        if expr is None:
            return
        if expr.op == "signal" and expr.value not in found:
            found.append(str(expr.value))
        visit_expr(expr.left)
        visit_expr(expr.right)

    def visit(node: Node) -> None:
        visit_expr(node.expr)
        if node.left is not None:
            visit(node.left)
        if node.right is not None:
            visit(node.right)

    visit(tree)
    return found


def validate(stl: str, available: dict[str, dict[str, str]]) -> list[str]:
    """Parse a formula and raise ``SpecError`` for unknown signal names."""
    names = referenced_signals(stl)
    unknown = [name for name in names if name not in available]
    if unknown:
        name = unknown[0]
        matches = get_close_matches(name, list(available), n=4, cutoff=0.35)
        suggestions = f" Close matches: {', '.join(matches)}." if matches else ""
        raise SpecError(f"Unknown signal {name!r}.{suggestions}", stl, stl.find(name) + 1)
    return names


def episode_signals(episode: str | Path, root: Path = Path(".qddot")) -> tuple[Path, dict[str, Any]]:
    """Resolve an episode id/path and read its recorded signal columns."""
    import duckdb

    candidate = Path(episode).expanduser()
    if not candidate.is_dir():
        candidate = root / "runs" / str(episode)
    signal_path = candidate / "signals.parquet"
    if not signal_path.is_file():
        raise FileNotFoundError(f"episode signals not found: {signal_path}")
    table = duckdb.connect(database=":memory:").execute(
        "SELECT * FROM read_parquet(?) ORDER BY t", [str(signal_path)]
    ).to_arrow_table()
    if table.num_rows == 0 or "t" not in table.column_names:
        raise ValueError(f"episode has no time samples: {signal_path}")
    return signal_path, {name: table[name].to_numpy() for name in table.column_names}


__all__ = ["episode_signals", "referenced_signals", "task_signals", "validate"]
