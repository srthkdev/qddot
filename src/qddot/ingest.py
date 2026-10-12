"""Turn recorded robot data into qddot episodes, so specs, `why`, `query` and `diff` work on real runs.

Sources:
- a LeRobot dataset directory (`meta/info.json` + `data/**/*.parquet`);
- an MCAP file: qddot's own JSON `/signals` channel, or ROS 2 `sensor_msgs/JointState`
  when `mcap-ros2-support` is installed;
- a CSV or Parquet table with a time column (`t`, `timestamp` or `time`).

Signals keep qddot's names where the source says what they are (LeRobot feature
names become `joint.<name>.pos` / `act.<name>.ctrl`); `rename` maps anything else.
Contacts cannot be recovered from logs, so only joint-limit and saturation events
are derived (when a task model gives the ranges).
"""

from __future__ import annotations

import json
import platform
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from qddot import __version__
from qddot.record.writer import write_episode
from qddot.sim.backend import load_backend
from qddot.sim.events import EventDetector
from qddot.task import Task, load_task

TIME_COLUMNS = ("t", "timestamp", "time")


def ingest(source: str | Path, task: Task | str | Path | None = None, episode: int = 0,
           rename: dict[str, str] | None = None, root: Path = Path(".qddot"),
           name: str | None = None) -> dict[str, Any]:
    """Write an episode from recorded data and return its verdict (specs only if a task is given)."""
    source = Path(source)
    times, columns = _read(source, episode)
    for new, old in (rename or {}).items():
        columns[new] = columns.pop(old)
    task = load_task(task) if isinstance(task, (str, Path)) else task
    rows = [{"t": float(t), **{k: float(v[i]) for k, v in columns.items()}} for i, t in enumerate(times)]
    events = _events(task, times, columns) if task else []
    for row, kinds in zip(rows, _indicators(times, events), strict=True):
        row.update({f"ev.{k}": float(k in kinds) for k in ("joint_limit", "saturation")})
    specs = []
    if task and task.specs:
        from qddot.spec import evaluate_specs

        signals = {k: np.asarray([r[k] for r in rows]) for k in rows[0] if k != "t"}
        specs = evaluate_specs(task.specs, np.asarray(times), signals)
    episode_id = name or f"{(task.name if task else source.stem)}-ingest-{source.stem}-e{episode}"
    failing = [s for s in specs if not s["passed"]]
    summary: dict[str, int] = {}
    for event in events:
        summary[event["kind"]] = summary.get(event["kind"], 0) + 1
    verdict = {
        "episode": episode_id, "task": task.name if task else None, "source": str(source),
        "passed": all(s["passed"] for s in specs), "specs": specs,
        "robustness": min((s["robustness"] for s in specs), default=None),
        "first_violation": {"spec": failing[0]["name"], "t": failing[0]["witness_t"]} if failing else None,
        "events_summary": summary, "errors": [], "perturb": {},
        "env": {"recorded": True, "python": platform.python_version(), "qddot": __version__},
        "stats": {"control_steps": len(rows) - 1, "physics_steps": 0, "wall_s": 0.0},
    }
    meta = {"task_path": str(task.path) if task else None, "source": str(source), "episode": episode,
            "recorded": True}
    write_episode(Path(root) / "runs" / episode_id, rows, events, verdict, meta, record_mcap=False)
    return verdict


def _read(source: Path, episode: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if (source / "meta" / "info.json").is_file():
        return _lerobot(source, episode)
    if source.suffix == ".mcap":
        return _mcap(source)
    if source.suffix in (".csv", ".parquet"):
        reader = "read_csv_auto" if source.suffix == ".csv" else "read_parquet"
        table = duckdb.connect().execute(f"SELECT * FROM {reader}(?)", [str(source)]).fetchnumpy()
        return _split_time({k: np.asarray(v, dtype=np.float64) for k, v in table.items()})
    raise ValueError(f"unsupported source: {source} (LeRobot dataset dir, .mcap, .csv or .parquet)")


def _split_time(columns: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    for name in TIME_COLUMNS:
        if name in columns:
            times = columns.pop(name)
            return times - times[0], columns
    raise ValueError(f"no time column; expected one of {TIME_COLUMNS}")


def _lerobot(root: Path, episode: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = json.loads((root / "meta" / "info.json").read_text())["features"]
    files = sorted(str(p) for p in (root / "data").rglob("*.parquet"))
    rows = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?) WHERE episode_index = ? ORDER BY frame_index", [files, episode]
    ).fetchnumpy()
    if not len(rows["frame_index"]):
        raise ValueError(f"episode {episode} not found in {root}")
    columns: dict[str, np.ndarray] = {}
    for key, spec in features.items():
        if key not in rows or spec.get("dtype") in ("image", "video", "string"):
            continue
        values = np.stack([np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in rows[key]])
        names = spec.get("names")
        names = names.get("motors", names) if isinstance(names, dict) else names
        for i in range(values.shape[1]):
            label = names[i] if names and i < len(names) else f"{i}"
            columns[_signal_name(key, str(label))] = values[:, i]
    columns["timestamp"] = np.asarray(rows["timestamp"], dtype=np.float64)
    for drop in ("frame_index", "episode_index", "index", "task_index"):
        columns.pop(_signal_name(drop, "0"), None)
    return _split_time(columns)


def _signal_name(feature: str, label: str) -> str:
    joint = re.sub(r"\.pos$", "", label)
    if feature == "observation.state":
        return f"joint.{joint}.pos"
    if feature == "action":
        return f"act.{joint}.ctrl"
    return f"{feature}.{label}" if label != "0" else feature


def _mcap(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from mcap.reader import make_reader

    decode = _ros2_joint_state_decoder()
    samples: list[tuple[float, dict[str, float]]] = []
    with path.open("rb") as stream:
        for schema, channel, message in make_reader(stream).iter_messages():
            t = message.log_time / 1e9
            if channel.topic == "/signals" and channel.message_encoding == "json":
                row = json.loads(message.data)
                samples.append((float(row.pop("t", t)), row))
            elif decode and schema is not None and schema.name == "sensor_msgs/msg/JointState":
                msg = decode(schema, message.data)
                row = {f"joint.{n}.pos": p for n, p in zip(msg.name, msg.position, strict=False)}
                row.update({f"joint.{n}.vel": v for n, v in zip(msg.name, msg.velocity, strict=False)})
                samples.append((t, row))
    if not samples:
        raise ValueError(f"no /signals JSON or ROS 2 JointState messages in {path}")
    keys = sorted({k for _, row in samples for k in row if not k.startswith(("ev.", "col_"))})
    times = np.asarray([t for t, _ in samples])
    return times - times[0], {k: np.asarray([row.get(k, np.nan) for _, row in samples], dtype=np.float64)
                              for k in keys}


def _ros2_joint_state_decoder():
    """CDR decoder for ROS 2 messages if `mcap-ros2-support` is installed, else None."""
    try:
        from mcap_ros2.decoder import DecoderFactory
    except ImportError:
        return None
    factory = DecoderFactory()
    return lambda schema, data: factory.decoder_for("cdr", schema)(data)


def _events(task: Task, times: np.ndarray, columns: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Joint-limit and saturation events from the task model's ranges (contacts are not observable)."""
    sim = load_backend(getattr(task, "backend", "mujoco"), str(task.model_path))
    detector = EventDetector()
    joints = [j for j in sim.joints if f"joint.{j.name}.pos" in columns]
    for i, t in enumerate(times):
        detector.begin_control_step()
        positions = np.asarray([columns[f"joint.{j.name}.pos"][i] for j in joints])
        detector.detect_physics([], joints, positions, float(t), [], i)
        for aid, act in enumerate(sim.actuators):
            value = columns.get(f"act.{act.name}.ctrl")
            if value is not None and act.limited:
                raw = float(value[i])
                excess = max(act.low - raw, raw - act.high, 0.0)
                detector.saturated(float(t), aid, act.name, not act.low < raw < act.high, excess)
    return detector.emitted


def _indicators(times: np.ndarray, events: list[dict[str, Any]]) -> list[set[str]]:
    kinds: list[set[str]] = [set() for _ in times]
    for event in events:
        index = int(np.searchsorted(times, event["t"]))
        if index < len(kinds):
            kinds[index].add(event["kind"])
    return kinds
