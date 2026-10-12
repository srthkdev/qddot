from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qddot.ingest import ingest
from qddot.sim.runner import run_episode
from qddot.task import load_task

ROOT = Path(__file__).resolve().parents[2]
CARTPOLE = ROOT / "tasks" / "cartpole_balance" / "task.yaml"
LQR = ROOT / "tests" / "backends" / "fixtures" / "cartpole_lqr.py"


def _lerobot(tmp: Path, angles: np.ndarray, push: np.ndarray) -> Path:
    root = tmp / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    features = {"observation.state": {"dtype": "float32", "shape": [2], "names": ["cart.pos", "pole.pos"]},
                "action": {"dtype": "float32", "shape": [1], "names": ["push"]},
                "observation.images.top": {"dtype": "video", "shape": [3, 8, 8]},
                "timestamp": {"dtype": "float32", "shape": [1]}}
    (root / "meta" / "info.json").write_text(json.dumps({"features": features}))
    n = len(angles)
    table = pa.table({"observation.state": [[0.0, float(a)] for a in angles], "action": [[float(p)] for p in push],
                      "timestamp": np.arange(n) * 0.01, "frame_index": np.arange(n),
                      "episode_index": np.zeros(n, dtype=int), "index": np.arange(n)})
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    return root


def test_lerobot_episode_gets_specs_and_events(tmp_path: Path) -> None:
    t = np.arange(400) * 0.01
    falling = 0.1 + 0.4 * t                       # a real robot that tipped over
    push = np.where(t > 1.0, 10.0, 0.0)           # and pinned its actuator at the limit
    verdict = ingest(_lerobot(tmp_path, falling, push), CARTPOLE, root=tmp_path / ".qddot")
    assert not verdict["passed"]
    assert verdict["first_violation"]["spec"] == "upright_and_hold"
    assert verdict["events_summary"].get("saturation") == 1
    signals = pq.read_table(tmp_path / ".qddot" / "runs" / verdict["episode"] / "signals.parquet")
    assert {"t", "joint.pole.pos", "joint.cart.pos", "act.push.ctrl", "ev.saturation"} <= set(signals.column_names)


def test_mcap_round_trip_reproduces_the_simulated_verdict(tmp_path: Path) -> None:
    sim = run_episode(load_task(CARTPOLE), LQR, root=tmp_path / "sim")
    recorded = ingest(sim.episode_dir / "log.mcap", CARTPOLE, root=tmp_path / "real")
    assert recorded["passed"] == sim.verdict["passed"]
    for a, b in zip(recorded["specs"], sim.verdict["specs"], strict=True):
        assert a["robustness"] == b["robustness"] or np.isclose(a["robustness"], b["robustness"])


def test_csv_without_task_is_queryable(tmp_path: Path) -> None:
    csv = tmp_path / "log.csv"
    csv.write_text("timestamp,joint.elbow.pos\n10.0,0.1\n10.5,0.2\n11.0,0.3\n")
    verdict = ingest(csv, root=tmp_path / ".qddot", name="bench-arm")
    assert verdict["passed"] and verdict["specs"] == []
    from qddot.store import connect

    rows = connect(tmp_path / ".qddot").execute("select max(t), max(\"joint.elbow.pos\") from signals").fetchone()
    assert rows == (1.0, 0.3)



def test_ros2_joint_state_bag(tmp_path: Path) -> None:
    import pytest

    writer_module = pytest.importorskip("mcap_ros2.writer")
    bag = tmp_path / "robot.mcap"
    schema_text = "std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n" \
        "================================================================================\nMSG: std_msgs/Header\n" \
        "builtin_interfaces/Time stamp\nstring frame_id\n" \
        "================================================================================\nMSG: builtin_interfaces/Time\n" \
        "int32 sec\nuint32 nanosec"
    with bag.open("wb") as stream:
        writer = writer_module.Writer(stream)
        schema = writer.register_msgdef("sensor_msgs/msg/JointState", schema_text)
        for i in range(300):  # 3 s at 100 Hz, pole held upright
            stamp = int(i * 1e7)
            writer.write_message("/joint_states", schema, {
                "header": {"stamp": {"sec": stamp // 10**9, "nanosec": stamp % 10**9}, "frame_id": ""},
                "name": ["cart", "pole"], "position": [0.0, 0.01], "velocity": [0.0, 0.0], "effort": []},
                log_time=stamp, publish_time=stamp)
        writer.finish()
    verdict = ingest(bag, CARTPOLE, root=tmp_path / ".qddot")
    signals = pq.read_table(tmp_path / ".qddot" / "runs" / verdict["episode"] / "signals.parquet")
    assert {"joint.cart.pos", "joint.pole.pos", "joint.pole.vel"} <= set(signals.column_names)
    upright = next(s for s in verdict["specs"] if s["name"] == "upright_and_hold")
    assert upright["passed"]
