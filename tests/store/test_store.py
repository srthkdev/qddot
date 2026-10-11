import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from qddot.store import connect


def test_connect_reads_episode_files_and_empty_roots(tmp_path: Path) -> None:
    empty = connect(tmp_path / "empty")
    assert empty.execute("select count(*) from signals").fetchone() == (0,)
    assert empty.execute("select count(*) from events").fetchone() == (0,)
    assert empty.execute("select count(*) from verdicts").fetchone() == (0,)

    episode = tmp_path / "data" / "runs" / "trial-1"
    episode.mkdir(parents=True)
    pq.write_table(pa.table({"t": [0.0, 0.1], "joint.x.pos": [0.2, 0.3]}), episode / "signals.parquet")
    pq.write_table(pa.table({"t": [0.1], "kind": ["collision"], "a": ["arm"], "b": ["table"], "value": [2.0], "detail": ["{}"]}), episode / "events.parquet")
    (episode / "verdict.json").write_text(
        json.dumps({"task": "reach", "controller": "ctrl.py", "seed": 3, "passed": False, "robustness": -0.1, "specs": [{"name": "reach", "passed": False}]}),
        encoding="utf-8",
    )

    connection = connect(tmp_path / "data")
    assert connection.execute("select episode, count(*) from signals group by episode").fetchone() == ("trial-1", 2)
    assert connection.execute("select kind, episode from events").fetchone() == ("collision", "trial-1")
    row = connection.execute("select task, seed, passed, robustness, specs from verdicts").fetchone()
    assert row[:4] == ("reach", 3, False, -0.1)
    assert "reach" in row[4]
