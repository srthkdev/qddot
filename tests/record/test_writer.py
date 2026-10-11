import json

import pyarrow.parquet as pq
from mcap.reader import make_reader

from qddot.record.writer import write_episode


def test_writer_preserves_event_schema_and_json(tmp_path):
    signals = [{"t": 0.0, "joint.a.pos": 0.25}, {"t": 0.1, "joint.a.pos": 0.3}]
    events = [{"t": 0.1, "kind": "contact_start", "a": "a", "b": "b",
               "value": 2.0, "detail": "{}"}]
    verdict = {"episode": "test", "passed": True}
    meta = {"seed": 3}
    write_episode(tmp_path, signals, events, verdict, meta)
    assert pq.read_table(tmp_path / "signals.parquet").to_pylist() == signals
    assert pq.read_table(tmp_path / "events.parquet").to_pylist() == events
    assert json.loads((tmp_path / "verdict.json").read_text()) == verdict
    assert json.loads((tmp_path / "meta.json").read_text()) == meta
    assert (tmp_path / "log.mcap").stat().st_size > 0
    with (tmp_path / "log.mcap").open("rb") as stream:
        messages = list(make_reader(stream).iter_messages())
    assert [channel.topic for _, channel, _ in messages] == [
        "/signals", "/events", "/signals", "/events"]
    assert all(channel.message_encoding == "json" and channel.schema_id == 0
               for _, channel, _ in messages)
