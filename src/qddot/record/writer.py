"""Episode artifact writers."""

from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from mcap.writer import Writer

EVENT_SCHEMA = pa.schema([
    ("t", pa.float64()), ("kind", pa.string()), ("a", pa.string()),
    ("b", pa.string()), ("value", pa.float64()), ("detail", pa.string()),
])


def write_episode(directory: Path, signal_rows: list[dict[str, float]],
                  events: list[dict[str, Any]], verdict: dict[str, Any],
                  meta: dict[str, Any], record_mcap: bool = True) -> None:
    """Write parquet, JSON, and optional MCAP outputs for one episode."""
    directory.mkdir(parents=True, exist_ok=True)
    if signal_rows:
        names = list(signal_rows[0])
        arrays = {name: pa.array([float(row[name]) for row in signal_rows], type=pa.float64())
                  for name in names}
        pq.write_table(pa.table(arrays), directory / "signals.parquet")
    else:
        pq.write_table(pa.table({"t": pa.array([], type=pa.float64())}), directory / "signals.parquet")
    pq.write_table(pa.Table.from_pylist(events, schema=EVENT_SCHEMA), directory / "events.parquet")
    _write_json(directory / "verdict.json", verdict)
    _write_json(directory / "meta.json", meta)
    if record_mcap:
        _write_mcap(directory / "log.mcap", signal_rows, events)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_mcap(path: Path, signals: list[dict[str, float]], events: list[dict[str, Any]]) -> None:
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start(profile="", library="qddot")
        signal_channel = writer.register_channel(topic="/signals", message_encoding="json", schema_id=0)
        event_channel = writer.register_channel(topic="/events", message_encoding="json", schema_id=0)
        event_buckets: list[list[dict[str, Any]]] = [[] for _ in signals]
        signal_times = [float(row.get("t", 0.0)) for row in signals]
        for event in events:
            event_time = float(event["t"])
            index = min(bisect_left(signal_times, event_time), max(0, len(signal_times) - 1))
            if event_buckets:
                event_buckets[index].append(event)
        for index, row in enumerate(signals):
            timestamp = max(0, int(float(row.get("t", 0.0)) * 1_000_000_000))
            payload = json.dumps(row, separators=(",", ":"), allow_nan=False).encode()
            writer.add_message(signal_channel, timestamp, payload, publish_time=timestamp)
            event_payload = json.dumps({"t": signal_times[index], "events": event_buckets[index]},
                                       separators=(",", ":"), allow_nan=False).encode()
            writer.add_message(event_channel, timestamp, event_payload, publish_time=timestamp)
        writer.finish()
