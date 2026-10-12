from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from typer.testing import CliRunner

from qddot import mcp_server
from qddot.commands.similar import register as register_similar
from qddot.memory import cards, store


def _episode(runs: Path, name: str, controller: Path, *, passed: bool = False) -> Path:
    directory = runs / name
    directory.mkdir(parents=True)
    task_path = runs.parent / "task.yaml"
    task_path.write_text('name: cartpole\nspecs:\n  - name: upright\n    stl: "x < 0.1"\n', encoding="utf-8")
    source = controller.read_text(encoding="utf-8")
    sha = hashlib.sha256(source.encode()).hexdigest()
    verdict = {
        "episode": name, "task": "cartpole", "controller": str(controller), "seed": 0,
        "passed": passed, "robustness": 1.0 if passed else -1.0,
        "specs": [{"name": "upright", "stl": "x < 0.1", "english": "x stays below 0.1",
                   "passed": passed, "robustness": 1.0 if passed else -1.0,
                   "witness_t": 0.0, "window": [0.0, 1.0]}],
        "events_summary": {}, "errors": [], "perturb": {},
    }
    meta = {"task_path": str(task_path), "controller_path": str(controller),
            "controller_sha256": sha, "seed": 0, "perturb": {}}
    (directory / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist([{"t": 0.0, "x": 1.0}, {"t": 1.0, "x": 0.5}]),
                   directory / "signals.parquet")
    pq.write_table(pa.Table.from_pylist([{"t": 0.2, "kind": "saturation", "a": "pole",
                                          "b": None, "value": 1.0, "detail": "{}"}]),
                   directory / "events.parquet")
    return directory


def test_card_contains_failure_english_causes_events_and_source(tmp_path: Path) -> None:
    controller = tmp_path / "buggy.py"
    controller.write_text("class Controller: pass\n", encoding="utf-8")
    episode = _episode(tmp_path / "runs", "fail", controller)
    card = cards.card_from_episode(episode)
    assert card["task"] == "cartpole"
    assert "x is below 0.1" in card["text"]
    assert "saturation" in card["text"]
    assert card["controller_src"] == controller.read_text(encoding="utf-8")
    assert card["passed"] is False


def test_later_pass_links_unified_controller_diff(tmp_path: Path, monkeypatch) -> None:
    controller = tmp_path / "controller.py"
    controller.write_text("gain = -1\n", encoding="utf-8")
    runs = tmp_path / "runs"
    failing = _episode(runs, "fail", controller)
    db_path = tmp_path / "memory.duckdb"
    monkeypatch.setattr(store, "_embed", lambda _text: [0.0] * 384)
    store.index_episode(failing, db_path)
    controller.write_text("gain = 1\n", encoding="utf-8")
    passing = _episode(runs, "pass", controller, passed=True)
    (passing / "verdict.json").touch()
    store.index_runs(runs, db_path)
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        row = connection.execute("SELECT fix_diff, controller_src FROM cards WHERE id = ?",
                                 [str(failing.resolve())]).fetchone()
    finally:
        connection.close()
    assert "-gain = -1" in row[0]
    assert "+gain = 1" in row[0]
    assert row[1] == "gain = -1\n"


def _insert_card(connection, episode: str, task: str, causes: list[str], vector: list[float], text: str) -> None:
    connection.execute(
        "INSERT INTO cards VALUES (?, ?, ?, now(), FALSE, ?, ?::JSON, NULL, ?)",
        [episode, task, episode, text, json.dumps(causes), f"source {episode}"],
    )
    connection.execute("INSERT INTO embeddings VALUES (?, ?)", [episode, vector])


def test_same_cause_card_ranks_first_and_different_source_only(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "memory.duckdb"
    connection = store.connect(db_path)
    try:
        _insert_card(connection, "match", "other-task", ["sign error"], [1.0] + [0.0] * 383, "close")
        _insert_card(connection, "weak", "cartpole", ["unrelated"], [0.0, 1.0] + [0.0] * 382, "weak")
    finally:
        connection.close()
    query = {"id": "query", "task": "cartpole", "episode": "query", "text": "query",
             "causes": ["sign error"], "controller_sha": "query-sha"}
    monkeypatch.setattr(store, "card_from_episode", lambda _path: query)
    monkeypatch.setattr(store, "_embed", lambda _text: [1.0] + [0.0] * 383)
    result = store.search("query", db_path, 3)
    assert [item["episode"] for item in result["results"]] == ["match", "weak"]
    assert result["results"][0]["score"] > result["results"][1]["score"]


def test_fallback_scores_token_overlap_in_duckdb(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "memory.duckdb"
    connection = store.connect(db_path)
    try:
        _insert_card(connection, "match", "cartpole", ["sign error"], [1.0] + [0.0] * 383,
                     "cartpole velocity sign error")
    finally:
        connection.close()
    query = {"id": "query", "task": "cartpole", "episode": "query",
             "text": "cartpole velocity sign error", "causes": ["sign error"],
             "controller_sha": "query-sha"}
    monkeypatch.setattr(store, "card_from_episode", lambda _path: query)
    monkeypatch.setattr(store, "_embed", lambda _text: None)
    result = store.search("query", db_path)
    assert result["results"][0]["score"] > 0
    assert "token overlap" in result["note"]


def test_similar_cli_prints_compact_result(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("qddot.commands.similar.index_runs", lambda *_args: 0)
    monkeypatch.setattr("qddot.commands.similar.search", lambda *_args: {
        "episode": "current", "note": "fastembed unavailable; using token overlap",
        "results": [{"score": 0.8, "task": "cartpole", "episode": "old", "causes": ["sign error"],
                     "fix_diff": "--- a\n+++ b\n@@\n-x\n+y"}],
    })
    app = typer.Typer()
    register_similar(app)
    result = CliRunner().invoke(app, [str(tmp_path / "episode"), "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "0.800 cartpole old" in result.stdout
    assert "-x" in result.stdout


def test_mcp_tool_registered() -> None:
    tools = asyncio.run(mcp_server.server.list_tools())
    assert "similar" in {item.name for item in tools}


def test_real_embedding_model_produces_384_dimensions() -> None:
    vector = store._embed("cartpole controller velocity sign failure")
    assert vector is not None
    assert len(vector) == 384
