"""DuckDB-backed failure cards and lightweight similarity retrieval."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb

from .cards import card_from_episode, link_fix

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_VECTOR_SIZE = 384


def database_path(root: str | Path = Path(".qddot"), global_store: bool = False) -> Path:
    """Resolve a project or user-wide memory database path."""
    if global_store:
        return Path.home() / ".qddot" / "memory.duckdb"
    return Path(root) / "memory.duckdb"


def connect(path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open a memory database and ensure its tables exist."""
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    connection.execute(
        """CREATE TABLE IF NOT EXISTS cards (
            id VARCHAR PRIMARY KEY, task VARCHAR, episode VARCHAR, created TIMESTAMPTZ,
            passed BOOLEAN, text VARCHAR, causes JSON, fix_diff VARCHAR, controller_src VARCHAR
        )"""
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (id VARCHAR PRIMARY KEY, vec FLOAT[384])"
    )
    return connection


@lru_cache(maxsize=1)
def _embedding_model() -> Any:
    from fastembed import TextEmbedding

    return TextEmbedding(_MODEL_NAME, cache_dir=str(Path.home() / ".cache" / "qddot" / "fastembed"))


def _embed(text: str) -> list[float] | None:
    try:
        vector = next(iter(_embedding_model().embed([text])))
    except (ImportError, OSError, RuntimeError, StopIteration):
        return None
    result = [float(value) for value in vector]
    return result if len(result) == _VECTOR_SIZE else None


def _fallback_note() -> str:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return "fastembed unavailable; using token overlap"
    return "embedding model unavailable; using token overlap"


def index_episode(episode_dir: str | Path, db_path: str | Path) -> dict[str, Any]:
    """Create or refresh one episode card and its embedding when available."""
    card = card_from_episode(episode_dir)
    connection = connect(db_path)
    try:
        current = connection.execute(
            "SELECT text, controller_src, fix_diff FROM cards WHERE id = ?", [card["id"]]
        ).fetchone()
        if current is not None and current[1]:
            card["controller_src"] = current[1]
            if not card["passed"]:
                card["fix_diff"] = link_fix(episode_dir, current[1]) or current[2]
        connection.execute(
            """INSERT OR REPLACE INTO cards
               (id, task, episode, created, passed, text, causes, fix_diff, controller_src)
               VALUES (?, ?, ?, ?::TIMESTAMPTZ, ?, ?, ?::JSON, ?, ?)""",
            [card["id"], card["task"], card["episode"], card["created"], card["passed"],
             card["text"], json.dumps(card["causes"]), card["fix_diff"], card["controller_src"]],
        )
        if current is None or current[0] != card["text"]:
            vector = _embed(card["text"])
            if vector is not None:
                connection.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                                   [card["id"], vector])
            else:
                connection.execute("DELETE FROM embeddings WHERE id = ?", [card["id"]])
        return card
    finally:
        connection.close()


def index_runs(runs_dir: str | Path, db_path: str | Path) -> int:
    """Index all recorded episodes, refreshing links when new fixes appear."""
    paths = sorted(Path(runs_dir).glob("*/verdict.json"))
    connection = connect(db_path)
    try:
        known = {row[0] for row in connection.execute("SELECT id FROM cards").fetchall()}
    finally:
        connection.close()
    new_paths = [path for path in paths if str(path.parent.resolve()) not in known]
    count = len(new_paths)
    for verdict_path in new_paths:
        index_episode(verdict_path.parent, db_path)
    # Revisit old failures only when a new passing run could add a fix link.
    new_pass = False
    for verdict_path in new_paths:
        try:
            new_pass |= bool(json.loads(verdict_path.read_text(encoding="utf-8")).get("passed"))
        except (OSError, ValueError):
            continue
    if new_pass:
        connection = connect(db_path)
        try:
            failures = connection.execute("SELECT id FROM cards WHERE passed = FALSE").fetchall()
        finally:
            connection.close()
        for (episode_dir,) in failures:
            if Path(episode_dir).exists():
                index_episode(episode_dir, db_path)
    return count


def _tokens(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def search(episode_dir: str | Path, db_path: str | Path, k: int = 3) -> dict[str, Any]:
    """Retrieve related failures using exact task/cause filters and similarity."""
    query_card = card_from_episode(episode_dir)
    vector = _embed(query_card["text"])
    connection = connect(db_path)
    try:
        same_cause = " OR ".join("json_contains(c.causes, ?::JSON)" for _ in query_card["causes"])
        exact_filter = "c.task = ?" + (f" OR ({same_cause})" if same_cause else "")
        params: list[Any] = [query_card["task"]]
        params.extend(json.dumps(cause) for cause in query_card["causes"])
        params.extend([query_card["id"], query_card["controller_sha"]])
        if vector is not None:
            score = "coalesce(array_cosine_similarity(e.vec, ?::FLOAT[384]), 0.0)"
            params = [*params, vector]
            note = None
        else:
            score = """CASE WHEN list_count(q.tokens) = 0 THEN 0.0
                ELSE list_count(list_intersect(c.tokens, q.tokens))::DOUBLE
                     / greatest(list_count(c.tokens), list_count(q.tokens), 1) END"""
            params = [*params, _tokens(query_card["text"])]
            note = _fallback_note()
        sql = f"""WITH candidates AS (
            SELECT c.*, sha256(c.controller_src) AS source_sha,
                   string_split(regexp_replace(lower(c.text), '[^a-z0-9]+', ' ', 'g'), ' ') AS tokens
            FROM cards c WHERE c.passed = FALSE AND ({exact_filter})
              AND c.id <> ? AND sha256(c.controller_src) <> ?
        ), ranked AS (
            SELECT c.id, c.task, c.episode, c.causes, c.fix_diff, {score} AS score
            FROM candidates c
            LEFT JOIN embeddings e ON e.id = c.id
            {'' if vector is not None else "CROSS JOIN (SELECT string_split(regexp_replace(lower(?), '[^a-z0-9]+', ' ', 'g'), ' ') AS tokens) q"}
        ) SELECT id, task, episode, causes, fix_diff, score FROM ranked
          ORDER BY score DESC, episode ASC LIMIT ?"""
        params.append(max(0, int(k)))
        rows = connection.execute(sql, params).fetchall()
    finally:
        connection.close()
    results = [{"episode": row[2], "task": row[1], "score": float(row[5]),
                "causes": json.loads(row[3]),
                "fix_diff": "\n".join((row[4] or "").splitlines()[:15]) or None}
               for row in rows]
    return {"episode": query_card["episode"], "results": results, "note": note}
