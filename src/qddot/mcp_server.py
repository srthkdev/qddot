"""FastMCP stdio tools for qddot's simulation and diagnosis APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # MCP SDK 2.x renamed FastMCP to MCPServer.
    from mcp.server.mcpserver import MCPServer as FastMCP

server = FastMCP("qddot")


def _verdict(result: Any) -> dict[str, Any]:
    verdict = dict(result.verdict)
    verdict.setdefault("episode_dir", str(result.episode_dir))
    return verdict


@server.tool(description="Run one controller in a task and return its compact verdict and episode path.")
def run(task: str, controller: str, seed: int = 0,
        perturb: dict[str, float | None] | None = None) -> dict[str, Any]:
    from qddot.sim.runner import run_episode
    from qddot.task import load_task

    try:
        return _verdict(run_episode(load_task(task), Path(controller), seed=seed, perturb=perturb))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@server.tool(description="Fuzz a controller with seeded task perturbations; returns pass rate and worst cases.")
def fuzz(task: str, controller: str, n: int = 50, workers: int = 1,
         seed: int = 0) -> dict[str, Any]:
    from qddot.sim.fuzz import fuzz as run_fuzz
    from qddot.task import load_task

    try:
        return run_fuzz(load_task(task), Path(controller), n=n, workers=workers, seed=seed)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _diagnostic(function_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        from importlib import import_module

        module = import_module("qddot.diag")
        function = getattr(module, function_name)
    except (ImportError, AttributeError):
        return {"error": "not available"}
    try:
        result = function(*args, **kwargs)
        return result if isinstance(result, dict) else {"result": result}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@server.tool(description="Explain a recorded episode's verdict, signals, nearby events, and likely causes.")
def why(episode_dir: str) -> dict[str, Any]:
    return _diagnostic("why", Path(episode_dir))


@server.tool(description="Compare two recorded episodes by signals, events, verdict, and run metadata.")
def diff(a_dir: str, b_dir: str) -> dict[str, Any]:
    return _diagnostic("diff", Path(a_dir), Path(b_dir))


@server.tool(description="Run SQL against qddot's signals, events, and verdicts views; returns at most 50 rows.")
def query(sql: str, root: str = ".qddot", limit: int = 50) -> dict[str, Any]:
    connection = None
    try:
        from qddot.store import connect

        connection = connect(Path(root))
        cursor = connection.execute(sql)
        columns = [str(column[0]) for column in cursor.description or []]
        max_rows = min(max(int(limit), 1), 50)
        rows = cursor.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        result_rows = [dict(zip(columns, row, strict=False)) for row in rows[:max_rows]]
        return {"columns": columns, "rows": result_rows, "truncated": truncated}
    except (duckdb.Error, ValueError, OSError) as exc:
        return {"error": str(exc)}
    finally:
        if connection is not None:
            connection.close()


@server.tool(description="List a task's formal specs and deterministic English back-translations.")
def spec(task: str) -> dict[str, Any]:
    try:
        from qddot.spec import to_english
        from qddot.task import load_task

        loaded = load_task(task)
        return {"task": loaded.name, "specs": [
            {"name": item["name"], "stl": item["stl"], "english": to_english(item["stl"])}
            for item in loaded.specs
        ]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@server.tool(description="Check a MuJoCo model for common physical and actuator setup problems.")
def check(model: str) -> dict[str, Any]:
    try:
        from qddot.check.static import check_model

        return {"findings": check_model(Path(model))}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@server.tool(description="Re-simulate an episode to a time and report state; optionally render a frame.")
def replay(episode_dir: str, t: float, frame: bool = False) -> dict[str, Any]:
    return _diagnostic("replay", Path(episode_dir), t, frame=frame)


# Compatibility alias makes the command intent clear to callers and tests.
mcp = server

if __name__ == "__main__":
    server.run(transport="stdio")
