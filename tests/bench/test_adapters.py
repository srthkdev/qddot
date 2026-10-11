from pathlib import Path

from bench.agents._common import claude_parse, codex_parse, opencode_parse
from bench.agents.claude_code import build_command as claude_command
from bench.agents.codex import build_command as codex_command
from bench.agents.opencode import build_command as opencode_command


def test_claude_parser_fixture():
    parsed = claude_parse('{"result":"Done.","usage":{"input_tokens":12,"output_tokens":4},"total_cost_usd":0.02}')
    assert parsed == {"final_message": "Done.", "tokens_in": 12, "tokens_out": 4, "cost_usd": 0.02}
    assert "--max-turns" in claude_command("sonnet", {})


def test_codex_parser_fixture():
    fixture = '{"type":"item.completed","item":{"type":"agent_message","text":"complete"}}\n{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}'
    parsed = codex_parse(fixture)
    assert parsed["final_message"] == "complete"
    assert (parsed["tokens_in"], parsed["tokens_out"]) == (10, 5)
    assert codex_command(Path("/tmp/wd"), "gpt-6-luna", {})[-1] == "-"


def test_opencode_parser_fixture():
    parsed = opencode_parse('{"part":{"type":"text","text":"works"}}\n{"part":{"tokens":{"input":7,"output":3}}}')
    assert parsed["final_message"] == "works"
    assert (parsed["tokens_in"], parsed["tokens_out"]) == (7, 3)
    assert opencode_command("provider/model") == ["opencode", "run", "--model", "provider/model"]
