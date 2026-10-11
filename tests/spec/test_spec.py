from __future__ import annotations

import time

import numpy as np
import pytest

from qddot.spec import SpecError, evaluate, evaluate_specs, parse, to_english


def test_parse_precedence_names_bounds_and_arithmetic() -> None:
    node = parse("not joint.elbow.pos + 2 * x >= 3 and ev.collision < 0.5 or a > 0 implies b == abs(c)")
    assert node.op == "implies"
    assert node.left.op == "or"
    assert node.left.left.op == "and"
    assert parse("always[0, 2.5](x > 0)").bound == (0.0, 2.5)


def test_constant_only_atom() -> None:
    result = evaluate("1 + 2 > 2", np.array([0.0]), {})
    assert result.robustness == 1.0


def test_parse_errors_have_column_and_caret() -> None:
    with pytest.raises(SpecError, match="column 5") as error:
        parse("x > and")
    assert "    ^" in str(error.value)
    with pytest.raises(SpecError, match="Bounds must satisfy"):
        parse("always[2, 1](x > 0)")


def test_unknown_signal_has_suggestion() -> None:
    with pytest.raises(SpecError, match="Did you mean joint.elbow.pos"):
        evaluate("joint.elbow.pso > 0", np.array([0.0]), {"joint.elbow.pos": np.array([1.0])})


def test_unknown_signal_without_close_match_lists_available_names() -> None:
    names = {f"signal_{index}": np.array([0.0]) for index in range(12)}
    with pytest.raises(SpecError) as error:
        evaluate("zzz > 0", np.array([0.0]), names)
    assert "Available signals:" in str(error.value)
    assert "signal_0" in str(error.value)
    assert "signal_9" in str(error.value)
    assert "signal_10" not in str(error.value)


def test_parenthesized_unary_and_division_arithmetic() -> None:
    times = np.array([0.0, 1.0])
    result = evaluate("abs((x - (-2)) / 2) < 1", times, {"x": np.array([-2.5, -2.0])})
    assert result.robustness == pytest.approx(0.75)
    assert result.witness_t == 0.0


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("x > 1", 1.0),
        ("x < 1", -1.0),
        ("x >= 1", 1.0),
        ("x <= 1", -1.0),
        ("x == 1", -1.0),
        ("not (x > 1)", -1.0),
        ("x > 0 and y > 0", 2.0),
        ("x > 0 or y > 0", 3.0),
        ("x > 0 implies y > 0", 3.0),
    ],
)
def test_atomic_boolean_robustness(formula: str, expected: float) -> None:
    result = evaluate(formula, np.array([0.0, 1.0]), {"x": np.array([2.0, 0.0]), "y": np.array([3.0, -3.0])})
    assert result.robustness == expected


def test_temporal_semantics_and_truncation() -> None:
    t = np.array([0.0, 0.5, 1.5, 3.0])
    x = np.array([0.0, 2.0, -1.0, 3.0])
    signals = {"x": x}
    assert evaluate("always[0,1](x > 0)", t, signals).robustness == 0.0
    assert evaluate("eventually[0,1](x > 0)", t, signals).robustness == 2.0
    assert evaluate("eventually[5,6](x > 0)", t, signals).robustness == -np.inf
    assert evaluate("always[5,6](x > 0)", t, signals).robustness == np.inf
    # At t=0, until robustness is max_j min(min(phi[0:j]), psi[j]).
    until = evaluate("(x > -2) until[0,1.5] (x > 0)", t, signals)
    assert until.robustness == 2.0


def test_witness_and_outer_temporal_window() -> None:
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    x = np.array([1.0, 0.5, 0.01, 0.01, 0.01, 1.0])
    result = evaluate("eventually[0,2.5](always[0,1](x < 0.02))", t, {"x": x})
    assert result.passed
    assert result.witness_t == 1.0
    assert result.window == (0.0, 2.5)


def test_english_nested_example_and_generic() -> None:
    assert to_english("eventually[0,2.5](always[0,1](dist_ee_target < 0.02))") == (
        "within 2.5 s, dist_ee_target becomes less than 0.02 and then stays below it for 1 s"
    )
    assert to_english("always[0,1](x > 2)") == "from 0 s to 1 s, x stays above 2"
    assert to_english("always(x < 3)") == "x stays below 3 for the whole run"
    assert to_english("not eventually(ev.joint_limit > 0.5)") == "no joint limit is ever hit"
    assert to_english("always(col_arm_table < 0.5)") == "no arm table collision occurs for the whole run"


def test_evaluate_specs_shape() -> None:
    result = evaluate_specs([{"name": "safe", "stl": "always(x > 0)"}], np.array([0.0, 1.0]), {"x": np.array([1.0, 2.0])})[0]
    assert set(result) == {"name", "stl", "english", "passed", "robustness", "witness_t", "window"}
    assert result["passed"] is True


def test_nested_performance() -> None:
    count = 10_000
    t = np.arange(count, dtype=np.float64) * 0.01
    signals = {"x": np.sin(t)}
    formula = "eventually[0,1](always[0,0.5](eventually[0,0.2](x > -2)))"
    started = time.perf_counter()
    result = evaluate(formula, t, signals)
    elapsed = time.perf_counter() - started
    assert result.passed
    assert elapsed < 0.05


def test_rtamt_crosscheck_when_available() -> None:
    rtamt = pytest.importorskip("rtamt")
    rng = np.random.default_rng(73)
    t = np.arange(30, dtype=np.float64)
    x = rng.normal(size=len(t))
    y = rng.normal(size=len(t))
    formulas = [
        "always[0,3](x > 0)",
        "eventually[1,4](x >= 0)",
        "always[0,2](x > 0) and eventually[0,3](y < 0)",
    ]
    for formula in formulas:
        reference_formula = formula.replace(",", ":")
        monitor = rtamt.StlDiscreteTimeOfflineSpecification()
        monitor.declare_var("x", "float")
        monitor.declare_var("y", "float")
        monitor.set_sampling_period(1.0)
        monitor.spec = reference_formula
        monitor.parse()
        reference = monitor.evaluate({"time": t.tolist(), "x": x.tolist(), "y": y.tolist()})
        ours = evaluate(formula, t, {"x": x, "y": y}).robustness
        assert ours == pytest.approx(float(reference[0][1]), abs=1e-9)
