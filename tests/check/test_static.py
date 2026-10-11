from pathlib import Path

from qddot.check.static import check_model


def _write_model(tmp_path: Path, xml: str) -> Path:
    path = tmp_path / "model.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def test_reports_requested_model_problems(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        """<mujoco><option timestep="0.02"/><worldbody>
        <geom name="floor" type="plane" size="1 1 0.1"/>
        <body name="massless" pos="0 0 1"/>
        <body name="unlimited" pos="0 0 0.5"><joint name="free_hinge" type="hinge"/>
          <geom type="sphere" size="0.05"/>
        </body>
        <body name="limited" pos="0 0 0.05"><joint name="hinge" type="hinge" range="-1 1"/>
          <geom type="sphere" size="0.1"/>
        </body>
        </worldbody><actuator><position name="servo" joint="hinge" kp="10" ctrlrange="-2 2"/>
        <motor name="free_motor" joint="free_hinge"/></actuator></mujoco>""",
    )
    issues = check_model(path)
    codes = {issue["code"] for issue in issues}
    assert "large_timestep" in codes
    assert "unlimited_actuated_joint" in codes
    assert "ctrlrange_outside_joint" in codes
    assert "initial_penetration" in codes
    assert "body_mass" in codes


def test_load_errors_are_returned_as_diagnostics(tmp_path: Path) -> None:
    path = _write_model(tmp_path, "<mujoco>")
    issues = check_model(path)
    assert issues[0]["severity"] == "error"
    assert issues[0]["code"] == "load_error"
    assert "could not load" in issues[0]["message"]


def test_vendored_models_load_without_errors() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "assets/robotstudio_so101/scene.xml",
        "assets/universal_robots_ur5e/scene.xml",
    ):
        path = root / relative
        assert path.exists()
        assert not [issue for issue in check_model(path) if issue["severity"] == "error"]
