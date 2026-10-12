"""Create a runnable qddot project around an MJCF or URDF robot model."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import mujoco
import yaml

MENAGERIE = {"so101": "robotstudio_so101", "ur5e": "universal_robots_ur5e", "aloha": "aloha"}


def scaffold_project(directory: Path, robot: str, task_name: str | None = None,
                     assets: Path | None = None) -> Path:
    """Create task, starter controller, local model assets, and gitignore."""
    destination = directory.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = _resolve_robot(robot, assets)
    if source.suffix.lower() == ".urdf":
        model_path = _convert_urdf(source, destination)
    else:
        model_path = _copy_model(source, destination)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    task_name = task_name or _slug(destination.name)
    metadata = _model_metadata(model)
    task = {
        "name": task_name,
        "description": "Hold the robot at its initial pose.",
        "model": model_path.relative_to(destination).as_posix(),
        "duration": 2.0,
        "control_dt": _control_dt(float(model.opt.timestep)),
        **({"init": {"keyframe": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, 0)}} if model.nkey else {}),
        "signals": metadata["signals"],
        "contact_pairs": [{"name": "robot_floor", "a": metadata["bodies"], "b": [metadata["floor_body"]]}],
        "perturb": {"friction_scale": [0.95, 1.05], "mass_scale": [0.99, 1.01],
                    "obs_noise_std": 0.0, "latency_ms": [0, 0], "jitter_ms": 0,
                    "controller_budget_ms": None},
        "specs": [
            {"name": "no_robot_floor_contact", "stl": "always(col_robot_floor < 0.5)"},
            {"name": "no_joint_limit", "stl": "always(ev.joint_limit < 0.5)"},
        ],
    }
    (destination / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    (destination / "controller.py").write_text(_controller_source(model_path), encoding="utf-8")
    _update_gitignore(destination)
    return destination / "task.yaml"


def _resolve_robot(robot: str, assets: Path | None) -> Path:
    candidate = Path(robot).expanduser()
    if candidate.suffix.lower() in {".xml", ".urdf"} or candidate.exists():
        if not candidate.is_file():
            raise FileNotFoundError(f"robot model does not exist: {candidate}")
        if candidate.suffix.lower() not in {".xml", ".urdf"}:
            raise ValueError("--robot path must point to an MJCF .xml or URDF .urdf file")
        return candidate.resolve()
    if robot not in MENAGERIE:
        raise ValueError(f"unknown Menagerie robot {robot!r}; choose {', '.join(MENAGERIE)} or pass a model path")
    assets_dir = assets.expanduser().resolve() if assets else Path(__file__).resolve().parents[3] / "assets"
    folder = assets_dir / MENAGERIE[robot]
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Menagerie assets for {robot!r} were not found at {folder}. "
            "Pass --assets PATH to a checkout containing the robot folder; pip users should pass "
            "a model path or a Menagerie checkout path."
        )
    preferred = {"so101": "scene.xml", "ur5e": "scene.xml", "aloha": "scene.xml"}[robot]
    model = folder / preferred
    if not model.is_file():
        raise FileNotFoundError(f"expected {preferred} in {folder}")
    return model.resolve()


def _copy_model(source: Path, destination: Path) -> Path:
    if source.parent.name in MENAGERIE.values():
        model_dir = destination / "assets" / source.parent.name
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.copytree(source.parent, model_dir)
        return model_dir / source.name
    model_dir = destination / "models" / source.stem
    if model_dir.exists():
        shutil.rmtree(model_dir)
    _copy_model_tree(source.parent, model_dir, destination)
    return model_dir / source.name


def _convert_urdf(source: Path, destination: Path) -> Path:
    model_dir = destination / "models" / source.stem
    if model_dir.exists():
        shutil.rmtree(model_dir)
    _copy_model_tree(source.parent, model_dir, destination)
    staged_urdf = model_dir / source.name
    converted = model_dir / f"{source.stem}.xml"
    model = mujoco.MjModel.from_xml_path(str(staged_urdf))
    mujoco.mj_saveLastXML(str(converted), model)
    spec = mujoco.MjSpec.from_file(str(converted))
    compiled = spec.compile()
    if not any(_geom_name(compiled, i) == "floor" for i in range(compiled.ngeom)):
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                pos=[0, 0, -0.05], size=[5, 5, 0.1], rgba=[0.25, 0.3, 0.35, 1])
    if compiled.nlight == 0:
        spec.worldbody.add_light(name="qddot_light", pos=[0, 0, 3], dir=[0, 0, -1])
    actuated: set[str] = set()
    for aid in range(compiled.nu):
        joint_id = int(compiled.actuator_trnid[aid, 0])
        name = mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            actuated.add(name)
    for jid in range(compiled.njnt):
        if int(compiled.jnt_type[jid]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        name = mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name or name in actuated:
            continue
        limited = bool(compiled.jnt_limited[jid])
        low, high = compiled.jnt_range[jid] if limited else (-3.14159, 3.14159)
        kp = 200.0
        spec.add_actuator(name=f"qddot_{name}", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          target=name, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          gainprm=[kp] + [0.0] * 9,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp, -20.0] + [0.0] * 7,
                          ctrllimited=1, ctrlrange=[float(low), float(high)])
    spec.compile()
    spec.to_file(str(converted))
    return converted


def _model_metadata(model: mujoco.MjModel) -> dict[str, Any]:
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    bodies = [name for name in body_names if name]
    floor_geom = next((i for i in range(model.ngeom)
                       if _geom_name(model, i).lower() == "floor"), None)
    floor_body = "floor"
    if floor_geom is not None:
        floor_body = _geom_name(model, floor_geom)
    site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite - 1) if model.nsite else None
    body_name = bodies[-1] if bodies else "world"
    ref_kind, ref = ("site_pos", site_name) if site_name else ("body_pos", body_name)
    return {"bodies": bodies, "floor_body": floor_body, "signals": [
        {"name": f"ee_{axis}", "kind": ref_kind, "ref": ref, "axis": axis} for axis in "xyz"
    ] + [{"name": "ee_height", "kind": ref_kind, "ref": ref, "axis": "z"}]}


def _copy_model_tree(source: Path, target: Path, destination: Path) -> None:
    ignored_defaults = shutil.ignore_patterns(".git", ".venv", ".qddot", "__pycache__")

    def ignore(current: str, names: list[str]) -> set[str]:
        ignored = set(ignored_defaults(current, names))
        ignored.update(
            name for name in names
            if (Path(current) / name).resolve() == destination
            or (Path(current) / name).resolve() in target.resolve().parents
        )
        return ignored

    shutil.copytree(source, target, ignore=ignore)


def _controller_source(model_path: Path) -> str:
    del model_path
    return '''"""Hold position actuators at the task's initial pose."""\n\nimport mujoco\n\n\nclass Controller:\n    def reset(self, info: dict) -> None:\n        model = mujoco.MjModel.from_xml_path(info["model_path"])\n        self.joint_for_actuator = {}\n        for aid in range(model.nu):\n            if int(model.actuator_biastype[aid]) != int(mujoco.mjtBias.mjBIAS_AFFINE):\n                continue\n            joint_id = int(model.actuator_trnid[aid, 0])\n            if joint_id < 0 or model.actuator_gainprm[aid, 0] <= 0:\n                continue\n            actuator = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)\n            joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)\n            if actuator and joint:\n                self.joint_for_actuator[actuator] = joint\n        self.targets = None\n\n    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:\n        del t\n        if self.targets is None:\n            self.targets = {actuator: obs[f"joint.{joint}.pos"]\n                            for actuator, joint in self.joint_for_actuator.items()}\n        return dict(self.targets)\n'''


def _update_gitignore(directory: Path) -> None:
    path = directory / ".gitignore"
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    if ".qddot/" not in content.splitlines():
        path.write_text(content.rstrip() + ("\n" if content.strip() else "") + ".qddot/\n", encoding="utf-8")


def _control_dt(timestep: float) -> float:
    return timestep * max(1, round(0.02 / timestep))


def _geom_name(model: mujoco.MjModel, index: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "robot_task"
