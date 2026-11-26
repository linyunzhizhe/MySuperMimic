"""Bumi robot configuration for the SuperMimic tracking environment.

The joint order, home pose, actuator limits and action scales are copied from
the Bumi MJCF and ``policy/bumi_ac.yaml``.  The scene path remains external so
the same code can run with the local checkout or the GPUFree checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


_SCENE_CANDIDATES = (
    Path(r"C:\AI2\projects\bumi\bumi_mujoco-main\bumi_robot\scene.xml"),
    Path("/root/gpufree-data/bumi_work/source/bumi_mujoco-main/bumi_robot/scene.xml"),
)


def get_bumi_scene_xml() -> Path:
    """Return the configured Bumi scene XML path."""

    override = os.environ.get("BUMI_SCENE_XML")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"BUMI_SCENE_XML does not exist: {path}")
        return path
    for path in _SCENE_CANDIDATES:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in _SCENE_CANDIDATES)
    raise FileNotFoundError(f"Bumi scene.xml not found; searched: {searched}")


BUMI_SCENE_XML = get_bumi_scene_xml

BUMI_JOINT_NAMES = (
    "waist_yaw_joint",
    "l_arm_pitch_joint",
    "l_arm_roll_joint",
    "l_arm_yaw_joint",
    "l_elbow_pitch_joint",
    "r_arm_pitch_joint",
    "r_arm_roll_joint",
    "r_arm_yaw_joint",
    "r_elbow_pitch_joint",
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
)

BUMI_BODY_NAMES = (
    "base_link",
    "waist_yaw_link",
    "l_arm_pitch_link",
    "l_arm_roll_link",
    "l_arm_yaw_link",
    "l_elbow_pitch_link",
    "r_arm_pitch_link",
    "r_arm_roll_link",
    "r_arm_yaw_link",
    "r_elbow_pitch_link",
    "l_leg_pitch_link",
    "l_leg_roll_link",
    "l_leg_yaw_link",
    "l_knee_pitch_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "r_leg_pitch_link",
    "r_leg_roll_link",
    "r_leg_yaw_link",
    "r_knee_pitch_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
)

# Order is exactly BUMI_JOINT_NAMES / the MJCF hinge-joint order.
BUMI_ACTION_SCALE = (
    0.127,
    0.115,
    0.115,
    0.115,
    0.115,
    0.115,
    0.115,
    0.115,
    0.115,
    0.250,
    0.250,
    0.112,
    0.333,
    0.250,
    0.250,
    0.250,
    0.250,
    0.112,
    0.333,
    0.250,
    0.250,
)

# Bumi's foot meshes are group=3 rather than stable foot geom names.  The
# environment therefore leaves foot-specific randomization disabled for now.
BUMI_FOOT_GEOM_REGEX = None

_KP = {
    "waist_yaw_joint": 53.0,
    "l_arm_pitch_joint": 12.0,
    "l_arm_roll_joint": 12.0,
    "l_arm_yaw_joint": 12.0,
    "l_elbow_pitch_joint": 12.0,
    "r_arm_pitch_joint": 12.0,
    "r_arm_roll_joint": 12.0,
    "r_arm_yaw_joint": 12.0,
    "r_elbow_pitch_joint": 12.0,
    "l_leg_pitch_joint": 60.0,
    "l_leg_roll_joint": 60.0,
    "l_leg_yaw_joint": 60.0,
    "l_knee_pitch_joint": 45.0,
    "l_ankle_pitch_joint": 20.0,
    "l_ankle_roll_joint": 20.0,
    "r_leg_pitch_joint": 60.0,
    "r_leg_roll_joint": 60.0,
    "r_leg_yaw_joint": 60.0,
    "r_knee_pitch_joint": 45.0,
    "r_ankle_pitch_joint": 20.0,
    "r_ankle_roll_joint": 20.0,
}

_KD = {
    "waist_yaw_joint": 3.4,
    "l_arm_pitch_joint": 0.4,
    "l_arm_roll_joint": 0.4,
    "l_arm_yaw_joint": 0.4,
    "l_elbow_pitch_joint": 0.4,
    "r_arm_pitch_joint": 0.4,
    "r_arm_roll_joint": 0.4,
    "r_arm_yaw_joint": 0.4,
    "r_elbow_pitch_joint": 0.4,
    "l_leg_pitch_joint": 3.0,
    "l_leg_roll_joint": 3.0,
    "l_leg_yaw_joint": 2.5,
    "l_knee_pitch_joint": 4.0,
    "l_ankle_pitch_joint": 1.0,
    "l_ankle_roll_joint": 1.0,
    "r_leg_pitch_joint": 3.0,
    "r_leg_roll_joint": 3.0,
    "r_leg_yaw_joint": 2.5,
    "r_knee_pitch_joint": 4.0,
    "r_ankle_pitch_joint": 1.0,
    "r_ankle_roll_joint": 1.0,
}

_EFFORT_LIMIT = {
    name: (
        5.5
        if "_arm_" in name
        else 10.0
        if "ankle" in name
        else 60.0
        if "leg_pitch" in name or "knee" in name
        else 27.0
    )
    for name in BUMI_JOINT_NAMES
}

_HOME_JOINT_POS = {
    "waist_yaw_joint": 0.0,
    "l_arm_pitch_joint": 0.0,
    "l_arm_roll_joint": 0.3,
    "l_arm_yaw_joint": 0.0,
    "l_elbow_pitch_joint": 0.0,
    "r_arm_pitch_joint": 0.0,
    "r_arm_roll_joint": -0.3,
    "r_arm_yaw_joint": 0.0,
    "r_elbow_pitch_joint": 0.0,
    "l_leg_pitch_joint": -0.1495,
    "l_leg_roll_joint": 0.0,
    "l_leg_yaw_joint": 0.0,
    "l_knee_pitch_joint": 0.3215,
    "l_ankle_pitch_joint": -0.172,
    "l_ankle_roll_joint": 0.0,
    "r_leg_pitch_joint": -0.1495,
    "r_leg_roll_joint": 0.0,
    "r_leg_yaw_joint": 0.0,
    "r_knee_pitch_joint": 0.3215,
    "r_ankle_pitch_joint": -0.172,
    "r_ankle_roll_joint": 0.0,
}


def get_spec() -> mujoco.MjSpec:
    """Load Bumi's MJCF and replace its torque motors with position actuators."""

    spec = mujoco.MjSpec.from_file(str(get_bumi_scene_xml()))
    # The source XML has vendor torque motors.  mjlab creates the position
    # actuators below, so retaining both would double the actuator count.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def _actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
    return tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(name,),
            stiffness=_KP[name] * (
                6.0 if name in {"l_ankle_pitch_joint", "r_ankle_pitch_joint"} else 3.0
            ),
            damping=_KD[name] * 3.0,
            effort_limit=_EFFORT_LIMIT[name],
        )
        for name in BUMI_JOINT_NAMES
    )

BUMI_ACTION_SCALE = {
    name: 0.25 * _EFFORT_LIMIT[name] / _KP[name]
    for name in BUMI_JOINT_NAMES
}

BUMI_ARTICULATION = EntityArticulationInfoCfg(
    actuators=_actuators(),
    soft_joint_pos_limit_factor=0.9,
)

BUMI_HOME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.4764),
    joint_pos=_HOME_JOINT_POS,
    joint_vel={".*": 0.0},
)


def get_bumi_robot_cfg() -> EntityCfg:
    """Return the mjlab entity configuration used by SuperMimic."""

    return EntityCfg(
        spec_fn=get_spec,
        init_state=BUMI_HOME,
        articulation=BUMI_ARTICULATION,
    )
