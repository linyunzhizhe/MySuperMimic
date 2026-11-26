"""Watch a trained tracking policy replay reference clips in an interactive viewer.

  uv run python scripts/play.py --ckpt logs/e5_bones_seed/ckpt_latest.pt --clips dance1_subject1
  uv run python scripts/play.py --ckpt logs/e3_lafan1_40_anchor/ckpt_latest.pt --npz-dir data/npz/lafan1 --num-envs 4

The grey/blue translucent robot is the reference pose; the solid robot is the policy.
Deterministic evaluation settings (no noise, no DR); every reset starts the clip from its first frame.
Native MuJoCo viewer when a display is available, otherwise viser (browser, http://localhost:8080).
Keys (native): space pause, right-arrow single step, +/- speed, backspace reset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots.bumi import get_bumi_robot_cfg


BUMI_TRACKED_BODIES = (
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


class _DictPolicy:
    """Adapts policy(obs_policy, obs_goal) to the viewer's policy(obs) protocol."""

    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs["policy"], obs["goal"])


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    num_envs: int = 1,
    device: str = "cuda:0",
    viewer: Literal["auto", "native", "viser"] = "auto",
    ghost: bool = True,
    sampling: Literal["start", "uniform"] = "start",
    robot: Literal["g1", "bumi"] = "g1",
    scene_xml: Path | None = None,
) -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)

    robot_cfg = None
    tracked_bodies = None
    anchor_body_name = None
    root_body_name = None
    com_body_name = None
    fk_model = None
    if robot == "bumi":
        if scene_xml is None:
            raise ValueError("--scene-xml is required when --robot bumi")
        robot_cfg = get_bumi_robot_cfg()
        tracked_bodies = BUMI_TRACKED_BODIES
        anchor_body_name = "base_link"
        root_body_name = "base_link"
        com_body_name = "base_link"
        fk_model = str(scene_xml)

    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        root_height_threshold=1.0e6,
        root_tilt_threshold=1.0e6,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode=sampling,
        record_transitions=False,
        seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        debug_vis=ghost,
        robot_cfg=robot_cfg,
        tracked_bodies=tracked_bodies,
        anchor_body_name=anchor_body_name,
        root_body_name=root_body_name,
        com_body_name=com_body_name,
        fk_model=fk_model,
    )
    env = RslRlVecEnvWrapper(make_env(args, device=device))
    pol = _DictPolicy(policy)
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    print(f"[INFO] {len(files)} clip(s), {num_envs} env(s), viewer={viewer}")
    try:
        if viewer == "native":
            NativeMujocoViewer(env, pol).run()
        else:
            ViserPlayViewer(env, pol).run()
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
