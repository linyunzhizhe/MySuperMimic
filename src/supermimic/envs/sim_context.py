"""Per-environment simulation context xi: friction, encoder bias, and COM offset."""

from __future__ import annotations

import numpy as np
import torch

XI_DIM = 33  # G1 default


class SimContext:
    def __init__(self, env, foot_geom_regex: str | None, com_body_name: str = "torso_link", entity_name: str = "robot"):
        self.env = env
        self.robot = env.scene[entity_name]
        self.foot_geom_global = None
        if foot_geom_regex is not None:
            local_ids, _ = self.robot.find_geoms(foot_geom_regex)
            if len(local_ids) == 0:
                raise ValueError(f"no geoms match {foot_geom_regex}")
            self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item())
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]), dtype=torch.float32, device=env.device
        )

    def read(self) -> torch.Tensor:
        model = self.env.sim.model
        n = self.env.num_envs
        if self.foot_geom_global is None:
            friction = torch.zeros(n, device=self.env.device, dtype=self.robot.data.encoder_bias.dtype)
        else:
            fr = model.geom_friction
            if fr.dim() == 3:  # expanded per world
                friction = fr[:, self.foot_geom_global, 0]
            else:
                friction = fr[self.foot_geom_global, 0].expand(n)
        ipos = model.body_ipos
        if ipos.dim() == 3:
            com = ipos[:, self.com_body_global] - self.com_nominal
        else:
            com = (ipos[self.com_body_global] - self.com_nominal).expand(n, 3)
        bias = self.robot.data.encoder_bias
        xi = torch.cat([friction.reshape(n, 1), bias, com], dim=-1)
        self.xi_dim = xi.shape[-1]
        assert xi.shape[0] == n and self.xi_dim == bias.shape[-1] + 4
        return xi.clone()
