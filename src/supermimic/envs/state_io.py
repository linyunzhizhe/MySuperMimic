"""Read and write the full simulator state as ``[qpos | qvel]``."""

from __future__ import annotations

import torch

NQ = 36  # G1 defaults retained for imports/checkpoints.
NV = 35
STATE_DIM = NQ + NV


class StateIO:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        # States are stored relative to each env's origin so that they are directly comparable
        # with the (origin-free) motion library and translation-invariant across envs.
        self.origins = env.scene.env_origins.clone()  # (N, 3)
        self.nq = int(self.sim.data.qpos.shape[-1])
        self.nv = int(self.sim.data.qvel.shape[-1])
        self.state_dim = self.nq + self.nv
        assert self.nq >= 7 and self.nv >= 6, (
            self.sim.data.qpos.shape,
            self.sim.data.qvel.shape,
        )

    def read(self) -> torch.Tensor:
        """Return a copy of ``[qpos | qvel]`` with shape ``(N, self.state_dim)``."""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        s = torch.cat([self.sim.data.qpos[:], self.sim.data.qvel[:]], dim=-1).clone()
        s[:, 0:3] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        state[:, 0:3] += self.origins[env_ids]
        self.sim.data.qpos[env_ids] = state[:, :self.nq]
        self.sim.data.qvel[env_ids] = state[:, self.nq:]
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()
