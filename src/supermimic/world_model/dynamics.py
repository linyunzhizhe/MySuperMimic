"""Dynamics MLP f_omega (paper Table A.4: MLP (2048, 2048, 2048))."""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_NUM_JOINTS = 29


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU, zero_last: bool = False, last_gain: float | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsMLP(nn.Module):
    """Learned state delta with dimensions derived from the active joint count.

    The default values preserve the original G1 layout. For another robot,
    passing ``act_dim=NJ`` is sufficient when the canonical state and context
    use the standard SuperMimic layout.
    """

    def __init__(
        self,
        feat_dim: int | None = None,
        act_dim: int = DEFAULT_NUM_JOINTS,
        xi_dim: int | None = None,
        hidden=(2048, 2048, 2048),
        out_dim: int | None = None,
    ):
        super().__init__()
        act_dim = int(act_dim)
        if act_dim <= 0:
            raise ValueError(f"act_dim must be positive, got {act_dim}")
        feat_dim = 13 + 2 * act_dim if feat_dim is None else int(feat_dim)
        xi_dim = act_dim + 4 if xi_dim is None else int(xi_dim)
        out_dim = 12 + 2 * act_dim if out_dim is None else int(out_dim)
        if feat_dim <= 0 or xi_dim <= 0 or out_dim <= 0:
            raise ValueError(f"invalid dynamics dimensions: feat={feat_dim}, xi={xi_dim}, out={out_dim}")
        self.mlp = MLP(feat_dim + act_dim + xi_dim, tuple(hidden), out_dim, zero_last=True)

    def forward(self, feat: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, action, xi_norm], dim=-1))
