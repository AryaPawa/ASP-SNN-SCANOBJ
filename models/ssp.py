"""
models/ssp.py — Slice Selection Policy with ablation modes.

Scores unvisited slices so the ASP loop can pick which slice to examine next.

Modes (set via cfg.ssp_mode, passed to __init__):
    'learned'   — attention-based scoring between LIF belief state and
                  per-slice geometry descriptors (the proposed method).
    'random'    — assign random scores to unvisited slices. The go/no-go
                  ablation baseline: if this matches 'learned', the active
                  perception thesis does not hold.
    'fps_order' — score slices by their FPS order (descending), i.e. always
                  walk slices in the deterministic FPS sequence. Tests whether
                  the learned policy beats a fixed, non-adaptive ordering.

All modes mask visited entries to -1e9 (finite, not -inf) so Gumbel-softmax
and argmax stay numerically stable even when all slices are visited (T >= M).
"""

import torch
import torch.nn as nn


class SSP(nn.Module):
    """Slice Selection Policy — supports learned / random / fps_order modes."""

    def __init__(self, belief_dim: int, geo_dim: int = 8,
                 d_ssp: int = 128, mode: str = 'learned'):
        super().__init__()
        assert mode in ('learned', 'random', 'fps_order'), \
            f"Unknown ssp_mode: {mode}"
        self.mode = mode
        # Projections are always created (so checkpoints are compatible across
        # modes), but only used when mode == 'learned'.
        self.key_proj   = nn.Linear(belief_dim, d_ssp, bias=False)
        self.query_proj = nn.Linear(geo_dim, d_ssp, bias=False)
        self.scale      = d_ssp ** -0.5

    def forward(
        self,
        belief:   torch.Tensor,   # [B, hidden_dim]
        geo:      torch.Tensor,   # [B, M, geo_dim]
        vis_mask: torch.Tensor,   # [B, M]  bool — True = already visited
    ) -> torch.Tensor:
        """
        Returns:
            scores: [B, M] with visited entries masked to -1e9.
        """
        B, M, _ = geo.shape
        device = geo.device

        if self.mode == 'learned':
            key   = self.key_proj(belief)                            # [B, d_ssp]
            query = self.query_proj(geo)                             # [B, M, d_ssp]
            scores = (query * key.unsqueeze(1)).sum(-1) * self.scale  # [B, M]

        elif self.mode == 'random':
            # Fresh random scores each timestep — no learning signal.
            scores = torch.rand(B, M, device=device)

        else:  # 'fps_order'
            # geo[:, :, 6] is max_dist, the key used to sort slices into FPS
            # order upstream. Here slices arrive already in that sorted order,
            # so a simple descending ramp picks them sequentially.
            ramp = torch.linspace(1.0, 0.0, M, device=device)        # [M]
            scores = ramp.unsqueeze(0).expand(B, M).contiguous()     # [B, M]

        return scores.masked_fill(vis_mask, -1e9)