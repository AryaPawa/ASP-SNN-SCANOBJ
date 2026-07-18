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

Category conditioning (A3 — Batch 1):
    When num_categories > 0 (ShapeNetPart), a learned category embedding is
    injected into the "key" projection so the SSP has a category-specific
    traversal prior from t=0. Adds ~1K parameters.

    Why this helps: for ShapeNetPart the category is known at input time.
    Without conditioning, the SSP burns early timesteps rediscovering "am I
    a chair or a lamp?" before it can specialize its traversal.
    With conditioning, the SSP can immediately learn category-specific
    policies (e.g. "for chairs, backrest first; for motorbikes, wheels first").

    When num_categories = 0 (ScanObjectNN classification, S3DIS scene seg),
    this branch is fully disabled — no extra parameters, exact backward
    compatibility with the original SSP.

    Reference: Mnih et al., Recurrent Models of Visual Attention (NeurIPS 2014)
    — precedent for conditioning attention on task metadata.

All modes mask visited entries to -1e9 (finite, not -inf) so Gumbel-softmax
and argmax stay numerically stable even when all slices are visited (T >= M).
"""

import torch
import torch.nn as nn


class SSP(nn.Module):
    """Slice Selection Policy — supports learned / random / fps_order modes."""

    def __init__(self, belief_dim: int, geo_dim: int = 8,
                 d_ssp: int = 128, mode: str = 'learned',
                 num_categories: int = 0, cat_emb_dim: int = 8):
        super().__init__()
        assert mode in ('learned', 'random', 'fps_order'), \
            f"Unknown ssp_mode: {mode}"
        self.mode = mode

        # Projections are always created (so checkpoints are compatible across
        # modes), but only used when mode == 'learned'.
        self.key_proj   = nn.Linear(belief_dim, d_ssp, bias=False)
        self.query_proj = nn.Linear(geo_dim, d_ssp, bias=False)
        self.scale      = d_ssp ** -0.5

        # ── Category-conditional SSP (A3) ──────────────────────────────
        # When enabled, a learned category embedding is added to the key so the
        # SSP knows the object class from t=0. Only used for datasets with
        # category labels (ShapeNetPart). Fully disabled when num_categories=0
        # (ScanObjectNN, S3DIS).
        self.num_categories = num_categories
        if num_categories > 0:
            self.cat_emb      = nn.Embedding(num_categories, cat_emb_dim)
            self.cat_key_proj = nn.Linear(cat_emb_dim, d_ssp, bias=False)
            # Small init so category signal starts as a gentle perturbation
            # on top of the belief-driven key, then grows as needed.
            nn.init.normal_(self.cat_emb.weight, std=0.02)
            nn.init.zeros_(self.cat_key_proj.weight)
        else:
            self.cat_emb      = None
            self.cat_key_proj = None

    def forward(
        self,
        belief:   torch.Tensor,           # [B, hidden_dim]
        geo:      torch.Tensor,           # [B, M, geo_dim]
        vis_mask: torch.Tensor,           # [B, M]  bool — True = visited
        cat_ids:  torch.Tensor = None,    # [B]  optional — category ids
    ) -> torch.Tensor:
        """
        Args:
            belief   : LIF belief state (or zeros at t=0)
            geo      : per-slice geometry descriptors
            vis_mask : True at slices already visited by the ASP loop
            cat_ids  : optional per-sample category ids (ShapeNetPart only).
                       Ignored when num_categories was 0 at construction.

        Returns:
            scores: [B, M] with visited entries masked to -1e9.
        """
        B, M, _ = geo.shape
        device = geo.device

        if self.mode == 'learned':
            key = self.key_proj(belief)                              # [B, d_ssp]

            # Category-conditional key augmentation (A3).
            # If category embeddings weren't created OR no cat_ids supplied,
            # this branch is a no-op — behavior identical to original SSP.
            if self.cat_emb is not None and cat_ids is not None:
                cat_key = self.cat_key_proj(self.cat_emb(cat_ids.long()))
                key = key + cat_key                                  # [B, d_ssp]

            query  = self.query_proj(geo)                             # [B, M, d_ssp]
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