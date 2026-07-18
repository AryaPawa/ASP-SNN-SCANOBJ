"""
models/asp_segmentor.py — ASP-SNN for point cloud segmentation.

Handles both:
    ShapeNetPart : use_category=True,  num_classes=50 parts, num_categories=16
    S3DIS        : use_category=False, num_classes=13 scene classes

Architecture:
    Shared with classifier:
        EdgeConv encoder -> [B, M, feat_dim]
        Transformer -> [B, M, feat_dim]
        SSP + ASP loop + LIF -> belief [B, hidden_dim]

    Segmentation-specific:
        PerPointBranch: pts_xyz [B,N,3] -> MLP -> [B,N, point_feat_dim]
        SegHead MLP: [local | global | point_feat | (cat_onehot) | xyz] -> [B,N, num_classes]

Batch 1 upgrades:
    A2 — Global context pathway ported from the classifier: mean+max pool over
         all M slice tokens, projected to feat_dim. This replaces the previous
         weak `mean over belief snapshots` global signal. The global_ctx is
         (a) fused into every ASP timestep via a learnable gate (so LIF sees
         it during temporal reasoning), and (b) passed directly as the
         `global_feat` input to the segmentation head (uncompressed by LIF
         bottleneck, matches classifier's design).

    A3 — Category-conditional SSP: for ShapeNetPart, cat_ids are propagated
         into the SSP module so the traversal policy is category-aware from
         t=0. Fully backward compatible with S3DIS (num_categories=0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EdgeConvFeatureExtractor
from .ssp import SSP
from .lif import MultiLayerLIF


class PerPointBranch(nn.Module):
    """
    Per-point xyz -> unique feature.  Breaks the problem where all ~128
    points in the same slice share an identical 512-dim feature.

    pts_xyz [B, N, 3] -> [B, N, out_dim]
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_dim, 1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: [B, N, in_dim] -> [B, N, out_dim]"""
        x = pts.transpose(1, 2)    # [B, in_dim, N]
        x = self.mlp(x)            # [B, out_dim, N]
        return x.transpose(1, 2)   # [B, N, out_dim]


class SegmentationHead(nn.Module):
    """
    Per-point MLP for segmentation.

    Input per point:
        ShapeNet: [local(512) | global(512) | point(64) | cat(16) | xyz(3)] = 1107
        S3DIS:    [local(512) | global(512) | point(64) | xyz(3)]           = 1091

    Note (A2): the `global_feat` input is now the mean+max pooled slice-token
    representation (dim = feat_dim), not the LIF-compressed belief mean.
    So the in_dim is now `feat_dim` for global, not hidden_dim.
    """

    def __init__(self, feat_dim: int = 512, point_feat_dim: int = 64,
                 num_classes: int = 50, num_categories: int = 0,
                 xyz_dim: int = 3):
        super().__init__()
        # feat_dim is used TWICE: once for local, once for global (A2).
        in_dim = feat_dim * 2 + point_feat_dim + num_categories + xyz_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, local_feats, global_feat, point_feats,
                cat_onehot, pts_xyz):
        """
        local_feats  : [B, N, feat_dim]
        global_feat  : [B, feat_dim]        (A2: pooled slice tokens)
        point_feats  : [B, N, point_feat_dim]
        cat_onehot   : [B, num_cats] or None
        pts_xyz      : [B, N, 3]

        Returns: [B, N, num_classes]
        """
        B, N, _ = local_feats.shape
        g = global_feat.unsqueeze(1).expand(B, N, -1)

        parts = [local_feats, g, point_feats]
        if cat_onehot is not None:
            c = cat_onehot.unsqueeze(1).expand(B, N, -1).float()
            parts.append(c)
        parts.append(pts_xyz)

        x = torch.cat(parts, dim=-1)    # [B, N, in_dim]
        x = x.reshape(B * N, -1)
        return self.mlp(x).reshape(B, N, -1)


class ASPSegmentor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_category = getattr(cfg, 'use_category', False)
        self.num_classes  = cfg.num_classes
        self.num_cats     = getattr(cfg, 'num_categories', 0) if self.use_category else 0

        in_ch = getattr(cfg, 'in_channels', 6)
        point_feat_dim = getattr(cfg, 'point_feat_dim', 64)

        # Shared encoder
        self.feature_extractor = EdgeConvFeatureExtractor(
            feat_dim=cfg.feat_dim,
            k_edge=cfg.k_edge,
            in_channels=in_ch,
            encoder_type=getattr(cfg, 'encoder_type', 'analog'),
            T_enc=getattr(cfg, 'encoder_T', 4),
            lif_leak=getattr(cfg, 'encoder_lif_leak', getattr(cfg, 'lif_leak', 0.9)),
            lif_threshold=getattr(cfg, 'encoder_lif_threshold', getattr(cfg, 'lif_threshold', 1.0)),
        )
        self.pos_proj = nn.Linear(3, cfg.feat_dim, bias=False)

        self.slice_transformer = nn.TransformerEncoderLayer(
            d_model=cfg.feat_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ffn_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )

        # ── SSP with category conditioning (A3) ────────────────────────
        # Segmentor passes num_categories only when the dataset provides
        # category labels (ShapeNetPart, use_category=True).
        # S3DIS gets num_categories=0 → SSP identical to original.
        self.ssp = SSP(
            belief_dim=cfg.hidden_dim,
            geo_dim=cfg.geo_dim,
            d_ssp=cfg.d_ssp,
            mode=getattr(cfg, 'ssp_mode', 'learned'),
            num_categories=self.num_cats,
            cat_emb_dim=getattr(cfg, 'cat_emb_dim', 8),
        )
        self.belief_to_feat = nn.Linear(cfg.hidden_dim, cfg.feat_dim, bias=False)
        self.belief_norm    = nn.LayerNorm(cfg.hidden_dim)

        # ── Global context pathway (A2) ────────────────────────────────
        # Mean+max pool over all M slice tokens -> feat_dim vector.
        # Serves two purposes:
        #   1) Fused into every ASP timestep's e_t so LIF has whole-object
        #      context during temporal reasoning (mirrors classifier design).
        #   2) Passed DIRECTLY as global_feat to the segmentation head, so
        #      per-point classification sees the strong pooled signal rather
        #      than the LIF-bottlenecked belief-mean used previously.
        # Reference: PointNet (Qi et al. CVPR 2017) for mean+max pooling as
        # a permutation-invariant set aggregator; Mnih et al. RAM (NeurIPS
        # 2014) for glimpse-plus-global-context design.
        self.global_context = nn.Sequential(
            nn.Linear(cfg.feat_dim * 2, cfg.feat_dim, bias=False),
            nn.LayerNorm(cfg.feat_dim),
            nn.GELU(),
        )
        # Learnable gate for how much global_ctx to fuse into ASP loop e_t.
        # Initialized so sigmoid ≈ 0.62 (mild fusion) — model can push toward
        # 0 (only focus slice) or 1 (all global) as needed.
        self.context_gate = nn.Parameter(torch.tensor(0.5))

        # LIF head — num_classes=1 stub (not used for segmentation output)
        self.lif_head = MultiLayerLIF(
            feat_dim=cfg.feat_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=1,
            num_layers=cfg.num_lif_layers,
            leak=cfg.lif_leak,
            threshold=cfg.lif_threshold,
            learnable_params=getattr(cfg, 'lif_learnable', False),
            use_mpbn=getattr(cfg, 'lif_use_mpbn', False),
        )

        self.register_buffer('gumbel_tau',
                             torch.tensor(float(cfg.tau_start)))

        # Per-point branch
        # For S3DIS we feed xyz+rgb+height (7 dims) to PerPointBranch
        pp_in = 3
        if getattr(cfg, 'use_height', False) and getattr(cfg, 'use_rgb', False):
            pp_in = 7  # xyz + rgb + height
        elif getattr(cfg, 'use_rgb', False):
            pp_in = 6  # xyz + rgb
        self.point_branch = PerPointBranch(in_dim=pp_in, out_dim=point_feat_dim)

        # Segmentation head — global_feat now has dim feat_dim (A2)
        self.seg_head = SegmentationHead(
            feat_dim=cfg.feat_dim,
            point_feat_dim=point_feat_dim,
            num_classes=self.num_classes,
            num_categories=self.num_cats,
            xyz_dim=3,
        )

    @staticmethod
    def aux_weights(T: int) -> list:
        if T == 1:
            return [1.0]
        return [0.1 + 0.9 * (t / (T - 1)) ** 2 for t in range(T)]

    def forward(self, slices, geo, sid_arr, cat_ids, pts_features,
                training=True):
        """
        Args:
            slices:       [B, M, K, C]  per-slice point clouds
            geo:          [B, M, 8]     geometry descriptors
            sid_arr:      [B, N]        slice-id per point (int)
            cat_ids:      [B]           category index (ShapeNet) or None
            pts_features: [B, N, F]     per-point features for PerPointBranch
                          ShapeNet: [B, N, 3] (xyz)
                          S3DIS:    [B, N, 7] (xyz + rgb + height)
            training:     bool

        Returns:
            part_logits:  [B, N, num_classes]
            belief_list:  list of [B, hidden_dim] per timestep
        """
        B, M, K, _ = slices.shape
        N      = sid_arr.shape[1]
        device = slices.device

        # Defensive: verify pts_features channel dim matches PerPointBranch
        pp_in_dim = self.point_branch.mlp[0].in_channels
        assert pts_features.shape[-1] == pp_in_dim, (
            f"pts_features has {pts_features.shape[-1]} channels but "
            f"PerPointBranch expects {pp_in_dim}. Check use_rgb/use_height config."
        )

        # Category one-hot (ShapeNet) or None (S3DIS)
        if self.use_category and cat_ids is not None:
            cat_onehot = F.one_hot(cat_ids.long(), self.num_cats)
        else:
            cat_onehot = None

        # Per-point branch
        point_feats = self.point_branch(pts_features)  # [B, N, point_feat_dim]

        # Encode all slices (original order — NOT sorted)
        all_feats = self.feature_extractor(slices)       # [B, M, feat_dim]
        pos       = self.pos_proj(geo[:, :, :3])
        all_feats = all_feats + pos
        all_feats = self.slice_transformer(all_feats)

        # ── Global context over ALL M slices (A2) ──────────────────────
        # Mean + max pool -> project to feat_dim. Used both inside the ASP
        # loop and as the global input to the seg head.
        ctx_mean   = all_feats.mean(dim=1)                # [B, feat_dim]
        ctx_max    = all_feats.max(dim=1).values          # [B, feat_dim]
        global_ctx = self.global_context(
            torch.cat([ctx_mean, ctx_max], dim=-1)
        )                                                 # [B, feat_dim]

        # Sort for SSP only
        order     = geo[:, :, 6].argsort(dim=1, descending=True)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        geo_ord          = geo[batch_idx, order]
        all_feats_sorted = all_feats[batch_idx, order]

        # ASP loop — no early exit for segmentation
        states      = self.lif_head.init_state(B, device)
        belief      = torch.zeros(B, self.cfg.hidden_dim, device=device)
        vis_mask    = torch.zeros(B, M, dtype=torch.bool, device=device)
        belief_list = []

        gate = torch.sigmoid(self.context_gate)          # A2

        for t in range(self.cfg.T):
            # A3: pass cat_ids into SSP for category-conditional scoring.
            # When num_categories=0 (S3DIS) or cat_ids is None, SSP ignores it.
            scores = self.ssp(belief, geo_ord, vis_mask, cat_ids=cat_ids)

            if training:
                w = F.gumbel_softmax(
                    scores, tau=self.gumbel_tau.item(), hard=True, dim=-1,
                )
            else:
                w = F.one_hot(scores.argmax(dim=-1), M).float()

            sel_idx  = w.argmax(dim=-1)
            vis_mask = vis_mask.clone()
            vis_mask[torch.arange(B, device=device), sel_idx] = True

            # Fuse selected slice + belief + global context (A2)
            e_t = (w.unsqueeze(-1) * all_feats_sorted).sum(dim=1)
            e_t = e_t + self.belief_to_feat(states[-1][0].detach())
            e_t = e_t + gate * global_ctx                # A2

            _, states, u_last = self.lif_head.step(e_t, states)
            belief = self.belief_norm(u_last.detach())
            belief_list.append(belief)

        # Per-point local features via direct lookup (original slice order)
        b_idx       = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        local_feats = all_feats[b_idx, sid_arr.long()]  # [B, N, feat_dim]

        # Extract xyz from pts_features (always first 3 dims)
        pts_xyz = pts_features[:, :, :3]

        # ── Segmentation head ──────────────────────────────────────────
        # A2 change: use global_ctx (strong pooled feature) as the global
        # input, not the belief-mean. The belief_list is still returned
        # for downstream loss variants that may use it (e.g. Batch 2 dense TET).
        part_logits = self.seg_head(
            local_feats, global_ctx, point_feats, cat_onehot, pts_xyz,
        )

        return part_logits, belief_list