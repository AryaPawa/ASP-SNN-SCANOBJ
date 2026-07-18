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

Batch 1 upgrades (already applied):
    A2 — Global context pathway (mean+max pool of all M slice tokens)
    A3 — Category-conditional SSP (ShapeNetPart cat_id fed to SSP)

Batch 2 upgrades (new):
    A0 — Dense TET loss support: per-timestep segmentation logits computed
         inside the ASP loop. When cfg.loss_mode == 'dense_tet', the seg
         head runs at every timestep with a per-timestep belief-modulated
         global signal, giving the LIF + seg pathway supervised gradient
         at every step (analog of Deng et al. TET, ICLR 2022, for dense
         prediction). Adds one small Linear (hidden_dim -> feat_dim) that
         is zero-initialized so this is a NO-OP when dense_tet is off.

    A1 — Soft feature propagation: replaces the hard slice lookup
         `local_feats = all_feats[b, sid_arr]` with a k-nearest inverse-
         distance weighted mixture over slice anchors. This kills the
         resolution-collapse problem where every ~128 points in a slice
         shared an identical 512-d feature vector. Direct analog of
         PointNet++'s Feature Propagation (Qi et al. NeurIPS 2017).
         Config-gated via `use_soft_fp` for A/B testing.

Return signature: `(part_logits, aux)` where `aux` is a dict:
    - aux['belief_list']         : list of [B, hidden_dim] per timestep
    - aux['per_timestep_logits'] : list of [B, N, num_classes] or None
Backward compatible with callers that do `part_logits, _ = model(...)`.
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
    """

    def __init__(self, feat_dim: int = 512, point_feat_dim: int = 64,
                 num_classes: int = 50, num_categories: int = 0,
                 xyz_dim: int = 3):
        super().__init__()
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
        global_feat  : [B, feat_dim]
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

        # SSP with category conditioning (A3)
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

        # Global context pathway (A2)
        self.global_context = nn.Sequential(
            nn.Linear(cfg.feat_dim * 2, cfg.feat_dim, bias=False),
            nn.LayerNorm(cfg.feat_dim),
            nn.GELU(),
        )
        self.context_gate = nn.Parameter(torch.tensor(0.5))

        # ── A0: Belief-to-seg-global projection ────────────────────────
        # Zero-initialized so this is a no-op when dense_tet is not enabled.
        # When dense_tet is on, this Linear learns to route belief info into
        # the seg head at every timestep, giving the LIF pathway direct
        # gradient signal from the per-point loss.
        self.belief_to_seg_global = nn.Linear(cfg.hidden_dim, cfg.feat_dim, bias=False)
        nn.init.zeros_(self.belief_to_seg_global.weight)

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
        pp_in = 3
        if getattr(cfg, 'use_height', False) and getattr(cfg, 'use_rgb', False):
            pp_in = 7  # xyz + rgb + height
        elif getattr(cfg, 'use_rgb', False):
            pp_in = 6  # xyz + rgb
        self.point_branch = PerPointBranch(in_dim=pp_in, out_dim=point_feat_dim)

        # Segmentation head
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

    # ══════════════════════════════════════════════════════════════════
    #  A1 — Soft Feature Propagation
    # ══════════════════════════════════════════════════════════════════
    def _soft_feature_propagation(
        self,
        all_feats:    torch.Tensor,   # [B, M, feat_dim]  (original order)
        geo:          torch.Tensor,   # [B, M, geo_dim]   (original order)
        pts_features: torch.Tensor,   # [B, N, F]         (first 3 = xyz)
        sid_arr:      torch.Tensor,   # [B, N]            (legacy fallback)
    ) -> torch.Tensor:
        """
        Inverse-distance weighted mixture of the fp_k nearest slice anchors
        for each point. Analog of PointNet++ Feature Propagation.

        Returns local_feats: [B, N, feat_dim]
        """
        B, M, feat_dim = all_feats.shape
        N = pts_features.shape[1]
        device = all_feats.device

        use_soft_fp = getattr(self.cfg, 'use_soft_fp', True)
        if not use_soft_fp:
            # Legacy hard lookup — kept for A/B ablation.
            b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
            return all_feats[b_idx, sid_arr.long()]

        k_fp   = getattr(self.cfg, 'fp_k', 3)
        fp_eps = getattr(self.cfg, 'fp_epsilon', 1e-8)

        anchor_xyz = geo[:, :, :3]                        # [B, M, 3]
        points_xyz = pts_features[:, :, :3]               # [B, N, 3]

        # cdist is memory-efficient vs. explicit broadcasting.
        # Use fp32 for numerical stability under AMP.
        dist = torch.cdist(points_xyz.float(), anchor_xyz.float())   # [B, N, M]

        # Guard against k > M (small-M configs)
        k_use = min(k_fp, M)
        knn_dist, knn_idx = dist.topk(k_use, dim=-1, largest=False)  # [B, N, k]

        # Inverse-distance weights (normalized to sum to 1 over k neighbours)
        inv_dist = 1.0 / (knn_dist + fp_eps)                          # [B, N, k]
        weights  = inv_dist / inv_dist.sum(dim=-1, keepdim=True)      # [B, N, k]

        # Gather features of the k nearest anchors
        b_idx_k = (torch.arange(B, device=device)
                   .view(B, 1, 1).expand(B, N, k_use))
        knn_feats = all_feats[b_idx_k, knn_idx]           # [B, N, k, feat_dim]

        # Weighted sum. Cast weights back to feature dtype for AMP.
        weights_typed = weights.to(knn_feats.dtype)
        local_feats = (weights_typed.unsqueeze(-1) * knn_feats).sum(dim=-2)
        return local_feats                                # [B, N, feat_dim]

    # ══════════════════════════════════════════════════════════════════
    #  Forward
    # ══════════════════════════════════════════════════════════════════
    def forward(self, slices, geo, sid_arr, cat_ids, pts_features,
                training=True):
        """
        Args:
            slices:       [B, M, K, C]  per-slice point clouds
            geo:          [B, M, 8]     geometry descriptors
            sid_arr:      [B, N]        slice-id per point (int; legacy fallback)
            cat_ids:      [B]           category index (ShapeNet) or None
            pts_features: [B, N, F]     per-point features for PerPointBranch
            training:     bool

        Returns:
            part_logits:  [B, N, num_classes]
            aux:          dict with keys:
                'belief_list'         : list of [B, hidden_dim] per timestep
                'per_timestep_logits' : list of [B, N, num_classes] or None
                                        (populated only when training and
                                         loss_mode == 'dense_tet')
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

        # A0 flag: are we producing per-timestep seg logits for dense TET?
        compute_per_t = training and (
            getattr(self.cfg, 'loss_mode', 'aux') == 'dense_tet'
        )

        # Category one-hot (ShapeNet) or None (S3DIS)
        if self.use_category and cat_ids is not None:
            cat_onehot = F.one_hot(cat_ids.long(), self.num_cats)
        else:
            cat_onehot = None

        # Per-point branch
        point_feats = self.point_branch(pts_features)     # [B, N, point_feat_dim]

        # Encode all slices (original order — NOT sorted)
        all_feats = self.feature_extractor(slices)        # [B, M, feat_dim]
        pos       = self.pos_proj(geo[:, :, :3])
        all_feats = all_feats + pos
        all_feats = self.slice_transformer(all_feats)

        # Global context over ALL M slices (A2)
        ctx_mean   = all_feats.mean(dim=1)                # [B, feat_dim]
        ctx_max    = all_feats.max(dim=1).values          # [B, feat_dim]
        global_ctx = self.global_context(
            torch.cat([ctx_mean, ctx_max], dim=-1)
        )                                                 # [B, feat_dim]

        # ── A1: soft feature propagation (computed once, reused per step) ──
        local_feats = self._soft_feature_propagation(
            all_feats, geo, pts_features, sid_arr,
        )                                                 # [B, N, feat_dim]

        # xyz for seg head (always first 3 dims of pts_features)
        pts_xyz = pts_features[:, :, :3]

        # Sort for SSP only
        order     = geo[:, :, 6].argsort(dim=1, descending=True)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        geo_ord          = geo[batch_idx, order]
        all_feats_sorted = all_feats[batch_idx, order]

        # ASP loop
        states              = self.lif_head.init_state(B, device)
        belief              = torch.zeros(B, self.cfg.hidden_dim, device=device)
        vis_mask            = torch.zeros(B, M, dtype=torch.bool, device=device)
        belief_list         = []
        per_timestep_logits = [] if compute_per_t else None

        gate    = torch.sigmoid(self.context_gate)         # A2
        u_last  = None                                     # will be set inside loop

        for t in range(self.cfg.T):
            # A3: pass cat_ids to SSP for category-conditional scoring
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
            e_t = e_t + gate * global_ctx

            _, states, u_last = self.lif_head.step(e_t, states)

            # ── A0: per-timestep seg logits with LIVE belief gradient ──
            # `belief_live` is NOT detached, so the per-timestep CE loss
            # backpropagates through belief_norm -> LIF cells at this step.
            # This is what makes dense TET actually train the LIF pathway
            # via the segmentation loss (analogous to how classifier TET
            # supervises fc_out at every step).
            if compute_per_t:
                belief_live = self.belief_norm(u_last)
                global_t = global_ctx + self.belief_to_seg_global(belief_live)
                logits_t = self.seg_head(
                    local_feats, global_t, point_feats, cat_onehot, pts_xyz,
                )
                per_timestep_logits.append(logits_t)

            # Detached belief for the NEXT iteration's SSP (unchanged
            # behaviour vs. Batch 1 — prevents recurrent graph growth).
            belief = self.belief_norm(u_last.detach())
            belief_list.append(belief)

        # ── Final part_logits ──────────────────────────────────────────
        if compute_per_t:
            # Reuse the last per-timestep logits (already computed with the
            # final belief). Avoids a redundant seg_head call.
            part_logits = per_timestep_logits[-1]
        else:
            # No dense TET: run seg head once at the end. Include belief
            # modulation for consistency (belief_to_seg_global is zero-init
            # so when dense TET was never on during training this reduces
            # to global_ctx alone — matching Batch 1 behaviour).
            final_belief = self.belief_norm(u_last) if u_last is not None else belief
            final_global = global_ctx + self.belief_to_seg_global(final_belief)
            part_logits = self.seg_head(
                local_feats, final_global, point_feats, cat_onehot, pts_xyz,
            )

        aux = {
            'belief_list':         belief_list,
            'per_timestep_logits': per_timestep_logits,
        }
        return part_logits, aux