"""
asp_classifier_trace_patch.py
─────────────────────────────
Runtime-installable instrumentation for ASPClassifier. Does NOT edit any
files in the repo. Adds one method — `forward_with_trace` — to a model
instance you pass in.

WHY A RUNTIME PATCH INSTEAD OF EDITING THE MODEL FILE?
  • Zero risk: if this file is deleted, everything works exactly as before.
  • Reversible: normal training and eval are unaffected.
  • Reviewable: everything lives in one small file that you can inspect.

WHAT IT DOES
  • Mirrors the standard eval forward pass step-for-step.
  • Disables the batch-wide `if margin > thr: break` so we always run
    all T timesteps — this is what lets us record the FULL trajectory
    the SSP would explore, and derive per-sample exit steps.
  • Records, for every sample in the batch:
       exit_step   : first timestep whose confidence margin > cfg.exit_threshold
                     (or T if the sample never exits)
       visit_order : the ORIGINAL (unsorted) slice index selected at every step
       per_step_logits : logits at every timestep
  • Predictions are formed per-sample using the logits AT that sample's
    exit step (or the mean over all T steps if it never exits — matches
    the existing eval semantics).

USAGE
─────
    from asp_classifier_trace_patch import install_trace_on

    model = ASPClassifier(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    install_trace_on(model)                       # <-- one line

    with torch.no_grad():
        preds, exit_step, visit_order, per_step_logits = \\
            model.forward_with_trace(slices, geo)

DESIGN NOTES
  • Works with both the "classic" ASPClassifier and the newer one with
    a global-context pathway (auto-detected by attribute presence).
  • Requires cfg.T and cfg.exit_threshold to be defined (they already are).
  • Batch-independent: per-sample exit steps are correct even in
    heterogeneous batches. This is the property the paper's Figure 5B
    depends on.
"""

import torch
import torch.nn.functional as F
from types import MethodType


def _forward_with_trace(self, slices, geo, exit_threshold=None):
    """
    Instrumented eval-only forward pass.  See module docstring.

    Args:
        slices:          [B, M, K, C]
        geo:             [B, M, geo_dim]
        exit_threshold:  optional override of self.cfg.exit_threshold

    Returns:
        preds            : [B]              int64 — argmax over per-sample exit logits
        exit_step        : [B]              int64 — 1..T; T means "never satisfied"
        visit_order      : [B, T]           int64 — ORIGINAL slice indices per step
        per_step_logits  : [B, T, num_class] float
    """
    B, M, K, _ = slices.shape
    device = slices.device
    T = int(self.cfg.T)
    thr = float(exit_threshold if exit_threshold is not None
                else self.cfg.exit_threshold)
    C = int(self.cfg.num_classes)

    # ── 1. Same slice-sorting as normal forward ─────────────────────
    order = geo[:, :, 6].argsort(dim=1, descending=True)     # [B, M]
    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    slices_ord = slices[batch_idx, order]
    geo_ord    = geo[batch_idx, order]

    # ── 2. Encode all slices once ───────────────────────────────────
    all_feats = self.feature_extractor(slices_ord)
    pos = self.pos_proj(geo_ord[:, :, :3])
    all_feats = all_feats + pos
    all_feats = self.slice_transformer(all_feats)

    # ── 3. Global context pathway (only present in newer models) ────
    if hasattr(self, 'global_context'):
        ctx_mean = all_feats.mean(dim=1)
        ctx_max  = all_feats.max(dim=1).values
        global_ctx = self.global_context(
            torch.cat([ctx_mean, ctx_max], dim=-1)
        )
        gate = torch.sigmoid(self.context_gate)
    else:
        global_ctx = None
        gate = None

    # ── 4. Trace buffers ────────────────────────────────────────────
    visit_order_sorted = torch.zeros(B, T, dtype=torch.long, device=device)
    per_step_logits    = torch.zeros(B, T, C, device=device)
    exit_step          = torch.full((B,), T, dtype=torch.long, device=device)
    never_exited       = torch.ones(B, dtype=torch.bool, device=device)
    logits_at_exit     = torch.zeros(B, C, device=device)

    # ── 5. ASP loop (identical to normal forward, minus the break) ──
    states = self.lif_head.init_state(B, device)
    belief = torch.zeros(B, self.cfg.hidden_dim, device=device)
    vis_mask = torch.zeros(B, M, dtype=torch.bool, device=device)

    for t in range(T):
        scores = self.ssp(belief, geo_ord, vis_mask)               # [B, M]
        # Inference: hard argmax (no Gumbel needed for tracing)
        sel_idx = scores.argmax(dim=-1)                            # [B]
        w = F.one_hot(sel_idx, M).float()

        visit_order_sorted[:, t] = sel_idx

        vis_mask = vis_mask.clone()
        vis_mask[torch.arange(B, device=device), sel_idx] = True

        e_t = (w.unsqueeze(-1) * all_feats).sum(dim=1)
        e_t = e_t + self.belief_to_feat(states[-1][0].detach())
        if global_ctx is not None:
            e_t = e_t + gate * global_ctx

        logits, states, u_last = self.lif_head.step(e_t, states)
        per_step_logits[:, t] = logits.detach()

        # ── Per-sample exit check (no batch-wide break!) ────────────
        probs  = logits.softmax(dim=-1).detach()
        top2   = probs.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]                           # [B]

        just_exited = never_exited & (margin > thr)
        exit_step = torch.where(
            just_exited,
            torch.full_like(exit_step, t + 1),      # 1-indexed
            exit_step,
        )
        logits_at_exit = torch.where(
            just_exited.unsqueeze(-1),
            logits.detach(),
            logits_at_exit,
        )
        never_exited = never_exited & ~just_exited

        belief = self.belief_norm(u_last.detach())

    # ── 6. For samples that never exited, use mean-of-T logits ──────
    final_mean = per_step_logits.mean(dim=1)
    logits_at_exit = torch.where(
        never_exited.unsqueeze(-1),
        final_mean,
        logits_at_exit,
    )

    # ── 7. Map sel indices from SORTED back to ORIGINAL space ───────
    # order[b, m'] = original index that was placed at sorted position m'
    # visit_order_sorted[b, t] is a sorted-space index → gather from order.
    visit_order = order.gather(1, visit_order_sorted)              # [B, T]

    preds = logits_at_exit.argmax(-1)
    return preds, exit_step, visit_order, per_step_logits


def install_trace_on(model):
    """
    Bind `forward_with_trace` as a method on the given model instance.

    Does not touch any class, so other models built later are unaffected.
    Idempotent: calling it twice is harmless.
    """
    if getattr(model, '_trace_installed', False):
        return model
    model.forward_with_trace = MethodType(_forward_with_trace, model)
    model._trace_installed = True
    return model