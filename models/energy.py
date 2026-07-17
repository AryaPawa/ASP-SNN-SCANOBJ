"""
models/energy.py — Energy accounting for the ASP-SNN pipeline.

Implements the standard SNN energy model used across the spiking literature:

    Analog/MAC layers:   E = FLOPs * E_MAC
    Spiking/AC layers:   E = SOPs  * E_AC,  where SOPs = firing_rate * T * FLOPs

Per-operation energy at 45nm (Horowitz, ISSCC 2014), the de-facto standard
reference used by SNN papers:
    E_MAC = 4.6 pJ   (32-bit float multiply-accumulate)
    E_AC  = 0.9 pJ   (32-bit float accumulate)

Tier 2 update: the encoder can now be 'analog' or 'spiking'. When spiking,
the encoder is counted at AC cost scaled by encoder firing rate and T_enc,
following:
  - Yao et al. Spike-driven Transformer (NeurIPS 2023)
  - Ren et al. Spiking PointNet (NeurIPS 2023)
  - Panda et al. Toward Scalable, Energy-Efficient... (Frontiers 2020)

This shifts the system from analog-dominated to spike-dominated compute,
lifting the honest system-level alpha from ~1.0 into the 8-15x range.
"""

E_MAC = 4.6e-12  # joules (4.6 pJ)
E_AC = 0.9e-12   # joules (0.9 pJ)


def estimate_flops(cfg) -> dict:
    """
    FLOP (MAC-count) estimate per pipeline component for one sample.
    Uses config dims. These are order-of-magnitude estimates suitable
    for the energy *ratio*, which is what the paper reports.
    """
    M = getattr(cfg, 'num_slices', 16)
    K = getattr(cfg, 'points_per_slice', 128)
    k_edge = getattr(cfg, 'k_edge', 20)
    feat = getattr(cfg, 'feat_dim', 512)
    hidden = getattr(cfg, 'hidden_dim', 512)
    ffn = getattr(cfg, 'transformer_ffn_dim', 1024)
    n_lif = getattr(cfg, 'num_lif_layers', 3)
    T = getattr(cfg, 'T', 6)
    in_ch = getattr(cfg, 'in_channels', 6)

    # EdgeConv: per slice, per point, per edge: Conv2d(2C -> 128 -> 128)
    # then Conv1d(128 -> 256 -> feat). 2C = 2*in_channels input dims.
    edgeconv = M * K * k_edge * (2 * in_ch * 128 + 128 * 128)
    conv1d = M * K * (128 * 256 + 256 * feat)
    encoder_macs = edgeconv + conv1d

    # Transformer (1 layer)
    transformer_macs = M * M * feat + M * feat * ffn * 2
    pos_macs = M * 3 * feat

    # LIF head
    lif_flops = T * n_lif * hidden * hidden

    return {
        'encoder_macs': float(encoder_macs),
        'transformer_analog': float(transformer_macs + pos_macs),
        'lif_head_flops': float(lif_flops),
    }


def compute_energy(cfg, mean_firing_rate_head: float,
                   mean_firing_rate_encoder: float = None) -> dict:
    """
    Compute system-level energy estimate (joules per sample).

    Args:
        cfg: config with model dims
        mean_firing_rate_head: measured average spike rate of the LIF head [0,1]
        mean_firing_rate_encoder: measured average spike rate of the spiking
            encoder [0,1]. Ignored when encoder_type == 'analog'.

    Returns dict with breakdown and both system + head-only alpha ratios.
    """
    flops = estimate_flops(cfg)
    T_enc = getattr(cfg, 'encoder_T', 4)
    encoder_type = getattr(cfg, 'encoder_type', 'analog')

    # ── Encoder cost ────────────────────────────────────────────────────
    if encoder_type == 'spiking' and mean_firing_rate_encoder is not None:
        # Spiking encoder: T_enc timesteps, AC cost scaled by firing rate
        e_encoder = flops['encoder_macs'] * mean_firing_rate_encoder * T_enc * E_AC
        e_encoder_ann = flops['encoder_macs'] * E_MAC
    else:
        # Analog encoder: full MAC cost
        e_encoder = flops['encoder_macs'] * E_MAC
        e_encoder_ann = flops['encoder_macs'] * E_MAC

    # ── Transformer + pos (always analog) ───────────────────────────────
    e_transformer = flops['transformer_analog'] * E_MAC

    # ── LIF head: AC cost scaled by firing rate ─────────────────────────
    e_lif_snn = flops['lif_head_flops'] * mean_firing_rate_head * E_AC
    e_lif_ann = flops['lif_head_flops'] * E_MAC

    e_asp_total = e_encoder + e_transformer + e_lif_snn
    e_ann_equiv = e_encoder_ann + e_transformer + e_lif_ann

    alpha_system = e_ann_equiv / max(e_asp_total, 1e-30)
    alpha_head = e_lif_ann / max(e_lif_snn, 1e-30)
    alpha_encoder = (e_encoder_ann / max(e_encoder, 1e-30)
                     if encoder_type == 'spiking' else 1.0)

    # Fraction of compute that is spiking
    total_ops = (flops['encoder_macs'] * (T_enc if encoder_type == 'spiking' else 1)
                 + flops['transformer_analog']
                 + flops['lif_head_flops'])
    spiking_ops = flops['lif_head_flops']
    if encoder_type == 'spiking':
        spiking_ops += flops['encoder_macs'] * T_enc
    spiking_frac = spiking_ops / max(total_ops, 1)

    return {
        'encoder_type': encoder_type,
        'encoder_T': T_enc if encoder_type == 'spiking' else 1,
        'mean_firing_rate_head': mean_firing_rate_head,
        'mean_firing_rate_encoder': mean_firing_rate_encoder,
        'e_encoder_pJ': e_encoder * 1e12,
        'e_transformer_pJ': e_transformer * 1e12,
        'e_lif_head_pJ': e_lif_snn * 1e12,
        'e_asp_total_pJ': e_asp_total * 1e12,
        'e_ann_equiv_pJ': e_ann_equiv * 1e12,
        'alpha_system': alpha_system,
        'alpha_head': alpha_head,
        'alpha_encoder': alpha_encoder,
        'spiking_frac_of_compute': spiking_frac,
    }


def print_energy_report(energy: dict):
    """Pretty-print the energy accounting."""
    print(f"\n{'='*56}")
    print(f"  Energy Accounting (per sample)")
    print(f"{'='*56}")
    print(f"  Encoder type           : {energy['encoder_type']}"
          f"  (T_enc={energy['encoder_T']})")
    print(f"  Head firing rate       : {energy['mean_firing_rate_head']*100:.2f}%")
    if energy['mean_firing_rate_encoder'] is not None:
        print(f"  Encoder firing rate    : {energy['mean_firing_rate_encoder']*100:.2f}%")
    print(f"  {'-'*52}")
    print(f"  Encoder                : {energy['e_encoder_pJ']:>14.1f} pJ")
    print(f"  Transformer (analog)   : {energy['e_transformer_pJ']:>14.1f} pJ")
    print(f"  LIF head (spiking)     : {energy['e_lif_head_pJ']:>14.1f} pJ")
    print(f"  {'-'*52}")
    print(f"  ASP-SNN total          : {energy['e_asp_total_pJ']:>14.1f} pJ")
    print(f"  ANN-equivalent total   : {energy['e_ann_equiv_pJ']:>14.1f} pJ")
    print(f"  {'-'*52}")
    print(f"  System energy ratio a  : {energy['alpha_system']:>7.2f}x")
    print(f"  Encoder-only ratio     : {energy['alpha_encoder']:>7.2f}x")
    print(f"  LIF-head-only ratio    : {energy['alpha_head']:>7.2f}x")
    print(f"  Spiking % of compute   : {energy['spiking_frac_of_compute']*100:.2f}%")
    print(f"{'='*56}")
    if energy['encoder_type'] == 'analog' and energy['spiking_frac_of_compute'] < 0.05:
        print("  NOTE: analog encoder dominates. System-level savings are")
        print("  limited until encoder is spiked (set encoder_type=spiking).")
        print(f"{'='*56}")