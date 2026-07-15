"""
models/energy.py — Energy accounting for the ASP-SNN pipeline.

Implements the standard SNN energy model used across the spiking literature
(e.g. Yao et al. Spike-driven Transformer; Hu et al.; Panda et al.):

    Analog/MAC layers:   E = FLOPs * E_MAC
    Spiking/AC layers:   E = SOPs  * E_AC,  where SOPs = firing_rate * T * FLOPs

Per-operation energy at 45nm (Horowitz, ISSCC 2014), the de-facto standard
reference used by SNN papers:
    E_MAC = 4.6 pJ   (32-bit float multiply-accumulate)
    E_AC  = 0.9 pJ   (32-bit float accumulate)

This module separates the pipeline into:
    - ANALOG components (EdgeConv encoder, transformer): always MAC
    - SPIKING components (LIF head): AC, scaled by measured firing rate

IMPORTANT HONESTY NOTE: with the current architecture the analog encoder
dominates total compute (~99%). This module reports the TRUE system-level
energy so the paper's efficiency claim is grounded in measurement, not a
favorable proxy. The energy ratio alpha = E_ann_equivalent / E_asp_snn will
be near 1.0 until the encoder itself is spiked (Tier 2 future work).
"""

E_MAC = 4.6e-12  # joules (4.6 pJ)
E_AC = 0.9e-12   # joules (0.9 pJ)


def estimate_flops(cfg) -> dict:
    """
    Rough FLOP (MAC-count) estimate per pipeline component for one sample.
    Uses config dims. These are order-of-magnitude estimates suitable for
    the energy *ratio*, which is what the paper reports.

    Returns dict of component -> MAC count.
    """
    M = getattr(cfg, 'num_slices', 16)
    K = getattr(cfg, 'points_per_slice', 128)
    k_edge = getattr(cfg, 'k_edge', 20)
    feat = getattr(cfg, 'feat_dim', 512)
    hidden = getattr(cfg, 'hidden_dim', 512)
    ffn = getattr(cfg, 'transformer_ffn_dim', 1024)
    n_lif = getattr(cfg, 'num_lif_layers', 3)
    T = getattr(cfg, 'T', 6)

    # EdgeConv: per slice, per point, per edge: Conv2d(2C->128->128)
    # then Conv1d(128->256->512). C=6 input channels -> 2C=12.
    edgeconv = M * K * k_edge * (12 * 128 + 128 * 128)
    conv1d = M * K * (128 * 256 + 256 * feat)
    encoder_macs = edgeconv + conv1d

    # Transformer (1 layer): attention (M x M x feat) + FFN (M x feat x ffn x 2)
    transformer_macs = M * M * feat + M * feat * ffn * 2

    # Positional projection: M x 3 x feat
    pos_macs = M * 3 * feat

    # LIF head (SPIKING): T timesteps x n_lif layers x (hidden x hidden)
    # These run as AC ops scaled by firing rate (applied later).
    lif_flops = T * n_lif * hidden * hidden

    return {
        'encoder_analog': float(encoder_macs),
        'transformer_analog': float(transformer_macs + pos_macs),
        'lif_spiking': float(lif_flops),
    }


def compute_energy(cfg, mean_firing_rate: float) -> dict:
    """
    Compute system-level energy estimate (joules per sample).

    Args:
        cfg: config with model dims
        mean_firing_rate: measured average spike rate of the LIF head in [0,1]

    Returns dict with energy breakdown and the honest alpha ratio.
    """
    flops = estimate_flops(cfg)

    # Analog components always pay full MAC cost
    e_encoder = flops['encoder_analog'] * E_MAC
    e_transformer = flops['transformer_analog'] * E_MAC

    # Spiking component: AC cost scaled by firing rate
    e_lif_snn = flops['lif_spiking'] * mean_firing_rate * E_AC

    # If the LIF head were a conventional ANN (dense MAC, no sparsity):
    e_lif_ann = flops['lif_spiking'] * E_MAC

    e_asp_total = e_encoder + e_transformer + e_lif_snn
    e_ann_equiv = e_encoder + e_transformer + e_lif_ann

    # System-level energy ratio (honest: includes the analog encoder)
    alpha_system = e_ann_equiv / max(e_asp_total, 1e-30)

    # Head-only ratio (what the SNN head buys you, in isolation)
    alpha_head = e_lif_ann / max(e_lif_snn, 1e-30)

    total_macs = sum(flops.values())
    spiking_frac = flops['lif_spiking'] / max(total_macs, 1)

    return {
        'mean_firing_rate': mean_firing_rate,
        'e_encoder_pJ': e_encoder * 1e12,
        'e_transformer_pJ': e_transformer * 1e12,
        'e_lif_snn_pJ': e_lif_snn * 1e12,
        'e_asp_total_pJ': e_asp_total * 1e12,
        'e_ann_equiv_pJ': e_ann_equiv * 1e12,
        'alpha_system': alpha_system,
        'alpha_head': alpha_head,
        'spiking_frac_of_compute': spiking_frac,
    }


def print_energy_report(energy: dict):
    """Pretty-print the energy accounting."""
    print(f"\n{'='*52}")
    print(f"  Energy Accounting (per sample)")
    print(f"{'='*52}")
    print(f"  Mean LIF firing rate   : {energy['mean_firing_rate']*100:.2f}%")
    print(f"  Encoder (analog)       : {energy['e_encoder_pJ']:>12.1f} pJ")
    print(f"  Transformer (analog)   : {energy['e_transformer_pJ']:>12.1f} pJ")
    print(f"  LIF head (spiking)     : {energy['e_lif_snn_pJ']:>12.1f} pJ")
    print(f"  {'-'*48}")
    print(f"  ASP-SNN total          : {energy['e_asp_total_pJ']:>12.1f} pJ")
    print(f"  ANN-equivalent total   : {energy['e_ann_equiv_pJ']:>12.1f} pJ")
    print(f"  {'-'*48}")
    print(f"  System energy ratio a  : {energy['alpha_system']:.3f}x")
    print(f"  LIF-head-only ratio    : {energy['alpha_head']:.3f}x")
    print(f"  Spiking % of compute   : {energy['spiking_frac_of_compute']*100:.2f}%")
    print(f"{'='*52}")
    if energy['spiking_frac_of_compute'] < 0.05:
        print("  NOTE: analog encoder dominates compute. System-level")
        print("  energy savings are limited until the encoder is spiked")
        print("  (Tier 2 future work). Head-only ratio shows SNN potential.")
        print(f"{'='*52}")