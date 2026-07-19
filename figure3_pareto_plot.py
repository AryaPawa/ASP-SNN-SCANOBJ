"""
figure3_pareto_plot.py — Efficiency vs. accuracy Pareto plot for the paper.

Generates the paper's Figure 3: a scatter of Instance mIoU vs. energy per
sample (log scale) with ANN baselines, SNN baselines, and the ASP-SNN
Batch 1-5 progression.

USAGE:
    python figure3_pareto_plot.py                    # writes figure3_pareto.pdf and .png
    python figure3_pareto_plot.py --format svg       # also writes .svg

UPDATE INSTRUCTIONS:
    Once you have real measurements from `eval_shapenet.py --energy`, edit
    the ASP_SNN_BATCHES dict below to replace projected values with
    measured ones. The chart re-renders automatically.

Baselines with published values are from:
    PointNet     : Qi et al. CVPR 2017
    PointNet++   : Qi et al. NeurIPS 2017
    DGCNN        : Wang et al. TOG 2019
    PointNeXt-S  : Qian et al. NeurIPS 2022
    SPM          : arXiv 2504.14371  (Spiking Point Mamba)
    S3DNet       : Elsevier 2025     (first SNN part-seg baseline)

Energy estimates for ANN baselines are FLOPs*E_MAC at 45nm (Horowitz ISSCC
2014, E_MAC=4.6 pJ). SNN baselines use their reported energy directly.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


# ══════════════════════════════════════════════════════════════════════
#  DATA — edit these dicts to update the plot
# ══════════════════════════════════════════════════════════════════════

# ANN baselines: (energy_uJ, mIoU_percent, label)
ANN_BASELINES = {
    'PointNet':     (5.0,  83.7),
    'PointNet++':   (8.0,  85.1),
    'DGCNN':        (14.0, 85.2),
    'PointNeXt-S':  (15.0, 87.0),
}

# SNN baselines: (energy_uJ, mIoU_percent, label)
SNN_BASELINES = {
    'SPM':          (1.5,  85.5),   # arXiv 2504.14371
    'S3DNet':       (2.5,  85.0),
}

# ASP-SNN batch progression: (energy_uJ, mIoU_percent, is_measured)
# Set is_measured=True once you have real numbers; the plot then drops
# the "(projected)" annotation for that point.
ASP_SNN_BATCHES = {
    'B1':   (15.0, 84.0, False),   # analog encoder + A2, A3, A4 fixes
    'B2':   (14.0, 85.0, False),   # + dense TET (A0) + soft F.P. (A1)
    'B3':   (17.0, 86.0, False),   # + boundary-aware (B1) + multiscale (B4)
    'B4':   (13.0, 85.8, False),   # + spiking seg head (B2)
    'B5':   (2.5,  86.0, False),   # + spiking encoder (B3)  <-- OUR TARGET
}

# ══════════════════════════════════════════════════════════════════════
#  Style — matched to Figure 2 (from the paper)
# ══════════════════════════════════════════════════════════════════════

COLOR_ANN     = '#888780'   # gray-400
COLOR_SNN     = '#378ADD'   # blue-400
COLOR_ASP     = '#1D9E75'   # teal-400
COLOR_OURS    = '#EF9F27'   # amber-200 (highlighted target)
COLOR_TEXT    = '#2C2C2A'
COLOR_MUTED   = '#5F5E5A'

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size':       10,
    'axes.linewidth':  0.6,
    'axes.edgecolor':  COLOR_MUTED,
    'axes.labelcolor': COLOR_TEXT,
    'xtick.color':     COLOR_MUTED,
    'ytick.color':     COLOR_MUTED,
    'text.color':      COLOR_TEXT,
    'axes.grid':       True,
    'grid.color':      '#D3D1C7',
    'grid.linewidth':  0.4,
    'grid.linestyle':  '--',
    'grid.alpha':      0.6,
    'legend.frameon':  False,
    'savefig.bbox':    'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype':    42,   # embed fonts as TrueType (required by AAAI)
    'ps.fonttype':     42,
})


def make_plot():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # ── ANN baselines ─────────────────────────────────────────────────
    ann_x = [v[0] for v in ANN_BASELINES.values()]
    ann_y = [v[1] for v in ANN_BASELINES.values()]
    ax.scatter(ann_x, ann_y, s=60, c=COLOR_ANN, edgecolors=COLOR_MUTED,
               linewidths=0.8, zorder=3, label='ANN baseline')

    for name, (x, y) in ANN_BASELINES.items():
        # Auto-place labels with small offsets to avoid overlap
        dx, dy = 0.15, 0.0
        ha = 'left'
        if name == 'PointNet':
            dx, dy, ha = 0, -0.25, 'center'
        elif name == 'DGCNN':
            dx, dy = -0.5, 0.15
            ha = 'right'
        ax.annotate(name, (x, y), xytext=(x * (1 + dx), y + dy),
                    fontsize=9, color=COLOR_MUTED, ha=ha,
                    va='center')

    # ── SNN baselines ─────────────────────────────────────────────────
    snn_x = [v[0] for v in SNN_BASELINES.values()]
    snn_y = [v[1] for v in SNN_BASELINES.values()]
    ax.scatter(snn_x, snn_y, s=60, c='#B5D4F4', edgecolors=COLOR_SNN,
               linewidths=0.8, zorder=3, label='SNN baseline')

    for name, (x, y) in SNN_BASELINES.items():
        dx, dy, ha = -0.3, 0.15, 'right'
        if name == 'S3DNet':
            dx, dy = -0.4, -0.25
        ax.annotate(name, (x, y), xytext=(x * (1 + dx), y + dy),
                    fontsize=9, color=COLOR_SNN, ha=ha, va='center')

    # ── ASP-SNN batch progression ─────────────────────────────────────
    asp_x = [v[0] for v in ASP_SNN_BATCHES.values()]
    asp_y = [v[1] for v in ASP_SNN_BATCHES.values()]

    # Trajectory: draw a dashed line connecting batches in sequence
    ax.plot(asp_x, asp_y, color=COLOR_ASP, linewidth=1.2,
            linestyle='--', alpha=0.6, zorder=2)

    # Non-Batch-5 points
    non_b5_x = [v[0] for k, v in ASP_SNN_BATCHES.items() if k != 'B5']
    non_b5_y = [v[1] for k, v in ASP_SNN_BATCHES.items() if k != 'B5']
    ax.scatter(non_b5_x, non_b5_y, s=55, c='#9FE1CB',
               edgecolors=COLOR_ASP, linewidths=0.8, zorder=4,
               label='ASP-SNN batch')

    # Non-Batch-5 labels
    for name, (x, y, _) in ASP_SNN_BATCHES.items():
        if name == 'B5':
            continue
        dx, dy, ha = 0.15, 0.15, 'left'
        if name == 'B4':
            dx, dy, ha = -0.4, -0.2, 'right'
        ax.annotate(name, (x, y), xytext=(x * (1 + dx), y + dy),
                    fontsize=9, color=COLOR_ASP, fontweight='bold',
                    ha=ha, va='center')

    # ── Batch 5 (highlighted target) ──────────────────────────────────
    x5, y5, _ = ASP_SNN_BATCHES['B5']
    ax.scatter([x5], [y5], s=280, c=COLOR_OURS, alpha=0.2,
               edgecolors='none', zorder=4)
    ax.scatter([x5], [y5], marker='*', s=280, c=COLOR_OURS,
               edgecolors='#854F0B', linewidths=1.0, zorder=5,
               label='ASP-SNN (Ours, B5)')
    ax.annotate('B5 (Ours)', (x5, y5), xytext=(x5 * 0.75, y5 + 0.4),
                fontsize=11, color='#412402', fontweight='bold',
                ha='center', va='center')

    # ── Direction hint (arrow indicating "better" corner) ────────────
    ax.annotate('better', xy=(0.55, 87.6),
                xytext=(1.6, 87.6),
                fontsize=9, color=COLOR_ASP, fontweight='bold',
                ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_ASP,
                                lw=0.8, shrinkA=0, shrinkB=0))

    # ── Axes ──────────────────────────────────────────────────────────
    ax.set_xscale('log')
    ax.set_xlim(0.5, 30)
    ax.set_ylim(83, 88)
    ax.set_xticks([1, 3, 10, 30])
    ax.set_xticklabels(['1', '3', '10', '30'])
    ax.set_yticks([83, 84, 85, 86, 87, 88])
    ax.set_xlabel('Energy per sample (\u03bcJ, log scale)', fontsize=11)
    ax.set_ylabel('Instance mIoU (%)', fontsize=11)
    ax.tick_params(axis='both', which='both', length=3, width=0.5)

    # ── Legend ────────────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_ANN, markeredgecolor=COLOR_MUTED,
               markersize=8, label='ANN baseline'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#B5D4F4', markeredgecolor=COLOR_SNN,
               markersize=8, label='SNN baseline'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#9FE1CB', markeredgecolor=COLOR_ASP,
               markersize=8, label='ASP-SNN batch'),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=COLOR_OURS, markeredgecolor='#854F0B',
               markersize=14, label='Ours (Batch 5)'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=9,
              handletextpad=0.5, columnspacing=1.0)

    # Note about projected values
    any_projected = any(not v[2] for v in ASP_SNN_BATCHES.values())
    if any_projected:
        ax.text(0.02, 0.02, 'ASP-SNN values shown are projected',
                transform=ax.transAxes, fontsize=8, style='italic',
                color=COLOR_MUTED, va='bottom', ha='left')

    return fig


def main():
    p = argparse.ArgumentParser(description="Generate the paper's Figure 3")
    p.add_argument('--out', type=str, default='figure3_pareto',
                   help='Output filename base (default: figure3_pareto)')
    p.add_argument('--format', choices=['pdf', 'png', 'svg', 'all'],
                   default='all', help='Output format(s)')
    args = p.parse_args()

    fig = make_plot()

    formats = ['pdf', 'png'] if args.format == 'all' else [args.format]
    if args.format == 'all':
        formats = ['pdf', 'png']

    for fmt in formats:
        path = f'{args.out}.{fmt}'
        fig.savefig(path, dpi=300)
        print(f'Wrote {path}')

    plt.close(fig)


if __name__ == '__main__':
    main()