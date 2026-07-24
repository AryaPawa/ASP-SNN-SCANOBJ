"""
render_pc.py — Headless point-cloud rendering utilities for the AAAI paper.

Powers three figures:
    Figure 1 (Teaser)   : side-by-side "fixed-order vs ASP" chair rendering
    Figure 5A (Qual SSP): grid of objects with SSP selection order overlaid
    Figure 8 (Seg qual) : point-cloud rendering coloured by per-point labels

Uses matplotlib 3D scatter rather than Open3D so the output is:
    - vector PDF (AAAI requirement for line-art / annotation layers),
    - headless-friendly (no X/GLX on SLURM nodes),
    - dependency-free beyond what's already in the project.

For 2048-point clouds the render is crisp; for >8k points the RGBA scatter
gets slow but not unusable. If you ever need photorealistic renders (e.g.
Fig 8 for S3DIS rooms), swap the backend to Open3D's headless renderer —
the trajectory/order overlay logic in this file is independent of the
scatter call.

USAGE
─────
Preview on a synthetic point cloud (no data needed):

    python render_pc.py --demo

Render Fig 5A style grid from saved samples (see docstring on
`save_slice_trajectory_figure` for the .npz layout):

    python render_pc.py --samples fig5a_samples.npz --out figure5a_ssp_trajectory

Render Fig 1 teaser (fixed vs ASP on a single chair):

    python render_pc.py --teaser fig1_teaser_sample.npz --out figure1_teaser
"""

import argparse
import os
from typing import Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)


# ══════════════════════════════════════════════════════════════════════
#  Style — matched to figure3_pareto_plot.py
# ══════════════════════════════════════════════════════════════════════
COLOR_ANN     = '#888780'
COLOR_SNN     = '#378ADD'
COLOR_ASP     = '#1D9E75'
COLOR_OURS    = '#EF9F27'
COLOR_TEXT    = '#2C2C2A'
COLOR_MUTED   = '#5F5E5A'
COLOR_UNVIS   = '#D3D1C7'   # grid-line grey — used for unvisited slices

# Selection-order colour ramp (bright → dim as t increases). Deliberately
# distinct from the pareto plot's semantic palette so a reader who has seen
# Figure 3 doesn't confuse trajectory colours with baseline categories.
ORDER_CMAP = plt.get_cmap('plasma')


plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size':          10,
    'axes.linewidth':     0.6,
    'axes.edgecolor':     COLOR_MUTED,
    'axes.labelcolor':    COLOR_TEXT,
    'xtick.color':        COLOR_MUTED,
    'ytick.color':        COLOR_MUTED,
    'text.color':         COLOR_TEXT,
    'legend.frameon':     False,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype':       42,   # TrueType — required by AAAI
    'ps.fonttype':        42,
})


# ══════════════════════════════════════════════════════════════════════
#  Core rendering primitives
# ══════════════════════════════════════════════════════════════════════

def _order_color(step: int, T: int) -> tuple:
    """Colour for a slice selected at timestep `step` out of T total steps."""
    if T <= 1:
        return ORDER_CMAP(0.0)
    return ORDER_CMAP(step / (T - 1) * 0.85)   # cap at 0.85 to avoid yellow


def _set_clean_3d(ax, points: np.ndarray, elev: float = 15.,
                  azim: float = -60., pad: float = 0.05):
    """Configure a 3d axis for a clean 'floating object' look."""
    # Equal-aspect cube around the data
    lo, hi = points.min(0), points.max(0)
    ctr, rng = (lo + hi) / 2, (hi - lo).max() * (1 + pad)
    ax.set_xlim(ctr[0] - rng/2, ctr[0] + rng/2)
    ax.set_ylim(ctr[1] - rng/2, ctr[1] + rng/2)
    ax.set_zlim(ctr[2] - rng/2, ctr[2] + rng/2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    # Strip chart junk
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for spine in ('x', 'y', 'z'):
        getattr(ax, f'{spine}axis').pane.set_visible(False)
    ax.grid(False)
    # Faint floor shadow via z-axis (optional aesthetic)
    ax.xaxis.line.set_color((0, 0, 0, 0))
    ax.yaxis.line.set_color((0, 0, 0, 0))
    ax.zaxis.line.set_color((0, 0, 0, 0))


def render_pc_with_trajectory(
    ax,
    points: np.ndarray,          # [N, 3]
    anchor_xyz: np.ndarray,      # [M, 3]
    sid_arr: np.ndarray,         # [N]  slice-id per point
    order: Sequence[int],        # visitation sequence, e.g. [7, 2, 11]
    T_max: Optional[int] = None,
    title: str = '',
    show_anchors: bool = True,
    show_arrows: bool = True,
):
    """
    Draw a point cloud with slice-membership colouring by SSP visitation order.

    Points inside visited slices are coloured by the timestep at which their
    slice was selected. Points inside unvisited slices are drawn muted grey.
    Anchor centroids are overlaid as large ringed markers with timestep
    numbers. Optional dashed arrows connect anchors in visitation order.

    Args:
        ax:         a matplotlib 3d Axes (from `add_subplot(projection='3d')`)
        points:     [N, 3] xyz
        anchor_xyz: [M, 3] slice-anchor centroids
        sid_arr:    [N] slice-id per point (from assign_points_to_slices)
        order:      list of slice indices in visitation order
        T_max:      colour-ramp normaliser; defaults to len(order)
        title:      subplot title
        show_anchors: draw ringed anchor markers
        show_arrows:  draw dashed arrows between successive anchors
    """
    T = T_max if T_max is not None else max(len(order), 1)

    # ── Per-point colour: visited-slice ramp, or muted grey ─────────
    N = len(points)
    colors = np.tile(np.array(list(matplotlib.colors.to_rgba(COLOR_UNVIS))),
                     (N, 1))
    alphas = np.full(N, 0.35)
    for t, s_idx in enumerate(order):
        mask = (sid_arr == s_idx)
        colors[mask] = _order_color(t, T)
        alphas[mask] = 0.95
    colors[:, 3] = alphas   # write alpha channel

    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, s=6, marker='o', linewidths=0, depthshade=False,
    )

    # ── Anchor markers ───────────────────────────────────────────────
    if show_anchors:
        for t, s_idx in enumerate(order):
            a = anchor_xyz[s_idx]
            c = _order_color(t, T)
            # Ring + number
            ax.scatter([a[0]], [a[1]], [a[2]], s=170, c=[c],
                       edgecolors='black', linewidths=0.8,
                       depthshade=False, zorder=10)
            ax.text(a[0], a[1], a[2], f' {t}', color='black',
                    fontsize=9, fontweight='bold', zorder=11)

    # ── Trajectory arrows ────────────────────────────────────────────
    if show_arrows and len(order) >= 2:
        for t in range(len(order) - 1):
            a0 = anchor_xyz[order[t]]
            a1 = anchor_xyz[order[t + 1]]
            ax.plot([a0[0], a1[0]], [a0[1], a1[1]], [a0[2], a1[2]],
                    color=COLOR_TEXT, linewidth=0.8, linestyle=(0, (3, 2)),
                    alpha=0.55, zorder=9)

    _set_clean_3d(ax, points)
    if title:
        ax.set_title(title, fontsize=10, pad=2)


# ══════════════════════════════════════════════════════════════════════
#  Multi-panel figure builders
# ══════════════════════════════════════════════════════════════════════

def save_slice_trajectory_figure(samples: dict, out_base: str,
                                 T_max: int = 6):
    """
    Build Figure 5A: grid of qualitative SSP trajectories.

    Layout: rows = objects (e.g. chair / airplane / table / lamp),
            cols = policy (learned / fps_order / random).

    `samples` must be a dict of the form:
        samples['objects'] = ['chair', 'airplane', 'table', 'lamp']
        samples['learned']  = [dict(points, anchor_xyz, sid_arr, order), ...]
        samples['fps_order']= [dict(points, anchor_xyz, sid_arr, order), ...]
        samples['random']   = [dict(points, anchor_xyz, sid_arr, order), ...]

    Each per-object dict has:
        points:     [N, 3] float
        anchor_xyz: [M, 3] float
        sid_arr:    [N]     int
        order:      list[int]  visitation sequence
    """
    objects = samples['objects']
    modes = [('learned', 'ASP-SNN (learned)'),
             ('fps_order', 'FPS-order baseline'),
             ('random', 'Random baseline')]

    n_rows, n_cols = len(objects), len(modes)
    fig = plt.figure(figsize=(2.2 * n_cols, 2.4 * n_rows))
    for r, obj_name in enumerate(objects):
        for c, (mode_key, mode_label) in enumerate(modes):
            ax = fig.add_subplot(n_rows, n_cols,
                                 r * n_cols + c + 1, projection='3d')
            s = samples[mode_key][r]
            render_pc_with_trajectory(
                ax,
                points=s['points'],
                anchor_xyz=s['anchor_xyz'],
                sid_arr=s['sid_arr'],
                order=s['order'],
                T_max=T_max,
                title=(mode_label if r == 0 else ''),
            )
            if c == 0:
                ax.text2D(-0.05, 0.5, obj_name, transform=ax.transAxes,
                          fontsize=10, fontweight='bold', rotation=90,
                          va='center', ha='center', color=COLOR_TEXT)

    # ── Colourbar-style legend for timestep order ───────────────────
    fig.subplots_adjust(bottom=0.10, left=0.06, right=0.98, top=0.94,
                        wspace=0.05, hspace=0.05)
    cbar_ax = fig.add_axes([0.30, 0.045, 0.40, 0.018])
    sm = plt.cm.ScalarMappable(cmap=ORDER_CMAP,
                               norm=plt.Normalize(vmin=0, vmax=T_max - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Selection order (t)', fontsize=9)
    cbar.set_ticks(range(T_max))
    cbar.ax.tick_params(labelsize=8, length=2, width=0.5)

    for ext in ('pdf', 'png'):
        path = f'{out_base}.{ext}'
        fig.savefig(path, dpi=300)
        print(f'Wrote {path}')
    plt.close(fig)


def save_teaser_figure(sample: dict, out_base: str, T_max: int = 6):
    """
    Build Figure 1 teaser: left = fixed FPS-order, right = ASP learned.

    `sample` fields:
        points:     [N, 3]
        anchor_xyz: [M, 3]
        sid_arr:    [N]
        order_fixed: list[int]   e.g. list(range(M))     — baseline
        order_asp:   list[int]   e.g. [7, 2, 11]         — early-exit
        exit_step_asp: int       # slices ASP actually used
    """
    fig = plt.figure(figsize=(7.0, 3.6))
    ax_left  = fig.add_subplot(1, 2, 1, projection='3d')
    ax_right = fig.add_subplot(1, 2, 2, projection='3d')

    # Baseline: use the full FPS order, colour ramp shows how many timesteps
    # a fixed-order SNN burns even after it's already confident.
    render_pc_with_trajectory(
        ax_left,
        points=sample['points'],
        anchor_xyz=sample['anchor_xyz'],
        sid_arr=sample['sid_arr'],
        order=sample['order_fixed'],
        T_max=len(sample['order_fixed']),
        title=f'Baseline SNN: all {len(sample["order_fixed"])} slices',
        show_arrows=False,   # too much clutter for a full 16-step walk
    )
    render_pc_with_trajectory(
        ax_right,
        points=sample['points'],
        anchor_xyz=sample['anchor_xyz'],
        sid_arr=sample['sid_arr'],
        order=sample['order_asp'],
        T_max=T_max,
        title=(f'ASP-SNN (ours): '
               f'{sample["exit_step_asp"]} slices, early-exit ✓'),
        show_arrows=True,
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02,
                        wspace=0.05)
    for ext in ('pdf', 'png'):
        path = f'{out_base}.{ext}'
        fig.savefig(path, dpi=300)
        print(f'Wrote {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  Demo (synthetic data) — for previewing the layout with no real samples
# ══════════════════════════════════════════════════════════════════════

def _make_synthetic_chair(n_points: int = 2048, rng=None) -> np.ndarray:
    """Rough chair: seat plate + backrest + 4 legs. Purely for layout demo."""
    if rng is None:
        rng = np.random.default_rng(0)
    seat = rng.uniform([-0.5, -0.5, 0.0], [0.5, 0.5, 0.05], (n_points // 3, 3))
    back = rng.uniform([-0.5, 0.45, 0.05], [0.5, 0.5, 1.0],
                       (n_points // 3, 3))
    legs = []
    for cx, cy in [(-.45, -.45), (-.45, .45), (.45, -.45), (.45, .45)]:
        legs.append(rng.uniform([cx - .03, cy - .03, -1.0],
                                [cx + .03, cy + .03, 0.0],
                                (n_points // 12, 3)))
    return np.concatenate([seat, back, *legs])[:n_points]


def _demo_standalone(out_base: str, T: int = 6):
    """Self-contained demo that does not depend on the repo's slicing.py."""
    rng = np.random.default_rng(0)
    pts = _make_synthetic_chair(rng=rng)
    N = len(pts)

    # Trivial FPS: pick M points greedily
    M = 16
    idx0 = int(rng.integers(0, N))
    anchors = [idx0]
    d = np.full(N, 1e9)
    for _ in range(M - 1):
        d = np.minimum(d, np.sum((pts - pts[anchors[-1]])**2, axis=1))
        anchors.append(int(np.argmax(d)))
    anchor_xyz = pts[anchors]

    # Assign each point to nearest anchor
    dmat = np.linalg.norm(pts[:, None] - anchor_xyz[None], axis=-1)
    sid_arr = dmat.argmin(1)

    # Fake three trajectories
    def fake_order(mode: str) -> list:
        if mode == 'learned':
            # Pick anchors whose y is highest (backrest) first, then seat, then legs
            keys = np.argsort(-anchor_xyz[:, 2])
            return keys[:T].tolist()
        if mode == 'fps_order':
            return list(range(T))
        # random
        r = list(range(M))
        rng2 = np.random.default_rng(42)
        rng2.shuffle(r)
        return r[:T]

    def _mk(mode):
        return dict(points=pts, anchor_xyz=anchor_xyz, sid_arr=sid_arr,
                    order=fake_order(mode))

    samples = {
        'objects': ['chair (demo)'],
        'learned':   [_mk('learned')],
        'fps_order': [_mk('fps_order')],
        'random':    [_mk('random')],
    }
    save_slice_trajectory_figure(samples, out_base + '_fig5a', T_max=T)

    # Teaser demo
    teaser = dict(
        points=pts, anchor_xyz=anchor_xyz, sid_arr=sid_arr,
        order_fixed=list(range(M)),
        order_asp=fake_order('learned')[:3],
        exit_step_asp=3,
    )
    save_teaser_figure(teaser, out_base + '_fig1', T_max=T)


# ══════════════════════════════════════════════════════════════════════
#  Sample-collection helper — how to build the .npz your figures consume
# ══════════════════════════════════════════════════════════════════════

SAMPLE_COLLECTION_SNIPPET = '''
    # ── Drop this into eval_scanobj.py (or eval_shapenet.py) just after
    #    the ASP loop finishes for a chosen batch. It captures everything
    #    render_pc.py needs.
    #
    # NOTE: model.asp_loop must return `visit_order` as a [B, T] tensor of
    # int slice-indices; add a couple of lines to asp_classifier.py to
    # log the argmax at each timestep if it doesn't already.

    import numpy as np

    picked = 0                                # per-object slot in the fig
    with torch.no_grad():
        for batch in test_loader:
            slices, geo, sid, xyz, labels = batch
            slices = slices.to(device); geo = geo.to(device)
            sid = sid.to(device); xyz = xyz.to(device)

            _, visit_order, exit_step = model(
                slices, geo, xyz, sid,
                return_trace=True,      # <-- one-line hook you add
            )

            for b in range(slices.size(0)):
                cls_name = idx_to_name[int(labels[b])]
                if cls_name not in wanted_objects:
                    continue

                # Anchor xyz can be reconstructed as the mean over each slice
                anchor_xyz_b = slices[b, :, :, :3].mean(dim=1).cpu().numpy()

                np.savez(f'fig5a_{cls_name}_{picked}.npz',
                         points=xyz[b].cpu().numpy(),
                         anchor_xyz=anchor_xyz_b,
                         sid_arr=sid[b].cpu().numpy(),
                         order_asp=visit_order[b, :exit_step[b]].cpu().numpy(),
                         order_fixed=np.arange(anchor_xyz_b.shape[0]),
                         exit_step_asp=int(exit_step[b]))
                picked += 1
                if picked >= max_samples:
                    return
'''


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--demo', action='store_true',
                   help='Render a synthetic demo (no data needed)')
    p.add_argument('--samples', type=str, default=None,
                   help='.npz produced by the eval-time collection snippet '
                        '(see SAMPLE_COLLECTION_SNIPPET). Renders Fig 5A.')
    p.add_argument('--teaser', type=str, default=None,
                   help='.npz for the Fig 1 teaser (fixed vs ASP).')
    p.add_argument('--out', type=str, default='render_pc_out',
                   help='Output filename base')
    p.add_argument('--T', type=int, default=6,
                   help='Total ASP timesteps (colour-ramp normaliser)')
    p.add_argument('--print-hook', action='store_true',
                   help='Print the eval-script hook snippet and exit')
    args = p.parse_args()

    if args.print_hook:
        print(SAMPLE_COLLECTION_SNIPPET)
        return

    if args.demo:
        _demo_standalone(args.out, T=args.T)
        return

    if args.teaser:
        data = np.load(args.teaser, allow_pickle=True)
        sample = {k: data[k] for k in data.files}
        # Coerce arrays to lists where the API expects sequences
        sample['order_fixed'] = list(sample['order_fixed'])
        sample['order_asp']   = list(sample['order_asp'])
        sample['exit_step_asp'] = int(sample['exit_step_asp'])
        save_teaser_figure(sample, args.out, T_max=args.T)
        return

    if args.samples:
        data = np.load(args.samples, allow_pickle=True)
        samples = data['samples'].item()   # dict was saved with allow_pickle
        save_slice_trajectory_figure(samples, args.out, T_max=args.T)
        return

    p.print_help()


if __name__ == '__main__':
    main()