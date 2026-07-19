import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Create output directory
out_dir = "aaai_figures"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# Figure 1: Active Perception Loop (Motorbike)
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 6, figsize=(18, 3))
fig.suptitle("Active perception loop — what the SSP sees on a motorbike", fontsize=16, y=1.05)

# Motorbike base components (simplified point cloud aesthetic)
def draw_motorbike(ax, alpha=0.3):
    # Wheels
    ax.add_patch(patches.Circle((2, 2), 1, fill=False, edgecolor='gray', lw=2, alpha=alpha, ls='--'))
    ax.add_patch(patches.Circle((7, 2), 1, fill=False, edgecolor='gray', lw=2, alpha=alpha, ls='--'))
    ax.plot([2, 2], [2, 2], 'ko', markersize=3, alpha=alpha) # hubs
    ax.plot([7, 7], [2, 2], 'ko', markersize=3, alpha=alpha)
    
    # Body/Frame
    ax.plot([2, 4.5, 7, 7.5], [2, 4.5, 4, 2], 'gray', lw=2, alpha=alpha)
    
    # Tank & Seat
    ax.add_patch(patches.Rectangle((3.5, 4.5), 2, 1, fill=False, edgecolor='gray', lw=2, alpha=alpha, ls='--'))
    
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

# States for t=1 to t=6
titles = [
    "t=1\ngrab rear wheel", 
    "t=2\ngrab gas tank", 
    "t=3\ngrab front wheel", 
    "t=4 (bnd-guided)\nrefine seat/tank", 
    "t=5 (bnd-guided)\nrefine wheel/frame", 
    "t=6 (output)\npart labels"
]

highlights = [
    {'center': (2, 2), 'radius': 1.2, 'color': '#DEB887', 'type': 'struct'},
    {'center': (4.5, 5), 'radius': 1.2, 'color': '#DEB887', 'type': 'struct'},
    {'center': (7, 2), 'radius': 1.2, 'color': '#DEB887', 'type': 'struct'},
    {'center': (3.5, 4.5), 'radius': 0.8, 'color': '#66CDAA', 'type': 'bnd'},
    {'center': (3, 3), 'radius': 0.8, 'color': '#66CDAA', 'type': 'bnd'},
]

for i, ax in enumerate(axes):
    draw_motorbike(ax, alpha=0.3 if i < 5 else 0.8)
    ax.set_title(titles[i], fontsize=11, pad=10)
    
    # Draw previous visits faintly
    if i < 5:
        for j in range(i):
            hl = highlights[j]
            ax.add_patch(patches.Circle(hl['center'], hl['radius'], fill=True, color=hl['color'], alpha=0.15))
            ax.add_patch(patches.Circle(hl['center'], hl['radius'], fill=False, edgecolor=hl['color'], lw=1, ls=':'))
            
        # Draw current visit strongly
        if i < len(highlights):
            hl = highlights[i]
            ax.add_patch(patches.Circle(hl['center'], hl['radius'], fill=True, color=hl['color'], alpha=0.6))
            ax.add_patch(patches.Circle(hl['center'], hl['radius'], fill=False, edgecolor='black', lw=1.5))
            ax.plot([hl['center'][0]], [hl['center'][1]], 'ko', markersize=4)

    # Output coloring
    if i == 5:
        # Color wheels blue
        ax.add_patch(patches.Circle((2, 2), 1, fill=True, color='#A9CCEA', alpha=0.8))
        ax.add_patch(patches.Circle((7, 2), 1, fill=True, color='#A9CCEA', alpha=0.8))
        # Color tank green
        ax.add_patch(patches.Rectangle((3.5, 4.5), 2, 1, fill=True, color='#C5E0B4', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "active_perception_loop.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "active_perception_loop.png"), bbox_inches='tight', dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 2: Hardware Efficiency (Pareto Curve)
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.grid(True, linestyle='--', alpha=0.6)

# Dummy data based on typical SNN vs ANN papers (MACs vs SOPs)
energy_ops = [15.2, 28.5, 42.1, 85.0] # Giga Ops
miou = [47.5, 52.3, 55.1, 58.0]
labels = ["ASP-SNN (T=2)", "ASP-SNN (T=4)", "ASP-SNN (T=6)", "PointNet (ANN)"]
colors = ['#FFA07A', '#FF6347', '#DC143C', '#4682B4']

for e, m, l, c in zip(energy_ops, miou, labels, colors):
    plt.scatter(e, m, color=c, s=150, label=l, edgecolors='black', zorder=5)

plt.plot(energy_ops[:3], miou[:3], color='gray', linestyle='--', zorder=4)

plt.title("Efficiency vs. Accuracy (S3DIS Area 5)", fontsize=14, pad=15)
plt.xlabel("Theoretical Energy Cost (Giga-Operations)", fontsize=12)
plt.ylabel("Mean Intersection over Union (mIoU %)", fontsize=12)
plt.legend(loc='lower right', fontsize=11)

# Annotate Pareto Frontier
plt.annotate('Pareto Frontier', xy=(28.5, 52.3), xytext=(20, 56),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
            fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "efficiency_pareto.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "efficiency_pareto.png"), bbox_inches='tight', dpi=300)
plt.close()

print(f"Successfully generated high-quality figures in {os.path.abspath(out_dir)}")
