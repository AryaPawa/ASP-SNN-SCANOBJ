import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = "aaai_figures"
os.makedirs(out_dir, exist_ok=True)

# ==============================================================================
# UNIQUE FIGURE 3: Gumbel-Softmax Temperature Stabilization
# Explains the Epoch 44 crash and justifies the tau=0.5 freeze strategy
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(9, 5))

epochs = np.arange(0, 250)

# Original Decay (tau_start=1.0, tau_decay=0.95, tau_end=0.1)
tau_original = np.maximum(0.1, 1.0 * (0.95 ** epochs))

# Stabilized Strategy (tau_start=0.5, tau_decay=1.0)
tau_stabilized = np.full_like(epochs, 0.5, dtype=float)

# Simulated Learning Rate (Cosine Annealing with Warmup)
lr = np.zeros_like(epochs, dtype=float)
warmup = 5
for e in epochs:
    if e < warmup:
        lr[e] = 0.1 + 0.9 * (e / warmup)
    else:
        progress = (e - warmup) / (250 - warmup)
        lr[e] = 0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
# Normalize LR for plotting
lr_norm = lr * 0.8 

# Plot Tau
line1 = ax1.plot(epochs, tau_original, color='#e74c3c', lw=3, label=r'Original $\tau$ Decay (Crash at Epoch ~45)')
line2 = ax1.plot(epochs, tau_stabilized, color='#2ecc71', lw=3, linestyle='--', label=r'Proposed Stabilized $\tau$ Freeze')

ax1.set_xlabel('Training Epoch', fontsize=12)
ax1.set_ylabel(r'Gumbel Temperature ($\tau$)', fontsize=12)
ax1.set_ylim(0, 1.1)

# Highlight the danger zone
ax1.axvspan(45, 250, color='#e74c3c', alpha=0.1)
ax1.text(120, 0.25, "Danger Zone:\n$\\tau \\to 0.1$ forces hard slice selection\nwhile LR is still high, causing gradient collapse", 
         fontsize=10, color='#c0392b', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e74c3c', boxstyle='round,pad=0.5'))

# Secondary axis for Learning Rate
ax2 = ax1.twinx()
line3 = ax2.plot(epochs, lr_norm, color='#34495e', lw=2, linestyle=':', label='Learning Rate Schedule')
ax2.set_ylabel('Normalized Learning Rate', fontsize=12)
ax2.set_ylim(0, 1.1)
ax2.set_yticks([])

# Combine legends
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=11)

plt.title("Stabilizing Active Slice Selection on S3DIS", fontsize=15, pad=15)
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "gumbel_stabilization_curve.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "gumbel_stabilization_curve.png"), bbox_inches='tight', dpi=300)
plt.close()

print("Generated Gumbel Stabilization Figure.")
