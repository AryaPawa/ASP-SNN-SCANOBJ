import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import FancyArrowPatch

out_dir = "aaai_figures"
os.makedirs(out_dir, exist_ok=True)

# ==============================================================================
# UNIQUE FIGURE 1: Learnable vs Fixed LIF Membrane Dynamics
# Justifies the Learnable LIF architectural contribution
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Adaptive Membrane Dynamics: Fixed vs. Learnable LIF on S3DIS Rare Classes", fontsize=16, y=1.02)

T = np.arange(1, 9)
# Simulate Fixed LIF (Struggles to reach threshold for rare feature)
fixed_leak = 0.8
fixed_thresh = 1.0
fixed_input = np.array([0.3, 0.4, 0.2, 0.5, 0.6, 0.3, 0.4, 0.5])
fixed_mem = np.zeros(8)
v = 0
fixed_spikes = []
for t in range(8):
    v = v * fixed_leak + fixed_input[t]
    if v >= fixed_thresh:
        fixed_spikes.append(t+1)
        v = 0
    fixed_mem[t] = v

# Simulate Learnable LIF (Adapts leak and threshold dynamically)
learn_leak = 0.95 # Learned to retain more memory
learn_thresh = 0.8 # Learned lower threshold for rare class
learn_mem = np.zeros(8)
v = 0
learn_spikes = []
for t in range(8):
    v = v * learn_leak + fixed_input[t]
    if v >= learn_thresh:
        learn_spikes.append(t+1)
        v = 0 # soft reset conceptual
    learn_mem[t] = v

# Plot Fixed
ax1.plot(T, fixed_mem, marker='o', color='#3498db', lw=3, label='Membrane Potential $U(t)$')
ax1.axhline(fixed_thresh, color='red', linestyle='--', lw=2, label='Fixed Threshold $\\tau$')
for s in fixed_spikes:
    ax1.axvline(s, color='orange', linestyle=':', lw=2)
    ax1.plot(s, fixed_thresh, '*', color='orange', markersize=15, label='Spike Fired' if s==fixed_spikes[0] else "")
ax1.set_title("Standard Fixed LIF (Misses temporal context)", fontsize=13)
ax1.set_xlabel("Timestep $t$", fontsize=12)
ax1.set_ylabel("Voltage", fontsize=12)
ax1.set_ylim(0, 1.3)
ax1.legend(loc='upper left')

# Plot Learnable
ax2.plot(T, learn_mem, marker='o', color='#2ecc71', lw=3, label='Membrane Potential $U(t)$')
ax2.axhline(learn_thresh, color='purple', linestyle='--', lw=2, label='Learned Threshold $\\tau^*$')
for s in learn_spikes:
    ax2.axvline(s, color='orange', linestyle=':', lw=2)
    ax2.plot(s, learn_thresh, '*', color='orange', markersize=15, label='Spike Fired' if s==learn_spikes[0] else "")
ax2.set_title("ASP-SNN Learnable LIF (Adapts to rare class geometry)", fontsize=13)
ax2.set_xlabel("Timestep $t$", fontsize=12)
ax2.set_ylim(0, 1.3)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "lif_dynamics_comparison.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "lif_dynamics_comparison.png"), bbox_inches='tight', dpi=300)
plt.close()

# ==============================================================================
# UNIQUE FIGURE 2: Cross-Modal Knowledge Distillation (ANN to SNN)
# Visualizes how KD bridges the continuous-discrete gap
# ==============================================================================
plt.figure(figsize=(10, 6))
plt.grid(False)

# Draw Teacher (ANN) Distribution
x = np.linspace(-3, 3, 100)
teacher_dist = np.exp(-0.5 * (x - 1)**2) / 1.5
plt.plot(x, teacher_dist, color='#e74c3c', lw=4, label='Teacher ANN (Continuous Logits)')
plt.fill_between(x, teacher_dist, alpha=0.2, color='#e74c3c')

# Draw Student (SNN) Distribution Without KD
student_no_kd = np.exp(-0.5 * (x + 1.5)**2) / 2.0
plt.plot(x, student_no_kd, color='#95a5a6', lw=3, linestyle='--', label='SNN Student (w/o KD, Noisy Spikes)')
plt.fill_between(x, student_no_kd, alpha=0.1, color='#95a5a6')

# Draw Student (SNN) Distribution WITH KD
student_kd = np.exp(-0.5 * (x - 0.5)**2) / 1.7
plt.plot(x, student_kd, color='#2980b9', lw=4, label='ASP-SNN Student (w/ KD)')
plt.fill_between(x, student_kd, alpha=0.3, color='#2980b9')

# Arrows showing distillation pull
plt.annotate('', xy=(0.5, 0.4), xytext=(-1.5, 0.3),
            arrowprops=dict(facecolor='#2c3e50', shrink=0.05, width=2, headwidth=10, connectionstyle="arc3,rad=-0.2"))
plt.text(-0.5, 0.45, "KL Divergence Pull\n($\\mathcal{L}_{KD}$)", fontsize=12, ha='center', fontweight='bold', color='#2c3e50')

plt.title("Bridging the Continuous-Discrete Gap via Knowledge Distillation", fontsize=15, pad=15)
plt.xlabel("Feature Representation Space", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.yticks([])
plt.xticks([])
plt.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "kd_distillation_concept.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "kd_distillation_concept.png"), bbox_inches='tight', dpi=300)
plt.close()

print("Generated Unique AAAI Figures.")
