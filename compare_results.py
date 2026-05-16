import pickle
import matplotlib.pyplot as plt
import numpy as np

# ─── METRICS LOAD KARO ──────────────────────────────────
print("Metrics load ho rahi hain...")

with open("models/cnn_metrics.pkl", "rb") as f:
    cnn = pickle.load(f)

with open("models/resnet_metrics.pkl", "rb") as f:
    resnet = pickle.load(f)

# ─── RESULTS PRINT KARO ─────────────────────────────────
print("\n========================================")
print("        DONO MODELS KA COMPARISON        ")
print("========================================")
print(f"{'Metric':<15} {'CNN':>10} {'ResNet50':>10} {'Winner':>10}")
print("-" * 50)

metrics = ['accuracy', 'precision', 'recall', 'f1_score']
labels  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for m, l in zip(metrics, labels):
    cnn_val    = cnn[m] * 100
    resnet_val = resnet[m] * 100
    winner     = "ResNet50" if resnet_val > cnn_val else "CNN"
    print(f"{l:<15} {cnn_val:>9.2f}% {resnet_val:>9.2f}% {winner:>10}")

print("========================================\n")

# ─── COMPARISON BAR CHART ───────────────────────────────
cnn_values    = [cnn[m] * 100    for m in metrics]
resnet_values = [resnet[m] * 100 for m in metrics]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, cnn_values,    width, label='CNN',     color='steelblue',  alpha=0.85)
bars2 = ax.bar(x + width/2, resnet_values, width, label='ResNet50', color='darkorange', alpha=0.85)

# Values upar likho
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('CNN vs ResNet50 — Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 110)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("results/comparison_chart.png")
plt.show()
print("Comparison chart save ho gaya: results/comparison_chart.png")

# ─── RADAR CHART ────────────────────────────────────────
categories = labels + [labels[0]]
cnn_vals    = cnn_values    + [cnn_values[0]]
resnet_vals = resnet_values + [resnet_values[0]]

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

ax.plot(angles, cnn_vals,    'o-', linewidth=2, label='CNN',      color='steelblue')
ax.fill(angles, cnn_vals,    alpha=0.2,                            color='steelblue')
ax.plot(angles, resnet_vals, 'o-', linewidth=2, label='ResNet50', color='darkorange')
ax.fill(angles, resnet_vals, alpha=0.2,                            color='darkorange')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0, 100)
ax.set_title('CNN vs ResNet50 — Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

plt.tight_layout()
plt.savefig("results/comparison_radar.png")
plt.show()
print("Radar chart save ho gaya: results/comparison_radar.png")

print("\n✅ Step 5 mukammal! Dono models compare ho gaye!")