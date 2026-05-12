"""
Figure 4 comparison: TCR-HE vs previously reported models (epitope hard split).
Data from published literature, TCR-HE uses our actual results.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "results/final"

# ── Literature data (from TCR-H paper Figure 4 / refs 21,23,25,36) ──
# Our actual TCR-HE
df = pd.read_csv(os.path.join(RESULTS_DIR, "results_table.csv"))
tcr_he = df[df["Model"] == "TCR-HE"].iloc[0]

comparison = {
    "AUC of ROC": {
        "ATM-TCR":     0.47,
        "ImRex":       0.55,
        "epiTCR":      0.75,
        "Pan-Peptide": 0.78,
        "TCR-HE":      tcr_he["AUC (predict)"],
    },
    "Precision": {
        "NetTCR":     0.53,
        "ERGO-LSTM":  0.52,
        "ERGO-AE":    0.57,
        "ATM-TCR":    0.51,
        "TCR-HE":     tcr_he["Precision"],
    },
    "Recall": {
        "NetTCR":     0.62,
        "ERGO-LSTM":  0.70,
        "ERGO-AE":    0.51,
        "ATM-TCR":    0.86,
        "TCR-HE":     tcr_he["Recall"],
    },
}

metrics = list(comparison.keys())
colors = plt.cm.Set2(np.linspace(0, 1, 3))
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, metric, color in zip(axes, metrics, colors):
    data = comparison[metric]
    models = list(data.keys())
    values = list(data.values())
    bars = ax.barh(range(len(models)), values, color=color, edgecolor="grey", linewidth=0.5, height=0.6)
    # Highlight TCR-HE bar
    tcr_idx = models.index("TCR-HE")
    bars[tcr_idx].set_color("#E74C3C")
    bars[tcr_idx].set_edgecolor("black")
    bars[tcr_idx].set_linewidth(1.5)
    for i, (m, v) in enumerate(zip(models, values)):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlim([0, 1.1])
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("Comparison with Previously Reported Models (Epitope Hard Split)", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "figure4_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figure 4 saved:", os.path.join(RESULTS_DIR, "figure4_comparison.png"))
