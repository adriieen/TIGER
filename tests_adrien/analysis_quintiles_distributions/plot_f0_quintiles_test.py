#!/usr/bin/env python3
"""
plot_f0_quintiles_test.py

Plots the Delta F0 distribution on the test set (EchoSet_cleaned_small)
superimposed with the training quintile boundaries to show coverage.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.autolayout": True,
})

DATA_ROOTS = [
    Path("/Data/adrien.dubois/EchoSet_cleaned_small"),
    Path("/users/eleves-a/2023/adrien.dubois/4A/project_ma1/TIGER/DataPreProcess/EchoSet_cleaned_small"),
    Path("/users/eleves-a/2023/adrien.dubois/4A/project_ma1/X_training/DataPreProcess/EchoSet_cleaned_small"),
]

root = next((r for r in DATA_ROOTS if (r / "egemaps_summary_train.parquet").exists()), None)
if root is None:
    raise FileNotFoundError("Could not find dataset directory containing parquet summaries.")

df_train = pd.read_parquet(root / "egemaps_summary_train.parquet")
df_test = pd.read_parquet(root / "egemaps_summary_test.parquet")

f0_col = "dist_F0semitoneFrom27.5Hz_sma3nz_amean"
deltas_train_st = df_train[f0_col].dropna().values
deltas_test_st = df_test[f0_col].dropna().values

# Training quintiles
quintiles_st = np.percentile(deltas_train_st, [20, 40, 60, 80])
q_edges = [0.0] + list(quintiles_st) + [np.inf]

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Compute test coverage per training quintile
test_coverages = []
for i in range(5):
    mask = (deltas_test_st >= q_edges[i]) & (deltas_test_st < q_edges[i+1])
    pct = mask.mean() * 100.0
    test_coverages.append(pct)

print("=" * 65)
print("COUVERTURE DU TEST SET SELON LES QUINTILES DE TRAIN :")
for i, pct in enumerate(test_coverages, 1):
    print(f"  Quintile {i} [{q_edges[i-1]:.2f} - {q_edges[i]:.2f} st] : {pct:.1f}% des paires de test")
print("=" * 65)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

sns.histplot(
    deltas_test_st,
    bins=60,
    kde=True,
    color="#e76f51",
    alpha=0.55,
    stat="density",
    edgecolor="black",
    ax=ax,
    label=f"Test Set Distribution (n={len(deltas_test_st):,})"
)

palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261"]
for idx, (th_st, color) in enumerate(zip(quintiles_st, palette), 1):
    pct_test_below = (deltas_test_st <= th_st).mean() * 100.0
    ax.axvline(
        th_st,
        color=color,
        linestyle="--",
        linewidth=2.0,
        label=f"Train Q{idx} Cut ({th_st:.2f} st) -> {pct_test_below:.1f}% of Test <= Cut"
    )

y_max = ax.get_ylim()[1]
for i in range(5):
    high_edge = min(q_edges[i+1], 18.0)
    mid_x = (q_edges[i] + high_edge) / 2.0
    ax.text(
        mid_x, y_max * 0.85, f"Q{i+1}\n({test_coverages[i]:.1f}%)",
        ha="center", va="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray")
    )

ax.set_title(r"Test Set $\Delta F_0$ Distribution Superimposed with Training Quintile Thresholds", fontsize=14, fontweight="bold")
ax.set_xlabel(r"Pairwise $\Delta F_0$ (semitones)", fontsize=12, fontweight="bold")
ax.set_ylabel("Density", fontsize=12, fontweight="bold")
ax.set_xlim(0, np.percentile(deltas_test_st, 99.5))
ax.legend(loc="upper right", frameon=True, fontsize=9.5)

out_test = OUTPUT_DIR / "distribution_f0_test_quintiles.png"
plt.savefig(out_test, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out_test}")

