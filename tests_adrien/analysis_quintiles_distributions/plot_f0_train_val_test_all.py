#!/usr/bin/env python3
"""
plot_f0_train_val_test_all.py

Comprehensive Delta F0 distribution comparison across Train, Valid, and Test sets
of EchoSet_cleaned_small with vertical dashed lines representing the 5 training quintiles.

Outputs:
- distribution_deltaf0_train_val_test_superimposed.png (Overlaid KDE + Histograms)
- distribution_deltaf0_train_val_test_panels.png (3-panel Train | Val | Test comparison)
- distribution_deltaf0_train_val_test_hz_panels.png (Hz equivalent 3-panel comparison)
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

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_train = pd.read_parquet(root / "egemaps_summary_train.parquet")
df_val = pd.read_parquet(root / "egemaps_summary_val.parquet")
df_test = pd.read_parquet(root / "egemaps_summary_test.parquet")

f0_col = "dist_F0semitoneFrom27.5Hz_sma3nz_amean"
tr_st = df_train[f0_col].dropna().values
val_st = df_val[f0_col].dropna().values
te_st = df_test[f0_col].dropna().values

def semitone_to_approx_hz(st, ref_f0=150.0):
    return ref_f0 * (2.0 ** (st / 12.0) - 1.0)

tr_hz = semitone_to_approx_hz(tr_st)
val_hz = semitone_to_approx_hz(val_st)
te_hz = semitone_to_approx_hz(te_st)

# Training quintiles
quintiles_st = np.percentile(tr_st, [20, 40, 60, 80])
quintiles_hz = semitone_to_approx_hz(quintiles_st)
q_edges_st = [0.0] + list(quintiles_st) + [np.inf]

palette_lines = ["#d90429", "#f77f00", "#fcbf49", "#0077b6"]

# Compute per-quintile coverage
def compute_coverage(data):
    covs = []
    for i in range(5):
        m = (data >= q_edges_st[i]) & (data < q_edges_st[i+1])
        covs.append(m.mean() * 100.0)
    return covs

tr_cov = compute_coverage(tr_st)
val_cov = compute_coverage(val_st)
te_cov = compute_coverage(te_st)

print("=" * 70)
print(f"SEUILS DES QUINTILES RETENUS POUR L'ENTRAÎNEMENT (Train Set, n={len(tr_st):,}) :")
for i, (qst, qhz) in enumerate(zip(quintiles_st, quintiles_hz), 1):
    print(f"  Quintile {i} Cut ({i*20}%): {qst:5.2f} st (~{qhz:5.1f} Hz) | Train: {tr_cov[i-1]:.1f}% | Val: {val_cov[i-1]:.1f}% | Test: {te_cov[i-1]:.1f}%")
print(f"  Quintile 5: > {quintiles_st[-1]:5.2f} st (~{quintiles_hz[-1]:5.1f} Hz) | Train: {tr_cov[4]:.1f}% | Val: {val_cov[4]:.1f}% | Test: {te_cov[4]:.1f}%")
print("=" * 70)

# =========================================================================
# FIGURE 1 : Superposition Train / Validation / Test (Histogramme + KDE)
# =========================================================================
fig, ax = plt.subplots(figsize=(14, 7))

sns.kdeplot(tr_st, color="#1d3557", linewidth=2.5, label=f"Train Set (n={len(tr_st):,}, mean={tr_st.mean():.2f} st)", ax=ax)
sns.kdeplot(val_st, color="#2a9d8f", linewidth=2.5, linestyle="--", label=f"Validation Set (n={len(val_st):,}, mean={val_st.mean():.2f} st)", ax=ax)
sns.kdeplot(te_st, color="#e76f51", linewidth=2.5, linestyle="-.", label=f"Test Set (n={len(te_st):,}, mean={te_st.mean():.2f} st)", ax=ax)

# Histograms with slight transparency
ax.hist(tr_st, bins=60, density=True, alpha=0.20, color="#1d3557", edgecolor="none")
ax.hist(val_st, bins=60, density=True, alpha=0.15, color="#2a9d8f", edgecolor="none")
ax.hist(te_st, bins=60, density=True, alpha=0.15, color="#e76f51", edgecolor="none")

# Dashed vertical lines for training quintiles
for idx, (th_st, color) in enumerate(zip(quintiles_st, palette_lines), 1):
    ax.axvline(
        th_st,
        color=color,
        linestyle=":",
        linewidth=2.4,
        label=f"Train Quintile {idx} Cut ({idx*20}%): {th_st:.2f} st"
    )

# Region annotations
y_top = ax.get_ylim()[1]
for i in range(5):
    high_bound = min(q_edges_st[i+1], 19.0)
    mid_x = (q_edges_st[i] + high_bound) / 2.0
    ax.text(
        mid_x, y_top * 0.88,
        f"Quintile {i+1}\nTr: {tr_cov[i]:.1f}%\nVal: {val_cov[i]:.1f}%\nTe: {te_cov[i]:.1f}%",
        ha="center", va="center", fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", alpha=0.85, edgecolor="#ced4da")
    )

ax.set_title(
    r"Complete $\Delta F_0$ Distribution: Train vs Valid vs Test Sets with Training Quintile Thresholds",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel(r"Mean Pairwise $\Delta F_0$ (semitones)", fontsize=13, fontweight="bold")
ax.set_ylabel("Density", fontsize=13, fontweight="bold")
ax.set_xlim(0, 20.0)
ax.legend(loc="upper right", frameon=True, fontsize=10)

out_superimposed = OUTPUT_DIR / "distribution_deltaf0_train_val_test_superimposed.png"
plt.savefig(out_superimposed, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out_superimposed}")


# =========================================================================
# FIGURE 2 : 3 Panneaux côte-à-côte (Train | Validation | Test) en demi-tons
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True, sharex=True)

splits_info = [
    ("Train Set", tr_st, "#1d3557", tr_cov, len(tr_st)),
    ("Validation Set", val_st, "#2a9d8f", val_cov, len(val_st)),
    ("Test Set", te_st, "#e76f51", te_cov, len(te_st)),
]

for i, (name, data, col_hist, cov, n_count) in enumerate(splits_info):
    ax = axes[i]
    sns.histplot(data, bins=50, kde=True, color=col_hist, alpha=0.6, stat="density", edgecolor="black", ax=ax)
    
    # Draw vertical dashed lines
    for idx, (th_st, color) in enumerate(zip(quintiles_st, palette_lines), 1):
        ax.axvline(th_st, color=color, linestyle="--", linewidth=1.8, label=f"Q{idx} Cut: {th_st:.2f} st" if i == 0 else None)
    
    # Annotate quintile percentages
    y_max = ax.get_ylim()[1]
    for q_idx in range(5):
        high_bound = min(q_edges_st[q_idx+1], 18.0)
        mid_x = (q_edges_st[q_idx] + high_bound) / 2.0
        ax.text(
            mid_x, y_max * 0.82, f"Q{q_idx+1}\n({cov[q_idx]:.1f}%)",
            ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="gray")
        )
        
    ax.set_title(f"{name} (n={n_count:,})\nMean: {data.mean():.2f} st | Med: {np.median(data):.2f} st", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Pairwise $\Delta F_0$ (semitones)", fontsize=11, fontweight="bold")
    if i == 0:
        ax.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 19.0)

plt.suptitle(r"$\Delta F_0$ Split Distributions Superimposed with Training Quintile Boundaries", y=1.03, fontsize=15, fontweight="bold")
plt.tight_layout()

out_panels = OUTPUT_DIR / "distribution_deltaf0_train_val_test_panels.png"
plt.savefig(out_panels, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out_panels}")


# =========================================================================
# FIGURE 3 : 3 Panneaux en Hz (Équivalent fréquentiel)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True, sharex=True)

splits_info_hz = [
    ("Train Set", tr_hz, "#1d3557", tr_cov, len(tr_hz)),
    ("Validation Set", val_hz, "#2a9d8f", val_cov, len(val_hz)),
    ("Test Set", te_hz, "#e76f51", te_cov, len(te_hz)),
]

for i, (name, data, col_hist, cov, n_count) in enumerate(splits_info_hz):
    ax = axes[i]
    sns.histplot(data, bins=50, kde=True, color=col_hist, alpha=0.6, stat="density", edgecolor="black", ax=ax)
    
    for idx, (th_hz, color) in enumerate(zip(quintiles_hz, palette_lines), 1):
        ax.axvline(th_hz, color=color, linestyle="--", linewidth=1.8, label=f"Q{idx} Cut: {th_hz:.1f} Hz" if i == 0 else None)
    
    ax.set_title(f"{name} (n={n_count:,})\nMean: {data.mean():.1f} Hz | Med: {np.median(data):.1f} Hz", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Pairwise $\Delta F_0$ (Hz approx)", fontsize=11, fontweight="bold")
    if i == 0:
        ax.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 350.0)

plt.suptitle(r"$\Delta F_0$ Split Distributions (Hz) Superimposed with Training Quintile Boundaries", y=1.03, fontsize=15, fontweight="bold")
plt.tight_layout()

out_panels_hz = OUTPUT_DIR / "distribution_deltaf0_train_val_test_hz_panels.png"
plt.savefig(out_panels_hz, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out_panels_hz}")

