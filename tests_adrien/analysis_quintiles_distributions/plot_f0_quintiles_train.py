#!/usr/bin/env python3
"""
plot_f0_quintiles_train.py

Plots the Delta F0 distribution on the training set (EchoSet_cleaned_small)
with vertical dashed lines indicating the 5 training quintile boundaries.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.autolayout": True,
})

# Locate parquet files
DATA_CANDIDATES = [
    Path("/Data/adrien.dubois/EchoSet_cleaned_small/egemaps_summary_train.parquet"),
    Path("/users/eleves-a/2023/adrien.dubois/4A/project_ma1/TIGER/DataPreProcess/EchoSet_cleaned_small/egemaps_summary_train.parquet"),
    Path("/users/eleves-a/2023/adrien.dubois/4A/project_ma1/X_training/DataPreProcess/EchoSet_cleaned_small/egemaps_summary_train.parquet"),
]

parquet_train = next((p for p in DATA_CANDIDATES if p.exists()), None)
if parquet_train is None:
    raise FileNotFoundError("Could not find egemaps_summary_train.parquet in candidate paths.")

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"--> Reading Training Parquet: {parquet_train}")
df_train = pd.read_parquet(parquet_train)

# Formula to convert semitones to approx Hz relative to a reference 150 Hz voice
def semitone_to_approx_hz(st, ref_f0=150.0):
    return ref_f0 * (2.0 ** (st / 12.0) - 1.0)

f0_col = "dist_F0semitoneFrom27.5Hz_sma3nz_amean"
deltas_st = df_train[f0_col].dropna().values
deltas_hz = semitone_to_approx_hz(deltas_st)

# Calculate quintiles on training set (20%, 40%, 60%, 80%)
quintiles_st = np.percentile(deltas_st, [20, 40, 60, 80])
quintiles_hz = semitone_to_approx_hz(quintiles_st)

print("=" * 65)
print("SEUILS DES QUINTILES SUR LE TRAIN SET :")
for i, (qst, qhz) in enumerate(zip(quintiles_st, quintiles_hz), 1):
    print(f"  Quintile {i} boundary (top {i*20}%): {qst:.2f} semitones ({qhz:.1f} Hz approx)")
print(f"  Max Delta F0: {deltas_st.max():.2f} semitones")
print("=" * 65)

# -------------------------------------------------------------
# PLOT 1: En demi-tons (semitones)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

sns.histplot(
    deltas_st,
    bins=60,
    kde=True,
    color="#1b4965",
    alpha=0.6,
    edgecolor="black",
    stat="density",
    ax=ax,
    label=f"EchoSet Cleaned Small - Train Set (n={len(deltas_st):,})"
)

palette = ["#e76f51", "#f4a261", "#e9c46a", "#2a9d8f"]
q_edges = [0.0] + list(quintiles_st) + [deltas_st.max()]

for idx, (th_st, color) in enumerate(zip(quintiles_st, palette), 1):
    ax.axvline(
        th_st,
        color=color,
        linestyle="--",
        linewidth=2.2,
        label=f"Q{idx} / Q{idx+1} Cut ({idx*20}%): {th_st:.2f} st"
    )

# Label quintile regions
y_max = ax.get_ylim()[1]
for i in range(5):
    mid_x = (q_edges[i] + min(q_edges[i+1], 18.0)) / 2.0
    ax.text(
        mid_x, y_max * 0.85, f"Quintile {i+1}\n(20%)",
        ha="center", va="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray")
    )

ax.set_title(r"$\Delta F_0$ Training Set Distribution with Quintile Thresholds", fontsize=14, fontweight="bold")
ax.set_xlabel(r"Pairwise $\Delta F_0$ (semitones)", fontsize=12, fontweight="bold")
ax.set_ylabel("Density", fontsize=12, fontweight="bold")
ax.set_xlim(0, np.percentile(deltas_st, 99.5))
ax.legend(loc="upper right", frameon=True, fontsize=10)

out_st = OUTPUT_DIR / "distribution_f0_train_quintiles_semitones.png"
plt.savefig(out_st, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out_st}")

# -------------------------------------------------------------
# PLOT 2: En Hz
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

sns.histplot(
    deltas_hz,
    bins=60,
    kde=True,
    color="#2b5c8f",
    alpha=0.6,
    edgecolor="black",
    stat="density",
    ax=ax,
    label=f"EchoSet Cleaned Small - Train Set (n={len(deltas_hz):,})"
)

for idx, (th_hz, th_st, color) in enumerate(zip(quintiles_hz, quintiles_st, palette), 1):
    ax.axvline(
        th_hz,
        color=color,
        linestyle="--",
        linewidth=2.2,
        label=f"Q{idx} Cut ({idx*20}%): {th_hz:.1f} Hz ({th_st:.2f} st)"
    )

ax.set_title(r"$\Delta F_0$ Training Set Distribution with Quintile Thresholds (Hz)", fontsize=14, fontweight="bold")
ax.set_xlabel(r"Pairwise $\Delta F_0$ (Hz approx relative to 150 Hz)", fontsize=12, fontweight="bold")
ax.set_ylabel("Density", fontsize=12, fontweight="bold")
ax.set_xlim(0, np.percentile(deltas_hz, 99.5))
ax.legend(loc="upper right", frameon=True, fontsize=10)

out_hz = OUTPUT_DIR / "distribution_f0_train_quintiles_hz.png"
plt.savefig(out_hz, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out_hz}")

