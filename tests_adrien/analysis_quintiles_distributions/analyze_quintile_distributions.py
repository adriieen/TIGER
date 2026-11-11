#!/usr/bin/env python3
"""
analyze_quintile_distributions.py

Analyze and plot distributions of delay, SNR, and Delta F0 across:
1) The 5 quintiles of EchoSet_cleaned_small (train set)
2) Comparison with Decile 1 of EchoSet_cleaned (train set)

Outputs:
- distribution_delay_per_quintile.png
- distribution_snr_per_quintile.png
- distribution_deltaf0_per_quintile.png
- comparison_decile1_vs_quintile1.png
- distributions_summary.csv
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 16,
})

OUTPUT_DIR = "/idiap/home/adubois/project_ma1/TIGER/tests_adrien/analysis_quintiles_distributions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Load EchoSet_cleaned_small training data & metadata
# -------------------------------------------------------------
print("Loading EchoSet_cleaned_small ranking & metadata...")
small_rank_file = "/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_cleaned_small/rankings/F0semitoneFrom27.5Hz_sma3nz_amean/mix.json"
with open(small_rank_file) as f:
    small_mix_entries = json.load(f)

records_small = []
for entry in small_mix_entries:
    path = entry[0]
    folder = os.path.dirname(path)
    m = re.search(r"(\d{2}/\d+_\d+)", path)
    pair_id = m.group(1) if m else None

    meta_path = os.path.join(folder, "metadata.txt")
    snr, delay = np.nan, np.nan
    if os.path.exists(meta_path):
        with open(meta_path) as mf:
            for line in mf:
                if line.startswith("snr:"):
                    snr = float(line.split(":")[1].strip())
                elif line.startswith("delay:"):
                    delay = float(line.split(":")[1].strip())

    records_small.append({
        "path": path,
        "pair_id": pair_id,
        "snr": snr,
        "delay": delay,
    })

df_small = pd.DataFrame(records_small)

# Merge delta F0 from parquet
parq_small_path = "/idiap/temp/adubois/EchoSet_cleaned_small/egemaps_summary_train.parquet"
df_parq_small = pd.read_parquet(parq_small_path)
f0_col = "dist_F0semitoneFrom27.5Hz_sma3nz_amean"

df_small = df_small.merge(df_parq_small[["pair_id", f0_col]], on="pair_id", how="left")
df_small.rename(columns={f0_col: "delta_f0_semitones"}, inplace=True)

# Assign quintiles based on ranking order
n_small = len(df_small)
q_size = n_small / 5
df_small["quintile"] = [f"Quintile {int(i // q_size) + 1}" for i in range(n_small)]

print(f"Loaded {len(df_small)} samples for EchoSet_cleaned_small.")

# -------------------------------------------------------------
# 2. Load EchoSet_cleaned Decile 1 for comparison
# -------------------------------------------------------------
print("Loading EchoSet_cleaned Decile 1...")
clean_rank_file = "/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_cleaned/rankings/F0semitoneFrom27.5Hz_sma3nz_amean/mix.json"
with open(clean_rank_file) as f:
    clean_mix_entries = json.load(f)

parq_clean_path = "/idiap/temp/adubois/EchoSet_cleaned/egemaps_summary_train.parquet"
df_parq_clean = pd.read_parquet(parq_clean_path)

records_clean = []
for entry in clean_mix_entries:
    path = entry[0]
    m = re.search(r"(\d{2}/\d+_\d+)", path)
    pair_id = m.group(1) if m else None
    records_clean.append({"path": path, "pair_id": pair_id})

df_clean = pd.DataFrame(records_clean).merge(df_parq_clean[["pair_id", f0_col]], on="pair_id", how="left")
df_clean.rename(columns={f0_col: "delta_f0_semitones"}, inplace=True)

n_clean = len(df_clean)
d_size = n_clean / 10
df_clean["decile"] = [f"Decile {int(i // d_size) + 1}" for i in range(n_clean)]

df_decile1 = df_clean[df_clean["decile"] == "Decile 1"].copy()
print(f"Loaded {len(df_decile1)} samples for EchoSet_cleaned Decile 1.")

# Save CSV summary stats
summary_stats = []
for q_name, grp in df_small.groupby("quintile"):
    summary_stats.append({
        "dataset": "EchoSet_cleaned_small",
        "subset": q_name,
        "count": len(grp),
        "delay_mean": grp["delay"].mean(),
        "delay_std": grp["delay"].std(),
        "snr_mean": grp["snr"].mean(),
        "snr_std": grp["snr"].std(),
        "delta_f0_mean_semitones": grp["delta_f0_semitones"].mean(),
        "delta_f0_median_semitones": grp["delta_f0_semitones"].median(),
        "delta_f0_min_semitones": grp["delta_f0_semitones"].min(),
        "delta_f0_max_semitones": grp["delta_f0_semitones"].max(),
    })

summary_stats.append({
    "dataset": "EchoSet_cleaned",
    "subset": "Decile 1",
    "count": len(df_decile1),
    "delay_mean": np.nan,
    "delay_std": np.nan,
    "snr_mean": np.nan,
    "snr_std": np.nan,
    "delta_f0_mean_semitones": df_decile1["delta_f0_semitones"].mean(),
    "delta_f0_median_semitones": df_decile1["delta_f0_semitones"].median(),
    "delta_f0_min_semitones": df_decile1["delta_f0_semitones"].min(),
    "delta_f0_max_semitones": df_decile1["delta_f0_semitones"].max(),
})

df_summary = pd.DataFrame(summary_stats)
summary_csv_path = os.path.join(OUTPUT_DIR, "distributions_summary.csv")
df_summary.to_csv(summary_csv_path, index=False)
print(f"Summary table saved to: {summary_csv_path}")

# -------------------------------------------------------------
# PLOT 1: Delay Distribution across Quintiles
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True, sharex=True)
colors = sns.color_palette("viridis", 5)

for i, (q_name, grp) in enumerate(df_small.groupby("quintile")):
    ax = axes[i]
    sns.histplot(grp["delay"], bins=20, kde=True, ax=ax, color=colors[i], stat="density")
    ax.set_title(f"{q_name}\n(mean={grp['delay'].mean():.2f}s, std={grp['delay'].std():.2f})")
    ax.set_xlabel("Delay $\\Delta$ (seconds)")
    if i == 0:
        ax.set_ylabel("Density")
    ax.set_xlim(-0.1, 3.1)
    ax.axvline(1.5, color="red", linestyle="--", alpha=0.7, label="Expected U[0, 3] mean (1.5s)")
    if i == 4:
        ax.legend(loc="upper right", fontsize=10)

plt.suptitle("EchoSet Cleaned Small: Speaker Pair Delay Distribution per Quintile", y=1.03)
plt.tight_layout()
fig_delay_path = os.path.join(OUTPUT_DIR, "distribution_delay_per_quintile.png")
plt.savefig(fig_delay_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_delay_path}")

# -------------------------------------------------------------
# PLOT 2: SNR Distribution across Quintiles
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True, sharex=True)
colors_snr = sns.color_palette("magma", 5)

for i, (q_name, grp) in enumerate(df_small.groupby("quintile")):
    ax = axes[i]
    sns.histplot(grp["snr"], bins=20, kde=True, ax=ax, color=colors_snr[i], stat="density")
    ax.set_title(f"{q_name}\n(mean={grp['snr'].mean():.2f} dB, std={grp['snr'].std():.2f})")
    ax.set_xlabel("SNR (dB)")
    if i == 0:
        ax.set_ylabel("Density")
    ax.set_xlim(-5.5, 5.5)
    ax.axvline(0.0, color="blue", linestyle="--", alpha=0.7, label="Expected U[-5, 5] mean (0 dB)")
    if i == 4:
        ax.legend(loc="upper right", fontsize=10)

plt.suptitle("EchoSet Cleaned Small: SDR / SNR Distribution per Quintile", y=1.03)
plt.tight_layout()
fig_snr_path = os.path.join(OUTPUT_DIR, "distribution_snr_per_quintile.png")
plt.savefig(fig_snr_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_snr_path}")

# -------------------------------------------------------------
# PLOT 3: Delta F0 Distribution per Quintile (Semitones)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
colors_f0 = sns.color_palette("crest", 5)

for i, (q_name, grp) in enumerate(df_small.groupby("quintile")):
    ax = axes[i]
    sns.histplot(grp["delta_f0_semitones"], bins=25, kde=True, ax=ax, color=colors_f0[i], stat="density")
    ax.set_title(f"{q_name}\n[min={grp['delta_f0_semitones'].min():.2f}, max={grp['delta_f0_semitones'].max():.2f}]\n(med={grp['delta_f0_semitones'].median():.2f} st)")
    ax.set_xlabel("$\\Delta F_0$ (semitones)")
    if i == 0:
        ax.set_ylabel("Density")
    ax.axvline(grp["delta_f0_semitones"].median(), color="black", linestyle=":", label=f"Median ({grp['delta_f0_semitones'].median():.2f})")
    ax.legend(loc="upper right", fontsize=10)

plt.suptitle("EchoSet Cleaned Small: $\\Delta F_0$ Distribution per Quintile", y=1.05)
plt.tight_layout()
fig_f0_path = os.path.join(OUTPUT_DIR, "distribution_deltaf0_per_quintile.png")
plt.savefig(fig_f0_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_f0_path}")

# -------------------------------------------------------------
# PLOT 4: Comparison Decile 1 (EchoSet_cleaned) vs Quintile 1 (EchoSet_cleaned_small)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

q1_data = df_small[df_small["quintile"] == "Quintile 1"]["delta_f0_semitones"]
d1_data = df_decile1["delta_f0_semitones"]

sns.histplot(d1_data, bins=30, kde=True, stat="density", color="royalblue", alpha=0.5,
             label=f"EchoSet_cleaned Decile 1 (N={len(d1_data)}, max={d1_data.max():.2f} st, med={d1_data.median():.2f} st)", ax=ax)
sns.histplot(q1_data, bins=30, kde=True, stat="density", color="crimson", alpha=0.5,
             label=f"EchoSet_cleaned_small Quintile 1 (N={len(q1_data)}, max={q1_data.max():.2f} st, med={q1_data.median():.2f} st)", ax=ax)

ax.set_xlabel("$\\Delta F_0$ (semitones)")
ax.set_ylabel("Density")
ax.set_title("$\\Delta F_0$ Comparison: Cleaned Decile 1 (0-10%) vs Small Quintile 1 (0-20%)", fontsize=15)
ax.axvline(d1_data.median(), color="blue", linestyle="--", label=f"Decile 1 Median ({d1_data.median():.2f} st)")
ax.axvline(q1_data.median(), color="red", linestyle="--", label=f"Quintile 1 Median ({q1_data.median():.2f} st)")
ax.legend(fontsize=11)

plt.tight_layout()
fig_comp_path = os.path.join(OUTPUT_DIR, "comparison_decile1_vs_quintile1.png")
plt.savefig(fig_comp_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_comp_path}")

print("\nAll plots generated successfully!")

