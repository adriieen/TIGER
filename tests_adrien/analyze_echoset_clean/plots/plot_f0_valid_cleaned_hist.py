import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

BASE_DIR = Path("/idiap/home/adubois/project_ma1/TIGER")
CLEANED_DIR = Path("/idiap/temp/adubois/EchoSet_cleaned")

# 1. Charger le parquet train pour récupérer les seuils de référence
train_parquet = CLEANED_DIR / "egemaps_summary_train.parquet"
if not train_parquet.exists():
    train_parquet = BASE_DIR / "tests_adrien/analyze_echoset_clean/egemaps_summary_train.parquet"

def semitone_to_hz(st):
    return 27.5 * (2.0 ** (st / 12.0))

print(f"--> Reading Train Parquet: {train_parquet}")
df_train = pd.read_parquet(train_parquet)

col_s1_tr = [c for c in df_train.columns if "f0" in c.lower() and "amean" in c.lower() and ("s1" in c.lower() or "_1" in c)]
col_s2_tr = [c for c in df_train.columns if "f0" in c.lower() and "amean" in c.lower() and ("s2" in c.lower() or "_2" in c)]

if col_s1_tr and col_s2_tr:
    f0_tr_s1 = semitone_to_hz(df_train[col_s1_tr[0]])
    f0_tr_s2 = semitone_to_hz(df_train[col_s2_tr[0]])
    deltas_train_hz = (f0_tr_s1 - f0_tr_s2).abs().dropna().values
else:
    f0_diff_tr = [c for c in df_train.columns if "f0" in c.lower() and "amean" in c.lower()][0]
    deltas_train_hz = 150.0 * (2.0 ** (df_train[f0_diff_tr].dropna().values / 12.0) - 1.0)

# Seuils des déciles d'entraînement
deciles_train_hz = np.percentile(deltas_train_hz, np.arange(10, 100, 10))

# 2. Chercher le parquet de validation
val_parquet_candidates = [
    CLEANED_DIR / "egemaps_summary_val.parquet",
]
val_parquet = None
for p in val_parquet_candidates:
    if p.exists():
        val_parquet = p
        break

if val_parquet is not None:
    print(f"--> Reading Validation Parquet: {val_parquet}")
    df_val = pd.read_parquet(val_parquet)
    col_s1_v = [c for c in df_val.columns if "f0" in c.lower() and "amean" in c.lower() and ("s1" in c.lower() or "_1" in c)]
    col_s2_v = [c for c in df_val.columns if "f0" in c.lower() and "amean" in c.lower() and ("s2" in c.lower() or "_2" in c)]
    if col_s1_v and col_s2_v:
        deltas_val_hz = (semitone_to_hz(df_val[col_s1_v[0]]) - semitone_to_hz(df_val[col_s2_v[0]])).abs().dropna().values
    else:
        f0_diff_v = [c for c in df_val.columns if "f0" in c.lower() and "amean" in c.lower()][0]
        deltas_val_hz = 150.0 * (2.0 ** (df_val[f0_diff_v].dropna().values / 12.0) - 1.0)

# 3. Figure comparative
OUTPUT_DIR = BASE_DIR / "tests_adrien/analyze_echoset_clean/plots_f0"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 6))

# Histogramme du jeu de validation (ou global)
sns.histplot(
    deltas_val_hz,
    bins=60,
    kde=True,
    color="#2a9d8f",
    alpha=0.55,
    stat="density",
    edgecolor="black",
    label=f"Validation Set Distribution (n={len(deltas_val_hz):,})",
    ax=ax
)

# Lignes des déciles de train
palette = sns.color_palette("rocket", 9)
for idx, (threshold, color) in enumerate(zip(deciles_train_hz, palette), 1):
    pct_val_below = (deltas_val_hz <= threshold).mean() * 100.0
    ax.axvline(
        threshold,
        color=color,
        linestyle="--",
        linewidth=1.8,
        label=f"Train D{idx} ({threshold:.1f} Hz) -> covers {pct_val_below:.1f}% of Val"
    )

ax.set_title(
    r"Validation Set $\Delta F_0$ Distribution Superimposed with Training Decile Thresholds",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel(r"Mean Pairwise $\Delta F_0$ (Hz)", fontsize=12, fontweight="bold")
ax.set_ylabel("Density", fontsize=12, fontweight="bold")
ax.set_xlim(0, np.percentile(deltas_val_hz, 99))

ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9.5)

out_path = OUTPUT_DIR / "f0_mean_val_vs_train_deciles.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Plot saved to: {out_path}")
