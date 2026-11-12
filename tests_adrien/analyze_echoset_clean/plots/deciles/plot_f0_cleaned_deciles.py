import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration des styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

BASE_DIR = Path("/idiap/home/adubois/project_ma1/TIGER")
CLEANED_DIR = Path("/idiap/temp/adubois/EchoSet_cleaned")

# 1. Localiser le fichier parquet de résumé eGeMAPS
parquet_candidates = [
    CLEANED_DIR / "egemaps_summary_train.parquet",
    BASE_DIR / "tests_adrien/analyze_echoset_clean/egemaps_summary_train.parquet",
    BASE_DIR / "egemaps_summary_train.parquet",
]
parquet_path = None
for p in parquet_candidates:
    if p.exists():
        parquet_path = p
        break

if parquet_path is None:
    found = list(BASE_DIR.glob("**/egemaps_summary_train.parquet"))
    parquet_path = found[0] if found else None

print(f"--> Lecture du parquet : {parquet_path}")
df = pd.read_parquet(parquet_path)

# Formule standard eGeMAPS : F0(Hz) = 27.5 * (2 ** (semitone / 12))
def semitone_to_hz(st):
    return 27.5 * (2.0 ** (st / 12.0))

# 2. Calcul des deltas en Hz
# On cherche les colonnes s1 et s2 pour convertir chaque voix en Hz avant de calculer le delta |f0_1 - f0_2|
col_s1 = [c for c in df.columns if "f0" in c.lower() and "amean" in c.lower() and ("s1" in c.lower() or "_1" in c)]
col_s2 = [c for c in df.columns if "f0" in c.lower() and "amean" in c.lower() and ("s2" in c.lower() or "_2" in c)]

if col_s1 and col_s2:
    f0_hz_s1 = semitone_to_hz(df[col_s1[0]])
    f0_hz_s2 = semitone_to_hz(df[col_s2[0]])
    deltas_hz = (f0_hz_s1 - f0_hz_s2).abs().dropna().values
    deltas_st = (df[col_s1[0]] - df[col_s2[0]]).abs().dropna().values
else:
    # Si seules les distances en demi-tons sont présentes, approximation autour de la voix moyenne (~150 Hz)
    f0_col_diff = [c for c in df.columns if "f0" in c.lower() and "amean" in c.lower()][0]
    deltas_st = df[f0_col_diff].dropna().values
    # Ratio : f2/f1 = 2^(st/12), delta_hz ~ 150 * (2^(st/12) - 1)
    deltas_hz = 150.0 * (2.0 ** (deltas_st / 12.0) - 1.0)

# 3. Calcul des déciles sur Delta F0 (Hz)
deciles_hz = np.percentile(deltas_hz, np.arange(10, 100, 10))
deciles_st = np.percentile(deltas_st, np.arange(10, 100, 10))

print("\n" + "=" * 65)
print("SEUILS DES DÉCILES SUR DELTA F0 (Hz et demi-tons) :")
for i, (dhz, dst) in enumerate(zip(deciles_hz, deciles_st), 1):
    print(f"  Décile {i:2d} (top {i*10:2d}%) : <= {dhz:6.2f} Hz  ({dst:.2f} demi-tons)")
print("=" * 65 + "\n")

# 4. Création de la figure
OUTPUT_DIR = BASE_DIR / "tests_adrien/analyze_echoset_clean/plots_f0"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 6))

# Histogramme global avec KDE en Hz
sns.histplot(
    deltas_hz,
    bins=50,
    kde=True,
    color="#1b4965",
    alpha=0.6,
    edgecolor="black",
    stat="density",
    ax=ax,
    label=f"EchoSet Cleaned (n={len(deltas_hz):,})"
)

# Lignes verticales des déciles
palette = sns.color_palette("mako", 9)
for idx, (th_hz, th_st, color) in enumerate(zip(deciles_hz, deciles_st, palette), 1):
    ax.axvline(
        th_hz,
        color=color,
        linestyle="--",
        linewidth=1.8,
        label=f"Decile {idx} (top {idx*10:2d}%): {th_hz:.1f} Hz ({th_st:.1f} st)"
    )

ax.set_title(
    r"$\Delta F_0$ Distribution with Decile Thresholds (EchoSet Cleaned)",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel(r"Mean Pairwise $\Delta F_0$ (Hz)", fontsize=12, fontweight="bold")
ax.set_ylabel("Density", fontsize=12, fontweight="bold")
ax.set_xlim(0, np.percentile(deltas_hz, 99))  # Tronque le 1% d'outliers extrêmes pour une belle visualisation

# Légende compacte à droite
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=10)

plot_file = OUTPUT_DIR / "f0_mean_cleaned_deciles_hz_distribution.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Plot en Hz sauvegardé avec succès dans : {plot_file}")
