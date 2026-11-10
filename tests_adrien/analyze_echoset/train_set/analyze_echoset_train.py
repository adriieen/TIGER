#!/usr/bin/env python3
"""
analyze_echoset_full.py

Multidimensional acoustic analysis on the EchoSet validation set:
- F0 (Fundamental frequency in Hz & semitones)
- Loudness (Perceived loudness in sones)
- F1, F2, F3 (Formants in Hz)
- F3 stddevNorm (Third Formant Normalized Standard Deviation / Variability)

Generates and saves all plots and summary metrics to valid_set/plots/.
"""

import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# GLOBAL & CONSISTENT COLOR PALETTE
# ==============================================================================
COLOR_MALE = "#1f77b4"       # Blue
COLOR_FEMALE = "#e377c2"     # Pink / Magenta
COLOR_MIXED = "#2ca02c"      # Green
COLOR_GLOBAL = "#7f7f7f"     # Grey
COLOR_THRESHOLD = "#d62728"  # Red
COLOR_LINE = "#222222"       # Dark Grey / Black

PAIR_COLORS = {
    "M/M": COLOR_MALE,
    "M/F": COLOR_MIXED,
    "F/F": COLOR_FEMALE,
    "All": COLOR_GLOBAL,
}

# ==============================================================================
# PATHS & CONFIGURATION
# ==============================================================================
DEV_CLEAN_CSV = Path(
    "/idiap/home/adubois/project_ma1/TIGER/tests_adrien/analyze_librispeech/train-clean-100/train-clean-100_egemaps_speakers.csv"
)
ECHOSET_VAL_JSON = Path(
    "/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet/train/s1.json"
)
INTERMEDIATE_VAL_PAIRS = Path(
    "/idiap/home/adubois/project_ma1/TIGER/echoset_pairs_train.json"
)
OUTPUT_PLOT_DIR = Path(
    "/idiap/home/adubois/project_ma1/TIGER/tests_adrien/analyze_echoset/train_set/plots"
)

PAIR_REGEX = re.compile(r"(\d+)_(\d+)")

# Configuration of features to analyze
FEATURES_CONFIG = {
    "f0": {
        "col": "F0semitoneFrom27.5Hz_sma3nz_amean",
        "label": "F0 (Hz)",
        "unit": "Hz",
        "title": "Fundamental Frequency (F0)",
        "is_f0_semitone": True,
        "threshold": 60.0,  # Critical threshold at 60 Hz
    },
    "loudness": {
        "col": "loudness_sma3_amean",
        "label": "Loudness (sone)",
        "unit": "sone",
        "title": "Native Loudness (LibriSpeech)",
        "is_f0_semitone": False,
        "threshold": None,
    },
    "f1": {
        "col": "F1frequency_sma3nz_amean",
        "label": "F1 (Hz)",
        "unit": "Hz",
        "title": "First Formant (F1)",
        "is_f0_semitone": False,
        "threshold": None,
    },
    "f1_stddevNorm": {
        "col": "F1frequency_sma3nz_stddevNorm",
        "label": "F1 StdDevNorm",
        "unit": "norm",
        "title": "First Formant Normalized StdDev (F1)",
        "is_f0_semitone": False,
        "threshold": None,
    },
    "f2": {
        "col": "F2frequency_sma3nz_amean",
        "label": "F2 (Hz)",
        "unit": "Hz",
        "title": "Second Formant (F2)",
        "is_f0_semitone": False,
        "threshold": None,
    },
    "f2_stddevNorm": {
        "col": "F2frequency_sma3nz_stddevNorm",
        "label": "F2 StdDevNorm",
        "unit": "norm",
        "title": "Second Formant Normalized StdDev (F2)",
        "is_f0_semitone": False,
        "threshold": None,
    },
    "f3": {
        "col": "F3frequency_sma3nz_amean",
        "label": "F3 (Hz)",
        "unit": "Hz",
        "title": "Third Formant (F3)",
        "is_f0_semitone": False,
        "threshold": None,
    },
    "f3_stddevNorm": {
        "col": "F3frequency_sma3nz_stddevNorm",
        "label": "F3 StdDevNorm",
        "unit": "norm",
        "title": "Third Formant Normalized StdDev (F3)",
        "is_f0_semitone": False,
        "threshold": None,
    },
        "spectral_flux": {
        "col": "spectralFlux_sma3_amean",
        "label": "Spectral Flux",
        "unit": "norm",
        "title": "Spectral Flux (Mean)",
        "is_f0_semitone": False,
        "threshold": None,
    },
    "spectral_flux_stddevNorm": {
        "col": "spectralFlux_sma3_stddevNorm",
        "label": "Spectral Flux StdDevNorm",
        "unit": "norm",
        "title": "Spectral Flux Normalized StdDev",
        "is_f0_semitone": False,
        "threshold": None,
    },

        # Harmonic purity
    "hnr": {
        "col": "HNRdBACF_sma3nz_amean",
        "label": "HNR (dB)",
        "unit": "dB",
        "title": "Harmonics-to-Noise Ratio (HNR)",
        "is_f0_semitone": False,
        "threshold": None,
    },
    # Spectral balance (50-1000 vs 1000-5000 Hz)
    "alpha_ratio": {
        "col": "alphaRatioV_sma3nz_amean",
        "label": "Alpha Ratio (dB)",
        "unit": "dB",
        "title": "Alpha Ratio Voiced",
        "is_f0_semitone": False,
        "threshold": None,
    },
    # Glottal opening quotient (pressed vs breathy voice)
    "h1_h2": {
        "col": "logRelF0-H1-H2_sma3nz_amean",
        "label": "H1-H2 (dB)",
        "unit": "dB",
        "title": "Harmonic Difference H1-H2",
        "is_f0_semitone": False,
        "threshold": None,
    },
    # Speech pacing
    "voiced_segments_per_sec": {
        "col": "VoicedSegmentsPerSec",
        "label": "Voiced Segments / sec",
        "unit": "seg/s",
        "title": "Speech Rate",
        "is_f0_semitone": False,
        "threshold": None,
    },
    # Macroscopic spectral envelope
    "mfcc2": {
        "col": "mfcc2_sma3_amean",
        "label": "MFCC 2",
        "unit": "norm",
        "title": "MFCC 2",
        "is_f0_semitone": False,
        "threshold": None,
    },
    # P80-P20 F0 dynamic range
    "f0_ambitus": {
        "col": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
        "label": "F0 Range P80-P20 (semitones)",
        "unit": "st",
        "title": "F0 Dynamic Range (P80-P20)",
        "is_f0_semitone": False,
        "threshold": None,
    },
    # Spectral slope
    "hammarberg_index": {
        "col": "hammarbergIndexV_sma3nz_amean",
        "label": "Hammarberg Index (dB)",
        "unit": "dB",
        "title": "Hammarberg Index Voiced",
        "is_f0_semitone": False,
        "threshold": None,
    },


}


def semitone_to_hz(semitones: float | np.ndarray) -> float | np.ndarray:
    return 27.5 * (2.0 ** (semitones / 12.0))


def find_column(df: pd.DataFrame, target_name: str) -> str:
    """Flexible column search (case & space insensitive)."""
    if target_name in df.columns:
        return target_name
    target_clean = target_name.lower().replace("_", "")
    for c in df.columns:
        if c.lower().replace("_", "") == target_clean:
            return c
    for c in df.columns:
        if target_name.lower() in c.lower():
            return c
    raise KeyError(f"Column '{target_name}' not found in CSV.")


def load_speaker_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Column 0 = speaker_id
    spk_col = df.columns[0]
    df[spk_col] = df[spk_col].astype(str).str.strip()
    df = df.rename(columns={spk_col: "speaker_id"}).set_index("speaker_id")

    # Column Gender
    gender_col = next((c for c in df.columns if "gender" in c.lower() or c.lower() == "sex"), df.columns[1])
    df["gender"] = df[gender_col].astype(str).str.strip().str.upper()

    # Extraction and conversion of configured features
    for feat_key, conf in FEATURES_CONFIG.items():
        actual_col = find_column(df, conf["col"])
        vals = pd.to_numeric(df[actual_col], errors="coerce")
        if conf["is_f0_semitone"]:
            df["f0_st"] = vals
            df["f0_val"] = semitone_to_hz(vals)
        else:
            df[f"{feat_key}_val"] = vals

    print(f"Loaded {len(df)} unique speakers from {csv_path.name}")
    return df


def load_validation_pairs() -> list[tuple[str, str]]:
    if INTERMEDIATE_VAL_PAIRS.is_file():
        with open(INTERMEDIATE_VAL_PAIRS, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [(str(p["id1"]), str(p["id2"])) for p in data]

    if ECHOSET_VAL_JSON.is_file():
        with open(ECHOSET_VAL_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = []
        for entry in data:
            if isinstance(entry, (list, tuple)) and entry:
                parent_name = Path(entry[0]).parent.name
                m = PAIR_REGEX.search(parent_name)
                if m:
                    pairs.append((m.group(1), m.group(2)))
        return pairs

    raise FileNotFoundError("Validation pairs file not found.")


def plot_feature_analysis(df_speakers: pd.DataFrame, df_pairs: pd.DataFrame, feat_key: str, conf: dict):
    """Generates the full set of figures for a given feature."""
    label = conf["label"]
    title = conf["title"]
    val_col1 = f"{feat_key}_1"
    val_col2 = f"{feat_key}_2"
    delta_col = f"delta_{feat_key}"
    threshold = conf["threshold"]

    # --------------------------------------------------------------------------
    # 1. Individual Speaker Distribution (Males vs Females with SHARED BINS)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    males = df_speakers[df_speakers["gender"] == "M"][f"{feat_key}_val"].dropna()
    females = df_speakers[df_speakers["gender"] == "F"][f"{feat_key}_val"].dropna()

    all_vals = pd.concat([males, females])
    common_bins = np.linspace(all_vals.min(), all_vals.max(), 16)  # Exactly 15 identical bins

    ax.hist(
        males,
        bins=common_bins,
        alpha=0.55,
        color=COLOR_MALE,
        edgecolor="black",
        label=f"Males (n={len(males)}, mean={males.mean():.2f})",
    )
    ax.hist(
        females,
        bins=common_bins,
        alpha=0.55,
        color=COLOR_FEMALE,
        edgecolor="black",
        label=f"Females (n={len(females)}, mean={females.mean():.2f})",
    )

    ax.set_xlabel(label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Speakers", fontsize=11, fontweight="bold")
    ax.set_title(f"Speaker Distribution: {title}", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    p1 = OUTPUT_PLOT_DIR / f"{feat_key}_1_speakers_distribution.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    # --------------------------------------------------------------------------
    # 2. Delta Histograms by Gender Configuration (M/M, M/F, F/F with SHARED BINS)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    max_delta = df_pairs[delta_col].dropna().max()
    common_delta_bins = np.linspace(0, max_delta, 25)

    for i, p_type in enumerate(["M/M", "M/F", "F/F"]):
        ax = axes[i]
        sub = df_pairs[df_pairs["pair_type"] == p_type][delta_col].dropna()
        n, bins, _ = ax.hist(
            sub,
            bins=common_delta_bins,
            color=PAIR_COLORS[p_type],
            alpha=0.65,
            edgecolor="black",
            density=True,
        )
        ax.axvline(
            sub.median(),
            color=COLOR_LINE,
            linestyle=":",
            linewidth=2,
            label=f"Median: {sub.median():.4f}",
        )

        extra_title = ""
        if threshold is not None:
            pct_above = (sub > threshold).mean() * 100.0
            pct_below = 100.0 - pct_above
            ax.axvline(
                threshold,
                color=COLOR_THRESHOLD,
                linestyle="--",
                linewidth=2,
                label=f"Threshold {threshold:.0f} {conf['unit']} ({pct_above:.1f}% > thresh)",
            )
            extra_title = f"\n{pct_above:.1f}% > {threshold:.0f} {conf['unit']} ({pct_below:.1f}% < thresh)"

        ax.set_title(f"{p_type} Pairs (N={len(sub):,}){extra_title}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Δ{label}", fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", frameon=True)
        ax.set_xlim(0, max_delta * 1.05)

    plt.suptitle(f"EchoSet Train: Distribution of Δ{label} by Gender Pair", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p2 = OUTPUT_PLOT_DIR / f"{feat_key}_2_delta_distribution_by_gender.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------------------------
    # 3. Comparative Boxplots of Delta
    # --------------------------------------------------------------------------
    plt.figure(figsize=(8, 5.5))
    data_to_plot = [
        df_pairs[df_pairs["pair_type"] == "M/M"][delta_col].dropna(),
        df_pairs[df_pairs["pair_type"] == "M/F"][delta_col].dropna(),
        df_pairs[df_pairs["pair_type"] == "F/F"][delta_col].dropna(),
        df_pairs[delta_col].dropna(),
    ]
    labels = ["M/M", "M/F", "F/F", "Overall (All)"]
    box = plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showmeans=True, notch=True)

    box_colors = [PAIR_COLORS["M/M"], PAIR_COLORS["M/F"], PAIR_COLORS["F/F"], PAIR_COLORS["All"]]
    for patch, col in zip(box["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    if threshold is not None:
        plt.axhline(threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.8, label=f"Critical Threshold ({threshold:.0f} {conf['unit']})")
        plt.legend(loc="upper left", frameon=True)

    plt.ylabel(f"Δ{label} = |Spk1 - Spk2|", fontsize=11, fontweight="bold")
    plt.title(f"Comparison of Δ{label} across EchoSet Val Pairs", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p3 = OUTPUT_PLOT_DIR / f"{feat_key}_3_delta_boxplots.png"
    plt.savefig(p3, dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 4. Bivariate Scatter (Speaker 1 vs Speaker 2)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(7.5, 6.5))
    for p_type in ["M/M", "F/F", "M/F"]:
        sub = df_pairs[df_pairs["pair_type"] == p_type]
        plt.scatter(sub[val_col1], sub[val_col2], label=f"{p_type}", color=PAIR_COLORS[p_type], alpha=0.35, s=16)

    min_v = min(df_pairs[val_col1].min(), df_pairs[val_col2].min())
    max_v = max(df_pairs[val_col1].max(), df_pairs[val_col2].max())
    x_range = np.linspace(min_v, max_v, 100)
    plt.plot(x_range, x_range, color=COLOR_LINE, linestyle="-", linewidth=1.2, label="Speaker 1 = Speaker 2")

    if threshold is not None:
        plt.plot(x_range, x_range + threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.3, label=f"|Δ| = {threshold:.0f} {conf['unit']}")
        plt.plot(x_range, x_range - threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.3)
        plt.fill_between(x_range, x_range - threshold, x_range + threshold, color=COLOR_THRESHOLD, alpha=0.08)

    plt.xlabel(f"{label} (Speaker 1)", fontsize=11, fontweight="bold")
    plt.ylabel(f"{label} (Speaker 2)", fontsize=11, fontweight="bold")
    plt.title(f"EchoSet Val: Bivariate Space of {title}", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    p4 = OUTPUT_PLOT_DIR / f"{feat_key}_4_bivariate_scatter.png"
    plt.savefig(p4, dpi=300)
    plt.close()

    print(f"  [✓] Plots generated for {title} -> {OUTPUT_PLOT_DIR}/{feat_key}_*.png")


def main():
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load features
    df_speakers = load_speaker_features(DEV_CLEAN_CSV)

    # 2. Load pairs
    pairs = load_validation_pairs()
    print(f"Number of loaded validation mixtures: {len(pairs):,}")

    df_pairs = pd.DataFrame(pairs, columns=["id1", "id2"])
    df_pairs["gender_1"] = df_pairs["id1"].map(df_speakers["gender"])
    df_pairs["gender_2"] = df_pairs["id2"].map(df_speakers["gender"])

    # Strict gender classification M/M, M/F, F/F
    df_pairs["pair_type"] = [
        "M/F" if (g1 == "M" and g2 == "F") or (g1 == "F" and g2 == "M") else f"{g1}/{g1}"
        for g1, g2 in zip(df_pairs["gender_1"], df_pairs["gender_2"])
    ]

    # Vectorized feature assignment and delta computation
    for feat_key, conf in FEATURES_CONFIG.items():
        v1 = df_pairs["id1"].map(df_speakers[f"{feat_key}_val"])
        v2 = df_pairs["id2"].map(df_speakers[f"{feat_key}_val"])
        df_pairs[f"{feat_key}_1"] = v1
        df_pairs[f"{feat_key}_2"] = v2
        df_pairs[f"delta_{feat_key}"] = (v1 - v2).abs()

    # 3. Console summary table
    print("\n" + "=" * 95)
    print("        ACOUSTIC DISTANCE (DELTAS) SUMMARY ON ECHOSET VALIDATION SET")
    print("=" * 95)

    rows = []
    for feat_key, conf in FEATURES_CONFIG.items():
        delta_col = f"delta_{feat_key}"
        unit = conf["unit"]
        for p_type in ["All", "M/M", "M/F", "F/F"]:
            sub = df_pairs if p_type == "All" else df_pairs[df_pairs["pair_type"] == p_type]
            vals = sub[delta_col].dropna()
            q25, q50, q75 = vals.quantile([0.25, 0.50, 0.75])
            unit_suffix = f" ({unit})" if unit else ""
            rows.append({
                "Feature": conf["title"],
                "Pair": p_type,
                "N": len(vals),
                f"Mean{unit_suffix}": vals.mean(),
                f"Median{unit_suffix}": q50,
                f"IQR{unit_suffix}": q75 - q25,
            })

    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 95 + "\n")

    # 4. Generate all figures
    print("Generating figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for feat_key, conf in FEATURES_CONFIG.items():
        plot_feature_analysis(df_speakers, df_pairs, feat_key, conf)

    print(f"\n[Success! All figures are saved to: {OUTPUT_PLOT_DIR}]")


if __name__ == "__main__":
    main()
