#!/usr/bin/env python3
"""
Statistical Framework: Acoustic Feature Discriminability for Low-Pitch-Difference Speaker Pairs.

Pass 1: Gender-stratified contrast (M/M, F/F, M/F) and balanced score S_same = 0.5 * (s_MM + s_FF).
Pass 2: Conditioned (Δf0 < 60 Hz) vs Unconditioned comparison and contrast retention ratio R.
Pass 3: Utterance-level intra-speaker variability and contrast-to-variability ratio Q = D_inter / D_intra.
Pass 4: Pitch sensitivity analysis (0-20 Hz, 20-40 Hz, 40-60 Hz).
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

# ==============================================================================
# 1. PATHS & CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (BASE_DIR / "../../..").resolve()

SPEAKERS_CSV_PATH = (
    PROJECT_ROOT
    / "tests_adrien"
    / "analyze_librispeech"
    / "train-clean-100"
    / "train-clean-100_egemaps_speakers.csv"
)

# Individual utterance embeddings directory
EMBEDDINGS_DIR = (
    PROJECT_ROOT
    / "tests_adrien"
    / "analyze_librispeech"
    / "train-clean-100"
    / "embeddings"
    / "egemaps"
)

ECHOSET_JSON_PATH = PROJECT_ROOT / "echoset_pairs_train.json"

OUTPUT_PLOT_DIR = BASE_DIR / "disciminative_features_corrected"
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Visual styling
PAIR_COLORS = {
    "M/M": "#1f77b4",   # Blue
    "F/F": "#e377c2",   # Pink
    "M/F": "#2ca02c",   # Green
    "S_same": "#d62728",# Red
}

CANDIDATE_FEATURES = {
    "f1_mean": {"col": "F1frequency_sma3nz_amean", "title": "F1 Mean", "unit": "Hz"},
    "f1_std": {"col": "F1frequency_sma3nz_stddevNorm", "title": "F1 StdDev Norm", "unit": ""},
    "f2_mean": {"col": "F2frequency_sma3nz_amean", "title": "F2 Mean", "unit": "Hz"},
    "f2_std": {"col": "F2frequency_sma3nz_stddevNorm", "title": "F2 StdDev Norm", "unit": ""},
    "f3_mean": {"col": "F3frequency_sma3nz_amean", "title": "F3 Mean", "unit": "Hz"},
    "f3_std": {"col": "F3frequency_sma3nz_stddevNorm", "title": "F3 StdDev Norm", "unit": ""},
    "spectral_flux_mean": {"col": "spectralFlux_sma3_amean", "title": "Spectral Flux Mean", "unit": ""},
    "spectral_flux_std": {"col": "spectralFlux_sma3_stddevNorm", "title": "Spectral Flux StdDev Norm", "unit": ""},
    "hnr": {"col": "HNRdBACF_sma3nz_amean", "title": "Harmonics-to-Noise Ratio (HNR)", "unit": "dB"},
    "h1_h2": {"col": "logRelF0-H1-H2_sma3nz_amean", "title": "Harmonic Energy (H1-H2)", "unit": "dB"},
    "alpha_ratio": {"col": "alphaRatioV_sma3nz_amean", "title": "Alpha Ratio (LF/HF)", "unit": "dB"},
    "loudness": {"col": "loudness_sma3_amean", "title": "Loudness", "unit": "sone"},
}

# ==============================================================================
# 2. DATA INGESTION & REFERENCE NORMALIZATION
# ==============================================================================
def load_data():
    print("=" * 85)
    print("1. DATA INGESTION & REFERENCE POPULATION SETUP")
    print("=" * 85)
    df_spk = pd.read_csv(SPEAKERS_CSV_PATH)
    df_spk["speaker_id"] = df_spk["speaker_id"].astype(int)

    # 1. F0 parsing & conversion to Hz
    if "F0semitoneFrom27.5Hz_sma3nz_amean" in df_spk.columns:
        df_spk["f0_hz"] = 27.5 * (2.0 ** (df_spk["F0semitoneFrom27.5Hz_sma3nz_amean"] / 12.0))
    elif "F0mean" in df_spk.columns:
        df_spk["f0_hz"] = df_spk["F0mean"].astype(float)
    else:
        f0_candidates = [c for c in df_spk.columns if "f0" in c.lower() and "mean" in c.lower()]
        if f0_candidates:
            df_spk["f0_hz"] = df_spk[f0_candidates[0]].astype(float)
        else:
            raise KeyError("Unable to locate F0 mean column in speaker CSV.")

    # 2. Fixed unconditioned reference standard deviations sigma[F, g]
    gender_stats = {}
    for feat_key, conf in CANDIDATE_FEATURES.items():
        col = conf["col"]
        if col in df_spk.columns:
            m_vals = df_spk[df_spk["gender"] == "M"][col].dropna()
            f_vals = df_spk[df_spk["gender"] == "F"][col].dropna()
            std_m = m_vals.std(ddof=1)
            std_f = f_vals.std(ddof=1)
            std_pooled = np.sqrt((std_m**2 + std_f**2) / 2.0)
            gender_stats[feat_key] = {
                "std_M": std_m,
                "std_F": std_f,
                "std_pooled": std_pooled,
            }
        else:
            print(f"[Warning] Column '{col}' missing. Skipping.")

    spk_dict = df_spk.set_index("speaker_id").to_dict(orient="index")

    # 3. EchoSet pairs ingestion
    with open(ECHOSET_JSON_PATH, "r") as f:
        raw_pairs = json.load(f)

    pair_records = []
    for item in raw_pairs:
        if isinstance(item, dict):
            if "id1" in item and "id2" in item:
                s1, s2 = int(item["id1"]), int(item["id2"])
            elif "speaker_1" in item and "speaker_2" in item:
                s1, s2 = int(item["speaker_1"]), int(item["speaker_2"])
            else:
                continue
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            s1, s2 = int(item[0]), int(item[1])
        else:
            continue

        if s1 not in spk_dict or s2 not in spk_dict:
            continue

        g1 = spk_dict[s1].get("gender", "U")
        g2 = spk_dict[s2].get("gender", "U")
        pair_type = f"{g1}/{g1}" if g1 == g2 else "M/F"

        f0_1 = spk_dict[s1]["f0_hz"]
        f0_2 = spk_dict[s2]["f0_hz"]
        delta_f0 = abs(f0_1 - f0_2)

        rec = {
            "speaker_1": s1,
            "speaker_2": s2,
            "pair_type": pair_type,
            "delta_f0": delta_f0,
        }

        # Compute raw and normalized deltas
        for feat_key in gender_stats.keys():
            c = CANDIDATE_FEATURES[feat_key]["col"]
            val1 = spk_dict[s1][c]
            val2 = spk_dict[s2][c]
            raw_d = abs(val1 - val2)
            rec[f"delta_{feat_key}"] = raw_d

            # Fixed denominator by group
            if pair_type == "M/M":
                sig = gender_stats[feat_key]["std_M"]
            elif pair_type == "F/F":
                sig = gender_stats[feat_key]["std_F"]
            else:
                sig = gender_stats[feat_key]["std_pooled"]

            rec[f"norm_{feat_key}"] = raw_d / sig if sig > 0 else 0.0

        pair_records.append(rec)

    df_pairs = pd.DataFrame(pair_records)
    print(f"Loaded {len(df_spk)} distinct speakers ({len(df_spk[df_spk['gender']=='M'])} M, {len(df_spk[df_spk['gender']=='F'])} F).")
    print(f"Loaded {len(df_pairs)} total pairs from EchoSet.")
    return df_spk, df_pairs, gender_stats


# ==============================================================================
# 3. PASS 1: GENDER-STRATIFIED & BALANCED SCORE (S_same)
# ==============================================================================
def run_pass_1(df_pairs, gender_stats):
    print("\n" + "=" * 85)
    print("PASS 1: LOW-PITCH POPULATION (Δf0 < 60 Hz) & BALANCED SCORE S_same")
    print("S_same = 0.5 * s[M/M] + 0.5 * s[F/F]")
    print("=" * 85)

    df_low = df_pairs[df_pairs["delta_f0"] < 60.0].copy()
    n_low = len(df_low)
    print(f"Low-pitch pairs (Δf0 < 60 Hz): {n_low:,} / {len(df_pairs):,} ({n_low/len(df_pairs)*100:.1f}%)")

    results = []
    for feat_key in gender_stats.keys():
        norm_col = f"norm_{feat_key}"
        raw_col = f"delta_{feat_key}"

        sub_mm = df_low[df_low["pair_type"] == "M/M"]
        sub_ff = df_low[df_low["pair_type"] == "F/F"]
        sub_mf = df_low[df_low["pair_type"] == "M/F"]

        s_mm = sub_mm[norm_col].mean()
        s_ff = sub_ff[norm_col].mean()
        s_mf = sub_mf[norm_col].mean()

        s_same = 0.5 * (s_mm + s_ff)
        abs_diff_mm_ff = abs(s_mm - s_ff)

        results.append({
            "feat_key": feat_key,
            "title": CANDIDATE_FEATURES[feat_key]["title"],
            "unit": CANDIDATE_FEATURES[feat_key]["unit"],
            "s_same": s_same,
            "s_mm": s_mm,
            "s_ff": s_ff,
            "diff_mm_ff": abs_diff_mm_ff,
            "s_mf": s_mf,
            "raw_mean_same": 0.5 * (sub_mm[raw_col].mean() + sub_ff[raw_col].mean()),
        })

    df_res = pd.DataFrame(results).sort_values(by="s_same", ascending=False).reset_index(drop=True)
    df_res["Rank_same"] = df_res.index + 1
    df_res["Rank_MM"] = df_res["s_mm"].rank(ascending=False, method="min").astype(int)
    df_res["Rank_FF"] = df_res["s_ff"].rank(ascending=False, method="min").astype(int)
    df_res["Rank_MF"] = df_res["s_mf"].rank(ascending=False, method="min").astype(int)

    table_disp = df_res[[
        "Rank_same", "title", "unit", "s_same", "s_mm", "Rank_MM", "s_ff", "Rank_FF", "diff_mm_ff", "s_mf", "Rank_MF"
    ]].copy()
    table_disp.columns = [
        "Rank", "Feature", "Unit", "S_same (σ)", "s[M/M] (σ)", "Rk_M", "s[F/F] (σ)", "Rk_F", "|M-F| (σ)", "s[M/F] (σ)", "Rk_MF"
    ]
    print(table_disp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Plot Pass 1: Horizontal Bar chart
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(df_res))
    bar_height = 0.25

    ax.barh(y_pos - bar_height, df_res["s_mm"], height=bar_height, color=PAIR_COLORS["M/M"], alpha=0.8, label="s[M/M]")
    ax.barh(y_pos, df_res["s_ff"], height=bar_height, color=PAIR_COLORS["F/F"], alpha=0.8, label="s[F/F]")
    ax.barh(y_pos + bar_height, df_res["s_same"], height=bar_height, color=PAIR_COLORS["S_same"], alpha=0.85, label="S_same (Balanced)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_res["title"], fontweight="bold", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Acoustic Contrast (in σ units)", fontsize=11, fontweight="bold")
    ax.set_title("Pass 1: Feature Ranking by Balanced Same-Gender Score (S_same | Δf0 < 60 Hz)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    fig.tight_layout()
    p_out = OUTPUT_PLOT_DIR / "1_feature_ranking_same_gender.png"
    fig.savefig(p_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Saved figure: {p_out.name}")

    return df_res, df_low


# ==============================================================================
# 4. PASS 2: CONDITIONED VS UNCONDITIONED COMPARISON (RATIO R)
# ==============================================================================
def run_pass_2(df_pairs, df_low, df_res_pass1, gender_stats):
    print("\n" + "=" * 85)
    print("PASS 2: CONDITIONED (Δf0 < 60 Hz) VS UNCONDITIONED COMPARISON")
    print("Retention Ratio R = s_low / s_all")
    print("=" * 85)

    rows = []
    for feat_key in gender_stats.keys():
        norm_col = f"norm_{feat_key}"

        # Unconditioned (All pairs)
        s_all_mm = df_pairs[df_pairs["pair_type"] == "M/M"][norm_col].mean()
        s_all_ff = df_pairs[df_pairs["pair_type"] == "F/F"][norm_col].mean()
        s_all_same = 0.5 * (s_all_mm + s_all_ff)

        # Conditioned (Low pitch pairs)
        s_low_mm = df_low[df_low["pair_type"] == "M/M"][norm_col].mean()
        s_low_ff = df_low[df_low["pair_type"] == "F/F"][norm_col].mean()
        s_low_same = 0.5 * (s_low_mm + s_low_ff)

        r_mm = s_low_mm / s_all_mm if s_all_mm > 0 else 1.0
        r_ff = s_low_ff / s_all_ff if s_all_ff > 0 else 1.0
        r_same = s_low_same / s_all_same if s_all_same > 0 else 1.0

        rows.append({
            "feat_key": feat_key,
            "title": CANDIDATE_FEATURES[feat_key]["title"],
            "s_all_same": s_all_same,
            "s_low_same": s_low_same,
            "R_same": r_same,
            "R_mm": r_mm,
            "R_ff": r_ff,
        })

    df_pass2 = pd.DataFrame(rows).sort_values(by="s_low_same", ascending=False).reset_index(drop=True)

    table_disp = df_pass2[["title", "s_all_same", "s_low_same", "R_same", "R_mm", "R_ff"]].copy()
    table_disp.columns = [
        "Feature", "S_all (σ)", "S_low (σ)", "R (Same)", "R (M/M)", "R (F/F)"
    ]
    print(table_disp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Plot Pass 2: Dumbbell / Retention plot
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y_pos = np.arange(len(df_pass2))

    for y, (_, row) in zip(y_pos, df_pass2.iterrows()):
        ax.plot([row["s_all_same"], row["s_low_same"]], [y, y], color="#999999", linewidth=2, zorder=1)

    ax.scatter(df_pass2["s_all_same"], y_pos, color="#7f7f7f", s=70, label="Unconditioned (All Pairs)", zorder=2)
    ax.scatter(df_pass2["s_low_same"], y_pos, color=PAIR_COLORS["S_same"], s=85, label="Conditioned (Δf0 < 60 Hz)", zorder=3)

    for y, (_, row) in zip(y_pos, df_pass2.iterrows()):
        ax.text(max(row["s_all_same"], row["s_low_same"]) + 0.03, y, f"R = {row['R_same']:.2f}",
                va="center", fontsize=9, fontweight="bold", color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_pass2["title"], fontweight="bold", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Balanced Same-Gender Score (in σ units)", fontsize=11, fontweight="bold")
    ax.set_title("Pass 2: Acoustic Contrast Retention (Unconditioned vs Δf0 < 60 Hz)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    fig.tight_layout()
    p_out = OUTPUT_PLOT_DIR / "2_conditioned_vs_unconditioned.png"
    fig.savefig(p_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Saved figure: {p_out.name}")

    return df_pass2

# ==============================================================================
# 5. PASS 3: INTRA-SPEAKER VARIABILITY & DUAL CONTRAST RATIOS (Q_same & Q_all)
# ==============================================================================
def run_pass_3(df_spk, df_low, gender_stats):
    print("\n" + "=" * 90)
    print("PASS 3: DUAL INTER/INTRA CONTRAST RATIOS (Q_same-gender & Q_all)")
    print(f"Reading individual utterance embeddings from: {EMBEDDINGS_DIR}")
    print("=" * 90)

    if not EMBEDDINGS_DIR.exists():
        print(f"[Error] Embeddings directory does not exist: {EMBEDDINGS_DIR}")
        return None

    # Get feature index mapping from opensmile or CSV columns
    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        feature_names = list(smile.feature_names)
    except Exception:
        df_tmp = pd.read_csv(SPEAKERS_CSV_PATH, nrows=1)
        feature_names = [c for c in df_tmp.columns if c not in ["speaker_id", "num_utterances", "gender", "f0_hz"]]

    feat_indices = {}
    for k, conf in CANDIDATE_FEATURES.items():
        c = conf["col"]
        if c in feature_names:
            feat_indices[k] = feature_names.index(c)

    # 1. Gather all utterance vectors per speaker
    print("Scanning embedding files and grouping utterances by speaker...")
    files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy")]
    if not files:
        print("[Warning] No .npy files found in embeddings directory.")
        return None

    spk_utterances = defaultdict(lambda: defaultdict(list))
    for fname in files:
        try:
            spk_id = int(fname.split("-")[0])
        except ValueError:
            continue

        arr = np.load(EMBEDDINGS_DIR / fname)
        if arr.ndim > 1:
            arr = arr.flatten()

        for k, idx in feat_indices.items():
            spk_utterances[spk_id][k].append(arr[idx])

    # Speaker gender map
    spk_gender = df_spk.set_index("speaker_id")["gender"].to_dict()

    # 2. Sample sizes statistics
    utt_counts_all = [len(feat_dict[list(feat_indices.keys())[0]]) for feat_dict in spk_utterances.values()]
    utt_counts_m = [
        len(feat_dict[list(feat_indices.keys())[0]]) 
        for spk_id, feat_dict in spk_utterances.items() 
        if spk_gender.get(spk_id) == "M"
    ]
    utt_counts_f = [
        len(feat_dict[list(feat_indices.keys())[0]]) 
        for spk_id, feat_dict in spk_utterances.items() 
        if spk_gender.get(spk_id) == "F"
    ]

    n_pairs_mm = len(df_low[df_low["pair_type"] == "M/M"])
    n_pairs_ff = len(df_low[df_low["pair_type"] == "F/F"])
    n_pairs_mf = len(df_low[df_low["pair_type"] == "M/F"])
    n_pairs_same = n_pairs_mm + n_pairs_ff
    n_pairs_all = len(df_low)

    print("-" * 90)
    print("SAMPLE SIZES (Δf0 < 60 Hz):")
    print(f" • Intra-speaker : {len(spk_utterances)} distinct speakers, mean {np.mean(utt_counts_all):.1f} utterances/speaker")
    print(f"                   (Males: {len(utt_counts_m)} spk, mean {np.mean(utt_counts_m):.1f} utt | Females: {len(utt_counts_f)} spk, mean {np.mean(utt_counts_f):.1f} utt)")
    print(f" • Inter-speaker : {n_pairs_all:,} total pairs with Δf0 < 60 Hz")
    print(f"                   -> Same-gender : {n_pairs_same:,} pairs (M/M: {n_pairs_mm:,}, F/F: {n_pairs_ff:,})")
    print(f"                   -> Cross-gender: {n_pairs_mf:,} pairs (M/F)")
    print("-" * 90)

    # 3. Compute D_intra (equal weight per speaker) and Inter distances
    records = []
    for k in feat_indices.keys():
        d_intra_by_spk_m = []
        d_intra_by_spk_f = []
        d_intra_by_spk_all = []

        # Equal-weighted intra-speaker distance across speakers
        for spk_id, feat_dict in spk_utterances.items():
            vals = np.array(feat_dict[k])
            if len(vals) < 2:
                continue
            diffs = np.abs(vals[:, None] - vals[None, :])
            spk_mean_intra = diffs[np.triu_indices(len(vals), k=1)].mean()

            d_intra_by_spk_all.append(spk_mean_intra)
            g = spk_gender.get(spk_id, "U")
            if g == "M":
                d_intra_by_spk_m.append(spk_mean_intra)
            elif g == "F":
                d_intra_by_spk_f.append(spk_mean_intra)

        # Intra distances (equal weight per speaker)
        d_intra_m = np.mean(d_intra_by_spk_m) if d_intra_by_spk_m else 1e-6
        d_intra_f = np.mean(d_intra_by_spk_f) if d_intra_by_spk_f else 1e-6
        d_intra_same = 0.5 * (d_intra_m + d_intra_f)
        d_intra_all = np.mean(d_intra_by_spk_all) if d_intra_by_spk_all else 1e-6

        # Inter distances on pitch-conditioned pairs (Δf0 < 60 Hz)
        raw_col = f"delta_{k}"
        d_inter_mm = df_low[df_low["pair_type"] == "M/M"][raw_col].mean()
        d_inter_ff = df_low[df_low["pair_type"] == "F/F"][raw_col].mean()
        d_inter_mf = df_low[df_low["pair_type"] == "M/F"][raw_col].mean()

        # Definition A: Balanced same-gender distance
        d_inter_same = 0.5 * (d_inter_mm + d_inter_ff)

        # Definition B: All-gender distance (unrestricted)
        d_inter_all = df_low[raw_col].mean()

        # Contrast ratios
        q_same = d_inter_same / d_intra_same if d_intra_same > 0 else 0.0
        q_all = d_inter_all / d_intra_all if d_intra_all > 0 else 0.0
        q_mm = d_inter_mm / d_intra_m if d_intra_m > 0 else 0.0
        q_ff = d_inter_ff / d_intra_f if d_intra_f > 0 else 0.0
        q_mf = d_inter_mf / d_intra_all if d_intra_all > 0 else 0.0

        records.append({
            "feat_key": k,
            "title": CANDIDATE_FEATURES[k]["title"],
            "unit": CANDIDATE_FEATURES[k]["unit"],
            "Q_same": q_same,
            "Q_all": q_all,
            "Q_MM": q_mm,
            "Q_FF": q_ff,
            "Q_MF": q_mf,
            "D_inter_same": d_inter_same,
            "D_inter_all": d_inter_all,
            "D_intra": d_intra_same,
        })

    df_q = pd.DataFrame(records).sort_values(by="Q_same", ascending=False).reset_index(drop=True)
    df_q["Rank_Q_same"] = df_q.index + 1
    df_q["Rank_Q_all"] = df_q["Q_all"].rank(ascending=False, method="min").astype(int)

    table_disp = df_q[[
        "Rank_Q_same", "title", "unit", "Q_same", "Q_all", "Rank_Q_all", "Q_MM", "Q_FF", "Q_MF", "D_inter_same", "D_inter_all", "D_intra"
    ]].copy()
    table_disp.columns = [
        "Rank", "Feature", "Unit", "Q (Same)", "Q (All)", "Rk_All", "Q(M/M)", "Q(F/F)", "Q(M/F)", "D_inter(Same)", "D_inter(All)", "D_intra"
    ]
    print(table_disp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Plot Pass 3: Grouped Bar chart comparing Q_same vs Q_all
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    y_pos = np.arange(len(df_q))
    bar_height = 0.35

    ax.barh(y_pos - bar_height / 2, df_q["Q_same"], height=bar_height, color=PAIR_COLORS["S_same"], alpha=0.85, label="Q (Same-gender)")
    ax.barh(y_pos + bar_height / 2, df_q["Q_all"], height=bar_height, color="#7f7f7f", alpha=0.75, label="Q (All-gender, incl. M/F)")

    # Baseline threshold Q = 1.0
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.5, label="Threshold Q = 1.0 (Inter = Intra)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_q["title"], fontweight="bold", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Contrast-to-Variability Ratio Q = D_inter / D_intra", fontsize=11, fontweight="bold")
    ax.set_title("Pass 3: Comparison of Inter/Intra Contrast Ratios (Q_same-gender vs Q_all)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, loc="lower right", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    fig.tight_layout()
    p_out = OUTPUT_PLOT_DIR / "3_contrast_to_variability_ratio_Q.png"
    fig.savefig(p_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Saved figure: {p_out.name}")

    return df_q

# ==============================================================================
# 6. PASS 4: SENSITIVITY ANALYSIS (PITCH STRATIFICATION UP TO 100 Hz)
# ==============================================================================
def run_sensitivity_analysis(df_pairs, gender_stats, df_res_pass1):
    print("\n" + "=" * 85)
    print("PASS 4: PITCH-THRESHOLD SENSITIVITY ANALYSIS (0 to 100 Hz)")
    print("Bins: [0, 20), [20, 40), [40, 60), [60, 80), [80, 100) Hz")
    print("=" * 85)

    bins = [
        ("0-20 Hz", (df_pairs["delta_f0"] >= 0) & (df_pairs["delta_f0"] < 20)),
        ("20-40 Hz", (df_pairs["delta_f0"] >= 20) & (df_pairs["delta_f0"] < 40)),
        ("40-60 Hz", (df_pairs["delta_f0"] >= 40) & (df_pairs["delta_f0"] < 60)),
        ("60-80 Hz", (df_pairs["delta_f0"] >= 60) & (df_pairs["delta_f0"] < 80)),
        ("80-100 Hz", (df_pairs["delta_f0"] >= 80) & (df_pairs["delta_f0"] < 100)),
        ("0-60 Hz (Baseline)", (df_pairs["delta_f0"] >= 0) & (df_pairs["delta_f0"] < 60)),
        ("0-100 Hz (Extended)", (df_pairs["delta_f0"] >= 0) & (df_pairs["delta_f0"] < 100)),
    ]

    top_features = df_res_pass1["feat_key"].head(11).tolist()
    sens_records = []

    for bin_name, mask in bins:
        sub_bin = df_pairs[mask]
        n_pairs = len(sub_bin)
        distinct_spk = len(set(sub_bin["speaker_1"]).union(set(sub_bin["speaker_2"])))

        row = {
            "Pitch_Bin": bin_name,
            "N_Pairs": n_pairs,
            "Distinct_Spk": distinct_spk,
        }

        for feat_key in top_features:
            norm_col = f"norm_{feat_key}"
            s_mm = sub_bin[sub_bin["pair_type"] == "M/M"][norm_col].mean()
            s_ff = sub_bin[sub_bin["pair_type"] == "F/F"][norm_col].mean()
            s_same = 0.5 * (s_mm + s_ff)
            row[CANDIDATE_FEATURES[feat_key]["title"]] = s_same

        sens_records.append(row)

    df_sens = pd.DataFrame(sens_records)
    print(df_sens.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Plot Sensitivity Lines (for the 5 sequential bins)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    plot_labels = ["0-20 Hz", "20-40 Hz", "40-60 Hz", "60-80 Hz", "80-100 Hz"]
    x_indices = np.arange(len(plot_labels))

    # Filter only the 5 individual bins for the line plot
    df_plot = df_sens.iloc[0:5]

    for feat_key in top_features:
        feat_title = CANDIDATE_FEATURES[feat_key]["title"]
        y_vals = df_plot[feat_title].values
        ax.plot(x_indices, y_vals, marker="o", linewidth=2.0, label=feat_title)

    # Threshold indicator at 60 Hz
    ax.axvline(2.5, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7, label="Collision Threshold (60 Hz)")

    ax.set_xticks(x_indices)
    ax.set_xticklabels(plot_labels, fontweight="bold", fontsize=10)
    ax.set_xlabel("Pitch Difference Bin (Δf0)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Balanced Score S_same (in σ units)", fontsize=11, fontweight="bold")
    ax.set_title("Pass 4: Pitch Sensitivity — Contrast Stability across Δf0 Bins up to 100 Hz", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    p_out = OUTPUT_PLOT_DIR / "4_pitch_sensitivity_analysis.png"
    fig.savefig(p_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Saved figure: {p_out.name}")

# ==============================================================================
# 7. COMBINED FIGURE: S_same (PASS 1) VS Q_all (PASS 3) ON DUAL AXES
# ==============================================================================
def plot_combined_s_same_vs_q(df_res_pass1, df_q):
    print("Generating combined dual-axis plot (S_same vs Q_all)...")
    
    # Merge and sort by Pass 3 Q_all descending
    df_merged = df_res_pass1.merge(df_q[["feat_key", "Q_all"]], on="feat_key")
    df_merged = df_merged.sort_values(by="Q_all", ascending=False).reset_index(drop=True)


    fig, ax1 = plt.subplots(figsize=(13, 6.2))
    ax2 = ax1.twinx()

    x = np.arange(len(df_merged))
    width = 0.38

    # Left axis (y1): Red bars for S_same
    bars1 = ax1.bar(
        x - width / 2, 
        df_merged["s_same"], 
        width, 
        color="#d62728", 
        alpha=0.85, 
        label="S_same (Pass 1 Contrast, in σ)"
    )

    # Right axis (y2): Blue bars for Q_all
    bars2 = ax2.bar(
        x + width / 2, 
        df_merged["Q_all"], 
        width, 
        color="#1f77b4", 
        alpha=0.85, 
        label="Q (Pass 3 All-gender, Inter/Intra)"
    )

    # Horizontal baseline for Q = 1.0 on right axis
    ax2.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2, alpha=0.7, label="Threshold Q = 1.0 (Inter = Intra)")

    # X-axis configuration
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_merged["title"], rotation=35, ha="right", fontweight="bold", fontsize=10)
    ax1.set_xlabel("Acoustic Feature (Sorted by Pass 3 Q_all)", fontsize=11, fontweight="bold")


    # Y1 configuration (Left - S_same)
    ax1.set_ylabel("Balanced Contrast Score S_same (in σ units)", color="#d62728", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(0, df_merged["s_same"].max() * 1.25)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Y2 configuration (Right - Q_all)
    ax2.set_ylabel("Contrast-to-Variability Ratio Q (All-gender)", color="#1f77b4", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0, df_merged["Q_all"].max() * 1.25)

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=9.5)

    plt.title(
        "Acoustic Contrast vs Robustness: S_same (Pass 1, Left) and Q_all (Pass 3, Right) | Δf0 < 60 Hz",
        fontsize=12,
        fontweight="bold",
        pad=15
    )

    fig.tight_layout()
    p_out = OUTPUT_PLOT_DIR / "5_combined_S_same_vs_Q_all.png"
    fig.savefig(p_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Saved figure: {p_out.name}")



# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    df_spk, df_pairs, gender_stats = load_data()
    df_res_pass1, df_low = run_pass_1(df_pairs, gender_stats)
    df_pass2 = run_pass_2(df_pairs, df_low, df_res_pass1, gender_stats)
    df_q = run_pass_3(df_spk, df_low, gender_stats)
    run_sensitivity_analysis(df_pairs, gender_stats, df_res_pass1)
    plot_combined_s_same_vs_q(df_res_pass1, df_q)


    print("\n" + "=" * 85)
    print(f"[COMPLETED] All 4 passes evaluated successfully.")
    print(f"All figures exported to:\n{OUTPUT_PLOT_DIR}")
    print("=" * 85)


if __name__ == "__main__":
    main()
