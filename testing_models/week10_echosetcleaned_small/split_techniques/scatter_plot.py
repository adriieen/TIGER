import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------------------------------------------------
# 1. Chemins
# -------------------------------------------------------------------------
BASE_DIR = Path("/idiap/home/adubois/project_ma1/TIGER")
PARQUET_TEST = Path("/idiap/temp/adubois/EchoSet_cleaned_small/egemaps_summary_test.parquet")
TOP20_FILE = BASE_DIR / "tests_adrien/analyze_echoset_clean/top20_features.txt"
EXP_DIR = BASE_DIR / "Experiments/checkpoint"
OUTPUT_HTML = BASE_DIR / "testing_models/week10_echosetcleaned_small/split_techniques/scatter_plot.html"

# Modèles à comparer pour le scatter plot sur EchoSet_cleaned_small
MODELS = {
    "Baseline": {
        "file": EXP_DIR / "TIGER-EchoSet_cleaned_small-baseline-6gpus/results/metrics.csv",
        "color": "#1f77b4",  # Bleu
    },
    "Even Split": {
        "file": EXP_DIR / "TIGER-EchoSet_cleaned_small-even_split-6gpus/results/metrics.csv",
        "color": "#2ca02c",  # Vert
    },
    "Normal Split": {
        "file": EXP_DIR / "TIGER-EchoSet_cleaned_small-normal_split-6gpus/results/metrics.csv",
        "color": "#d62728",  # Rouge
    },
}

def clean_pair_key(s):
    s = str(s).replace("\\", "/").strip()
    m = re.search(r"(\d{2}/\d+_\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d{2})_(\d+_\d+)", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"(\d+_\d+)", s)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(s))[0]

# -------------------------------------------------------------------------
# 2. Chargement des features
# -------------------------------------------------------------------------
if not TOP20_FILE.exists():
    raise FileNotFoundError(f"Fichier top20_features.txt introuvable : {TOP20_FILE}")

with open(TOP20_FILE, "r") as f:
    top20_raw = [line.strip() for line in f if line.strip() and not line.startswith("#")]

print(f"--> {len(top20_raw)} features cibles chargées.")
print(f"--> Lecture du parquet : {PARQUET_TEST}")
df_test = pd.read_parquet(PARQUET_TEST)
pair_col = [c for c in df_test.columns if any(k in c.lower() for k in ["pair", "key", "utt"])][0]
df_test["clean_id"] = df_test[pair_col].apply(clean_pair_key)
df_test = df_test.drop_duplicates(subset=["clean_id"]).set_index("clean_id")

matched_features = []
for feat in top20_raw:
    candidates = [c for c in df_test.columns if c == feat or c == f"dist_{feat}" or c.endswith(feat)]
    if candidates:
        matched_features.append((feat, candidates[0]))
    else:
        print(f"[Warning] Feature non trouvée dans le parquet : {feat}")

# -------------------------------------------------------------------------
# 3. Chargement des résultats metrics.csv (SI-SDRi et SDRi)
# -------------------------------------------------------------------------
df_models_sisdri = {}
df_models_sdri = {}

for name, m_info in MODELS.items():
    csv_path = m_info["file"]
    if not csv_path.exists():
        print(f"[Warning] Fichier absent : {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    key_col = "key" if "key" in df.columns else df.columns[0]
    df["clean_id"] = df[key_col].apply(clean_pair_key)
    
    # Identifier colonnes SI-SDRi et SDRi
    sisdri_col = [c for c in df.columns if any(k in c.lower() for k in ["si-snr_i", "si_sdr_i", "si_sdri", "sisdr_i", "sisdri"])]
    sisdri_col = sisdri_col[0] if sisdri_col else ("si-snr" if "si-snr" in df.columns else df.columns[4])

    sdri_col = [c for c in df.columns if any(k in c.lower() for k in ["sdr_i", "sdri"])]
    sdri_col = sdri_col[0] if sdri_col else df.columns[2]
        
    df_dedup = df.drop_duplicates(subset=["clean_id"]).set_index("clean_id")
    df_models_sisdri[name] = df_dedup[sisdri_col].to_dict()
    df_models_sdri[name] = df_dedup[sdri_col].to_dict()
    print(f"[{name}] Chargé ({len(df_dedup)} paires). SI-SDRi: {sisdri_col}, SDRi: {sdri_col}")

# -------------------------------------------------------------------------
# 4. Configuration des vues (Unités explicites)
# -------------------------------------------------------------------------
view_configs = []
for feat_label, col_name in matched_features:
    clean_name = feat_label.replace("dist_", "").replace("_sma3nz_amean", "")
    low_col = col_name.lower()
    
    is_f0 = "f0" in low_col or "semitone" in low_col
    is_formant = any(f in low_col for f in ["f1frequency", "f2frequency", "f3frequency", "f1", "f2", "f3"]) and "norm" not in low_col
    
    if is_f0:
        view_configs.append({
            "dropdown_label": f"{clean_name} [Hz]",
            "col_name": col_name,
            "mode": "f0_hz",
            "x_axis_title": f"Δ {clean_name} (Hertz [Hz])",
            "title_feat": f"Δ {clean_name} [Hz]"
        })
        view_configs.append({
            "dropdown_label": f"{clean_name} [Semitones (Log)]",
            "col_name": col_name,
            "mode": "raw",
            "x_axis_title": f"Δ {clean_name} (Semitones [st] - Log Scale)",
            "title_feat": f"Δ {clean_name} [Semitones]"
        })
    elif is_formant:
        view_configs.append({
            "dropdown_label": f"{clean_name} [Hz]",
            "col_name": col_name,
            "mode": "raw",
            "x_axis_title": f"Δ {clean_name} (Hertz [Hz])",
            "title_feat": f"Δ {clean_name} [Hz]"
        })
        view_configs.append({
            "dropdown_label": f"{clean_name} [Log (Semitones)]",
            "col_name": col_name,
            "mode": "formant_log",
            "x_axis_title": f"Δ {clean_name} (Semitones [st] - Log Scale)",
            "title_feat": f"Δ {clean_name} [Log Semitones]"
        })
    else:
        unit_str = "Ratio" if "ratio" in low_col else ("dB" if "slope" in low_col else "Distance L1")
        view_configs.append({
            "dropdown_label": clean_name,
            "col_name": col_name,
            "mode": "raw",
            "x_axis_title": f"Δ {clean_name} ({unit_str})",
            "title_feat": f"Δ {clean_name}"
        })

def get_x_data(cfg):
    col = cfg["col_name"]
    raw_series = df_test[col].copy()
    if cfg["mode"] == "f0_hz":
        vals = 150.0 * (2.0 ** (raw_series / 12.0) - 1.0)
    elif cfg["mode"] == "formant_log":
        vals = 12.0 * np.log2(1.0 + np.maximum(0, raw_series) / 500.0)
    else:
        vals = raw_series
    return vals.to_dict()

# -------------------------------------------------------------------------
# 5. Construction de la Figure Plotly
# -------------------------------------------------------------------------
fig = go.Figure()
model_names = list(df_models_sisdri.keys())
num_models = len(model_names)
num_views = len(view_configs)
METRICS = ["SI-SDRi", "SDRi"]

traces_fit_indices = []
trace_counter = 0

for metric_idx, metric_name in enumerate(METRICS):
    models_dict = df_models_sisdri if metric_name == "SI-SDRi" else df_models_sdri
    
    for v_idx, cfg in enumerate(view_configs):
        vals_dict = get_x_data(cfg)
        is_visible = (metric_idx == 0 and v_idx == 0)
        
        # 1. Nuages de points avec Médiane ET Moyenne dans la légende
        for m_name in model_names:
            sdr_dict = models_dict[m_name]
            common = [p for p in sdr_dict if p in vals_dict and pd.notna(vals_dict[p])]
            
            x_vals = [vals_dict[p] for p in common]
            y_vals = [sdr_dict[p] for p in common]
            
            if y_vals:
                mean_val = np.mean(y_vals)
                median_val = np.median(y_vals)
                legend_name = f"{m_name} (med: {median_val:.2f} | mean: {mean_val:.2f} dB)"
            else:
                legend_name = f"{m_name} (N/A)"
            
            unit = "Hz" if "hz" in cfg["mode"] else ("st" if "log" in cfg["mode"] or "semitone" in cfg["mode"] else "val")
            text_hover = [
                f"Pair: {p}<br>X: {x:.2f} {unit}<br>{metric_name}: {y:.2f} dB"
                for p, x, y in zip(common, x_vals, y_vals)
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    name=legend_name,
                    marker=dict(
                        color=MODELS[m_name]["color"],
                        size=5,
                        opacity=0.45
                    ),
                    text=text_hover,
                    hoverinfo="text",
                    visible=is_visible
                )
            )
            trace_counter += 1

        # 2. Lignes de Fit Linéaire (filtrage strict des NaN/Inf)
        for m_name in model_names:
            sdr_dict = models_dict[m_name]
            common = [p for p in sdr_dict if p in vals_dict and pd.notna(vals_dict[p])]
            x_vals = np.array([vals_dict[p] for p in common], dtype=np.float64)
            y_vals = np.array([sdr_dict[p] for p in common], dtype=np.float64)
            
            valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            x_clean = x_vals[valid_mask]
            y_clean = y_vals[valid_mask]

            x_line, y_line = [], []
            fit_label = f"{m_name} Fit: N/A"

            if len(x_clean) > 5 and np.ptp(x_clean) > 1e-6:
                try:
                    a, b = np.polyfit(x_clean, y_clean, 1)
                    x_line = np.linspace(np.min(x_clean), np.max(x_clean), 100)
                    y_line = a * x_line + b
                    fit_label = f"{m_name} Fit: {a:.3f}x + {b:.2f}"
                except Exception:
                    x_line, y_line = [], []

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=fit_label,
                    line=dict(color=MODELS[m_name]["color"], width=3, dash="solid"),
                    hoverinfo="name",
                    visible=is_visible
                )
            )
            traces_fit_indices.append(trace_counter)
            trace_counter += 1

traces_per_view = 2 * num_models
total_traces = len(fig.data)

# -------------------------------------------------------------------------
# 6. Menus Interactifs (Maintien robuste des titres et polices 16pt)
# -------------------------------------------------------------------------
feature_buttons = []
for v_idx, cfg in enumerate(view_configs):
    vis_mask = [False] * total_traces
    start = v_idx * traces_per_view
    for t in range(start, start + traces_per_view):
        vis_mask[t] = True
        
    feature_buttons.append(
        dict(
            label=cfg["dropdown_label"],
            method="update",
            args=[
                {"visible": vis_mask},
                {
                    "title.text": f"<b>SI-SDRi vs {cfg['title_feat']} (EchoSet Cleaned Small)</b>",
                    "xaxis.title.text": f"<b>{cfg['x_axis_title']}</b>",
                    "yaxis.title.text": "<b>SI-SDRi (dB) [Separation Improvement]</b>"
                }
            ]
        )
    )

toggle_fit_buttons = [
    dict(label="Show Fit Lines", method="restyle", args=[{"visible": True}, traces_fit_indices]),
    dict(label="Hide Fit Lines", method="restyle", args=[{"visible": False}, traces_fit_indices])
]

metric_buttons = []
half_total = total_traces // 2
for m_idx, m_name in enumerate(METRICS):
    vis_mask = [False] * total_traces
    offset = m_idx * half_total
    for t in range(offset, offset + traces_per_view):
        vis_mask[t] = True
    metric_buttons.append(
        dict(
            label=m_name,
            method="update",
            args=[
                {"visible": vis_mask},
                {
                    "yaxis.title.text": f"<b>{m_name} (dB) [Separation Improvement]</b>"
                }
            ]
        )
    )

first_cfg = view_configs[0]

fig.update_layout(
    title=dict(
        text=f"<b>SI-SDRi vs {first_cfg['title_feat']} (EchoSet Cleaned Small)</b>",
        font=dict(size=18, family="Arial", color="black"),
        x=0.02,
        y=0.98
    ),
    xaxis=dict(
        title=f"<b>{first_cfg['x_axis_title']}</b>",
        title_font=dict(size=16, family="Arial", color="black"),
        tickfont=dict(size=13),
        gridcolor="#e5e5e5"
    ),
    yaxis=dict(
        title="<b>SI-SDRi (dB) [Separation Improvement]</b>",
        title_font=dict(size=16, family="Arial", color="black"),
        tickfont=dict(size=13),
        gridcolor="#e5e5e5"
    ),
    template="plotly_white",
    hovermode="closest",
    updatemenus=[
        # 1. Menu déroulant des Features placé dans la colonne latérale droite
        dict(
            active=0,
            buttons=feature_buttons,
            direction="down",
            showactive=True,
            x=1.02,
            xanchor="left",
            y=0.90,
            yanchor="top",
            font=dict(size=12)
        ),
        # 2. Boutons Fit Lines : placés aussi dans la colonne latérale droite sous le dropdown
        dict(
            type="buttons",
            direction="down",
            active=0,
            buttons=toggle_fit_buttons,
            x=1.02,
            xanchor="left",
            y=0.62,
            yanchor="top",
            font=dict(size=12)
        ),
        # 3. Boutons Metric (SI-SDRi / SDRi) : placés dans la colonne latérale droite sous le Fit
        dict(
            type="buttons",
            direction="down",
            active=0,
            buttons=metric_buttons,
            x=1.02,
            xanchor="left",
            y=0.45,
            yanchor="top",
            font=dict(size=12)
        )
    ],
    # Légende en haut : 100% libre et dégagée sur toute la largeur
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.0,
        font=dict(size=13)
    ),
    margin=dict(l=80, r=280, t=140, b=80),  # r=280 pour tous les contrôles à droite, t=140 pour le titre et les 2 lignes de légendes
    width=1450,
    height=800
)

OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(OUTPUT_HTML))
print(f"✅ Scatter plot interactif généré avec succès dans : {OUTPUT_HTML}")

