import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------------------------------------------------
# 1. Paths and Setup
# -------------------------------------------------------------------------
BASE_DIR = Path("/idiap/home/adubois/project_ma1/TIGER")
PARQUET_TEST = Path("/idiap/temp/adubois/EchoSet_cleaned_small/egemaps_summary_test.parquet")
EXP_DIR = BASE_DIR / "Experiments/checkpoint"
OUTPUT_HTML = BASE_DIR / "testing_models/week10_echosetcleaned_small/f0_MOE/matrix_plot.html"

# The 6 models in order: Quintiles 1 to 5 + Uniform Control
MODEL_CONFIGS = [
    ("Train Quintile 1", EXP_DIR / "TIGER-EchoSet_cleaned_small-quintile1_3_gpus_X_11_11_snr"),
    ("Train Quintile 2", EXP_DIR / "TIGER-EchoSet_cleaned_small-quintile2_3_gpus_X_11_11_snr"),
    ("Train Quintile 3", EXP_DIR / "TIGER-EchoSet_cleaned_small-quintile3_3_gpus_X_11_11_snr_fixed"),
    ("Train Quintile 4", EXP_DIR / "TIGER-EchoSet_cleaned_small-quintile4_3_gpus_X_11_11_snr"),
    ("Train Quintile 5", EXP_DIR / "TIGER-EchoSet_cleaned_small-quintile5_3_gpus_X_11_11_snr"),
    ("Uniform sample", EXP_DIR / "TIGER-EchoSet_cleaned_small-control_uniform_810_3_gpus_X"),
]

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
# 2. Partitioning Test Set into 5 Quintiles based on Delta F0
# -------------------------------------------------------------------------
print(f"--> Reading test parquet: {PARQUET_TEST}")
df_test = pd.read_parquet(PARQUET_TEST)
pair_col = [c for c in df_test.columns if any(k in c.lower() for k in ["pair", "key", "utt"])][0]
df_test["clean_id"] = df_test[pair_col].apply(clean_pair_key)
df_test = df_test.drop_duplicates(subset=["clean_id"]).set_index("clean_id")

f0_col = "dist_F0semitoneFrom27.5Hz_sma3nz_amean"
if f0_col not in df_test.columns:
    f0_cols = [c for c in df_test.columns if "f0" in c.lower() and "amean" in c.lower()]
    f0_col = f0_cols[0]

print(f"--> Ranking test set using F0 feature: {f0_col}")
df_test_sorted = df_test.sort_values(by=f0_col)
total_test_pairs = len(df_test_sorted)
print(f"--> Total test pairs: {total_test_pairs}")

test_subsets_pairs = {}
column_labels = []
chunk_size = total_test_pairs // 5

for j in range(5):
    start_idx = j * chunk_size
    end_idx = (j + 1) * chunk_size if j < 4 else total_test_pairs
    subset = df_test_sorted.iloc[start_idx:end_idx]
    
    min_st = subset[f0_col].min()
    max_st = subset[f0_col].max()
    min_hz = 150.0 * (2.0 ** (min_st / 12.0) - 1.0)
    max_hz = 150.0 * (2.0 ** (max_st / 12.0) - 1.0)
    
    label = f"Test Q{j+1}<br>[{min_hz:.1f}-{max_hz:.1f} Hz]<br>({min_st:.2f}-{max_st:.2f} st)"
    column_labels.append(label)
    test_subsets_pairs[j] = set(subset.index)

# Colonne 6 : Overall Test Set (toutes les 2650 paires)
column_labels.append("<b>Overall Test Set</b><br>(All 2650 pairs)")
test_subsets_pairs[5] = set(df_test_sorted.index)

# -------------------------------------------------------------------------
# 3. Computing the 4 Matrices (Mean / Median for SI-SDRi and SDRi)
# -------------------------------------------------------------------------
n_models = len(MODEL_CONFIGS)
num_cols = len(column_labels)  # 6 colonnes : Q1 à Q5 + Overall

mat_sisdri_mean = np.zeros((n_models, num_cols))
mat_sisdri_med = np.zeros((n_models, num_cols))
mat_sdri_mean = np.zeros((n_models, num_cols))
mat_sdri_med = np.zeros((n_models, num_cols))

for i, (m_label, m_path) in enumerate(MODEL_CONFIGS):
    csv_file = m_path / "results/metrics.csv"
    if not csv_file.exists():
        print(f"[Warning] metrics.csv missing for {m_label} in {m_path}")
        continue
    
    df_metrics = pd.read_csv(csv_file)
    key_col = "key" if "key" in df_metrics.columns else df_metrics.columns[0]
    df_metrics["clean_id"] = df_metrics[key_col].apply(clean_pair_key)
    df_metrics = df_metrics.drop_duplicates(subset=["clean_id"]).set_index("clean_id")
    
    sisdri_col = [c for c in df_metrics.columns if any(k in c.lower() for k in ["si-snr_i", "si_sdr_i", "si_sdri", "sisdr_i", "sisdri"])]
    sisdri_col = sisdri_col[0] if sisdri_col else ("si-snr" if "si-snr" in df_metrics.columns else df_metrics.columns[4])
    
    sdri_col = [c for c in df_metrics.columns if any(k in c.lower() for k in ["sdr_i", "sdri"])]
    sdri_col = sdri_col[0] if sdri_col else df_metrics.columns[2]
    
    for j in range(num_cols):
        common = [p for p in test_subsets_pairs[j] if p in df_metrics.index]
        if common:
            mat_sisdri_mean[i, j] = df_metrics.loc[common, sisdri_col].mean()
            mat_sisdri_med[i, j] = df_metrics.loc[common, sisdri_col].median()
            mat_sdri_mean[i, j] = df_metrics.loc[common, sdri_col].mean()
            mat_sdri_med[i, j] = df_metrics.loc[common, sdri_col].median()
        else:
            mat_sisdri_mean[i, j] = mat_sisdri_med[i, j] = np.nan
            mat_sdri_mean[i, j] = mat_sdri_med[i, j] = np.nan

model_y_labels = [m[0] for m in MODEL_CONFIGS]

# -------------------------------------------------------------------------
# 4. Annotations Format : Best in Bold, Second Best in Italic (per column)
# -------------------------------------------------------------------------
def make_annotations(matrix):
    annot = []
    best_rows = {}
    second_rows = {}
    
    for j in range(matrix.shape[1]):
        col_vals = matrix[:, j]
        valid_indices = [idx for idx in range(len(col_vals)) if not np.isnan(col_vals[idx])]
        sorted_indices = sorted(valid_indices, key=lambda idx: col_vals[idx], reverse=True)
        
        best_rows[j] = sorted_indices[0] if len(sorted_indices) > 0 else None
        second_rows[j] = sorted_indices[1] if len(sorted_indices) > 1 else None

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            formatted_val = f"{val:.2f} dB"
            if i == best_rows[j]:
                text = f"<b>{formatted_val}</b>"
                font_dict = dict(color="#000000", size=15, family="Arial")
            elif i == second_rows[j]:
                text = f"<i>{formatted_val}</i>"
                font_dict = dict(color="#333333", size=14, family="Arial")
            else:
                text = f"{formatted_val}"
                font_dict = dict(color="#555555", size=13, family="Arial")

            annot.append(
                dict(
                    x=column_labels[j],
                    y=model_y_labels[i],
                    text=text,
                    showarrow=False,
                    font=font_dict
                )
            )
    return annot

# -------------------------------------------------------------------------
# 5. Clean Black & White Matrix Plot
# -------------------------------------------------------------------------
fig = go.Figure()

white_background = np.zeros_like(mat_sisdri_med)
white_colorscale = [[0.0, "#ffffff"], [1.0, "#ffffff"]]

traces_data = [
    (mat_sisdri_med, "SI-SDRi Median", True),
    (mat_sisdri_mean, "SI-SDRi Mean", False),
    (mat_sdri_med, "SDRi Median", False),
    (mat_sdri_mean, "SDRi Mean", False),
]

all_annotations = []
for z_mat, name, vis in traces_data:
    all_annotations.append(make_annotations(z_mat))
    custom_hover = np.empty((n_models, num_cols), dtype=object)
    for i in range(n_models):
        for j in range(num_cols):
            custom_hover[i, j] = f"Model: {model_y_labels[i]}<br>Subset: {column_labels[j].replace('<br>', ' ')}<br>{name}: {z_mat[i, j]:.2f} dB"
            
    fig.add_trace(
        go.Heatmap(
            z=white_background,
            x=column_labels,
            y=model_y_labels,
            colorscale=white_colorscale,
            showscale=False,
            hoverongaps=False,
            hoverinfo="text",
            text=custom_hover,
            name=name,
            visible=vis
        )
    )

# -------------------------------------------------------------------------
# 6. Interactive Menus (Statistic: Median / Mean & Metric: SI-SDRi / SDRi)
# -------------------------------------------------------------------------
stat_buttons = [
    dict(
        label="Median",
        method="update",
        args=[
            {"visible": [True, False, False, False]},
            {
                "title.text": "<b>Cross-Performance Matrix: MEDIAN SI-SDRi (with Overall Test Set)</b><br><span style='font-size:12px;color:#555;'>Best per column in <b>bold</b>, second best in <i>italic</i></span>",
                "annotations": all_annotations[0]
            }
        ]
    ),
    dict(
        label="Mean",
        method="update",
        args=[
            {"visible": [False, True, False, False]},
            {
                "title.text": "<b>Cross-Performance Matrix: MEAN SI-SDRi (with Overall Test Set)</b><br><span style='font-size:12px;color:#555;'>Best per column in <b>bold</b>, second best in <i>italic</i></span>",
                "annotations": all_annotations[1]
            }
        ]
    )
]

metric_buttons = [
    dict(
        label="SI-SDRi",
        method="update",
        args=[
            {"visible": [True, False, False, False]},
            {
                "title.text": "<b>Cross-Performance Matrix: MEDIAN SI-SDRi (with Overall Test Set)</b><br><span style='font-size:12px;color:#555;'>Best per column in <b>bold</b>, second best in <i>italic</i></span>",
                "annotations": all_annotations[0]
            }
        ]
    ),
    dict(
        label="SDRi",
        method="update",
        args=[
            {"visible": [False, False, True, False]},
            {
                "title.text": "<b>Cross-Performance Matrix: MEDIAN SDRi (with Overall Test Set)</b><br><span style='font-size:12px;color:#555;'>Best per column in <b>bold</b>, second best in <i>italic</i></span>",
                "annotations": all_annotations[2]
            }
        ]
    )
]

fig.update_layout(
    title=dict(
        text="<b>Cross-Performance Matrix: MEDIAN SI-SDRi (with Overall Test Set)</b><br><span style='font-size:12px;color:#555;'>Best per column in <b>bold</b>, second best in <i>italic</i></span>",
        font=dict(size=18, family="Arial", color="black"),
        x=0.05,
        y=0.96
    ),
    xaxis=dict(
        title="<b>Test Set Evaluation Subsets (Ranked by increasing acoustic distance ΔF0)</b>",
        title_font=dict(size=15, family="Arial", color="black"),
        tickfont=dict(size=12, color="black"),
        side="bottom",
        showgrid=True,
        gridcolor="#e0e0e0",
        gridwidth=1.5,
        linecolor="#333333",
        linewidth=1.5,
        mirror=True
    ),
    yaxis=dict(
        title="<b>Training Models</b>",
        title_font=dict(size=15, family="Arial", color="black"),
        tickfont=dict(size=13, color="black"),
        autorange="reversed",
        showgrid=True,
        gridcolor="#e0e0e0",
        gridwidth=1.5,
        linecolor="#333333",
        linewidth=1.5,
        mirror=True
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    template="plotly_white",
    annotations=all_annotations[0],
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            buttons=stat_buttons,
            x=0.05,
            xanchor="left",
            y=1.14,
            yanchor="top",
            font=dict(size=13)
        ),
        dict(
            type="buttons",
            direction="right",
            active=0,
            buttons=metric_buttons,
            x=0.35,
            xanchor="left",
            y=1.14,
            yanchor="top",
            font=dict(size=13)
        )
    ],
    margin=dict(l=220, r=60, t=140, b=100),
    width=1300,
    height=650
)

OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(OUTPUT_HTML))
print(f"✅ Clean Black & White Matrix Plot with Overall column saved to: {OUTPUT_HTML}")
