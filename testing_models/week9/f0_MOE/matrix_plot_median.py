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
PARQUET_TEST = Path("/idiap/temp/adubois/EchoSet_cleaned/egemaps_summary_test.parquet")
EXP_DIR = BASE_DIR / "Experiments/checkpoint"
OUTPUT_HTML = BASE_DIR / "testing_models/f0_MOE/matrix_plot.html"

MODEL_DIRS = []
for d in range(1, 6):
    pattern = f"OGLR-TIGER-EchoSet-decile{d}_f0-reduced"
    matches = sorted(list(EXP_DIR.glob(f"{pattern}*")))
    if matches:
        MODEL_DIRS.append((f"Train Decile {d}", matches[0]))

if not MODEL_DIRS:
    raise FileNotFoundError(f"No models found in {EXP_DIR}")

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
# 2. Partitioning Test Set into 10 Deciles
# -------------------------------------------------------------------------
df_test = pd.read_parquet(PARQUET_TEST)
pair_col = [c for c in df_test.columns if any(k in c.lower() for k in ["pair", "key", "utt"])][0]
df_test["clean_id"] = df_test[pair_col].apply(clean_pair_key)
df_test = df_test.drop_duplicates(subset=["clean_id"]).set_index("clean_id")

f0_col = [c for c in df_test.columns if "f0" in c.lower() and "amean" in c.lower()][0]
df_test_sorted = df_test.sort_values(by=f0_col)
total_test_pairs = len(df_test_sorted)

test_deciles_pairs = {}
decile_labels = []
chunk_size = total_test_pairs // 10

for j in range(10):
    start_idx = j * chunk_size
    end_idx = (j + 1) * chunk_size if j < 9 else total_test_pairs
    subset = df_test_sorted.iloc[start_idx:end_idx]
    
    min_st = subset[f0_col].min()
    max_st = subset[f0_col].max()
    min_hz = 150.0 * (2.0 ** (min_st / 12.0) - 1.0)
    max_hz = 150.0 * (2.0 ** (max_st / 12.0) - 1.0)
    
    decile_labels.append(f"Test D{j+1}<br>[{min_hz:.1f}-{max_hz:.1f} Hz]")
    test_deciles_pairs[j] = set(subset.index)

# -------------------------------------------------------------------------
# 3. Computing the 4 Matrices (Mean / Median for SI-SDRi and SDRi)
# -------------------------------------------------------------------------
n_models = len(MODEL_DIRS)
mat_sisdri_mean = np.zeros((n_models, 10))
mat_sisdri_med = np.zeros((n_models, 10))
mat_sdri_mean = np.zeros((n_models, 10))
mat_sdri_med = np.zeros((n_models, 10))

for i, (m_label, m_path) in enumerate(MODEL_DIRS):
    csv_file = m_path / "results/metrics.csv"
    if not csv_file.exists():
        continue
    
    df_metrics = pd.read_csv(csv_file)
    key_col = "key" if "key" in df_metrics.columns else df_metrics.columns[0]
    df_metrics["clean_id"] = df_metrics[key_col].apply(clean_pair_key)
    df_metrics = df_metrics.drop_duplicates(subset=["clean_id"]).set_index("clean_id")
    
    sisdri_col = [c for c in df_metrics.columns if c in ["si_sdr_i", "si_sdri", "sisdr_i", "sisdri"]]
    sisdri_col = sisdri_col[0] if sisdri_col else ("si_sdr" if "si_sdr" in df_metrics.columns else df_metrics.columns[1])
    
    sdri_col = [c for c in df_metrics.columns if c in ["sdr_i", "sdri"]]
    sdri_col = sdri_col[0] if sdri_col else ("sdr" if "sdr" in df_metrics.columns else sisdri_col)
    
    for j in range(10):
        common = [p for p in test_deciles_pairs[j] if p in df_metrics.index]
        if common:
            mat_sisdri_mean[i, j] = df_metrics.loc[common, sisdri_col].mean()
            mat_sisdri_med[i, j] = df_metrics.loc[common, sisdri_col].median()
            mat_sdri_mean[i, j] = df_metrics.loc[common, sdri_col].mean()
            mat_sdri_med[i, j] = df_metrics.loc[common, sdri_col].median()
        else:
            mat_sisdri_mean[i, j] = mat_sisdri_med[i, j] = np.nan
            mat_sdri_mean[i, j] = mat_sdri_med[i, j] = np.nan

model_y_labels = [m[0] for m in MODEL_DIRS]

# -------------------------------------------------------------------------
# 4. Interactive Plotly Heatmaps
# -------------------------------------------------------------------------
def make_annotations(matrix):
    annot = []
    mean_val = np.nanmean(matrix)
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            annot.append(
                dict(
                    x=decile_labels[j],
                    y=model_y_labels[i],
                    text=f"<b>{val:.2f} dB</b>",
                    showarrow=False,
                    font=dict(color="white" if val < mean_val else "black", size=13, family="Arial")
                )
            )
    return annot

fig = go.Figure()

# 4 Traces: [0] SI-SDRi Median (default), [1] SI-SDRi Mean, [2] SDRi Median, [3] SDRi Mean
traces_data = [
    (mat_sisdri_med, "Viridis", "SI-SDRi Median", True),
    (mat_sisdri_mean, "Viridis", "SI-SDRi Mean", False),
    (mat_sdri_med, "Cividis", "SDRi Median", False),
    (mat_sdri_mean, "Cividis", "SDRi Mean", False),
]

all_annotations = []
for z_mat, cscale, name, vis in traces_data:
    all_annotations.append(make_annotations(z_mat))
    fig.add_trace(
        go.Heatmap(
            z=z_mat,
            x=decile_labels,
            y=model_y_labels,
            colorscale=cscale,
            colorbar=dict(title=dict(text=f"<b>{name} (dB)</b>", side="right")),
            hoverongaps=False,
            name=name,
            visible=vis
        )
    )

# -------------------------------------------------------------------------
# 5. English Interactive Menus (Statistic & Metric)
# -------------------------------------------------------------------------
stat_buttons = [
    dict(
        label="Median",
        method="update",
        args=[
            {"visible": [True, False, False, False]},
            {
                "title.text": "<b>Cross-Performance Matrix: MEDIAN SI-SDRi (Model vs Test Decile ΔF0)</b>",
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
                "title.text": "<b>Cross-Performance Matrix: MEAN SI-SDRi (Model vs Test Decile ΔF0)</b>",
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
                "title.text": "<b>Cross-Performance Matrix: MEDIAN SI-SDRi (Model vs Test Decile ΔF0)</b>",
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
                "title.text": "<b>Cross-Performance Matrix: MEDIAN SDRi (Model vs Test Decile ΔF0)</b>",
                "annotations": all_annotations[2]
            }
        ]
    )
]

fig.update_layout(
    title=dict(
        text="<b>Cross-Performance Matrix: MEDIAN SI-SDRi (Model vs Test Decile ΔF0)</b>",
        font=dict(size=18, family="Arial", color="black"),
        x=0.05,
        y=0.96
    ),
    xaxis=dict(
        title="<b>Test Set Deciles (Ranked by increasing acoustic distance ΔF0)</b>",
        title_font=dict(size=15, family="Arial", color="black"),
        tickfont=dict(size=12, color="black"),
        side="bottom"
    ),
    yaxis=dict(
        title="<b>Training Model</b>",
        title_font=dict(size=15, family="Arial", color="black"),
        tickfont=dict(size=13, color="black"),
        autorange="reversed"
    ),
    template="plotly_white",
    annotations=all_annotations[0],
    updatemenus=[
        # Statistic Selector (Median vs Mean)
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
        # Metric Selector (SI-SDRi vs SDRi)
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
    margin=dict(l=150, r=60, t=140, b=100),
    width=1200,
    height=600
)

OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(OUTPUT_HTML))
print(f"✅ English Matrix Plot saved to: {OUTPUT_HTML}")