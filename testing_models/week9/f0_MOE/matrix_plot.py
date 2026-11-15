import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------------------------------------------------
# 1. Chemins et Configuration
# -------------------------------------------------------------------------
BASE_DIR = Path("/idiap/home/adubois/project_ma1/TIGER")
PARQUET_TEST = Path("/idiap/temp/adubois/EchoSet_cleaned/egemaps_summary_test.parquet")
EXP_DIR = BASE_DIR / "Experiments/checkpoint"
OUTPUT_HTML = BASE_DIR / "testing_models/f0_MOE/matrix_plot.html"

# Détection automatique des dossiers des 5 modèles dans Experiments/checkpoint
MODEL_DIRS = []
for d in range(1, 6):
    pattern = f"OGLR-TIGER-EchoSet-decile{d}_f0-reduced"
    matches = sorted(list(EXP_DIR.glob(f"{pattern}*")))
    if matches:
        MODEL_DIRS.append((f"Train Decile {d}", matches[0]))
    else:
        print(f"[Warning] Aucun dossier trouvé pour le décile {d} avec le motif {pattern}")

if not MODEL_DIRS:
    raise FileNotFoundError(f"Aucun dossier de modèle OGLR-TIGER trouvé dans {EXP_DIR}")

print(f"--> {len(MODEL_DIRS)} modèles détectés : {[m[0] for m in MODEL_DIRS]}")

# -------------------------------------------------------------------------
# 2. Fonction de nettoyage des clés de paire
# -------------------------------------------------------------------------
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
# 3. Chargement du Parquet et Découpage en 10 Déciles du Test Set
# -------------------------------------------------------------------------
print(f"--> Lecture du parquet de test : {PARQUET_TEST}")
df_test = pd.read_parquet(PARQUET_TEST)

pair_col = [c for c in df_test.columns if any(k in c.lower() for k in ["pair", "key", "utt"])][0]
df_test["clean_id"] = df_test[pair_col].apply(clean_pair_key)
df_test = df_test.drop_duplicates(subset=["clean_id"]).set_index("clean_id")

# Trouver la colonne Delta F0
f0_cols = [c for c in df_test.columns if "f0" in c.lower() and "amean" in c.lower()]
f0_col = f0_cols[0]
print(f"--> Colonne F0 utilisée pour le ranking de test : {f0_col}")

# Tri des paires par distance F0 croissante
df_test_sorted = df_test.sort_values(by=f0_col)
total_test_pairs = len(df_test_sorted)
print(f"--> {total_test_pairs} paires dans le test set.")

# Découpage en 10 sous-ensembles de cardinal égal
test_deciles_pairs = {}
decile_labels = []

chunk_size = total_test_pairs // 10
for j in range(10):
    start_idx = j * chunk_size
    end_idx = (j + 1) * chunk_size if j < 9 else total_test_pairs
    subset = df_test_sorted.iloc[start_idx:end_idx]
    
    # Valeurs min et max en Hz (conversion relative depuis demi-tons)
    min_st = subset[f0_col].min()
    max_st = subset[f0_col].max()
    min_hz = 150.0 * (2.0 ** (min_st / 12.0) - 1.0)
    max_hz = 150.0 * (2.0 ** (max_st / 12.0) - 1.0)
    
    label = f"Test D{j+1}<br>[{min_hz:.1f}-{max_hz:.1f}Hz]"
    decile_labels.append(label)
    test_deciles_pairs[j] = set(subset.index)

# -------------------------------------------------------------------------
# 4. Chargement des metrics.csv pour chaque modèle
# -------------------------------------------------------------------------
matrix_sisdri = np.zeros((len(MODEL_DIRS), 10))
matrix_sdri = np.zeros((len(MODEL_DIRS), 10))

for i, (m_label, m_path) in enumerate(MODEL_DIRS):
    csv_file = m_path / "results/metrics.csv"
    if not csv_file.exists():
        print(f"[Warning] metrics.csv absent dans {m_path}")
        continue
    
    df_metrics = pd.read_csv(csv_file)
    key_col = "key" if "key" in df_metrics.columns else df_metrics.columns[0]
    df_metrics["clean_id"] = df_metrics[key_col].apply(clean_pair_key)
    df_metrics = df_metrics.drop_duplicates(subset=["clean_id"]).set_index("clean_id")
    
    # Colonnes SI-SDRi et SDRi
    sisdri_col = [c for c in df_metrics.columns if c in ["si_sdr_i", "si_sdri", "sisdr_i", "sisdri"]]
    sisdri_col = sisdri_col[0] if sisdri_col else ("si_sdr" if "si_sdr" in df_metrics.columns else df_metrics.columns[1])
    
    sdri_col = [c for c in df_metrics.columns if c in ["sdr_i", "sdri"]]
    sdri_col = sdri_col[0] if sdri_col else ("sdr" if "sdr" in df_metrics.columns else sisdri_col)
    
    # Calcul de la moyenne pour chaque décile de test j
    for j in range(10):
        target_pairs = test_deciles_pairs[j]
        # Paires communes présentes dans ce décile
        common = [p for p in target_pairs if p in df_metrics.index]
        if common:
            matrix_sisdri[i, j] = df_metrics.loc[common, sisdri_col].mean()
            matrix_sdri[i, j] = df_metrics.loc[common, sdri_col].mean()
        else:
            matrix_sisdri[i, j] = np.nan
            matrix_sdri[i, j] = np.nan

model_y_labels = [m[0] for m in MODEL_DIRS]

# -------------------------------------------------------------------------
# 5. Création de la Heatmap Interactive Plotly
# -------------------------------------------------------------------------
def make_annotations(matrix):
    annotations = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=decile_labels[j],
                    y=model_y_labels[i],
                    text=f"<b>{val:.2f} dB</b>",
                    showarrow=False,
                    font=dict(color="white" if val < np.nanmean(matrix) else "black", size=13, family="Arial")
                )
            )
    return annotations

fig = go.Figure()

# 1. Trace SI-SDRi (active par défaut)
fig.add_trace(
    go.Heatmap(
        z=matrix_sisdri,
        x=decile_labels,
        y=model_y_labels,
        colorscale="Viridis",
        colorbar=dict(title=dict(text="<b>SI-SDRi (dB)</b>", side="right")),        hoverongaps=False,
        name="SI-SDRi",
        visible=True
    )
)

# 2. Trace SDRi (inactive par défaut)
fig.add_trace(
    go.Heatmap(
        z=matrix_sdri,
        x=decile_labels,
        y=model_y_labels,
        colorscale="Cividis",
        colorbar=dict(title=dict(text="<b>SDRi (dB)</b>", side="right")),        hoverongaps=False,
        name="SDRi",
        visible=False
    )
)

annot_sisdri = make_annotations(matrix_sisdri)
annot_sdri = make_annotations(matrix_sdri)

# -------------------------------------------------------------------------
# 6. Boutons de bascule SI-SDRi vs SDRi
# -------------------------------------------------------------------------
buttons = [
    dict(
        label="Afficher SI-SDRi",
        method="update",
        args=[
            {"visible": [True, False]},
            {
                "title": "<b>Matrix of cross-performance : mean SI-SDRi (Model vs Test Decile ΔF0)</b>",

                "annotations": annot_sisdri
            }
        ]
    ),
    dict(
        label="Afficher SDRi",
        method="update",
        args=[
            {"visible": [False, True]},
            {
                "title": "<b>Matrix of cross-performance : mean SDRi (Model vs Test Decile ΔF0)</b>",
                "annotations": annot_sdri
            }
        ]
    )
]

fig.update_layout(
    title=dict(
        text="<b>Matrix of cross-performance : mean SI-SDRi (Model vs Test Decile ΔF0)</b>",

        font=dict(size=18, family="Arial", color="black"),
        x=0.05,
        y=0.96
    ),
    xaxis=dict(
        title="<b>Deciles of the test Set (ordered by increasing acoustic distance ΔF0)</b>",
        title_font=dict(size=15, family="Arial", color="black"),
        tickfont=dict(size=12, color="black"),
        side="bottom"
    ),
    yaxis=dict(
        title="<b>Expert model : </b>",
        title_font=dict(size=15, family="Arial", color="black"),
        tickfont=dict(size=13, color="black"),
        autorange="reversed"  # Affiche le modèle Decile 1 tout en haut
    ),
    template="plotly_white",
    annotations=annot_sisdri,
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            buttons=buttons,
            x=0.05,
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

# -------------------------------------------------------------------------
# 7. Sauvegarde
# -------------------------------------------------------------------------
OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(OUTPUT_HTML))
print(f"\n✅ Matrice interactive générée avec succès dans :")
print(f"   {OUTPUT_HTML}")