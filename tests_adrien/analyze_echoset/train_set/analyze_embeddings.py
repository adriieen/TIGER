#!/usr/bin/env python3
"""
Vérification du nombre d'extraits par locuteur dans le dossier embeddings
vs le CSV train-clean-100_egemaps_speakers.csv.
"""

import os
from collections import Counter
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/idiap/home/adubois/project_ma1/TIGER")
EMBEDDINGS_DIR = PROJECT_ROOT / "tests_adrien/analyze_librispeech/train-clean-100/embeddings/egemaps"
CSV_PATH = PROJECT_ROOT / "tests_adrien/analyze_librispeech/train-clean-100/train-clean-100_egemaps_speakers.csv"

# 1. Compte des fichiers .npy par locuteur
files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy")]
spk_counts_emb = Counter()

for fname in files:
    try:
        spk_id = int(fname.split("-")[0])
        spk_counts_emb[spk_id] += 1
    except ValueError:
        pass

# 2. Chargement du CSV de référence
df_spk = pd.read_csv(CSV_PATH)

# 3. Tableau comparatif
rows = []
for _, row in df_spk.iterrows():
    spk_id = int(row["speaker_id"])
    gender = row.get("gender", "U")
    expected_csv = int(row["num_utterances"])
    found_emb = spk_counts_emb.get(spk_id, 0)
    diff = found_emb - expected_csv

    rows.append({
        "speaker_id": spk_id,
        "gender": gender,
        "n_utterances_embeddings": found_emb,
        "num_utterances_csv": expected_csv,
        "diff": diff,
    })

df_comp = pd.DataFrame(rows).sort_values(by="speaker_id")

print("=" * 80)
print(f"Total fichiers .npy trouvés : {len(files):,}")
print(f"Total locuteurs dans le CSV : {len(df_spk)}")
print(f"Total locuteurs trouvés dans les embeddings : {len(spk_counts_emb)}")
print("=" * 80)

# Affichage des 30 premiers locuteurs (ou de tous les locuteurs)
print(df_comp.to_string(index=False))

# Vérification d'anomalies
discrepancies = df_comp[df_comp["diff"] != 0]
if len(discrepancies) > 0:
    print("\n" + "!" * 80)
    print(f"ATTENTION : {len(discrepancies)} locuteurs ont un nombre d'extraits différent entre le CSV et les embeddings :")
    print(discrepancies.to_string(index=False))
    print("!" * 80)
else:
    print("\n Parfait : 100% des 251 locuteurs ont exactement le bon nombre d'extraits (diff = 0 partout) !")
