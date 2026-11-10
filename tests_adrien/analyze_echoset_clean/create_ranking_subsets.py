#!/usr/bin/env python3
"""
create_ranking_subsets.py

Génère des listes de données ordonnées (rankings complets) pour chaque feature acoustique.
Pour chaque feature spécifiée :
- Toutes les paires du dataset sont triées par distance L1 croissante (du plus proche au plus éloigné).
- Le triplet TIGER (mix.json, s1.json, s2.json) est généré dans :
  [out_base_dir]/[feature_name]/

Ainsi, au moment du training, TIGER peut trancher à la volée le percentile voulu
(ex: les 10% premiers = les plus proches, les 10% derniers = les plus éloignés),
sans avoir à régénérer de fichiers.
"""

import os
import json
import argparse
import pandas as pd


def create_feature_ranking(df, feature_name, out_base_dir):
    col = f"dist_{feature_name}"
    if col not in df.columns:
        raise ValueError(
            f"Feature '{col}' non trouvée dans le résumé parquet.\n"
            f"Exemples de colonnes disponibles : {[c for c in df.columns if c.startswith('dist_')][:5]}..."
        )

    # Tri par distance croissante (index 0 = plus proche, index -1 = plus éloigné)
    df_sorted = df.sort_values(by=col, ascending=True)

    out_dir = os.path.join(out_base_dir, feature_name)
    os.makedirs(out_dir, exist_ok=True)

    mix_infos = []
    s1_infos = []
    s2_infos = []

    # audio_base_dir = "/idiap/temp/adubois/EchoSet_cleaned_small/train"
    audio_base_dir = "/idiap/temp/adubois/EchoSet_cleaned_small/test"
    for _, row in df_sorted.iterrows():
        # Reconstruire le chemin absolu à partir de pair_id (ex: "00/19_150")
        pair_dir = os.path.join(audio_base_dir, str(row["pair_id"]))
        mix_infos.append([os.path.join(pair_dir, "mix.wav"), 96000])
        s1_infos.append([os.path.join(pair_dir, "s1.wav"), 96000])
        s2_infos.append([os.path.join(pair_dir, "s2.wav"), 96000])


    with open(os.path.join(out_dir, "mix.json"), "w") as f:
        json.dump(mix_infos, f, indent=4)
    with open(os.path.join(out_dir, "s1.json"), "w") as f:
        json.dump(s1_infos, f, indent=4)
    with open(os.path.join(out_dir, "s2.json"), "w") as f:
        json.dump(s2_infos, f, indent=4)

    min_dist = df_sorted[col].iloc[0]
    max_dist = df_sorted[col].iloc[-1]
    median_dist = df_sorted[col].median()

    print(f"Ranking '{feature_name}' créé avec succès dans {out_dir}")
    print(f"  -> Total paires: {len(mix_infos)} | Distance L1 min: {min_dist:.4f}, median: {median_dist:.4f}, max: {max_dist:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Création de classements ordonnés complets (rankings) par feature pour TIGER"
    )
    parser.add_argument(
        "--summary_parquet",
        type=str,
        default="/idiap/temp/adubois/EchoSet_cleaned_small/egemaps_summary_test.parquet",
        help="Chemin vers le fichier summary parquet de eGeMAPS",
    )
    parser.add_argument(
        "--feature_name",
        type=str,
        default=None,
        help="Nom d'une seule feature à classer",
    )
    parser.add_argument(
        "--features_file",
        type=str,
        default=None,
        help="Fichier texte listant les features (ex: top20_features.txt)",
    )
    parser.add_argument(
        "--out_base_dir",
        type=str,
        default="/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_cleaned_small/test_rankings",
        help="Dossier de sortie racine pour les rankings ordonnés",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.summary_parquet):
        raise FileNotFoundError(f"Fichier parquet introuvable : {args.summary_parquet}")

    print(f"Chargement du résumé parquet : {args.summary_parquet}...")
    df = pd.read_parquet(args.summary_parquet)
    print(f"Total d'échantillons chargés : {len(df)}")

    # Collecter la liste des features à traiter
    features_to_process = []
    if args.feature_name:
        features_to_process.append(args.feature_name)

    if args.features_file:
        with open(args.features_file, "r") as f:
            for line in f:
                feat = line.strip()
                if feat and not feat.startswith("#"):
                    features_to_process.append(feat)

    if not features_to_process:
        raise ValueError("Veuillez spécifier --feature_name ou --features_file.")

    # Dédupliquer tout en gardant l'ordre
    seen = set()
    features = [x for x in features_to_process if not (x in seen or seen.add(x))]

    print(f"\nGénération de {len(features)} rankings ordonnés dans : {args.out_base_dir}\n")

    for idx, feat in enumerate(features, 1):
        print(f"[{idx}/{len(features)}] Traitement de : {feat}")
        create_feature_ranking(df, feat, args.out_base_dir)

    print("\nTous les rankings ordonnés ont été générés avec succès !")


if __name__ == "__main__":
    main()
