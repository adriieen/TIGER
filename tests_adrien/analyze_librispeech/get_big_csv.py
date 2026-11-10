import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import opensmile


def aggregate_speaker_features(
    embeddings_dir: str,
    output_csv_path: str,
    speakers_txt_path: str = None
):
    """
    Regroupe les embeddings eGeMAPS par locuteur, calcule la moyenne arithmétique
    sur l'ensemble de ses extraits et sauvegarde le résultat dans un fichier CSV.
    """
    if not os.path.isdir(embeddings_dir):
        raise FileNotFoundError(f"Le dossier spécifié n'existe pas : {embeddings_dir}")

    # 1. Récupération des 88 noms officiels des features eGeMAPS
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    feature_names = smile.feature_names

    # 2. Indexation des fichiers par locuteur (format: <speaker_id>-<chapter_id>-<utterance_id>.npy)
    speaker_files = defaultdict(list)
    for fname in os.listdir(embeddings_dir):
        if fname.endswith(".npy"):
            speaker_id = fname.split("-")[0]
            speaker_files[speaker_id].append(os.path.join(embeddings_dir, fname))

    print(f"Trouvé {len(speaker_files)} locuteurs uniques pour {sum(len(v) for v in speaker_files.values())} extraits.")

    # 3. (Optionnel) Chargement des métadonnées (ex: genre M/F) depuis SPEAKERS.txt
    speaker_metadata = {}
    if speakers_txt_path and os.path.exists(speakers_txt_path):
        print(f"Chargement des métadonnées depuis {speakers_txt_path}...")
        with open(speakers_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    sid = parts[0]
                    gender = parts[1]
                    speaker_metadata[sid] = {"gender": gender}

    # 4. Calcul de la moyenne arithmétique pour chaque locuteur
    records = []
    for speaker_id, file_paths in sorted(speaker_files.items()):
        # Charger tous les tableaux pour ce locuteur (chacun de forme (1, 88) ou (88,))
        arrays = [np.load(fp).squeeze() for fp in file_paths]
        stacked = np.stack(arrays, axis=0)  # forme (N_utterances, 88)
        mean_features = np.mean(stacked, axis=0)  # forme (88,)

        record = {
            "speaker_id": speaker_id,
            "num_utterances": len(file_paths),
        }
        if speaker_id in speaker_metadata:
            record["gender"] = speaker_metadata[speaker_id]["gender"]

        # Ajout des 88 features
        for name, val in zip(feature_names, mean_features):
            record[name] = val

        records.append(record)

    # 5. Création du DataFrame et export en CSV
    df = pd.DataFrame(records)
    
    # Création du dossier parent si besoin
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Grand CSV enregistré avec succès dans : {output_csv_path}")
    print(f"Dimensions du tableau : {df.shape} (locuteurs: {df.shape[0]}, colonnes: {df.shape[1]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agrégation des features eGeMAPS par locuteur")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="/idiap/home/adubois/project_ma1/TIGER/tests_adrien/analyze_librispeech/train-clean-100/embeddings/egemaps",
        help="Dossier contenant les fichiers .npy des embeddings eGeMAPS"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/idiap/home/adubois/project_ma1/TIGER/tests_adrien/analyze_librispeech/train-clean-100/train-clean-100_egemaps_speakers.csv",
        help="Chemin de sauvegarde du fichier CSV de sortie"
    )
    parser.add_argument(
        "--speakers_txt",
        type=str,
        default="/idiap/home/adubois/project_ma1/TIGER/tests_adrien/SPEAKERS.txt",
        help="Chemin vers le fichier SPEAKERS.txt de LibriSpeech (pour ajouter le genre M/F si disponible)"
    )

    args = parser.parse_args()
    aggregate_speaker_features(args.embeddings_dir, args.output_csv, args.speakers_txt)
