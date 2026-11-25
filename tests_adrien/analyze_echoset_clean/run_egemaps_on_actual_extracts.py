#!/usr/bin/env python3
"""
run_egemaps.py

Extrait les descripteurs acoustiques openSMILE (eGeMAPSv02 Functionals = 88 features)
sur les fichiers extraits s1.wav et s2.wav du dataset EchoSet_cleaned.

Pour chaque dossier [split]/[occ]/[spk1]_[spk2]/ :
- s1.wav -> [spk1]_[spk2]__[occ]__[spk1].npy
- s2.wav -> [spk1]_[spk2]__[occ]__[spk2].npy

Génère également un fichier consolidé egemaps_summary_[split].parquet
pour faciliter le calcul de distance L1 et la segmentation du dataset.
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from tqdm import tqdm

from opensmile.core.smile import Smile
from opensmile.core.define import FeatureSet, FeatureLevel


# Instance openSMILE globale pour chaque processus worker
_worker_smile = None


def init_worker(feature_set: str):
    """Initialise une instance openSMILE par sous-processus."""
    global _worker_smile
    if feature_set == "egemaps":
        _worker_smile = Smile(
            feature_set=FeatureSet.eGeMAPSv02,
            feature_level=FeatureLevel.Functionals,
        )
    elif feature_set == "compare":
        _worker_smile = Smile(
            feature_set=FeatureSet.ComParE_2016,
            feature_level=FeatureLevel.Functionals,
        )
    else:
        raise ValueError(f"Feature set inconnu: {feature_set}")


def process_pair_folder(task):
    """
    Traite un sous-dossier contenant s1.wav et s2.wav.
    """
    global _worker_smile
    (pair_dir, split, occ, spk1, spk2, overwrite) = task

    s1_path = os.path.join(pair_dir, "s1.wav")
    s2_path = os.path.join(pair_dir, "s2.wav")

    out_s1 = os.path.join(pair_dir, f"{spk1}_{spk2}__{occ}__{spk1}.npy")
    out_s2 = os.path.join(pair_dir, f"{spk1}_{spk2}__{occ}__{spk2}.npy")

    res = {
        "split": split,
        "occ": occ,
        "spk1": spk1,
        "spk2": spk2,
        "pair_id": f"{occ}/{spk1}_{spk2}",
        "pair_dir": pair_dir,
    }

    # Vérification si déjà calculé
    need_s1 = overwrite or not os.path.exists(out_s1)
    need_s2 = overwrite or not os.path.exists(out_s2)

    try:
        # Traitement s1
        if need_s1:
            if not os.path.exists(s1_path):
                return None
            df_s1 = _worker_smile.process_file(s1_path)
            feat_s1 = df_s1.to_numpy().flatten()
            np.save(out_s1, feat_s1)
        else:
            feat_s1 = np.load(out_s1).flatten()

        # Traitement s2
        if need_s2:
            if not os.path.exists(s2_path):
                return None
            df_s2 = _worker_smile.process_file(s2_path)
            feat_s2 = df_s2.to_numpy().flatten()
            np.save(out_s2, feat_s2)
        else:
            feat_s2 = np.load(out_s2).flatten()

        res["feat_s1"] = feat_s1
        res["feat_s2"] = feat_s2
        return res

    except Exception as e:
        print(f"Erreur sur {pair_dir}: {e}")
        return None


def collect_tasks(base_dir, split, overwrite=False):
    """
    Parcourt l'arborescence : base_dir/[split]/[occ]/[spk1]_[spk2]/
    """
    split_dir = os.path.join(base_dir, split)
    if not os.path.isdir(split_dir):
        print(f"Dossier non trouvé: {split_dir}")
        return []

    tasks = []
    # occ = "00", "01", etc.
    for occ in sorted(os.listdir(split_dir)):
        occ_dir = os.path.join(split_dir, occ)
        if not os.path.isdir(occ_dir) or occ.startswith("."):
            continue

        for pair_name in sorted(os.listdir(occ_dir)):
            pair_dir = os.path.join(occ_dir, pair_name)
            if not os.path.isdir(pair_dir) or pair_name.startswith("."):
                continue

            # Extraire spk1 et spk2 depuis le nom du dossier "spk1_spk2"
            parts = pair_name.split("_")
            if len(parts) == 2:
                spk1, spk2 = parts[0], parts[1]
                tasks.append((pair_dir, split, occ, spk1, spk2, overwrite))

    return tasks


def run_extraction(args):
    # Récupérer les noms des colonnes de features avec un Smile témoin
    smile_probe = Smile(
        feature_set=FeatureSet.eGeMAPSv02 if args.feature_set == "egemaps" else FeatureSet.ComParE_2016,
        feature_level=FeatureLevel.Functionals,
    )
    feature_names = smile_probe.feature_names

    for split in args.subsets:
        print(f"\n--- Collecte des dossiers pour '{split}' dans {args.data_dir} ---")
        tasks = collect_tasks(args.data_dir, split, overwrite=args.overwrite)
        print(f"Nombre de paires trouvées : {len(tasks)}")

        if not tasks:
            continue

        results = []
        num_workers = args.num_workers
        print(f"Extraction avec {num_workers} workers...")

        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(args.feature_set,)) as executor:
            for r in tqdm(executor.map(process_pair_folder, tasks, chunksize=16), total=len(tasks), desc=f"eGeMAPS {split}"):
                if r is not None:
                    results.append(r)

        # Création du fichier consolidé pour vos segmentations
        if results and args.save_summary:
            out_summary = os.path.join(args.data_dir, f"{args.feature_set}_summary_{split}.parquet")
            print(f"Création de la table récapitulative : {out_summary}")

            records = []
            for item in results:
                f1 = item["feat_s1"]
                f2 = item["feat_s2"]
                l1_dist = np.abs(f1 - f2)

                row = {
                    "pair_id": item["pair_id"],
                    "occ": item["occ"],
                    "spk1": item["spk1"],
                    "spk2": item["spk2"],
                    "mean_l1": float(np.mean(l1_dist)),
                }
                # Ajouter les distances L1 individuelles pour chaque feature F
                for f_name, d_val in zip(feature_names, l1_dist):
                    row[f"dist_{f_name}"] = float(d_val)

                records.append(row)

            df_summary = pd.DataFrame(records)
            df_summary.to_parquet(out_summary, index=False)
            print(f"Table sauvegardée avec succès ({len(df_summary)} lignes, {len(feature_names)} features).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction openSMILE sur EchoSet_cleaned")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/idiap/temp/adubois/EchoSet_cleaned_small",
        help="Chemin de base de EchoSet_cleaned",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Subsets à traiter (train, val, test)",
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="egemaps",
        choices=["egemaps", "compare"],
        help="Feature set openSMILE (défaut: egemaps)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Nombre de processus parallèles",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculer même si le fichier .npy existe déjà",
    )
    parser.add_argument(
        "--save_summary",
        action="store_true",
        default=True,
        help="Générer un fichier summary .parquet par split",
    )

    args = parser.parse_args()
    run_extraction(args)
