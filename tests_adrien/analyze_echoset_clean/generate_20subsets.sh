#!/bin/bash
set -e

# Dossier du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="/idiap/temp/adubois/miniconda3/envs/tiger/bin/python"

# Fichiers et dossiers par défaut
SUMMARY_PARQUET="/idiap/temp/adubois/EchoSet_cleaned/egemaps_summary_train.parquet"
FEATURES_FILE="${SCRIPT_DIR}/top20_features.txt"
OUT_BASE_DIR="/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_cleaned/rankings"

if [ ! -f "$FEATURES_FILE" ]; then
    echo "Erreur : Fichier introuvable : $FEATURES_FILE"
    exit 1
fi

if [ ! -f "$SUMMARY_PARQUET" ]; then
    echo "Erreur : Fichier summary parquet introuvable : $SUMMARY_PARQUET"
    exit 1
fi

echo "=========================================================="
echo "Génération des rankings ordonnés complets pour les 20 features"
echo "Fichier de features : $FEATURES_FILE"
echo "Résumé parquet      : $SUMMARY_PARQUET"
echo "Dossier de sortie   : $OUT_BASE_DIR"
echo "=========================================================="

# Lancer la génération de tous les rankings en une seule passe rapide
$PYTHON_EXEC "${SCRIPT_DIR}/create_ranking_subsets.py" \
    --summary_parquet "$SUMMARY_PARQUET" \
    --features_file "$FEATURES_FILE" \
    --out_base_dir "$OUT_BASE_DIR"

echo "Terminé !"
