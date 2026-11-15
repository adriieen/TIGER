#!/bin/bash
# evaluate.sh
# Script d'évaluation en série pour les quintiles q3, q4, q5 et les 3 splits

set -e  # Arrête le script en cas d'erreur bloquante

# Se placer dans la racine du repo TIGER
cd /idiap/home/adubois/project_ma1/TIGER

# Chemin vers l'interpréteur Python de l'environnement tiger
PYTHON="/idiap/temp/adubois/miniconda3/envs/tiger/bin/python"

# S'assurer que le module local look2hear est bien accessible
export PYTHONPATH="/idiap/home/adubois/project_ma1/TIGER:${PYTHONPATH}"

# Liste ordonnée des fichiers de configuration à évaluer
CONFIGS=(
    "/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned_small/X_training/f0_quintiles/q3.yml"
    "/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned_small/X_training/f0_quintiles/q4.yml"
    "/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned_small/X_training/f0_quintiles/q5.yml"
    "/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned_small/X_training/splits/baseline.yml"
    "/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned_small/X_training/splits/even_split.yml"
    "/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned_small/X_training/splits/normal_split.yml"
)

TOTAL=${#CONFIGS[@]}
echo "=========================================================="
echo "Démarrage des évaluations en série (${TOTAL} modèles)"
echo "Début : $(date)"
echo "Machine : $(hostname)"
echo "=========================================================="

COUNT=1
for CONF in "${CONFIGS[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo "[${COUNT}/${TOTAL}] $(date)"
    echo "Configuration : ${CONF}"
    echo "----------------------------------------------------------"

    if [ ! -f "${CONF}" ]; then
        echo "[ERREUR] Le fichier de configuration ${CONF} n'existe pas !"
        exit 1
    fi

    # Lancement de l'évaluation
    $PYTHON audio_test_modified.py --conf_dir "${CONF}"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERREUR] Échec de l'évaluation pour ${CONF} (code: ${EXIT_CODE})"
        exit $EXIT_CODE
    else
        echo "[SUCCÈS] Évaluation terminée avec succès pour ${CONF}"
    fi

    COUNT=$((COUNT + 1))
done

echo ""
echo "=========================================================="
echo "Toutes les ${TOTAL} évaluations sont terminées avec succès !"
echo "Fin : $(date)"
echo "=========================================================="
