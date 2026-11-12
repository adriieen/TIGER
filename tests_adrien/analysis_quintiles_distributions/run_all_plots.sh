#!/bin/bash
# run_all_plots.sh
# Execute all distribution plotting scripts in analysis_quintiles_distributions

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

echo "=========================================================="
echo "Generating all Quintile & Split Distribution Plots in $DIR"
echo "=========================================================="

echo "[1/4] Running plot_f0_train_val_test_all.py..."
$PYTHON "$DIR/plot_f0_train_val_test_all.py"

echo "[2/4] Running plot_f0_quintiles_train.py..."
$PYTHON "$DIR/plot_f0_quintiles_train.py"

echo "[3/4] Running plot_f0_quintiles_valid.py..."
$PYTHON "$DIR/plot_f0_quintiles_valid.py"

echo "[4/4] Running plot_f0_quintiles_test.py..."
$PYTHON "$DIR/plot_f0_quintiles_test.py"

echo "=========================================================="
echo "✅ All plots generated successfully!"
echo "=========================================================="

