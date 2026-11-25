#!/usr/bin/env python3
"""
eval_router.py

Evaluate `DeltaF0Router` on EchoSet_cleaned_small Test Set (2,650 pairs):
- Generates 5x5 confusion matrix (counts & row-normalized %) in interactive Plotly HTML.
- Reports Top-1 Accuracy, Adjacent-1 Accuracy (|y_pred - y_true| <= 1), and MACE.
- Exports `test_routing_predictions.csv` with predicted expert indices, softmax
  probabilities, and mapped TIGER checkpoint directories for downstream MoE inference.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import plotly.graph_objects as go

from router_model import DeltaF0Router
from train_router import load_split_dataframe, DeltaF0MixtureDataset

EXPERT_CHECKPOINTS = {
    0: "TIGER-EchoSet_cleaned_small-quintile1_3_gpus_X_11_11_snr",
    1: "TIGER-EchoSet_cleaned_small-quintile2_3_gpus_X_11_11_snr",
    2: "TIGER-EchoSet_cleaned_small-quintile3_3_gpus_X_11_11_snr_fixed",
    3: "TIGER-EchoSet_cleaned_small-quintile4_3_gpus_X_11_11_snr",
    4: "TIGER-EchoSet_cleaned_small-quintile5_3_gpus_X_11_11_snr",
}
CLASS_NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix_html(cm: np.ndarray, out_html: Path, title_suffix: str, acc: float, adj1: float, mace: float):
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

    annotations = []
    for i in range(5):
        for j in range(5):
            pct = cm_norm[i, j]
            cnt = cm[i, j]
            annotations.append(
                dict(
                    x=CLASS_NAMES[j],
                    y=CLASS_NAMES[i],
                    text=f"<b>{pct:.1f}%</b><br>({cnt})",
                    showarrow=False,
                    font=dict(color="white" if pct > 50.0 else "black", size=14, family="Arial"),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            zmin=0,
            zmax=100,
            colorbar=dict(title=dict(text="<b>Row %</b>", side="right")),
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Row %: %{z:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>DeltaF0Router Confusion Matrix — Test Set ({title_suffix})</b><br>"
                f"<span style='font-size:13px;color:#444;'>Top-1 Acc: {acc*100:.2f}% | "
                f"Adjacent (±1) Acc: {adj1*100:.2f}% | MACE: {mace:.3f}</span>"
            ),
            font=dict(size=18, family="Arial", color="black"),
            x=0.05,
            y=0.95,
        ),
        xaxis=dict(title="<b>Predicted Quintile</b>", title_font=dict(size=15), tickfont=dict(size=13), side="bottom"),
        yaxis=dict(title="<b>True Quintile (Train Thresholds)</b>", title_font=dict(size=15), tickfont=dict(size=13), autorange="reversed"),
        annotations=annotations,
        template="plotly_white",
        width=760,
        height=680,
        margin=dict(l=100, r=80, t=120, b=80),
    )

    fig.write_html(str(out_html))
    print(f"--> Saved confusion matrix to: {out_html}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeltaF0Router on EchoSet_cleaned_small Test Set")
    parser.add_argument("--data_root", type=str, default="/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_cleaned_small")
    parser.add_argument("--parquet_root", type=str, default="/idiap/temp/adubois/EchoSet_cleaned_small")
    parser.add_argument("--ckpt_path", type=str, default="/idiap/home/adubois/project_ma1/TIGER/mixture_of_experts/checkpoints/best_router.pt")
    parser.add_argument("--thresholds_path", type=str, default="/idiap/home/adubois/project_ma1/TIGER/mixture_of_experts/checkpoints/thresholds.json")
    parser.add_argument("--out_dir", type=str, default="/idiap/home/adubois/project_ma1/TIGER/mixture_of_experts/eval_results")
    parser.add_argument("--crop_sec", type=float, default=None, help="None = full 6s utterance, or e.g. 2.0 for 2s center crop")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    parq_root = Path(args.parquet_root)

    with open(args.thresholds_path, "r") as f:
        thresh_info = json.load(f)
    thresholds_st = thresh_info["thresholds_semitones"]
    f0_col = thresh_info.get("feature", "dist_F0semitoneFrom27.5Hz_sma3nz_amean")

    df_test = load_split_dataframe(
        data_root / "test/mix.json",
        parq_root / "egemaps_summary_test.parquet",
        thresholds_st,
        f0_col,
    )
    print(f"--> Test samples loaded: {len(df_test)}")
    print(f"--> True class distribution: {df_test['label'].value_counts().sort_index().to_dict()}")

    test_ds = DeltaF0MixtureDataset(df_test, crop_sec=args.crop_sec, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    mode = ckpt.get("mode", "classification")
    channels = tuple(ckpt.get("channels", (32, 64, 128, 256)))

    model = DeltaF0Router(num_classes=5, channels=channels, mode=mode).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device, non_blocking=True)
            idx, probs = model.predict_expert(x, thresholds=thresholds_st)
            all_preds.append(idx.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    y_pred = np.concatenate(all_preds)
    probs_all = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels)

    acc = float(np.mean(y_pred == y_true))
    adj1 = float(np.mean(np.abs(y_pred - y_true) <= 1))
    mace = float(np.mean(np.abs(y_pred - y_true)))

    dur_tag = f"{args.crop_sec}s" if args.crop_sec is not None else "6.0s_full"
    print(f"\n=== Test Set Results ({dur_tag}) ===")
    print(f"Top-1 Accuracy      : {acc*100:.2f}%")
    print(f"Adjacent (±1) Acc   : {adj1*100:.2f}%")
    print(f"Mean Abs Class Err  : {mace:.4f}")

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=5)
    print("\nConfusion Matrix (rows=True Q1..Q5, cols=Pred Q1..Q5):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

    plot_confusion_matrix_html(
        cm,
        out_dir / f"confusion_matrix_test_{dur_tag}.html",
        title_suffix=dur_tag,
        acc=acc,
        adj1=adj1,
        mace=mace,
    )

    df_out = df_test.copy()
    df_out["true_quintile"] = [CLASS_NAMES[i] for i in y_true]
    df_out["pred_idx"] = y_pred
    df_out["pred_quintile"] = [CLASS_NAMES[i] for i in y_pred]
    for k in range(5):
        df_out[f"prob_{CLASS_NAMES[k]}"] = probs_all[:, k]
    df_out["assigned_checkpoint"] = [EXPERT_CHECKPOINTS[int(i)] for i in y_pred]

    csv_out = out_dir / f"test_routing_predictions_{dur_tag}.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"--> Exported routing predictions to: {csv_out}")


if __name__ == "__main__":
    main()
