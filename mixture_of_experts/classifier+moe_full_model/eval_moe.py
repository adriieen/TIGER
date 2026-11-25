#!/usr/bin/env python3
"""
eval_moe.py

Evaluate the trained `JointMoETIGER` checkpoint in Hard-Routing mode on `EchoSet_cleaned_small/test`
(2,650 pairs), saving:
- `metrics.csv` (standard TIGER MetricsTracker format: snt_id, sdr, sdr_i, si-snr, si-snr_i)
- `routing_predictions.csv` (true_quintile, pred_quintile, pred_delta_f0, true_delta_f0, probs)
- `confusion_matrix_test.html` (5x5 router confusion matrix)
- Summary printout (Overall + per-test-quintile SI-SDRi & SDRi)
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import plotly.graph_objects as go

TIGER_ROOT = Path("/idiap/home/adubois/project_ma1/TIGER")
if str(TIGER_ROOT) not in sys.path:
    sys.path.insert(0, str(TIGER_ROOT))

from look2hear.metrics import MetricsTracker
from dataset import load_split_records, MoEEchoSetDataset
from model import JointMoETIGER

CLASS_NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def plot_cm_html(cm: np.ndarray, out_html: Path, acc: float, adj1: float, mace: float):
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
    annotations = []
    for i in range(5):
        for j in range(5):
            pct, cnt = cm_norm[i, j], cm[i, j]
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
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>JointMoETIGER Router Confusion Matrix — Test Set</b><br>"
                f"<span style='font-size:13px;color:#444;'>Top-1 Acc: {acc*100:.2f}% | "
                f"Adjacent (±1) Acc: {adj1*100:.2f}% | MACE: {mace:.3f}</span>"
            ),
            x=0.05,
            y=0.95,
            font=dict(size=17),
        ),
        xaxis=dict(title="<b>Predicted Quintile</b>", tickfont=dict(size=13)),
        yaxis=dict(title="<b>True Quintile</b>", tickfont=dict(size=13), autorange="reversed"),
        annotations=annotations,
        template="plotly_white",
        width=760,
        height=680,
        margin=dict(l=100, r=80, t=120, b=80),
    )
    fig.write_html(str(out_html))


def main():
    parser = argparse.ArgumentParser(description="Evaluate JointMoETIGER in Hard-Routing mode")
    parser.add_argument("--data_root", type=str, default="/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_cleaned_small")
    parser.add_argument("--parquet_root", type=str, default="/idiap/temp/adubois/EchoSet_cleaned_small")
    parser.add_argument("--ckpt_path", type=str, default="/idiap/home/adubois/project_ma1/TIGER/mixture_of_experts/classifier+moe_full_model/checkpoints/run_default/best_moe.pt")
    parser.add_argument("--out_dir", type=str, default="/idiap/home/adubois/project_ma1/TIGER/mixture_of_experts/classifier+moe_full_model/eval_results/run_default")
    parser.add_argument("--segment", type=float, default=None, help="None = full 6.0s test utterance")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    parq_root = Path(args.parquet_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    thresh_info = ckpt["thresholds_info"]
    thresholds_st = thresh_info["thresholds_semitones"]
    centers_st = thresh_info["centers_semitones"]
    f0_col = thresh_info["feature"]
    temperature = ckpt.get("config", {}).get("temperature", 1.0)

    model = JointMoETIGER(
        centers_semitones=centers_st,
        thresholds_semitones=thresholds_st,
        temperature=temperature,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_records = load_split_records(data_root / "test", parq_root / "egemaps_summary_test.parquet", thresholds_st, f0_col)
    test_ds = MoEEchoSetDataset(test_records, sample_rate=16000, segment=args.segment, is_train=False)

    metrics = MetricsTracker(save_file=str(out_dir / "metrics.csv"))
    routing_rows = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="Evaluating JointMoETIGER (Hard)"):
            mix, sources, true_df0, true_label = test_ds[idx]
            rec = test_records[idx]

            mix_dev = mix.unsqueeze(0).to(device)
            sources_dev = sources.to(device)

            est_sources, pred_df0, probs, pred_idx = model.forward_hard(mix_dev)
            est_sources_squeezed = est_sources.squeeze(0)

            metrics(
                mix=mix_dev.squeeze(0),
                clean=sources_dev,
                estimate=est_sources_squeezed,
                key=rec["mix_path"],
            )

            p_np = probs.squeeze(0).cpu().numpy()
            routing_rows.append({
                "pair_id": rec["pair_id"],
                "mix_path": rec["mix_path"],
                "true_delta_f0": float(true_df0.item()),
                "pred_delta_f0": float(pred_df0.item()),
                "true_label": int(true_label.item()),
                "pred_label": int(pred_idx.item()),
                **{f"prob_{CLASS_NAMES[k]}": float(p_np[k]) for k in range(5)},
            })

    metrics.final()

    df_route = pd.DataFrame(routing_rows)
    df_route.to_csv(out_dir / "routing_predictions.csv", index=False)

    y_true = df_route["true_label"].values
    y_pred = df_route["pred_label"].values
    acc = float(np.mean(y_true == y_pred))
    adj1 = float(np.mean(np.abs(y_true - y_pred) <= 1))
    mace = float(np.mean(np.abs(y_true - y_pred)))

    cm = np.zeros((5, 5), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plot_cm_html(cm, out_dir / "confusion_matrix_test.html", acc, adj1, mace)

    df_m = pd.read_csv(out_dir / "metrics.csv")
    df_merged = pd.concat([df_route, df_m[["sdr", "sdr_i", "si-snr", "si-snr_i"]]], axis=1)

    print(f"\n=== JointMoETIGER Test Results (N={len(df_merged)}) ===")
    print(f"Router Top-1 Acc: {acc*100:.2f}% | Adj(±1) Acc: {adj1*100:.2f}% | MACE: {mace:.3f}")
    print(f"Overall SI-SDRi -> Mean: {df_merged['si-snr_i'].mean():.2f} dB | Median: {df_merged['si-snr_i'].median():.2f} dB")
    print(f"Overall SDRi    -> Mean: {df_merged['sdr_i'].mean():.2f} dB | Median: {df_merged['sdr_i'].median():.2f} dB")
    print("\nBreakdown by True Quintile:")
    for k in range(5):
        sub = df_merged[df_merged["true_label"] == k]
        print(f"  {CLASS_NAMES[k]} (N={len(sub)}): Med SI-SDRi={sub['si-snr_i'].median():.2f} dB (Mean={sub['si-snr_i'].mean():.2f} dB)")


if __name__ == "__main__":
    main()
