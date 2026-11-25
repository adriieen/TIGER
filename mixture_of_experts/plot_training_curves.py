#!/usr/bin/env python3
"""
plot_training_curves.py

Automatically discovers all `checkpoints/*/training_history.csv` files and generates
clean PNG plots in each model folder:
- `training_curves.png` (Loss, Accuracy %, MACE)
- `checkpoints/comparison_training_curves.png` (Global comparison)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
})

CKPT_ROOT = Path("/idiap/home/adubois/project_ma1/TIGER/mixture_of_experts/checkpoints")


def plot_single_model(csv_path: Path):
    model_name = csv_path.parent.name
    df = pd.read_csv(csv_path)

    best_idx = df["val6s_acc"].idxmax()
    best_ep = int(df.loc[best_idx, "epoch"])
    best_acc = float(df.loc[best_idx, "val6s_acc"]) * 100.0
    best_mace = float(df.loc[best_idx, "val6s_mace"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # 1. Loss
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax.plot(df["epoch"], df["val6s_loss"], label="Val 6s Loss", color="#d62728", lw=2)
    ax.plot(df["epoch"], df["val2s_loss"], label="Val 2s Loss", color="#ff7f0e", lw=1.8, ls=":")
    ax.axvline(best_ep, color="gray", ls="--", lw=1.3, label=f"Best Ep ({best_ep})")
    ax.set_ylabel("Loss")
    ax.set_title("Loss (Train vs Val 6s / Val 2s)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2)

    # 2. Accuracy
    ax = axes[1]
    ax.plot(df["epoch"], df["train_acc"] * 100, label="Train Acc (%)", color="#1f77b4", lw=2)
    ax.plot(df["epoch"], df["val6s_acc"] * 100, label="Val 6s Acc (%)", color="#d62728", lw=2)
    ax.plot(df["epoch"], df["val2s_acc"] * 100, label="Val 2s Acc (%)", color="#ff7f0e", lw=1.8, ls=":")
    ax.plot(df["epoch"], df["val6s_adj1"] * 100, label="Val 6s Adj±1 (%)", color="#2ca02c", lw=1.6, ls="--")
    ax.axvline(best_ep, color="gray", ls="--", lw=1.3)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Top-1 & Adjacent (±1) Accuracy (%)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", ncol=2)

    # 3. MACE
    ax = axes[2]
    ax.plot(df["epoch"], df["val6s_mace"], label="Val 6s MACE", color="#9467bd", lw=2)
    ax.plot(df["epoch"], df["val2s_mace"], label="Val 2s MACE", color="#8c564b", lw=1.8, ls=":")
    ax.axvline(best_ep, color="gray", ls="--", lw=1.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MACE")
    ax.set_title("Mean Absolute Class Error (MACE)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.suptitle(
        f"DeltaF0Router ({model_name}) — Best Val(6s) Acc: {best_acc:.2f}% @ Epoch {best_ep} (MACE={best_mace:.3f})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = csv_path.parent / "training_curves.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved {out_png}")
    return model_name, df


def main():
    csv_files = sorted(CKPT_ROOT.glob("*/training_history.csv"))
    if not csv_files:
        print(f"No training_history.csv found under {CKPT_ROOT}")
        return

    all_runs = [plot_single_model(p) for p in csv_files]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    for i, (name, df) in enumerate(all_runs):
        color = palette[i % len(palette)]
        axes[0].plot(df["epoch"], df["val6s_acc"] * 100, label=f"{name} (max {df['val6s_acc'].max()*100:.2f}%)", color=color, lw=2)
        axes[1].plot(df["epoch"], df["val6s_mace"], label=f"{name} (min {df['val6s_mace'].min():.3f})", color=color, lw=2)

    axes[0].set_ylabel("Val 6s Accuracy (%)")
    axes[0].set_title("Validation (6s) Top-1 Accuracy Comparison", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val 6s MACE")
    axes[1].set_title("Validation (6s) MACE Comparison", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    comp_png = CKPT_ROOT / "comparison_training_curves.png"
    plt.savefig(comp_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved {comp_png}")


if __name__ == "__main__":
    main()
