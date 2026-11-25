#!/usr/bin/env python3
"""
train_router.py

Training script for `DeltaF0Router` on EchoSet_cleaned_small.
- Computes fixed 5-quintile Delta F0 thresholds from the training set and saves `thresholds.json`.
- Labels train (4,054) and val (4,604) mixtures using those fixed thresholds.
- Trains with on-the-fly 2.0s active-speaker-aware random cropping, random gain U[0.7, 1.3],
  and SpecAugment light time masking (inside DeltaF0Router).
- Evaluates on validation set and saves `best_router.pt` maximizing Top-1 Accuracy
  (while tracking Mean Absolute Class Error and Adjacent-1 Accuracy).
"""

import os
import re
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from router_model import DeltaF0Router


class GaussianSoftOrdinalLoss(nn.Module):
    """
    Soft-label ordinal cross-entropy: builds a Gaussian target distribution
    centered on the true quintile y, rewarding +/-1 adjacent predictions and
    strongly penalizing >= 2 quintile errors.
    """

    def __init__(self, num_classes: int = 5, sigma: float = 0.85):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        classes = torch.arange(num_classes, dtype=torch.float32)
        self.register_buffer("classes", classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: (B,) in {0..4}
        diff = self.classes.unsqueeze(0) - targets.unsqueeze(1).float()
        soft_targets = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()


def clean_pair_key(s: str) -> str:
    s = str(s).replace("\\", "/").strip()
    m = re.search(r"(\d{2}/\d+_\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d{2})_(\d+_\d+)", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"(\d+_\d+)", s)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(s))[0]


def load_split_dataframe(
    mix_json_path: Path,
    parquet_path: Path,
    thresholds: list,
    f0_col: str = "dist_F0semitoneFrom27.5Hz_sma3nz_amean",
) -> pd.DataFrame:
    """Load mix.json paths, merge with eGeMAPS Delta F0, and assign class labels 0..4."""
    with open(mix_json_path, "r") as f:
        mix_entries = json.load(f)

    df_parq = pd.read_parquet(parquet_path)
    pair_col = [c for c in df_parq.columns if any(k in c.lower() for k in ["pair", "key", "utt"])][0]
    df_parq["clean_id"] = df_parq[pair_col].apply(clean_pair_key)
    df_parq = df_parq.drop_duplicates(subset=["clean_id"]).set_index("clean_id")

    records = []
    for entry in mix_entries:
        mix_path = entry[0]
        cid = clean_pair_key(mix_path)
        if cid in df_parq.index:
            df0 = float(df_parq.loc[cid, f0_col])
            if np.isfinite(df0):
                label = int(np.digitize(df0, thresholds))
                records.append({
                    "pair_id": cid,
                    "mix_path": mix_path,
                    "delta_f0": df0,
                    "label": label,
                })

    return pd.DataFrame(records)


class DeltaF0MixtureDataset(Dataset):
    """Dataset returning raw 16kHz mono waveforms and Delta F0 quintile labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        sample_rate: int = 16000,
        crop_sec: float = 2.0,
        is_train: bool = True,
        gain_range=(0.7, 1.3),
        min_active_rms_db: float = -20.0,
    ):
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.crop_samples = int(round(crop_sec * sample_rate)) if crop_sec is not None else None
        self.is_train = is_train
        self.gain_range = gain_range
        self.rms_ratio_thresh = 10.0 ** (min_active_rms_db / 20.0)  # -20 dB -> 0.1

    def __len__(self) -> int:
        return len(self.df)

    def _read_mono(self, path: str) -> np.ndarray:
        wav, _ = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]
        return wav

    def _select_active_crop(self, mix: np.ndarray, mix_path: str) -> np.ndarray:
        """
        Select a random `crop_samples` window where both speakers are active
        (window RMS >= -20 dB relative to 6s RMS) when s1.wav/s2.wav are available.
        """
        L = len(mix)
        C = self.crop_samples
        if C is None or L <= C:
            if L < C:
                return np.pad(mix, (0, C - L), mode="constant")
            return mix

        sample_dir = os.path.dirname(mix_path)
        s1_path = os.path.join(sample_dir, "s1.wav")
        s2_path = os.path.join(sample_dir, "s2.wav")
        has_sources = os.path.exists(s1_path) and os.path.exists(s2_path)

        if not has_sources:
            start = random.randint(0, L - C)
            return mix[start : start + C]

        s1 = self._read_mono(s1_path)
        s2 = self._read_mono(s2_path)
        rms1_full = float(np.sqrt(np.mean(s1 ** 2) + 1e-12))
        rms2_full = float(np.sqrt(np.mean(s2 ** 2) + 1e-12))

        best_start = 0
        best_score = -1.0
        for _ in range(8):
            start = random.randint(0, L - C)
            r1 = float(np.sqrt(np.mean(s1[start : start + C] ** 2) + 1e-12)) / rms1_full
            r2 = float(np.sqrt(np.mean(s2[start : start + C] ** 2) + 1e-12)) / rms2_full
            score = min(r1, r2)
            if score >= self.rms_ratio_thresh:
                return mix[start : start + C]
            if score > best_score:
                best_score = score
                best_start = start

        return mix[best_start : best_start + C]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        mix_path = row["mix_path"]
        mix = self._read_mono(mix_path)

        if self.is_train:
            mix = self._select_active_crop(mix, mix_path)
            gain = random.uniform(self.gain_range[0], self.gain_range[1])
            mix = mix * gain
        elif self.crop_samples is not None:
            L = len(mix)
            C = self.crop_samples
            if L > C:
                start = (L - C) // 2
                mix = mix[start : start + C]
            elif L < C:
                mix = np.pad(mix, (0, C - L), mode="constant")

        return torch.from_numpy(mix.astype(np.float32)), int(row["label"]), float(row["delta_f0"])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, mode: str, thresholds_st: list):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    for x, y, df0 in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        df0 = df0.to(device, non_blocking=True, dtype=torch.float32)

        out = model(x)
        if mode == "regression":
            pred_f0 = out.squeeze(-1)
            loss = criterion(pred_f0, df0)
            preds = torch.tensor(np.digitize(pred_f0.cpu().numpy(), thresholds_st), dtype=torch.long)
        else:
            loss = criterion(out, y)
            preds = torch.argmax(out, dim=-1).cpu()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        all_preds.append(preds.numpy())
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    top1_acc = float(np.mean(all_preds == all_labels))
    adj1_acc = float(np.mean(np.abs(all_preds - all_labels) <= 1))
    mace = float(np.mean(np.abs(all_preds - all_labels)))
    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, top1_acc, adj1_acc, mace


def main():
    parser = argparse.ArgumentParser(description="Train DeltaF0Router on EchoSet_cleaned_small")
    parser.add_argument("--data_root", type=str, default="/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_cleaned_small")
    parser.add_argument("--parquet_root", type=str, default="/idiap/temp/adubois/EchoSet_cleaned_small")
    parser.add_argument("--save_dir", type=str, default="/idiap/home/adubois/project_ma1/TIGER/mixture_of_experts/checkpoints/175k_ordinal")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--dropout", type=float, default=0.4)
    loss_group = parser.add_mutually_exclusive_group()
    loss_group.add_argument("--smooth_l1", action="store_true", help="Use SmoothL1 regression on Delta F0 (default)")
    loss_group.add_argument("--gaussian_target", action="store_true", help="Use Gaussian soft-label ordinal cross-entropy")
    loss_group.add_argument("--one_hot", action="store_true", help="Use standard CrossEntropyLoss on one-hot targets")
    parser.add_argument("--channels", type=int, nargs=4, default=[8, 16, 32, 64])
    parser.add_argument("--sigma", type=float, default=0.85, help="Gaussian sigma for soft ordinal loss")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_crop_sec", type=float, default=2.0)
    parser.add_argument("--val_crop_sec", type=float, default=None, help="None = full 6s utterance for val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="Real-work-dataset")
    parser.add_argument("--wandb_name", type=str, default="DeltaF0Router-EchoSet_small")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()
    mode = "classification" if (args.gaussian_target or args.one_hot) else "regression"

    #use_wandb = not args.no_wandb
    use_wandb = False
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    parq_root = Path(args.parquet_root)
    f0_col = "dist_F0semitoneFrom27.5Hz_sma3nz_amean"

    # 1. Compute fixed quintile thresholds strictly from the training set
    train_parq_path = parq_root / "egemaps_summary_train.parquet"
    df_train_parq = pd.read_parquet(train_parq_path)
    with open(data_root / "train/mix.json", "r") as f:
        train_mix = json.load(f)

    pair_col = [c for c in df_train_parq.columns if any(k in c.lower() for k in ["pair", "key", "utt"])][0]
    df_train_parq["clean_id"] = df_train_parq[pair_col].apply(clean_pair_key)
    df_train_parq = df_train_parq.drop_duplicates(subset=["clean_id"]).set_index("clean_id")

    train_ids = [clean_pair_key(e[0]) for e in train_mix]
    train_f0_vals = df_train_parq.loc[[i for i in train_ids if i in df_train_parq.index], f0_col].values
    train_f0_vals = train_f0_vals[np.isfinite(train_f0_vals)]

    thresholds_st = [float(q) for q in np.quantile(train_f0_vals, [0.2, 0.4, 0.6, 0.8])]
    thresholds_hz = [float(150.0 * (2.0 ** (st / 12.0) - 1.0)) for st in thresholds_st]

    thresh_info = {
        "feature": f0_col,
        "thresholds_semitones": thresholds_st,
        "thresholds_hz_ref150": thresholds_hz,
        "num_train_samples": int(len(train_f0_vals)),
    }
    with open(save_dir / "thresholds.json", "w") as f:
        json.dump(thresh_info, f, indent=2)
    print(f"--> Saved train quintile thresholds (st): {[round(t, 4) for t in thresholds_st]}")
    print(f"--> Approximate Hz equivalents (ref 150Hz): {[round(h, 2) for h in thresholds_hz]}")

    # 2. Load Train & Val DataFrames with fixed thresholds
    df_train = load_split_dataframe(data_root / "train/mix.json", parq_root / "egemaps_summary_train.parquet", thresholds_st, f0_col)
    df_val = load_split_dataframe(data_root / "val/mix.json", parq_root / "egemaps_summary_val.parquet", thresholds_st, f0_col)

    print(f"--> Train samples: {len(df_train)} | Class counts: {df_train['label'].value_counts().sort_index().to_dict()}")
    print(f"--> Val   samples: {len(df_val)}   | Class counts: {df_val['label'].value_counts().sort_index().to_dict()}")

    train_ds = DeltaF0MixtureDataset(df_train, crop_sec=args.train_crop_sec, is_train=True)
    val_ds_6s = DeltaF0MixtureDataset(df_val, crop_sec=args.val_crop_sec, is_train=False)
    val_ds_2s = DeltaF0MixtureDataset(df_val, crop_sec=2.0, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader_6s = DataLoader(val_ds_6s, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader_2s = DataLoader(val_ds_2s, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 3. Model, Loss, Optimizer, Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Training on device: {device}")

    model = DeltaF0Router(
        num_classes=5,
        channels=tuple(args.channels),
        dropout=args.dropout,
        mode=mode,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"--> DeltaF0Router ({mode}, channels={args.channels}) parameters: {n_params:,}")

    if mode == "regression":
        criterion = nn.SmoothL1Loss(beta=1.0)
    elif args.one_hot:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    else:
        criterion = GaussianSoftOrdinalLoss(num_classes=5, sigma=args.sigma).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_acc = -1.0
    best_mace = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y, df0 in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            df0 = df0.to(device, non_blocking=True, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            if mode == "regression":
                pred_f0 = out.squeeze(-1)
                loss = criterion(pred_f0, df0)
                preds_np = np.digitize(pred_f0.detach().cpu().numpy(), thresholds_st)
                correct = int((preds_np == y.cpu().numpy()).sum())
            else:
                loss = criterion(out, y)
                correct = int((torch.argmax(out, dim=-1) == y).sum().item())

            loss.backward()
            optimizer.step()

            bs = y.size(0)
            train_loss_sum += loss.item() * bs
            train_correct += correct
            train_total += bs

        scheduler.step()
        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        val_loss_6s, val_acc_6s, val_adj1_6s, val_mace_6s = evaluate(model, val_loader_6s, criterion, device, mode, thresholds_st)
        val_loss_2s, val_acc_2s, val_adj1_2s, val_mace_2s = evaluate(model, val_loader_2s, criterion, device, mode, thresholds_st)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} ({dt:.1f}s, lr={lr_now:.2e}) | "
            f"Train Loss={train_loss:.4f} Acc={train_acc*100:.2f}% | "
            f"Val(6s) Acc={val_acc_6s*100:.2f}% Adj1={val_adj1_6s*100:.2f}% MACE={val_mace_6s:.3f} | "
            f"Val(2s) Acc={val_acc_2s*100:.2f}% MACE={val_mace_2s:.3f}"
        )

        epoch_metrics = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val6s_loss": val_loss_6s,
            "val6s_acc": val_acc_6s,
            "val6s_adj1": val_adj1_6s,
            "val6s_mace": val_mace_6s,
            "val2s_loss": val_loss_2s,
            "val2s_acc": val_acc_2s,
            "val2s_adj1": val_adj1_2s,
            "val2s_mace": val_mace_2s,
        }
        history.append(epoch_metrics)
        if use_wandb:
            wandb.log(epoch_metrics, step=epoch)

        # Save best checkpoint maximizing Val(6s) Top-1 Accuracy (tie-break by lowest MACE)
        if (val_acc_6s > best_acc) or (np.isclose(val_acc_6s, best_acc) and val_mace_6s < best_mace):
            best_acc = val_acc_6s
            best_mace = val_mace_6s
            torch.save(
                {
                    "epoch": epoch,
                    "mode": mode,
                    "channels": args.channels,
                    "model_state_dict": model.state_dict(),
                    "val6s_acc": val_acc_6s,
                    "val6s_mace": val_mace_6s,
                    "val2s_acc": val_acc_2s,
                    "val2s_mace": val_mace_2s,
                    "thresholds": thresh_info,
                },
                save_dir / "best_router.pt",
            )
            print(f"   --> [Saved best_router.pt] Val(6s) Acc={best_acc*100:.2f}%, MACE={best_mace:.3f}")

    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict(), "thresholds": thresh_info}, save_dir / "last_router.pt")
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)
    print(f"\n✅ Training complete! Best Val(6s) Top-1 Acc: {best_acc*100:.2f}% (MACE={best_mace:.3f})")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
