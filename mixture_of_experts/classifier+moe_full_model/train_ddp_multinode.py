#!/usr/bin/env python3
"""
train_ddp_multinode.py

Standalone Multi-Node DistributedDataParallel (DDP) training script for `JointMoETIGER`
(175k SmoothL1 DeltaF0Router + 5 TIGER Tiny specialists, K=5 quintiles, all from scratch).

Designed for manual / deferred `torchrun` launches across 6 nodes (1 GPU each, RTX 4000 Ada 20GB):
- Extended rendezvous timeout (2h) for asynchronous manual node startups
- Rank-0 guarded directory creation, threshold computation, checkpointing, and WandB logging
- Method B separation objective: sum_k p_k * PIT_neg_sisdr(est_k, targets) + lambda_cls * SmoothL1(pred_df0, true_df0)
- Exact float32 PIT SI-SDR computation under bfloat16/fp16 autocast
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import timedelta

# Clear residual Slurm variables before torch.distributed initialization via torchrun
for _v in ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID", "SLURM_NTASKS"):
    os.environ.pop(_v, None)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

TIGER_ROOT = Path("/idiap/home/adubois/project_ma1/TIGER")
if str(TIGER_ROOT) not in sys.path:
    sys.path.insert(0, str(TIGER_ROOT))

from look2hear.losses import PITLossWrapper, pairwise_neg_sisdr
from dataset import compute_train_thresholds, load_split_records, MoEEchoSetDataset
from model import JointMoETIGER


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=2),
        )
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_sums(values: list, device: torch.device) -> list:
    """All-reduce sum of float tensors across all DDP ranks."""
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


@torch.no_grad()
def evaluate_ddp(
    model_raw: JointMoETIGER,
    loader: DataLoader,
    pit_wrapper: PITLossWrapper,
    lambda_cls: float,
    device: torch.device,
    use_amp: bool,
):
    model_raw.eval()

    sum_hard_loss = 0.0
    sum_soft_loss = 0.0
    sum_cls_loss = 0.0
    sum_correct = 0.0
    sum_adj1 = 0.0
    sum_abs_err = 0.0
    total_n = 0.0
    expert_counts = [0.0] * 5

    for mix, sources, true_df0, labels in loader:
        mix = mix.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)
        true_df0 = true_df0.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bs = mix.size(0)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred_df0, probs, pred_idx, ests_list = model_raw.forward_train(mix)

        # Hard-routed estimates
        hard_ests = mix.new_zeros((bs, 2, mix.shape[-1]), dtype=torch.float32)
        for k in range(5):
            mask = (pred_idx == k)
            if mask.any():
                hard_ests[mask] = ests_list[k][mask].float()
            expert_counts[k] += float(mask.sum().item())

        pw_hard = pairwise_neg_sisdr(hard_ests, sources.float())
        hard_loss_vec, _ = pit_wrapper.find_best_perm(pw_hard)

        soft_losses = []
        for k in range(5):
            pw_k = pairwise_neg_sisdr(ests_list[k].float(), sources.float())
            lk, _ = pit_wrapper.find_best_perm(pw_k)
            soft_losses.append(lk)
        soft_losses = torch.stack(soft_losses, dim=1)
        soft_loss_vec = (probs.float() * soft_losses).sum(dim=-1)

        cls_loss_vec = F.smooth_l1_loss(pred_df0.float(), true_df0.float(), beta=1.0, reduction="none")

        sum_hard_loss += float(hard_loss_vec.sum().item())
        sum_soft_loss += float(soft_loss_vec.sum().item())
        sum_cls_loss += float(cls_loss_vec.sum().item())
        sum_correct += float((pred_idx == labels).sum().item())
        sum_adj1 += float((torch.abs(pred_idx - labels) <= 1).sum().item())
        sum_abs_err += float(torch.abs(pred_idx - labels).sum().item())
        total_n += float(bs)

    reduced = reduce_sums(
        [sum_hard_loss, sum_soft_loss, sum_cls_loss, sum_correct, sum_adj1, sum_abs_err, total_n] + expert_counts,
        device,
    )
    tot = max(1.0, reduced[6])
    return {
        "val_hard_loss": reduced[0] / tot,
        "val_soft_sep_loss": reduced[1] / tot,
        "val_cls_loss": reduced[2] / tot,
        "val_total_loss": (reduced[1] + lambda_cls * reduced[2]) / tot,
        "val_router_acc": reduced[3] / tot,
        "val_router_adj1": reduced[4] / tot,
        "val_router_mace": reduced[5] / tot,
        "expert_counts": [int(c) for c in reduced[7:]],
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-node DDP training for JointMoETIGER")
    parser.add_argument("--data_root", type=str, default="/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_cleaned_small")
    parser.add_argument("--parquet_root", type=str, default="/idiap/temp/adubois/EchoSet_cleaned_small")
    parser.add_argument("--save_dir", type=str, default="/idiap/home/adubois/project_ma1/TIGER/mixture_of_experts/classifier+moe_full_model/checkpoints/run_default")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--segment", type=float, default=3.0, help="Training segment length in seconds")
    parser.add_argument("--val_segment", type=float, default=None, help="Validation segment length (None = full 6.0s)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_cls", type=float, default=2.0, help="Weight of SmoothL1 router loss")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for soft routing weights")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=20, help="ReduceLROnPlateau patience")
    parser.add_argument("--no_amp", action="store_true", help="Disable bfloat16 autocast")
    parser.add_argument("--sync_bn", action="store_true", default=True, help="Convert BatchNorm to SyncBatchNorm for DDP")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="Real-work-dataset")
    parser.add_argument("--wandb_name", type=str, default="JointMoETIGER-EchoSet_small")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and torch.cuda.is_available()

    # Different data-augmentation seed per rank for crop diversity, same model init seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    save_dir = Path(args.save_dir)
    data_root = Path(args.data_root)
    parq_root = Path(args.parquet_root)
    thresh_path = save_dir / "thresholds.json"

    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        thresh_info = compute_train_thresholds(
            data_root / "train/mix.json",
            parq_root / "egemaps_summary_train.parquet",
        )
        with open(thresh_path, "w") as f:
            json.dump(thresh_info, f, indent=2)
        with open(save_dir / "train_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"[Rank 0] World size: {world_size} | Save dir: {save_dir}")
        print(f"[Rank 0] Train thresholds (st): {[round(t, 4) for t in thresh_info['thresholds_semitones']]}")
        print(f"[Rank 0] Quintile centers (st): {[round(c, 4) for c in thresh_info['centers_semitones']]}")

    if dist.is_initialized():
        dist.barrier()

    with open(thresh_path, "r") as f:
        thresh_info = json.load(f)

    thresholds_st = thresh_info["thresholds_semitones"]
    centers_st = thresh_info["centers_semitones"]
    f0_col = thresh_info["feature"]

    train_records = load_split_records(data_root / "train", parq_root / "egemaps_summary_train.parquet", thresholds_st, f0_col)
    val_records = load_split_records(data_root / "val", parq_root / "egemaps_summary_val.parquet", thresholds_st, f0_col)

    train_ds = MoEEchoSetDataset(train_records, sample_rate=16000, segment=args.segment, is_train=True)
    val_ds = MoEEchoSetDataset(val_records, sample_rate=16000, segment=args.val_segment, is_train=False)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if dist.is_initialized() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = JointMoETIGER(
        centers_semitones=centers_st,
        thresholds_semitones=thresholds_st,
        temperature=args.temperature,
    ).to(device)

    if dist.is_initialized() and args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if is_main:
        r_params = sum(p.numel() for p in model.router.parameters())
        e_params = sum(p.numel() for p in model.experts.parameters())
        print(f"[Rank 0] Params -> Router: {r_params:,} | 5x TIGER Tiny: {e_params:,} | Total: {r_params + e_params:,}")

    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if dist.is_initialized() else model
    model_raw = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model

    pit_wrapper = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx", threshold_byloss=False).to(device)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)

    use_wandb = is_main and (not args.no_wandb)
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    best_val_hard = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        ddp_model.train()
        sum_tot, sum_sep, sum_cls, sum_acc, sum_adj1, tot_n = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for mix, sources, true_df0, labels in train_loader:
            mix = mix.to(device, non_blocking=True)
            sources = sources.to(device, non_blocking=True)
            true_df0 = true_df0.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bs = mix.size(0)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred_df0, probs, pred_idx, ests_list = ddp_model(mix)

            # Method B separation loss in exact float32: sum_k p_k * PIT_neg_sisdr(est_k, sources)
            expert_losses = []
            for k in range(5):
                pw_k = pairwise_neg_sisdr(ests_list[k].float(), sources.float())
                lk, _ = pit_wrapper.find_best_perm(pw_k)  # (B,)
                expert_losses.append(lk)
            expert_losses = torch.stack(expert_losses, dim=1)  # (B, 5)
            sep_loss_vec = (probs.float() * expert_losses).sum(dim=-1)  # (B,)
            sep_loss = sep_loss_vec.mean()

            # Router SmoothL1 regression loss against true Delta F0
            cls_loss = F.smooth_l1_loss(pred_df0.float(), true_df0.float(), beta=1.0)
            total_loss = sep_loss + args.lambda_cls * cls_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.grad_clip)
            optimizer.step()

            sum_tot += float(total_loss.item()) * bs
            sum_sep += float(sep_loss.item()) * bs
            sum_cls += float(cls_loss.item()) * bs
            sum_acc += float((pred_idx == labels).sum().item())
            sum_adj1 += float((torch.abs(pred_idx - labels) <= 1).sum().item())
            tot_n += float(bs)

        tr_red = reduce_sums([sum_tot, sum_sep, sum_cls, sum_acc, sum_adj1, tot_n], device)
        tr_tot_n = max(1.0, tr_red[5])
        train_total = tr_red[0] / tr_tot_n
        train_sep = tr_red[1] / tr_tot_n
        train_cls = tr_red[2] / tr_tot_n
        train_acc = tr_red[3] / tr_tot_n
        train_adj1 = tr_red[4] / tr_tot_n

        val_stats = evaluate_ddp(model_raw, val_loader, pit_wrapper, args.lambda_cls, device, use_amp)
        scheduler.step(val_stats["val_hard_loss"])

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        if is_main:
            print(
                f"Ep {epoch:03d}/{args.epochs:03d} ({dt:.1f}s, lr={lr_now:.2e}) | "
                f"Tr Tot={train_total:.3f} (Sep={train_sep:.3f}, Cls={train_cls:.3f}, RAcc={train_acc*100:.1f}%) | "
                f"Val Hard={val_stats['val_hard_loss']:.3f} Soft={val_stats['val_soft_sep_loss']:.3f} "
                f"RAcc={val_stats['val_router_acc']*100:.1f}% Adj1={val_stats['val_router_adj1']*100:.1f}% MACE={val_stats['val_router_mace']:.3f} | "
                f"Experts={val_stats['expert_counts']}"
            )
            row = {
                "epoch": epoch,
                "lr": lr_now,
                "train_total_loss": train_total,
                "train_sep_loss": train_sep,
                "train_cls_loss": train_cls,
                "train_router_acc": train_acc,
                "train_router_adj1": train_adj1,
                "val_hard_loss": val_stats["val_hard_loss"],
                "val_soft_sep_loss": val_stats["val_soft_sep_loss"],
                "val_cls_loss": val_stats["val_cls_loss"],
                "val_total_loss": val_stats["val_total_loss"],
                "val_router_acc": val_stats["val_router_acc"],
                "val_router_adj1": val_stats["val_router_adj1"],
                "val_router_mace": val_stats["val_router_mace"],
            }
            history.append(row)
            if use_wandb:
                wandb.log(row, step=epoch)

            if val_stats["val_hard_loss"] < best_val_hard:
                best_val_hard = val_stats["val_hard_loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_raw.state_dict(),
                        "val_hard_loss": best_val_hard,
                        "val_router_acc": val_stats["val_router_acc"],
                        "thresholds_info": thresh_info,
                        "config": vars(args),
                    },
                    save_dir / "best_moe.pt",
                )
                print(f"   --> [Saved best_moe.pt] Val Hard Loss={best_val_hard:.3f} (Val Router Acc={val_stats['val_router_acc']*100:.1f}%)")

            if epoch % 10 == 0 or epoch == args.epochs:
                torch.save({"epoch": epoch, "model_state_dict": model_raw.state_dict(), "thresholds_info": thresh_info, "config": vars(args)}, save_dir / "last_moe.pt")
                pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)

        if dist.is_initialized():
            dist.barrier()

    if is_main:
        print(f"\n✅ Joint MoE DDP training finished! Best Val Hard neg-SI-SDR: {best_val_hard:.3f}")
        if use_wandb:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
