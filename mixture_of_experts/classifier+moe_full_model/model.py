#!/usr/bin/env python3
"""
model.py

Joint MoE Speech Separation Model (`JointMoETIGER`):
- Router: `DeltaF0Router` (175k params, channels=(8,16,32,64), mode="regression"),
  initialized from scratch, trained with SmoothL1 (lambda=2.0) + gradient from MoE separation.
- 5 TIGER Tiny Specialists (K=5 quintiles), initialized from scratch:
  out_channels=24, in_channels=64, num_blocks=4, upsampling_depth=5,
  win=640, stride=160, num_sources=2, sample_rate=16000.

Training objective (Method B):
  L_total = sum_k p_k * L_PIT(est_k, targets) + lambda_cls * SmoothL1(pred_df0, true_df0)
Testing / Validation Hard-Routing:
  Routes each mixture strictly to expert k* = argmax_k p_k.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

TIGER_ROOT = Path("/idiap/home/adubois/project_ma1/TIGER")
MOE_ROOT = TIGER_ROOT / "mixture_of_experts"
for p in (str(TIGER_ROOT), str(MOE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from look2hear.models import TIGER
from router_model import DeltaF0Router


class JointMoETIGER(nn.Module):
    def __init__(
        self,
        centers_semitones: list,
        thresholds_semitones: list,
        temperature: float = 1.0,
        router_channels: tuple = (8, 16, 32, 64),
        router_dropout: float = 0.4,
        num_experts: int = 5,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature

        self.register_buffer("centers", torch.tensor(centers_semitones, dtype=torch.float32))
        self.register_buffer("thresholds", torch.tensor(thresholds_semitones, dtype=torch.float32))

        # 1. Router initialized from scratch (SmoothL1 regression head)
        self.router = DeltaF0Router(
            num_classes=num_experts,
            sample_rate=16000,
            channels=router_channels,
            dropout=router_dropout,
            mode="regression",
        )

        # 2. 5 TIGER Tiny Specialists initialized from scratch
        self.experts = nn.ModuleList([
            TIGER(
                out_channels=24,
                in_channels=64,
                num_blocks=4,
                upsampling_depth=5,
                win=640,
                stride=160,
                num_sources=2,
                sample_rate=16000,
            )
            for _ in range(num_experts)
        ])

    def route(self, mix: torch.Tensor):
        """
        Compute continuous Delta F0 prediction (B,), soft routing weights p (B, K),
        and discrete quintile indices (B,) via thresholds.
        """
        pred_df0 = self.router(mix).squeeze(-1)  # (B,)
        # Differentiable soft weights around quintile centers
        dists = torch.abs(pred_df0.unsqueeze(-1) - self.centers.unsqueeze(0))  # (B, K)
        probs = F.softmax(-dists / self.temperature, dim=-1)  # (B, K)
        pred_idx = torch.bucketize(pred_df0, self.thresholds, right=True)
        return pred_df0, probs, pred_idx

    def forward_train(self, mix: torch.Tensor):
        """
        Forward pass for training (Method B):
        Runs router + all K=5 TIGER Tiny experts.
        Returns:
            pred_df0: (B,)
            probs: (B, K)
            ests_list: list of K tensors, each of shape (B, 2, L)
        """
        pred_df0, probs, pred_idx = self.route(mix)
        ests_list = [expert(mix) for expert in self.experts]
        return pred_df0, probs, pred_idx, ests_list

    def forward(self, mix: torch.Tensor):
        return self.forward_train(mix)

    @torch.no_grad()
    def forward_hard(self, mix: torch.Tensor):
        """
        Hard-routing inference pass: each sample is routed only to its assigned specialist.
        Returns:
            ests: (B, 2, L)
            pred_df0: (B,)
            probs: (B, K)
            pred_idx: (B,)
        """
        was_training = self.training
        self.eval()
        pred_df0, probs, pred_idx = self.route(mix)
        B, L = mix.shape
        ests = mix.new_zeros((B, 2, L))

        for k in range(self.num_experts):
            mask = (pred_idx == k)
            if mask.any():
                ests[mask] = self.experts[k](mix[mask])

        if was_training:
            self.train()
        return ests, pred_df0, probs, pred_idx
