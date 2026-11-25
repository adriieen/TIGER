#!/usr/bin/env python3
"""
router_model.py

Autonomous definition of `DeltaF0Router`:
A lightweight 2D-ResNet acoustic regime classifier operating on raw 16 kHz mono
mixed speech waveforms `(B, L)` to predict logits over the 5 Delta F0 quintiles
(Q1 to Q5) for dynamic routing to specialized TIGER checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2D(nn.Module):
    """Standard 2D Residual Block with 1x1 projection shortcut on resolution changes."""

    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act(out)


class DeltaF0Router(nn.Module):
    """
    STFT Front-end + ResNet2D Backbone for 5-class Delta F0 quintile routing.

    STFT Front-end:
        - fs = 16000 Hz
        - n_fft = 2000 (df = 8.0 Hz/bin), win_length = 2000 (125 ms), hop_length = 500 (31.25 ms)
        - Hann window registered as buffer
        - Log-magnitude: log(|X| + 1e-6)
        - Frequency crop: first 375 bins [0 Hz, 3000 Hz[
        - Per-utterance CMVN over (F, T)
        - Optional light Time Masking (<= 15% frames) during training only
    """

    def __init__(
        self,
        num_classes: int = 5,
        sample_rate: int = 16000,
        n_fft: int = 2000,
        win_length: int = 2000,
        hop_length: int = 500,
        freq_bins: int = 375,
        channels: tuple = (8, 16, 32, 64),
        dropout: float = 0.4,
        time_mask_max_ratio: float = 0.15,
        mode: str = "classification",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.freq_bins = freq_bins
        self.channels = tuple(channels)
        self.time_mask_max_ratio = time_mask_max_ratio
        self.mode = mode

        c1, c2, c3, c4 = self.channels
        self.register_buffer("window", torch.hann_window(win_length))

        # Stem: (B, 1, 375, T) -> (B, c1, 188, T)
        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                c1,
                kernel_size=(7, 5),
                stride=(2, 1),
                padding=(3, 2),
                bias=False,
            ),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )

        # Residual Stages
        self.stage1 = nn.Sequential(
            ResidualBlock2D(c1, c1, stride=(1, 1)),
            ResidualBlock2D(c1, c1, stride=(1, 1)),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock2D(c1, c2, stride=(2, 2)),
            ResidualBlock2D(c2, c2, stride=(1, 1)),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock2D(c2, c3, stride=(2, 2)),
            ResidualBlock2D(c3, c3, stride=(1, 1)),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock2D(c3, c4, stride=(2, 2)),
            ResidualBlock2D(c4, c4, stride=(1, 1)),
        )

        # Duration-invariant aggregation & head
        out_dim = 1 if mode == "regression" else num_classes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c4, out_dim),
        )

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert raw waveform (B, L) or (B, 1, L) into normalized log-spectrogram (B, 1, 375, T).
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        X = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        # Magnitude + log compression
        M = torch.abs(X) + 1e-6
        S = torch.log(M)

        # Keep [0 Hz, 3000 Hz[ -> 375 bins
        S = S[:, : self.freq_bins, :]

        # Per-utterance CMVN over (F, T)
        mean = S.mean(dim=(-2, -1), keepdim=True)
        std = S.std(dim=(-2, -1), keepdim=True) + 1e-6
        S_norm = (S - mean) / std

        # Light Time Masking (SpecAugment) during training only (no frequency masking!)
        if self.training and self.time_mask_max_ratio > 0.0:
            B, _, T = S_norm.shape
            max_mask = max(1, int(T * self.time_mask_max_ratio))
            for b in range(B):
                mask_len = int(torch.randint(0, max_mask + 1, (1,), device=S_norm.device).item())
                if mask_len > 0 and mask_len < T:
                    t0 = int(torch.randint(0, T - mask_len + 1, (1,), device=S_norm.device).item())
                    S_norm[b, :, t0 : t0 + mask_len] = 0.0

        return S_norm.unsqueeze(1)  # (B, 1, 375, T)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape (B, L) or (B, 1, L) at 16 kHz.
        Returns:
            logits: Raw class logits of shape (B, 5) for [Q1, Q2, Q3, Q4, Q5].
        """
        x = self.extract_features(waveform)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.head(x)

    @torch.no_grad()
    def predict_expert(self, waveform: torch.Tensor, thresholds=None):
        """
        Inference helper for dynamic routing.

        Args:
            waveform: Tensor of shape (B, L) or (L,) at 16 kHz.
            thresholds: List/Tensor of 4 thresholds (semitones) when mode='regression'.
        Returns:
            expert_idx: Tensor of argmax/bucketized class indices in {0, 1, 2, 3, 4} (Q1..Q5).
            probs: Softmax probability tensor of shape (B, 5).
        """
        was_training = self.training
        self.eval()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        out = self.forward(waveform)
        if self.mode == "regression":
            if thresholds is None:
                raise ValueError("`thresholds` must be provided when mode='regression'.")
            thresh_t = torch.as_tensor(thresholds, device=out.device, dtype=out.dtype)
            pred_f0 = out.squeeze(-1)
            expert_idx = torch.bucketize(pred_f0, thresh_t, right=True)
            # Soft assignment weights based on distance to quintile centers
            centers = torch.tensor(
                [
                    thresh_t[0] * 0.5,
                    0.5 * (thresh_t[0] + thresh_t[1]),
                    0.5 * (thresh_t[1] + thresh_t[2]),
                    0.5 * (thresh_t[2] + thresh_t[3]),
                    thresh_t[3] + 2.0,
                ],
                device=out.device,
                dtype=out.dtype,
            )
            probs = F.softmax(-torch.abs(pred_f0.unsqueeze(-1) - centers), dim=-1)
        else:
            probs = F.softmax(out, dim=-1)
            expert_idx = torch.argmax(probs, dim=-1)
        if was_training:
            self.train()
        return expert_idx, probs
