#!/usr/bin/env python3
"""
dataset.py

Dataset & threshold utilities for joint training of `DeltaF0Router` + 5 TIGER Tiny
specialists on `EchoSet_cleaned_small`.

Each sample returns:
    (mixture, sources, delta_f0, label)
where:
    - mixture: (L,) float32 tensor
    - sources: (2, L) float32 tensor
    - delta_f0: float32 scalar (semitones, from eGeMAPS train/val/test parquet)
    - label: int64 scalar in {0, 1, 2, 3, 4} (fixed train quintile index Q1..Q5)
"""

import os
import re
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


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


def compute_train_thresholds(
    train_mix_json: Path,
    train_parquet: Path,
    f0_col: str = "dist_F0semitoneFrom27.5Hz_sma3nz_amean",
) -> dict:
    """Compute exact 5-quintile boundaries and representative bin centers from the training set."""
    with open(train_mix_json, "r") as f:
        train_mix = json.load(f)

    df_p = pd.read_parquet(train_parquet)
    pair_col = [c for c in df_p.columns if any(k in c.lower() for k in ["pair", "key", "utt"])][0]
    df_p["clean_id"] = df_p[pair_col].apply(clean_pair_key)
    df_p = df_p.drop_duplicates(subset=["clean_id"]).set_index("clean_id")

    train_ids = [clean_pair_key(e[0]) for e in train_mix]
    vals = df_p.loc[[i for i in train_ids if i in df_p.index], f0_col].values
    vals = np.sort(vals[np.isfinite(vals)])

    thresholds = [float(q) for q in np.quantile(vals, [0.2, 0.4, 0.6, 0.8])]
    # Empirical medians inside each train quintile serve as clean centers for the RBF/Laplace softmax
    labels = np.digitize(vals, thresholds)
    centers = [float(np.median(vals[labels == k])) for k in range(5)]

    return {
        "feature": f0_col,
        "thresholds_semitones": thresholds,
        "centers_semitones": centers,
        "num_train_samples": int(len(vals)),
    }


def load_split_records(
    split_dir: Path,
    parquet_path: Path,
    thresholds: list,
    f0_col: str = "dist_F0semitoneFrom27.5Hz_sma3nz_amean",
) -> list:
    """Align mix/s1/s2 JSON paths with eGeMAPS Delta F0 labels."""
    with open(split_dir / "mix.json", "r") as f:
        mix_infos = json.load(f)
    with open(split_dir / "s1.json", "r") as f:
        s1_infos = json.load(f)
    with open(split_dir / "s2.json", "r") as f:
        s2_infos = json.load(f)

    df_p = pd.read_parquet(parquet_path)
    pair_col = [c for c in df_p.columns if any(k in c.lower() for k in ["pair", "key", "utt"])][0]
    df_p["clean_id"] = df_p[pair_col].apply(clean_pair_key)
    df_p = df_p.drop_duplicates(subset=["clean_id"]).set_index("clean_id")

    records = []
    for m_ent, s1_ent, s2_ent in zip(mix_infos, s1_infos, s2_infos):
        mix_path = m_ent[0]
        cid = clean_pair_key(mix_path)
        if cid in df_p.index:
            df0 = float(df_p.loc[cid, f0_col])
            if np.isfinite(df0):
                label = int(np.digitize(df0, thresholds))
                records.append({
                    "pair_id": cid,
                    "mix_path": mix_path,
                    "s1_path": s1_ent[0],
                    "s2_path": s2_ent[0],
                    "n_samples": int(m_ent[1]),
                    "delta_f0": df0,
                    "label": label,
                })
    return records


class MoEEchoSetDataset(Dataset):
    """Dataset yielding (mixture, sources, delta_f0, label) for joint Router + MoE training."""

    def __init__(
        self,
        records: list,
        sample_rate: int = 16000,
        segment: float = 3.0,
        is_train: bool = True,
        min_active_rms_db: float = -20.0,
    ):
        self.records = records
        self.sample_rate = sample_rate
        self.seg_len = int(round(segment * sample_rate)) if segment is not None else None
        self.is_train = is_train
        self.rms_ratio_thresh = 10.0 ** (min_active_rms_db / 20.0)

    def __len__(self) -> int:
        return len(self.records)

    def _read_mono(self, path: str) -> np.ndarray:
        wav, _ = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]
        return wav

    def _pick_active_start(self, s1: np.ndarray, s2: np.ndarray, total_len: int) -> int:
        """Pick a random crop start where both speakers have RMS >= -20 dB vs full clip."""
        C = self.seg_len
        if total_len <= C:
            return 0
        rms1_full = float(np.sqrt(np.mean(s1 ** 2) + 1e-12))
        rms2_full = float(np.sqrt(np.mean(s2 ** 2) + 1e-12))

        best_start = 0
        best_score = -1.0
        for _ in range(8):
            start = random.randint(0, total_len - C)
            r1 = float(np.sqrt(np.mean(s1[start : start + C] ** 2) + 1e-12)) / rms1_full
            r2 = float(np.sqrt(np.mean(s2[start : start + C] ** 2) + 1e-12)) / rms2_full
            score = min(r1, r2)
            if score >= self.rms_ratio_thresh:
                return start
            if score > best_score:
                best_score = score
                best_start = start
        return best_start

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        mix = self._read_mono(rec["mix_path"])
        s1 = self._read_mono(rec["s1_path"])
        s2 = self._read_mono(rec["s2_path"])

        L = len(mix)
        if self.seg_len is not None:
            C = self.seg_len
            if L < C:
                pad = C - L
                mix = np.pad(mix, (0, pad), mode="constant")
                s1 = np.pad(s1, (0, pad), mode="constant")
                s2 = np.pad(s2, (0, pad), mode="constant")
            elif L > C:
                if self.is_train:
                    start = self._pick_active_start(s1, s2, L)
                else:
                    start = (L - C) // 2
                mix = mix[start : start + C]
                s1 = s1[start : start + C]
                s2 = s2[start : start + C]

        sources = np.stack([s1, s2], axis=0).astype(np.float32)  # (2, L)
        mix = mix.astype(np.float32)  # (L,)

        return (
            torch.from_numpy(mix),
            torch.from_numpy(sources),
            torch.tensor(rec["delta_f0"], dtype=torch.float32),
            torch.tensor(rec["label"], dtype=torch.long),
        )
