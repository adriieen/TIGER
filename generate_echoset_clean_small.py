#!/usr/bin/env python3
"""
generate_echoset_clean.py

Generate clean spatial/non-reverberant speech mixtures for EchoSet clean dataset
using LibriSpeech subsets (train-clean-100, dev-clean, test-clean) and speaker pair
definitions from EchoSet JSON files.

Constraints & Features:
- Duration: T = 6.0 seconds (96,000 samples @ 16 kHz, mono WAV)
- Speaker pairs: loaded from echoset_pairs_[train|val|test].json
- Disambiguation: occurrences of identical pairs are placed into subfolders 00/, 01/, etc.
- Audio synthesis:
    s1: starts at t=0, 6s slice if longer, zero-padded if shorter.
    s2: delayed by delta ~ U[0, max_delay] seconds (default max_delay=3.0s, guaranteeing
        at least 3s of active speech overlap), scaled by alpha according to SDR ~ U[-5dB, 5dB].
        Truncated if exceeding the 6-second window.
    Normalize: peak-normalization with safety margin (0.9 / max_peak) applied jointly
               so that s1 + s2 == mix without clipping.
- Outputs:
    - mix.wav, s1.wav, s2.wav (with optional symlinks to spk1_reverb.wav, spk2_reverb.wav)
    - Metadata JSONs: mix.json, s1.json, s2.json (TIGER compatible format)
- Multiprocessing support for fast parallel generation.
"""

import os
import sys
import json
import glob
import random
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import soundfile as sf
from tqdm import tqdm


SUBSET_CONFIGS = {
    "train": {
        "pairs_file": "echoset_pairs_train.json",
        "librispeech_subdir": "train-clean-100",
    },
    "val": {
        "pairs_file": "echoset_pairs_val.json",
        "librispeech_subdir": "dev-clean",
    },
    "test": {
        "pairs_file": "echoset_pairs_test.json",
        "librispeech_subdir": "test-clean",
    },
}

TARGET_SR = 16000
TARGET_DURATION = 6.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 96000 samples


def index_librispeech_speakers(librispeech_subdir_path):
    """
    Index all .flac audio files for each speaker in the given LibriSpeech subset.
    Returns: dict mapping speaker_id (str) -> list of file paths (str).
    """
    print(f"Indexing LibriSpeech files in {librispeech_subdir_path}...")
    speaker_files = defaultdict(list)
    
    if not os.path.isdir(librispeech_subdir_path):
        raise FileNotFoundError(f"LibriSpeech directory not found: {librispeech_subdir_path}")

    # LibriSpeech structure: <subset>/<speaker_id>/<chapter_id>/*.flac
    with os.scandir(librispeech_subdir_path) as spk_entries:
        for spk_entry in spk_entries:
            if spk_entry.is_dir() and not spk_entry.name.startswith("."):
                spk_id = spk_entry.name
                for root, _, files in os.walk(spk_entry.path):
                    for file in files:
                        if file.endswith(".flac"):
                            speaker_files[spk_id].append(os.path.join(root, file))

    print(f"Found {len(speaker_files)} speakers with total "
          f"{sum(len(v) for v in speaker_files.values())} audio files.")
    return speaker_files


def process_single_pair(args_tuple):
    """
    Worker function to generate a single clean mixture sample.
    """
    (
        sample_idx,
        spk1_id,
        spk2_id,
        folder_prefix,
        spk1_file_list,
        spk2_file_list,
        out_dir_sample,
        seed,
        create_symlinks,
        max_delay,
    ) = args_tuple

    # Set independent deterministic seed for reproducibility
    rng = random.Random(seed)

    # 1. Choose random audio file for spk1 and spk2
    f1 = rng.choice(spk1_file_list)
    f2 = rng.choice(spk2_file_list)

    # 2. Read audio signals (16 kHz mono)
    x1, sr1 = sf.read(f1, dtype="float32")
    x2, sr2 = sf.read(f2, dtype="float32")

    if x1.ndim > 1:
        x1 = x1[:, 0]
    if x2.ndim > 1:
        x2 = x2[:, 0]

    # Calculate raw total energy E(s) = sum(s^2)
    e1 = float(np.sum(x1 ** 2))
    e2 = float(np.sum(x2 ** 2))
    e1 = max(e1, 1e-12)
    e2 = max(e2, 1e-12)

    # 3. SDR uniformly pooled between -5 dB and 5 dB
    # SDR = 10 * log10(E1 / E2) - 20 * log10(alpha)
    # => alpha = sqrt(E1 / E2) * 10^(-SDR / 20)
    sdr = rng.uniform(-5.0, 5.0)
    alpha = np.sqrt(e1 / e2) * (10.0 ** (-sdr / 20.0))

    # 4. Prepare s1 (duration T = 6.0s = 96000 samples, starts at t=0)
    if len(x1) > TARGET_SAMPLES:
        start1 = rng.randint(0, len(x1) - TARGET_SAMPLES)
        s1 = x1[start1 : start1 + TARGET_SAMPLES]
    elif len(x1) < TARGET_SAMPLES:
        s1 = np.pad(x1, (0, TARGET_SAMPLES - len(x1)), mode="constant")
    else:
        s1 = x1.copy()

    # 5. Prepare s2 (delayed by delta in [0, max_delay] seconds, scaled by alpha)
    # With max_delay <= 3.0s, at least 3.0s of overlap is guaranteed
    delta_sec = rng.uniform(0.0, max_delay)
    delay_samples = int(delta_sec * TARGET_SR)
    delay_samples = min(max(delay_samples, 0), TARGET_SAMPLES - 1)

    # Scale x2
    x2_scaled = (alpha * x2).astype(np.float32)

    # s2 container
    s2 = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    avail_len = TARGET_SAMPLES - delay_samples
    to_copy = min(len(x2_scaled), avail_len)
    s2[delay_samples : delay_samples + to_copy] = x2_scaled[:to_copy]

    # 6. Normalize mixture and sources to avoid clipping
    mix_unnorm = s1 + s2
    max_peak = float(np.max(np.abs(mix_unnorm)))
    if max_peak > 0:
        norm_factor = 0.9 / max_peak
    else:
        norm_factor = 1.0

    s1_norm = (s1 * norm_factor).astype(np.float32)
    s2_norm = (s2 * norm_factor).astype(np.float32)
    mix_norm = s1_norm + s2_norm

    # 7. Write WAV files (16-bit PCM, 16 kHz)
    os.makedirs(out_dir_sample, exist_ok=True)
    mix_path = os.path.join(out_dir_sample, "mix.wav")
    s1_path = os.path.join(out_dir_sample, "s1.wav")
    s2_path = os.path.join(out_dir_sample, "s2.wav")

    sf.write(mix_path, mix_norm, TARGET_SR, subtype="PCM_16")
    sf.write(s1_path, s1_norm, TARGET_SR, subtype="PCM_16")
    sf.write(s2_path, s2_norm, TARGET_SR, subtype="PCM_16")

    # 8. Save metadata (SNR of spk1 relative to spk2, delay in seconds)
    metadata_path = os.path.join(out_dir_sample, "metadata.txt")
    with open(metadata_path, "w") as f_meta:
        f_meta.write(f"snr: {sdr:.6f}\n")
        f_meta.write(f"delay: {delta_sec:.6f}\n")

    # Optional symlinks for backward compatibility with process_echoset.py
    if create_symlinks:
        spk1_reverb_path = os.path.join(out_dir_sample, "spk1_reverb.wav")
        spk2_reverb_path = os.path.join(out_dir_sample, "spk2_reverb.wav")
        if not os.path.exists(spk1_reverb_path):
            try:
                os.symlink("s1.wav", spk1_reverb_path)
            except OSError:
                pass
        if not os.path.exists(spk2_reverb_path):
            try:
                os.symlink("s2.wav", spk2_reverb_path)
            except OSError:
                pass

    return (
        sample_idx,
        mix_path,
        s1_path,
        s2_path,
        TARGET_SAMPLES,
    )


def generate_subset(
    subset,
    pairs_json_path,
    librispeech_subdir_path,
    output_base_dir,
    output_json_dirs,
    num_workers=8,
    base_seed=42,
    create_symlinks=True,
    max_samples=None,
    max_delay=3.0,
):
    print(f"\n=======================================================")
    print(f"Generating EchoSet Clean for subset: '{subset}'")
    print(f"Pairs file: {pairs_json_path}")
    print(f"LibriSpeech source: {librispeech_subdir_path}")
    print(f"Output directory: {os.path.join(output_base_dir, subset)}")
    print(f"Max delay: {max_delay:.2f}s (Guaranteed overlap >= {TARGET_DURATION - max_delay:.2f}s)")
    print(f"=======================================================")

    # 1. Load pairs
    with open(pairs_json_path, "r") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} speaker pairs.")
    if subset == "train":
        pairs = pairs[::5]
        print(f"Subsampled '{subset}' to 20% (1 sample every 5): {len(pairs)} pairs retained.")
    if max_samples is not None and max_samples < len(pairs):
        pairs = pairs[:max_samples]
        print(f"Limiting to first {max_samples} pairs for this run.")

    # 2. Index speaker audio files in LibriSpeech
    speaker_files = index_librispeech_speakers(librispeech_subdir_path)

    # 3. Prepare task list with occurrence disambiguation
    pair_occurrence_count = defaultdict(int)
    tasks = []

    for idx, pair in enumerate(pairs):
        id1 = str(pair["id1"])
        id2 = str(pair["id2"])

        if id1 not in speaker_files or len(speaker_files[id1]) == 0:
            raise KeyError(f"Speaker ID {id1} has no audio files in {librispeech_subdir_path}")
        if id2 not in speaker_files or len(speaker_files[id2]) == 0:
            raise KeyError(f"Speaker ID {id2} has no audio files in {librispeech_subdir_path}")

        occ = pair_occurrence_count[(id1, id2)]
        pair_occurrence_count[(id1, id2)] += 1
        folder_prefix = f"{occ:02d}"

        # Sample output dir: [output_base_dir]/[subset]/[00|01]/[id1]_[id2]
        out_dir_sample = os.path.join(output_base_dir, subset, folder_prefix, f"{id1}_{id2}")
        sample_seed = base_seed + idx * 31

        tasks.append((
            idx,
            id1,
            id2,
            folder_prefix,
            speaker_files[id1],
            speaker_files[id2],
            out_dir_sample,
            sample_seed,
            create_symlinks,
            max_delay,
        ))

    # 4. Execute tasks with multiprocessing
    results = [None] * len(tasks)
    print(f"Processing {len(tasks)} samples using {num_workers} workers...")

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for res in tqdm(executor.map(process_single_pair, tasks, chunksize=16),
                            total=len(tasks), desc=f"Subset {subset}"):
                sample_idx = res[0]
                results[sample_idx] = res
    else:
        for t in tqdm(tasks, desc=f"Subset {subset}"):
            res = process_single_pair(t)
            sample_idx = res[0]
            results[sample_idx] = res

    # 5. Build JSON metadata records
    mix_infos = []
    s1_infos = []
    s2_infos = []

    for res in results:
        _, mix_path, s1_path, s2_path, n_samples = res
        mix_infos.append([mix_path, n_samples])
        s1_infos.append([s1_path, n_samples])
        s2_infos.append([s2_path, n_samples])

    # 6. Save JSON metadata files in all specified output JSON directories
    for json_dir in output_json_dirs:
        target_dir = os.path.join(json_dir, subset)
        os.makedirs(target_dir, exist_ok=True)
        
        with open(os.path.join(target_dir, "mix.json"), "w") as f:
            json.dump(mix_infos, f, indent=4)
        with open(os.path.join(target_dir, "s1.json"), "w") as f:
            json.dump(s1_infos, f, indent=4)
        with open(os.path.join(target_dir, "s2.json"), "w") as f:
            json.dump(s2_infos, f, indent=4)
        print(f"Metadata saved to: {target_dir}/(mix.json, s1.json, s2.json)")

    print(f"Subset '{subset}' completed successfully ({len(mix_infos)} samples generated).")


def main():
    parser = argparse.ArgumentParser(
        description="Generate clean speech mixtures for EchoSet clean dataset."
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["test", "val", "train"],
        choices=["train", "val", "test", "all"],
        help="Subsets to process (default: test val train)",
    )
    parser.add_argument(
        "--pairs_dir",
        type=str,
        default="/idiap/home/adubois/project_ma1/TIGER",
        help="Directory containing echoset_pairs_[train|val|test].json",
    )
    parser.add_argument(
        "--librispeech_dir",
        type=str,
        default="/idiap/resource/database/LibriSpeech",
        help="Root directory of LibriSpeech dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/idiap/temp/adubois/EchoSet_cleaned_small",
        help="Base directory to save generated audio files",
    )
    parser.add_argument(
        "--out_json_dir",
        type=str,
        default="/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_cleaned_small",
        help="Additional directory to save output JSON files for TIGER",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
        help="Number of parallel worker processes (default: $SLURM_CPUS_PER_TASK or 8)",
    )
    parser.add_argument(
        "--max_delay",
        type=float,
        default=3.0,
        help="Maximum delay in seconds for speaker 2 (default: 3.0s, guaranteeing at least 3s of overlap)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples to generate per subset (useful for test runs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no_symlinks",
        action="store_true",
        help="Disable creation of spk1_reverb.wav / spk2_reverb.wav symlinks",
    )

    args = parser.parse_args()

    subsets = args.subsets
    if "all" in subsets:
        subsets = ["test", "val", "train"]

    # Target JSON directories: both in the audio output directory and the TIGER DataPreProcess directory
    json_dirs = [args.output_dir]
    if args.out_json_dir and args.out_json_dir != args.output_dir:
        json_dirs.append(args.out_json_dir)

    for subset in subsets:
        cfg = SUBSET_CONFIGS[subset]
        pairs_json_path = os.path.join(args.pairs_dir, cfg["pairs_file"])
        librispeech_subdir_path = os.path.join(args.librispeech_dir, cfg["librispeech_subdir"])

        generate_subset(
            subset=subset,
            pairs_json_path=pairs_json_path,
            librispeech_subdir_path=librispeech_subdir_path,
            output_base_dir=args.output_dir,
            output_json_dirs=json_dirs,
            num_workers=args.num_workers,
            base_seed=args.seed,
            create_symlinks=not args.no_symlinks,
            max_samples=args.max_samples,
            max_delay=args.max_delay,
        )

    print("\nAll requested subsets generated successfully!")


if __name__ == "__main__":
    main()
