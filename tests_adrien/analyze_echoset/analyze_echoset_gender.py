#!/usr/bin/env python3
"""
analyze_echoset_gender.py

Analyzes:
1. Gender pair distribution (M/M, M/F, F/F, Unknown)
2. Dataset quality pair distribution (clean/clean, clean/other, other/other)
3. Unique speaker distribution across clean vs other subsets
across the train, val, and test splits of the EchoSet dataset.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# ==============================================================================
# CONFIGURABLE CONSTANTS
# ==============================================================================
# Absolute path to LibriSpeech SPEAKERS metadata file
SPEAKERS_TXT_PATH = "/idiap/home/adubois/project_ma1/TIGER/tests_adrien/SPEAKERS.txt"

# Base directory for EchoSet splits containing {split}/s1.json
ECHOSET_BASE_DIR = "/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet"

# Directory where intermediate pair files and final statistics will be saved
OUTPUT_DIR = "."

# Splits to process
SPLITS = ["train", "val", "test"]

# Regex pattern to extract ID1 and ID2 from the pair directory name (e.g. '1995_5683')
PAIR_REGEX = re.compile(r"(\d+)_(\d+)")


def load_speakers_metadata(speakers_path: str | Path) -> dict[str, dict[str, str]]:
    """
    Parses SPEAKERS.txt and returns a dictionary mapping:
    speaker_id (str) -> {
        "gender": 'M' | 'F',
        "subset": str (e.g. 'train-clean-360'),
        "quality": 'clean' | 'other' | 'unknown'
    }
    """
    path = Path(speakers_path)
    if not path.is_file():
        raise FileNotFoundError(f"SPEAKERS file not found at: {path}")

    speaker_metadata: dict[str, dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comment lines and empty lines
            if not line or line.startswith(";"):
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 3:
                reader_id = parts[0]
                gender = parts[1].upper()
                subset = parts[2]

                subset_lower = subset.lower()
                if "clean" in subset_lower:
                    quality = "clean"
                elif "other" in subset_lower:
                    quality = "other"
                else:
                    quality = "unknown"

                if reader_id:
                    speaker_metadata[reader_id] = {
                        "gender": gender,
                        "subset": subset,
                        "quality": quality,
                    }

    print(f"Loaded {len(speaker_metadata)} speaker records from: {path}")
    return speaker_metadata


def extract_pairs_from_split(
    s1_json_path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    """
    Reads an s1.json file containing entries formatted as [path_to_wavfile, length].
    Extracts ID1 and ID2 from the immediate parent directory of each wav file.
    """
    with open(s1_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted_pairs: list[dict[str, str]] = []
    unparseable: list[str] = []

    for entry in data:
        if not entry or not isinstance(entry, (list, tuple)):
            continue
        wav_path_str = entry[0]
        parent_dir = Path(wav_path_str).parent.name

        match = PAIR_REGEX.search(parent_dir)
        if match:
            extracted_pairs.append({
                "id1": match.group(1),
                "id2": match.group(2),
            })
        else:
            unparseable.append(wav_path_str)

    return extracted_pairs, unparseable


def categorize_gender_pair(
    id1: str, id2: str, speaker_metadata: dict[str, dict[str, str]]
) -> tuple[str, list[str]]:
    """
    Categorizes pair into 'M/M', 'F/F', 'M/F', or 'Unknown'.
    """
    s1 = speaker_metadata.get(id1)
    s2 = speaker_metadata.get(id2)

    missing = []
    if s1 is None:
        missing.append(id1)
    if s2 is None:
        missing.append(id2)

    if missing:
        return "Unknown", missing

    g1 = s1["gender"]
    g2 = s2["gender"]

    if g1 == "M" and g2 == "M":
        return "M/M", []
    elif g1 == "F" and g2 == "F":
        return "F/F", []
    elif (g1 == "M" and g2 == "F") or (g1 == "F" and g2 == "M"):
        return "M/F", []
    else:
        return "Unknown", []


def categorize_quality_pair(
    id1: str, id2: str, speaker_metadata: dict[str, dict[str, str]]
) -> tuple[str, list[str]]:
    """
    Categorizes pair into 'clean/clean', 'clean/other', 'other/other', or 'Unknown'.
    """
    s1 = speaker_metadata.get(id1)
    s2 = speaker_metadata.get(id2)

    missing = []
    if s1 is None:
        missing.append(id1)
    if s2 is None:
        missing.append(id2)

    if missing:
        return "Unknown", missing

    q1 = s1["quality"]
    q2 = s2["quality"]

    if q1 == "clean" and q2 == "clean":
        return "clean/clean", []
    elif q1 == "other" and q2 == "other":
        return "other/other", []
    elif (q1 == "clean" and q2 == "other") or (q1 == "other" and q2 == "clean"):
        return "clean/other", []
    else:
        return "Unknown", []


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load speaker metadata
    try:
        speaker_metadata = load_speakers_metadata(SPEAKERS_TXT_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load SPEAKERS metadata: {e}", file=sys.stderr)
        sys.exit(1)

    all_stats: dict[str, dict] = {}
    overall_gender_counts: Counter = Counter()
    overall_quality_counts: Counter = Counter()
    overall_unique_speakers: set[str] = set()

    # 2. Process each split
    for split in SPLITS:
        s1_file = Path(ECHOSET_BASE_DIR) / split / "s1.json"
        if not s1_file.is_file():
            print(f"[WARNING] Split file not found: {s1_file}. Skipping split '{split}'.", file=sys.stderr)
            continue

        print(f"\nProcessing '{split}' split from: {s1_file}")
        pairs, unparseable = extract_pairs_from_split(s1_file)

        if unparseable:
            print(f"  [WARNING] Could not extract IDs from {len(unparseable)} paths in split '{split}'.", file=sys.stderr)

        # Save intermediate pairs file
        pairs_output_file = out_dir / f"echoset_pairs_{split}.json"
        with open(pairs_output_file, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2)
        print(f"  Saved {len(pairs)} extracted pairs -> {pairs_output_file}")

        # Count gender and quality
        gender_counts: Counter = Counter()
        quality_counts: Counter = Counter()
        missing_ids_set: set[str] = set()
        split_speakers: set[str] = set()

        for p in pairs:
            id1, id2 = p["id1"], p["id2"]
            split_speakers.add(id1)
            split_speakers.add(id2)

            g_cat, g_missing = categorize_gender_pair(id1, id2, speaker_metadata)
            gender_counts[g_cat] += 1
            if g_missing:
                missing_ids_set.update(g_missing)

            q_cat, _ = categorize_quality_pair(id1, id2, speaker_metadata)
            quality_counts[q_cat] += 1

        overall_unique_speakers.update(split_speakers)
        overall_gender_counts.update(gender_counts)
        overall_quality_counts.update(quality_counts)

        if missing_ids_set:
            print(
                f"  [WARNING] {len(missing_ids_set)} speaker ID(s) missing from SPEAKERS.txt in '{split}' split: "
                f"{sorted(missing_ids_set)}",
                file=sys.stderr,
            )

        total_pairs = len(pairs)

        # Unique speakers stats
        spk_quality_counts: Counter = Counter()
        spk_subset_counts: Counter = Counter()
        for spk in split_speakers:
            info = speaker_metadata.get(spk)
            if info:
                spk_quality_counts[info["quality"]] += 1
                spk_subset_counts[info["subset"]] += 1
            else:
                spk_quality_counts["Unknown"] += 1
                spk_subset_counts["Unknown"] += 1

        total_split_speakers = len(split_speakers)

        gender_percentages = {
            cat: (gender_counts[cat] / total_pairs * 100.0) if total_pairs > 0 else 0.0
            for cat in ["M/M", "M/F", "F/F", "Unknown"]
        }
        quality_percentages = {
            cat: (quality_counts[cat] / total_pairs * 100.0) if total_pairs > 0 else 0.0
            for cat in ["clean/clean", "clean/other", "other/other", "Unknown"]
        }
        spk_quality_percentages = {
            cat: (spk_quality_counts[cat] / total_split_speakers * 100.0) if total_split_speakers > 0 else 0.0
            for cat in ["clean", "other", "Unknown"]
        }

        all_stats[split] = {
            "total_pairs": total_pairs,
            "gender_pair_distribution": {
                "counts": {cat: gender_counts[cat] for cat in ["M/M", "M/F", "F/F", "Unknown"]},
                "percentages": {cat: round(gender_percentages[cat], 2) for cat in ["M/M", "M/F", "F/F", "Unknown"]},
            },
            "dataset_quality_pair_distribution": {
                "counts": {cat: quality_counts[cat] for cat in ["clean/clean", "clean/other", "other/other", "Unknown"]},
                "percentages": {cat: round(quality_percentages[cat], 2) for cat in ["clean/clean", "clean/other", "other/other", "Unknown"]},
            },
            "unique_speakers": {
                "total": total_split_speakers,
                "counts": {cat: spk_quality_counts[cat] for cat in ["clean", "other", "Unknown"]},
                "percentages": {cat: round(spk_quality_percentages[cat], 2) for cat in ["clean", "other", "Unknown"]},
                "subset_counts": dict(spk_subset_counts),
            },
            "missing_speaker_ids": sorted(missing_ids_set),
        }

    # Overall statistics
    total_all_pairs = sum(overall_gender_counts.values())
    total_all_spk = len(overall_unique_speakers)

    all_spk_quality: Counter = Counter()
    all_spk_subsets: Counter = Counter()
    for spk in overall_unique_speakers:
        info = speaker_metadata.get(spk)
        if info:
            all_spk_quality[info["quality"]] += 1
            all_spk_subsets[info["subset"]] += 1
        else:
            all_spk_quality["Unknown"] += 1
            all_spk_subsets["Unknown"] += 1

    all_stats["overall"] = {
        "total_pairs": total_all_pairs,
        "gender_pair_distribution": {
            "counts": {cat: overall_gender_counts[cat] for cat in ["M/M", "M/F", "F/F", "Unknown"]},
            "percentages": {
                cat: round((overall_gender_counts[cat] / total_all_pairs * 100.0), 2) if total_all_pairs > 0 else 0.0
                for cat in ["M/M", "M/F", "F/F", "Unknown"]
            },
        },
        "dataset_quality_pair_distribution": {
            "counts": {cat: overall_quality_counts[cat] for cat in ["clean/clean", "clean/other", "other/other", "Unknown"]},
            "percentages": {
                cat: round((overall_quality_counts[cat] / total_all_pairs * 100.0), 2) if total_all_pairs > 0 else 0.0
                for cat in ["clean/clean", "clean/other", "other/other", "Unknown"]
            },
        },
        "unique_speakers": {
            "total": total_all_spk,
            "counts": {cat: all_spk_quality[cat] for cat in ["clean", "other", "Unknown"]},
            "percentages": {
                cat: round((all_spk_quality[cat] / total_all_spk * 100.0), 2) if total_all_spk > 0 else 0.0
                for cat in ["clean", "other", "Unknown"]
            },
            "subset_counts": dict(all_spk_subsets),
        },
    }

    # ==========================================================================
    # 3. Print Summary Tables
    # ==========================================================================

    # --- TABLE 1: Gender Distribution ---
    print("\n" + "=" * 94)
    print("                    1. ECHOSET GENDER PAIR DISTRIBUTION SUMMARY")
    print("=" * 94)
    print(f"{'Split':<10} | {'Total':<10} | {'M/M Count (%)':<16} | {'M/F Count (%)':<16} | {'F/F Count (%)':<16} | {'Unknown (%)':<14}")
    print("-" * 94)
    for split in SPLITS:
        if split not in all_stats:
            continue
        g = all_stats[split]["gender_pair_distribution"]
        total = all_stats[split]["total_pairs"]
        print(
            f"{split:<10} | {total:<10} | "
            f"{g['counts']['M/M']} ({g['percentages']['M/M']:5.2f}%) | "
            f"{g['counts']['M/F']} ({g['percentages']['M/F']:5.2f}%) | "
            f"{g['counts']['F/F']} ({g['percentages']['F/F']:5.2f}%) | "
            f"{g['counts']['Unknown']} ({g['percentages']['Unknown']:5.2f}%)"
        )
    print("-" * 94)
    ov_g = all_stats["overall"]["gender_pair_distribution"]
    print(
        f"{'TOTAL':<10} | {total_all_pairs:<10} | "
        f"{ov_g['counts']['M/M']} ({ov_g['percentages']['M/M']:5.2f}%) | "
        f"{ov_g['counts']['M/F']} ({ov_g['percentages']['M/F']:5.2f}%) | "
        f"{ov_g['counts']['F/F']} ({ov_g['percentages']['F/F']:5.2f}%) | "
        f"{ov_g['counts']['Unknown']} ({ov_g['percentages']['Unknown']:5.2f}%)"
    )
    print("=" * 94)

    # --- TABLE 2: Dataset Quality (Clean vs Other) Pairs ---
    print("\n" + "=" * 98)
    print("                2. ECHOSET DATASET PAIR DISTRIBUTION (Clean vs Other)")
    print("=" * 98)
    print(f"{'Split':<10} | {'Total':<10} | {'clean/clean (%)':<18} | {'clean/other (%)':<18} | {'other/other (%)':<18} | {'Unknown (%)':<12}")
    print("-" * 98)
    for split in SPLITS:
        if split not in all_stats:
            continue
        q = all_stats[split]["dataset_quality_pair_distribution"]
        total = all_stats[split]["total_pairs"]
        print(
            f"{split:<10} | {total:<10} | "
            f"{q['counts']['clean/clean']} ({q['percentages']['clean/clean']:5.2f}%) | "
            f"{q['counts']['clean/other']} ({q['percentages']['clean/other']:5.2f}%) | "
            f"{q['counts']['other/other']} ({q['percentages']['other/other']:5.2f}%) | "
            f"{q['counts']['Unknown']} ({q['percentages']['Unknown']:5.2f}%)"
        )
    print("-" * 98)
    ov_q = all_stats["overall"]["dataset_quality_pair_distribution"]
    print(
        f"{'TOTAL':<10} | {total_all_pairs:<10} | "
        f"{ov_q['counts']['clean/clean']} ({ov_q['percentages']['clean/clean']:5.2f}%) | "
        f"{ov_q['counts']['clean/other']} ({ov_q['percentages']['clean/other']:5.2f}%) | "
        f"{ov_q['counts']['other/other']} ({ov_q['percentages']['other/other']:5.2f}%) | "
        f"{ov_q['counts']['Unknown']} ({ov_q['percentages']['Unknown']:5.2f}%)"
    )
    print("=" * 98)

    # --- TABLE 3: Unique Speaker Breakdown ---
    print("\n" + "=" * 94)
    print("                3. UNIQUE SPEAKERS BREAKDOWN (Clean vs Other)")
    print("=" * 94)
    print(f"{'Split':<10} | {'Unique Spk':<12} | {'Clean Count (%)':<20} | {'Other Count (%)':<20} | {'Unknown (%)':<12}")
    print("-" * 94)
    for split in SPLITS:
        if split not in all_stats:
            continue
        u = all_stats[split]["unique_speakers"]
        print(
            f"{split:<10} | {u['total']:<12} | "
            f"{u['counts']['clean']} ({u['percentages']['clean']:5.2f}%) | "
            f"{u['counts']['other']} ({u['percentages']['other']:5.2f}%) | "
            f"{u['counts']['Unknown']} ({u['percentages']['Unknown']:5.2f}%)"
        )
    print("-" * 94)
    ov_u = all_stats["overall"]["unique_speakers"]
    print(
        f"{'TOTAL':<10} | {ov_u['total']:<12} | "
        f"{ov_u['counts']['clean']} ({ov_u['percentages']['clean']:5.2f}%) | "
        f"{ov_u['counts']['other']} ({ov_u['percentages']['other']:5.2f}%) | "
        f"{ov_u['counts']['Unknown']} ({ov_u['percentages']['Unknown']:5.2f}%)"
    )
    print("=" * 94 + "\n")

    # 4. Save JSON summary
    summary_path = out_dir / "echoset_gender_distribution.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Full statistics saved to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
