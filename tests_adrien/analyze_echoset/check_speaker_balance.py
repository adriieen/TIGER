#!/usr/bin/env python3
"""
check_speaker_balance.py

Vérifie l'équité de présence des locuteurs dans EchoSet et génère les fichiers
de coefficients de biais (alpha et poids de correction) dans tests_adrien/echoset_bias/.
"""

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

SPEAKERS_TXT = "/idiap/home/adubois/project_ma1/TIGER/tests_adrien/SPEAKERS.txt"
ECHOSET_BASE_DIR = "/idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet"
BIAS_OUTPUT_DIR = "/idiap/home/adubois/project_ma1/TIGER/tests_adrien/echoset_bias"
SPLITS = ["train", "val", "test"]
PAIR_REGEX = re.compile(r"(\d+)_(\d+)")


def load_speakers_metadata(speakers_path: str) -> dict[str, dict]:
    """Charge le genre, le subset et la durée (minutes) de chaque locuteur LibriSpeech."""
    metadata = {}
    p = Path(speakers_path)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                parts = [x.strip() for x in line.split("|")]
                if len(parts) >= 4 and parts[0]:
                    try:
                        mins = float(parts[3])
                    except ValueError:
                        mins = 0.0
                    metadata[parts[0]] = {
                        "gender": parts[1].upper(),
                        "subset": parts[2],
                        "minutes": mins,
                        "name": parts[4] if len(parts) > 4 else "",
                    }
    return metadata


def load_pairs(split: str) -> list[tuple[str, str]]:
    """Charge les paires depuis echoset_pairs_{split}.json ou s1.json."""
    candidates = [
        Path(f"echoset_pairs_{split}.json"),
        Path("TIGER") / f"echoset_pairs_{split}.json",
    ]
    for c in candidates:
        if c.is_file():
            with open(c, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [(p["id1"], p["id2"]) for p in data]

    # Fallback vers s1.json
    s1_path = Path(ECHOSET_BASE_DIR) / split / "s1.json"
    if not s1_path.is_file():
        return []

    with open(s1_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for entry in data:
        if isinstance(entry, (list, tuple)) and entry:
            parent_name = Path(entry[0]).parent.name
            m = PAIR_REGEX.search(parent_name)
            if m:
                pairs.append((m.group(1), m.group(2)))
    return pairs


def ascii_histogram(values: list[int], bins: int = 5) -> str:
    if not values:
        return ""
    v_min, v_max = min(values), max(values)
    if v_min == v_max:
        return f"[{v_min}]: " + "#" * len(values)
    step = (v_max - v_min) / bins
    bucket_counts = [0] * bins
    for v in values:
        idx = min(int((v - v_min) / step), bins - 1)
        bucket_counts[idx] += 1
    max_c = max(bucket_counts) if max(bucket_counts) > 0 else 1
    lines = []
    for i in range(bins):
        b_start = int(v_min + i * step)
        b_end = int(v_min + (i + 1) * step)
        bar = "█" * int(bucket_counts[i] / max_c * 20)
        lines.append(f"  [{b_start:>4} - {b_end:>4}] : {bucket_counts[i]:>4} locuteurs {bar}")
    return "\n".join(lines)


def main():
    spk_meta = load_speakers_metadata(SPEAKERS_TXT)
    out_dir = Path(BIAS_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("           ANALYSE DE L'ÉQUITÉ ET CALCUL DES FACTEURS DE BIAIS ALPHA")
    print("=" * 95)

    summary_rows = []

    for split in SPLITS:
        pairs = load_pairs(split)
        if not pairs:
            print(f"[!] Pas de données pour le split '{split}'")
            continue

        total_pairs = len(pairs)
        total_occurrences = total_pairs * 2

        speaker_counts = Counter()
        for id1, id2 in pairs:
            speaker_counts[id1] += 1
            speaker_counts[id2] += 1

        counts = list(speaker_counts.values())
        num_speakers = len(counts)
        mean_c = statistics.mean(counts)
        median_c = statistics.median(counts)
        std_c = statistics.stdev(counts) if num_speakers > 1 else 0.0
        min_c, max_c = min(counts), max(counts)
        cv = (std_c / mean_c * 100) if mean_c > 0 else 0.0

        # Paires de locuteurs uniques
        unique_spk_pairs = set(tuple(sorted([p[0], p[1]])) for p in pairs)
        max_possible_pairs = num_speakers * (num_speakers - 1) // 2
        pct_pairs_covered = (len(unique_spk_pairs) / max_possible_pairs * 100) if max_possible_pairs > 0 else 0.0

        summary_rows.append((split, num_speakers, total_occurrences, min_c, max_c, mean_c, median_c, std_c, cv))

        # Calcul des durées totales LibriSpeech pour ce subset
        total_ls_minutes = sum(spk_meta.get(spk, {}).get("minutes", 0.0) for spk in speaker_counts)

        # Construction du dictionnaire de biais pour ce split
        split_bias_data = {
            "split": split,
            "num_speakers": num_speakers,
            "total_pairs": total_pairs,
            "total_speaker_occurrences": total_occurrences,
            "mean_occurrences_per_speaker": round(mean_c, 2),
            "unique_speaker_pairs_formed": len(unique_spk_pairs),
            "max_possible_speaker_pairs": max_possible_pairs,
            "percentage_pairs_covered": round(pct_pairs_covered, 2),
            "alpha_lookup": {},  # Format rapide: { "spk_id": alpha }
            "speakers": {},      # Format détaillé avec métadonnées
        }

        for spk, count in sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True):
            meta = spk_meta.get(spk, {})
            alpha_uniform = count / mean_c
            weight_to_balance = 1.0 / alpha_uniform if alpha_uniform > 0 else 0.0

            # Comparaison par rapport aux minutes de parole dans LibriSpeech
            spk_minutes = meta.get("minutes", 0.0)
            if total_ls_minutes > 0 and spk_minutes > 0:
                expected_ls_prop = spk_minutes / total_ls_minutes
                actual_echoset_prop = count / total_occurrences
                alpha_vs_librispeech_duration = actual_echoset_prop / expected_ls_prop
            else:
                alpha_vs_librispeech_duration = None

            split_bias_data["alpha_lookup"][spk] = round(alpha_uniform, 4)
            split_bias_data["speakers"][spk] = {
                "gender": meta.get("gender", "Unknown"),
                "librispeech_subset": meta.get("subset", ""),
                "librispeech_minutes": spk_minutes,
                "echoset_count": count,
                "echoset_frequency": round(count / total_occurrences, 6),
                "alpha_uniform": round(alpha_uniform, 4),
                "weight_to_balance": round(weight_to_balance, 4),
                "alpha_vs_librispeech_duration": (
                    round(alpha_vs_librispeech_duration, 4)
                    if alpha_vs_librispeech_duration is not None else None
                ),
            }

        # Sauvegarde JSON
        bias_file = out_dir / f"{split}.json"
        with open(bias_file, "w", encoding="utf-8") as f:
            json.dump(split_bias_data, f, indent=2)

        # Affichage console
        print(f"\n--- Split: {split.upper()} ({num_speakers} locuteurs, {total_pairs:,} mélanges) ---")
        print(f"  • Paires de locuteurs formées : {len(unique_spk_pairs):,} / {max_possible_pairs:,} ({pct_pairs_covered:.1f}% des combinaisons possibles)")
        print(f"  • Présence théorique égale : {mean_c:.1f} apparitions / locuteur")
        print(f"  • Réalité : Min = {min_c} (α = {min_c/mean_c:.2f}) | Max = {max_c} (α = {max_c/mean_c:.2f}) | Écart = x{max_c/min_c:.2f}")
        print(f"  • Moyenne ± Écart-type : {mean_c:.1f} ± {std_c:.1f} (CV = {cv:.1f}%)")
        print("  • Distribution des fréquences :")
        print(ascii_histogram(counts, bins=5))
        print(f"  -> Fichier de biais sauvegardé : {bias_file}")

    print("\n" + "=" * 95)
    print(f"{'Split':<8} | {'Nb Spk':<8} | {'Total Occ':<10} | {'Min':<6} | {'Max':<6} | {'Moyenne':<9} | {'Médiane':<9} | {'Écart-type (CV%)':<16}")
    print("-" * 95)
    for r in summary_rows:
        cv_str = f"{r[7]:.1f} ({r[8]:.1f}%)"
        print(f"{r[0]:<8} | {r[1]:<8} | {r[2]:<10} | {r[3]:<6} | {r[4]:<6} | {r[5]:<9.1f} | {r[6]:<9.1f} | {cv_str:<16}")
    print("=" * 95)


if __name__ == "__main__":
    main()
