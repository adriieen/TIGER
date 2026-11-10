import json
import os
import random

# Fixer la seed pour la reproductibilité (optionnel mais recommandé)
random.seed(42)

factor = 0.01
splits = ["train", "val", "test"]
sources = ["mix", "s1", "s2"]

input_dir = "EchoSet"
output_dir = "EchoSet_1:100"

for split in splits:
    split_in_dir = os.path.join(input_dir, split)
    split_out_dir = os.path.join(output_dir, split)
    os.makedirs(split_out_dir, exist_ok=True)

    # Charger les 3 fichiers (mix, s1, s2)
    data = {}
    for src in sources:
        filepath = os.path.join(split_in_dir, f"{src}.json")
        with open(filepath, "r") as f:
            data[src] = json.load(f)

    # Vérifier que toutes les sources ont bien la même taille
    total_samples = len(data["mix"])
    for src in sources:
        assert len(data[src]) == total_samples, (
            f"Taille incohérente pour {split}/{src}.json "
            f"({len(data[src])} vs {total_samples})"
        )

    # Échantillonner les mêmes indices aléatoires
    k = int(total_samples * factor)
    sampled_indices = random.sample(range(total_samples), k)
    # Optionnel : trier les indices pour préserver l'ordre initial
    sampled_indices.sort()

    print(f"[{split}] Échantillons: {total_samples} -> {k}")

    # Extraire les mêmes éléments et sauvegarder
    for src in sources:
        sampled_data = [data[src][i] for i in sampled_indices]
        out_filepath = os.path.join(split_out_dir, f"{src}.json")
        with open(out_filepath, "w") as f:
            json.dump(sampled_data, f, indent=4)

print("Done")
