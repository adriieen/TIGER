import os
import shutil
import re

base_dir = "/idiap/home/adubois/project_ma1/TIGER/configs/large_windows"
src_dir = os.path.join(base_dir, "tiger-tiny")
target_dir = os.path.join(base_dir, "tiger-1:100")

# 1. Si tiger-1:10 n'existe pas, on le duplique depuis tiger-tiny
if not os.path.exists(target_dir):
    shutil.copytree(src_dir, target_dir)
    print(f"Copié {src_dir} -> {target_dir}")

# 2. Parcours de tous les .yml du dossier cible
for fname in list(os.listdir(target_dir)):
    if not (fname.endswith('.yml') or fname.endswith('.yaml')):
        continue

    filepath = os.path.join(target_dir, fname)
    with open(filepath, 'r') as f:
        content = f.read()

    # Remplacement des chemins train, valid, test
    content = re.sub(
        r'train_dir:\s*.*', 
        'train_dir: /idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_1:100/train', 
        content
    )
    content = re.sub(
        r'valid_dir:\s*.*', 
        'valid_dir: /idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_1:100/val', 
        content
    )
    content = re.sub(
        r'test_dir:\s*.*', 
        'test_dir: /idiap/home/adubois/project_ma1/TIGER/DataPreProcess/EchoSet_1:100/test', 
        content
    )

# Remplacement simple dans exp_name
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("exp_name:"):
            lines[i] = line.replace("tiny", "1:100")
    content = "\n".join(lines) + "\n"

    # Sauvegarde des modifications
    with open(filepath, 'w') as f:
        f.write(content)

    # Renommage du fichier s'il contient 'tiger-tiny'
    if 'tiger-tiny' in fname:
        new_fname = fname.replace('tiger-tiny', 'tiger-1:100')
        new_filepath = os.path.join(target_dir, new_fname)
        os.rename(filepath, new_filepath)
        print(f"Renommé & mis à jour : {fname} -> {new_fname}")
    else:
        print(f"Mis à jour : {fname}")

print("Terminé avec succès !")