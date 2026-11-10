import opensmile
import numpy as np
import pandas as pd

file_path = "/idiap/home/adubois/project_ma1/TIGER/tests_adrien/embeddings/egemaps/84-121123-0000.npy"
array = np.load(file_path).squeeze()  # Vecteur 1D de taille 88

# Récupérer les noms officiels des 88 colonnes eGeMAPS depuis openSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
feature_names = smile.feature_names

# Création d'une Series pandas avec le nom de chaque feature
features = pd.Series(array, index=feature_names)

# Filtrer les caractéristiques d'intérêt (Pitch, Formants, Jitter, Shimmer, Loudness)
keywords = ["F0", "F1", "F2", "F3", "jitter", "shimmer", "loudness"]
selected_features = features[features.index.str.contains("|".join(keywords), case=False)]

print("=" * 65)
print(" CARACTÉRISTIQUES VOCALES CLÉS (PITCH, FORMANTS, HARMONIQUES)")
print("=" * 65)
for name, val in selected_features.items():
    print(f"{name:<50} : {val:.3f}")
