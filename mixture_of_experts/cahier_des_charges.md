# Cahier des Charges : `DeltaF0Router` (Gating Network ResNet-2D)

## 1. Objectif du module
Implémenter en PyTorch un classifieur léger de régimes acoustiques (`DeltaF0Router`) prenant en entrée le signal audio mélangé brut `x` de forme `(B, L)` (mono, 16 kHz) et prédisant les logits sur les 5 quintiles de Delta F0 (`Q1` à `Q5`) pour aiguiller dynamiquement les échantillons vers les checkpoints TIGER spécialisés correspondants.

---

## 2. Extraction des caractéristiques (Front-end STFT)
Intégrer l'extraction spectrale directement dans le `forward` du module (via `torch.stft` et une fenêtre de Hann enregistrée via `register_buffer`) pour garantir la portabilité entre entraînement et inférence.

* **Fréquence d'échantillonnage (fs) :** `16000` Hz
* **Paramètres STFT :**
  * `n_fft = 2000` (résolution fréquentielle exacte `df = 16000 / 2000 = 8.0` Hz/bin)
  * `win_length = 2000` (125 ms)
  * `hop_length = 500` (31.25 ms, soit exactement 32 trames/seconde)
  * `window = torch.hann_window(2000)`
  * `center = True`, `return_complex = True`
* **Calcul du Log-Spectrogramme :**
  1. Calculer le module : `M = torch.abs(X) + 1e-6`
  2. Appliquer la compression logarithmique : `S = torch.log(M)`
* **Troncature fréquentielle (Crop basses/moyennes fréquences) :**
  * La sortie brute de `rfft` comporte `1001` bins (0 à 8000 Hz).
  * Ne conserver que les **375 premiers bins** (`S[:, :375, :]`), couvrant exactement la bande `[0 Hz, 3000 Hz[` où se concentrent F0 et les premiers formants.
* **Normalisation par instance (CMVN par énoncé) :**
  * Normaliser chaque spectrogramme indépendamment sur les axes `(F, T)` pour rendre le routeur invariant au gain global et au SNR global :
    `S_norm = (S - S.mean(dim=(-2, -1), keepdim=True)) / (S.std(dim=(-2, -1), keepdim=True) + 1e-6)`
* **Dimensions de sortie du front-end :** Tenseur `S_norm` de forme `(B, 1, 375, T)` (avec `T = 65` pour 2.0 s et `T = 193` pour 6.0 s).

---

## 3. Architecture du Backbone (`ResNet2D`)
Encodeur résiduel 2D compact (~350k–500k paramètres) inspiré de ResNet-18, utilisant des convolutions asymétriques/isotropes et des raccourcis de projection `1x1` de stride 2 lors des changements de résolution.

### Bloc de base (`ResidualBlock2D`)
Chaque bloc prend `(in_channels, out_channels, stride=(s_f, s_t))` :
* **Branche principale :**
  1. `Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)`
  2. `BatchNorm2d(out_channels)`
  3. `GELU()`
  4. `Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)`
  5. `BatchNorm2d(out_channels)`
* **Branche résiduelle (Shortcut) :**
  * Si `stride != (1, 1)` ou `in_channels != out_channels` : `Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)` + `BatchNorm2d(out_channels)`
  * Sinon : `nn.Identity()`
* **Sortie :** `GELU(main + shortcut)`

### Topologie complète
1. **Stem (Extraction initiale) :**
   * `Conv2d(1, 32, kernel_size=(7, 5), stride=(2, 1), padding=(3, 2), bias=False)` -> `BatchNorm2d(32)` -> `GELU()`
   * *(Note : le stride `(2, 1)` réduit l'axe fréquentiel de `375 -> 188` tout en préservant toute la résolution temporelle dès l'entrée).*
2. **Stage 1 :** `ResidualBlock2D(32, 32, stride=(1, 1))` + `ResidualBlock2D(32, 32, stride=(1, 1))`
3. **Stage 2 :** `ResidualBlock2D(32, 64, stride=(2, 2))` + `ResidualBlock2D(64, 64, stride=(1, 1))`
4. **Stage 3 :** `ResidualBlock2D(64, 128, stride=(2, 2))` + `ResidualBlock2D(128, 128, stride=(1, 1))`
5. **Stage 4 :** `ResidualBlock2D(128, 256, stride=(2, 2))` + `ResidualBlock2D(256, 256, stride=(1, 1))`
6. **Agrégation & Tête de classification :**
   * `AdaptiveAvgPool2d((1, 1))` -> `Flatten()` -> Tenseur de forme `(B, 256)` (garantit l'invariance à la durée audio 2 s vs 6 s).
   * `Dropout(p=0.2)`
   * `Linear(256, 5)` -> Logits bruts `(B, 5)` correspondant aux classes `[Q1, Q2, Q3, Q4, Q5]`.

---

## 4. Protocole d'entraînement et de validation

* **Données d'entrée :**
  * Train set : 4 054 mélanges d'EchoSet_cleaned_small, labellisés `y` dans `{0, 1, 2, 3, 4}` selon les seuils de Delta F0 définis sur le train set (`[1.82, 3.98, 6.63, 9.93]` demi-tons).
  * Val set : 4 604 mélanges, labellisés avec **les mêmes seuils fixes** extraits du train set (pas de recalcul de quantiles sur le val/test set pour éviter tout décalage de frontières de classes).
* **Data Augmentation (On-the-fly sur waveform avant STFT) :**
  * **Random Crop temporel :** Tirage aléatoire d'une fenêtre de 2.0 s (`32000` échantillons) à chaque itération d'entraînement (aligné avec la durée d'entraînement de TIGER). Si l'on utilise un découpage fixe préalable en segments de 2.0 s, filtrer les fenêtres où l'un des deux locuteurs est inactif/silencieux (énergie RMS < -20 dB par rapport à sa moyenne sur 6 s) ou recalculer Delta F0 localement. En validation/test, évaluer sur la fenêtre standardisée ou l'énoncé entier.
  * **Random Gain :** Multiplication par un facteur aléatoire uniforme dans `[0.7, 1.3]`.
  * **Time Masking léger (SpecAugment) :** Masquage aléatoire d'au plus 15 % des trames temporelles après CMVN pour forcer le réseau à ne pas dépendre d'une seule voyelle isolée. *(Ne pas appliquer de Frequency Masking, qui détruirait le motif harmonique vertical nécessaire à l'estimation de Delta F0).*
* **Fonction de perte :** `nn.CrossEntropyLoss(label_smoothing=0.05)` (le label smoothing atténue la pénalité sur les échantillons situés exactement sur les frontières entre deux quintiles adjacents).
* **Optimiseur & Scheduler :**
  * `AdamW(lr=1e-3, weight_decay=1e-2)`
  * `CosineAnnealingLR(T_max=epochs, eta_min=1e-5)`
  * Batch size : `32` ou `64` (entraînement sur 1 seul GPU, `num_workers=4` à `6`, `pin_memory=True`).
* **Critère de sauvegarde :** Sauvegarder le checkpoint maximisant l'**Accuracy Top-1** sur le validation set, ainsi que le **Mean Absolute Class Error** `mean(|y_pred - y_true|)` (métrique critique : confondre `Q1` avec `Q2` est bénin, confondre `Q1` avec `Q5` est catastrophique).

---

## 5. Livrables attendus de l'agent de code
1. **`router_model.py` :** Définition autonome de `DeltaF0Router` avec méthode `.predict_expert(waveform)` renvoyant l'indice `argmax` (Hard-Routing) et les probabilités `softmax` (pour analyse ou Top-2).
2. **`train_router.py` :** Script d'entraînement lisant les métadonnées Delta F0 existantes, calculant les seuils de quintiles sur le train set, sauvegardant `thresholds.json` et `best_router.pt`.
3. **`eval_router.py` :** Génération de la matrice de confusion `5x5` sur le Test Set (2 650 paires) et export des prédictions d'assignation pour branchement direct sur les 5 checkpoints TIGER.