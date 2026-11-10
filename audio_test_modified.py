import os
import random
from typing import Union
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from scipy.io import wavfile
import warnings
import torchaudio

warnings.filterwarnings("ignore")

import look2hear.models
import look2hear.datas
from look2hear.metrics import MetricsTracker
from look2hear.utils import (
    tensors_to_device,
    RichProgressBarTheme,
    MyMetricsTextColumn,
    BatchesProcessedColumn,
)

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="local/mixit_conf.yml",
    help="Full path to configuration file",
)
parser.add_argument(
    "--save_wavs",
    action="store_true",
    help="Whether to save separated audio outputs to disk",
)


def load_model(conf, model_path, device):
    audionet_name = conf["audionet"]["audionet_name"]
    audionet_config = conf["audionet"]["audionet_config"]

    # 1. Instancier l'architecture
    model_class = getattr(look2hear.models, audionet_name)
    model = model_class(**audionet_config)

    # 2. Charger les poids via torch.load natif
    print(f"--> Loading weights from: {model_path}")
    pkg = torch.load(model_path, map_location="cpu")

    if isinstance(pkg, dict) and "state_dict" in pkg:
        state_dict = pkg["state_dict"]
    elif isinstance(pkg, dict) and "model_state_dict" in pkg:
        state_dict = pkg["model_state_dict"]
    elif isinstance(pkg, dict):
        state_dict = pkg
    else:
        state_dict = pkg.state_dict()

    # Nettoyer les clés générées par PyTorch Lightning ('audio_model.xxx' -> 'xxx')
    cleaned_state = {}
    for k, v in state_dict.items():
        new_k = k.replace("audio_model.", "")
        cleaned_state[new_k] = v

    model.load_state_dict(cleaned_state, strict=False)
    model.to(device)
    model.eval()
    return model


def main(config):
    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "•",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metricscolumn,
    )

    train_conf = config["train_conf"]
    exp_name = train_conf["exp"]["exp_name"]
    exp_dir = os.path.join(os.getcwd(), "Experiments", "checkpoint", exp_name)

    train_conf["main_args"] = {"exp_dir": exp_dir}
    model_path = os.path.join(exp_dir, "best_model.pth")

    # Si best_model.pth n'existe pas, essayer last.ckpt
    if not os.path.exists(model_path):
        fallback_ckpt = os.path.join(exp_dir, "last.ckpt")
        if os.path.exists(fallback_ckpt):
            print(f"[Notice] 'best_model.pth' not found. Falling back to: {fallback_ckpt}")
            model_path = fallback_ckpt
        else:
            raise FileNotFoundError(f"No checkpoint found in: {exp_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chargement du modèle
    model = load_model(train_conf, model_path, device)

    # Instanciation des données de test
    datamodule = getattr(look2hear.datas, train_conf["datamodule"]["data_name"])(
        **train_conf["datamodule"]["data_config"]
    )
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    ex_save_dir = os.path.join(exp_dir, "results")
    os.makedirs(ex_save_dir, exist_ok=True)
    metrics = MetricsTracker(save_file=os.path.join(ex_save_dir, "metrics.csv"))

    save_wavs = config.get("save_wavs", False)

    with torch.no_grad():
        with progress:
            for idx in progress.track(range(len(test_set))):
                mix, sources, key = tensors_to_device(test_set[idx], device=device)

                # Inférence : mix[None] ajoute la dimension batch [1, T]
                est_sources = model(mix[None])

                mix_np = mix
                sources_np = sources
                est_sources_np = est_sources.squeeze(0)

                # Évaluation et log des métriques
                metrics(
                    mix=mix_np,
                    clean=sources_np,
                    estimate=est_sources_np,
                    key=key,
                )

                # Optionnel : sauvegarde d'échantillons audio
                if save_wavs and idx < 10:
                    wav_save_dir = os.path.join(ex_save_dir, f"sample_{idx}")
                    for i in range(est_sources_np.shape[0]):
                        spk_dir = os.path.join(wav_save_dir, f"s{i + 1}")
                        os.makedirs(spk_dir, exist_ok=True)
                        filename = os.path.basename(key)
                        torchaudio.save(
                            os.path.join(spk_dir, filename),
                            est_sources_np[i].unsqueeze(0).cpu(),
                            train_conf["datamodule"]["data_config"]["sample_rate"],
                        )

                if idx % 20 == 0:
                    metricscolumn.update(metrics.update())

    print("\n[Evaluation Complete]")
    metrics.final()


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Charger la configuration
    with open(args.conf_dir, "r") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf

    main(arg_dic)