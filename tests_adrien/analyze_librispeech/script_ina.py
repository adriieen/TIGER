import os
import argparse
import numpy as np
import soundfile as sf
from opensmile.core.smile import Smile
from opensmile.core.define import FeatureSet, FeatureLevel


def extract_os_features(folder_path, feature_set, dataset="Librispeech"):
    if feature_set == 'egemaps':
        smile_lld = Smile(feature_set=FeatureSet.eGeMAPSv02, feature_level=FeatureLevel.Functionals)
    elif feature_set == 'compare':
        smile_lld = Smile(feature_set=FeatureSet.ComParE_2016, feature_level=FeatureLevel.Functionals)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}. Expected 'egemaps' or 'compare'.")

    # Create the save path
    save_folder = os.path.join("embeddings", feature_set)
    os.makedirs(save_folder, exist_ok=True)

    # Récupération des fichiers audio selon le dataset
    if dataset != "Librispeech":
        # Dossier plat (fichiers directement dans folder_path)
        audio_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith((".wav", ".flac"))
        ]
    else:
        # Recherche récursive pour LibriSpeech (speaker_id/chapter_id/*.flac)
        audio_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(folder_path)
            for f in files
            if f.endswith((".wav", ".flac"))
        ]

    print(f"Found {len(audio_files)} audio files to process.")

    for file_path in audio_files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(save_folder, f"{base_name}.npy")

        speaker_id = filename.split('-')[0]
        if speaker_id == "5867" or speaker_id == "587" or speaker_id[0] in ['6', '7', '8', '9']:
            try:
                # openSMILE peut traiter directement le fichier (.wav, .flac, etc.)
                df = smile_lld.process_file(file_path)
                y = df.to_numpy()  # shape: (1, n_features)
                
                print(f"Saving {filename}: shape = {y.shape}")
                np.save(output_file, y)
                print(f"Saved embeddings for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenSmile data processing")
    parser.add_argument(
        "--folder_path",
        type=str,
        default="/idiap/resource/database/LibriSpeech/train-clean-100",
        help="Path to the folder containing the audio files"
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="egemaps",
        choices=["egemaps", "compare"],
        help="Feature set to use for OpenSmile"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Librispeech",
        help="Dataset name ('Librispeech' or other)"
    )

    args = parser.parse_args()
    extract_os_features(args.folder_path, args.feature_set, args.dataset)
