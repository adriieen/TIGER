import os
import soundfile as sf

base_in = "/idiap/home/adubois/project_ma1/TIGER/test"
base_out = "/idiap/home/adubois/project_ma1/TIGER/tests_adrien/audio_samples"

splits = {
    "train": (0.0, 4.0),
    "val": (1.0, 5.0),
    "test": (2.0, 6.0),
}

file_mapping = {
    "mix.wav": "mix.wav",
    "s1.wav": "spk1_reverb.wav",
    "s2.wav": "spk2_reverb.wav",
}

for split_name, (start_sec, end_sec) in splits.items():
    out_dir = os.path.join(base_out, split_name)
    os.makedirs(out_dir, exist_ok=True)
    for in_name, out_name in file_mapping.items():
        src_path = os.path.join(base_in, in_name)
        data, sr = sf.read(src_path)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        sliced = data[start_sample:end_sample]
        dst_path = os.path.join(out_dir, out_name)
        sf.write(dst_path, sliced, sr)
        print(f"Créé : {dst_path} ({len(sliced)/sr:.1f}s)")

print("done")
