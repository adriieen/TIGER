import soundfile as sf

s = 0

for i in range(3):
    for k in range(10):
        if i == 2 and k > 5:
            break
        info = sf.info(f"/idiap/resource/database/LibriSpeech/train-clean-100/7312/92432/7312-92432-00{i}{k}.flac")

        duration_sec = info.duration          # Calculé automatiquement
        s += duration_sec
        num_samples = info.frames             # Nombre d'échantillons
        sample_rate = info.samplerate         # Fréquence (ex: 16000 Hz)

        # Vérification : duration_sec == num_samples / sample_rate
        print(f"Durée : {duration_sec:.2f} s ({num_samples:,} samples à {sample_rate} Hz)")

print(s)