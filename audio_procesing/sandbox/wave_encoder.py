import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the audio file
# librosa automatically converts audio to mono and resamples to 22050 Hz by default
audio_path = r"C:\Users\betas\OneDrive\Documentos\Elite_Grup\bogota_wro\audio_procesing\sandbox\output_raw.wav"
y, sr = librosa.load(audio_path)

# 2. Compute the Short-Time Fourier Transform (STFT)
stft_matrix = librosa.stft(y)

# 3. Convert amplitude to decibels (dB)
stft_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)

# 4. Plot the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(stft_db, sr=sr, x_axis="time", y_axis="hz", cmap="magma")

# 5. Add visual details
plt.colorbar(label="Decibels (dB)")
plt.title("Audio Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()
