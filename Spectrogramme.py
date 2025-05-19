import numpy as np
import librosa
import matplotlib.pyplot as plt

# --- Imports des audios ---
fichier = "Audio_HD/C2.wav"
audio, sr = librosa.load(fichier, sr = 44100, offset = 2, duration = 8)

# --- Spectrogramme ---
S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=3000)

# --- Transformée de Fourier ---
freqs = np.fft.rfftfreq(len(audio), 1 / sr)
spectre = np.abs(np.fft.rfft(audio))

# --- Affichage ---

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=3000, ax=ax1)

fig.colorbar(img, ax=ax1, format='%+2.0f dB')
ax1.set(title='Spectogramme')
ax2.plot(freqs, spectre, color = 'black')
ax2.set(title='Transformée de Fourier', xlim = (0,3000))

plt.show()