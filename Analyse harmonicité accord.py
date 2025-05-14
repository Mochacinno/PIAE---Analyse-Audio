import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import itertools

freq_A1 = 27.5
notes = ['C', 'C#', 'D', 'D#', 'E', 'F',
         'F#', 'G', 'G#', 'A', 'A#', 'B']


def freq_to_note(freq):
    A_index = 9  # Position de A dans la liste des notes
    if freq <= 0:
        return None
    n_demi_tons = round(12 * np.log2(freq / freq_A1))
    note_index = (A_index + n_demi_tons) % 12
    octave = (A_index + n_demi_tons) // 12
    return notes[note_index] + str(octave)


def trouve_pics(audio, sr, tolerance_hz=5):
    amplitude = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    freq_resolution = freqs[1] - freqs[0]
    min_distance = int(tolerance_hz / freq_resolution)

    i_peaks, _ = find_peaks(amplitude, height=15, distance=min_distance)
    freq_peaks = sorted(freqs[i_peaks])
    print(freq_peaks)

    return freq_peaks


# --- Import audio ---
fichier = "Audio_HD/domisol4.wav"
nb_notes = int(input("Nombre de notes dans l'accord : "))

audio, sr = librosa.load(fichier, sr=44100)
audio = audio[sr:5 * sr]  # trim

# --- Transformée de Fourier ---
freqs = np.fft.rfftfreq(len(audio), 1 / sr)
spectre = np.abs(np.fft.rfft(audio))

# --- Détection des pics ---
freq_peaks = trouve_pics(audio, sr, tolerance_hz=5)
dico_accords = {}
noms_note = []
f_max = 2500 #Hz

for i in range(nb_notes):
    fondamentale = freq_peaks[i]
    nom_note = freq_to_note(fondamentale)
    noms_note.append(nom_note)
    dico_accords[nom_note] = [fondamentale]
    j = 0
    while fondamentale * (j + 2) < 2100 :
        dico_accords[nom_note].append(fondamentale * (j + 2))
        j+=1

# --- Recherche de fréquences communes entre toutes les paires de notes ---
pics_en_commun = []

for nom1, nom2 in itertools.combinations(noms_note, 2):
    for f1 in dico_accords[nom1]:
        note_f1 = freq_to_note(f1)
        for f2 in dico_accords[nom2]:
            note_f2 = freq_to_note(f2)
            if note_f1 == note_f2:
                if f1 not in pics_en_commun :
                    pics_en_commun.append(f1)

pics_en_commun = sorted([float(pic) for pic in pics_en_commun])
print("Les spectres de chaque note présentent", len(pics_en_commun), "fréquences communes :", pics_en_commun, "c'est à dire les notes :", [freq_to_note(pic) for pic in pics_en_commun])

# --- Affichage ---
colors = ["green", "blue", "orange"]
fig, ax = plt.subplots()

# --- Spectre ---
ax.plot(freqs, spectre, label=fichier, color='black')

# --- Harmoniques ---
for i, (nom, accord) in enumerate(dico_accords.items()):
    color = colors[i % len(colors)]
    ax.scatter(x=accord[0], y = (i+1)* 100, color=color, marker='+', alpha=1, label=nom)
    for freq in accord[1:]:
        ax.scatter(x=freq, y = (i+1)* 100, color=color, marker='+', alpha=1)

# --- Pics en commun ---
for pic in pics_en_commun:
    note = freq_to_note(pic)
    ax.scatter(x=pic,y = -5, color='red', marker = '+', alpha=1, label=note)

ax.set_xlim(0, 2100)
ax.legend()
plt.show()
