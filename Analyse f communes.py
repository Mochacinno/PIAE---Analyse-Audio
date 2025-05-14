import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


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
    
    freq_resolution = freqs[1]-freqs[0]
    min_distance = int(tolerance_hz/freq_resolution)
    
    i_peaks, _ = find_peaks(amplitude, height=15, distance=min_distance)
    freq_peaks = sorted(freqs[i_peaks])
    print(freq_peaks)
    
    return freq_peaks

# --- Imports des audios ---
fichier = "domisol4.wav"
nb_notes = int(input("Nombre de notes dans l'accord :"))

audio, sr = librosa.load(fichier)
audio = audio[sr:5*sr]   # trim

# --- Transformée de Fourier ---
freqs = np.fft.rfftfreq(len(audio), 1 / sr)
spectre = np.abs(np.fft.rfft(audio))



# --- Détection fondamentales ---
freq_peaks = trouve_pics(audio, sr,tolerance_hz=5)
dico_accords = {}
noms_note = []
nb_harmoniques = 6

for i in range(nb_notes):
    fondamentale = freq_peaks[i]
    nom_note = freq_to_note(fondamentale)
    noms_note.append(nom_note)
    dico_accords[nom_note] = [fondamentale]
    for j in range(nb_harmoniques):
        dico_accords[nom_note].append(fondamentale*(j+2))

pics_en_commun = []

# Parcourir toutes les combinaisons de fréquences
for f1 in dico_accords[noms_note[0]]:
    note_f1 = freq_to_note(f1)
    count = 0
    for key, peaks in dico_accords.items():  # Pour chaque fichier dans le dictionnaire
        if key != noms_note[0]:  # Ignorer le 1er fichier car on l'a déjà parcouru
            for f2 in peaks:
                note_f2 = freq_to_note(f2)
                if note_f1 == note_f2 :
                    count += 1
                    break 
            if count >= 1:  # Si trouvé dans au moins un autre fichier
                break  # On peut sortir de la boucle des fichiers

    if count >= 1:  # Présence dans au moins 2 listes
        pics_en_commun.append(float(f1))



print("Les spectres de chaque note présentent", len(pics_en_commun), "fréquences communes :", pics_en_commun)



# --- Affichage ---
colors = ["green","blue","orange"]
fig, ax = plt.subplots()

# --- Spectre ---
ax.plot(freqs, spectre,label=fichier, color='black')

# --- Harmoniques ---
for i, (nom, accord) in enumerate(dico_accords.items()):
    color = colors[i]
    ax.axvline(x=accord[0], color=color, linestyle='--', alpha=1,label=nom)
    for freq in accord[1:] : 
        ax.axvline(x=freq, color=color, linestyle='--', alpha=1)

# --- Pics en commun ---        
for pic in pics_en_commun : 
    note = freq_to_note(pic)
    ax.axvline(x=pic, color='red', linestyle='-', alpha=1,label=note)

ax.set_xlim(0,4000)
ax.legend()
plt.show()
