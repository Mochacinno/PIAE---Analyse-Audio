import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import itertools


# --- Paramètres ---
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

# --- Détection des pics ---
def trouve_pics(audio, sr, seuil_detection_pourcent = 0.05, tolerance_hz=5):
    amplitude = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    
    freq_resolution = freqs[1]-freqs[0]
    min_distance = int(tolerance_hz/freq_resolution)
    
    i_peaks, _ = find_peaks(amplitude, height=np.max(amplitude) * seuil_detection_pourcent, distance=min_distance)
    freq_peaks = sorted(freqs[i_peaks])
    
    return i_peaks, freq_peaks

# --- Imports des audios ---
fichiers = ["Audio_YT/C4 YT.mp3","Audio_YT/E4 YT.mp3","Audio_YT/G4 YT.mp3"]
tolerance_hz = 3

dico = {}

for f in fichiers :
    dico[f]=[]
    audio, sr = librosa.load(f, sr = 44100, offset = 2, duration =2)
    
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    spectre = np.abs(np.fft.rfft(audio))
    
    freq_resolution = freqs[1]-freqs[0]
    min_distance = int(tolerance_hz/freq_resolution)
    
    i_peaks, _ = find_peaks(spectre, height=np.max(spectre) * 0.05 , distance=min_distance)
    freq_peaks = freqs[i_peaks]
    
    dico[f].append(audio) #0
    dico[f].append(freqs) #1
    dico[f].append(spectre) #2
    dico[f].append(freq_peaks) #3



freqs = dico["Audio_YT/C4 YT.mp3"][1]
spectre_C4 = dico["Audio_YT/C4 YT.mp3"][2]
spectre_E4 = dico["Audio_YT/E4 YT.mp3"][2]
spectre_G4 = dico["Audio_YT/G4 YT.mp3"][2]



pics_en_commun = []

for nom1, nom2 in itertools.combinations(fichiers, 2):
    for f1 in dico[nom1][3]:
        note_f1 = freq_to_note(f1)
        for f2 in dico[nom2][3]:
            note_f2 = freq_to_note(f2)
            if note_f1 == note_f2:
                if f1 not in pics_en_commun :
                    pics_en_commun.append(f1)

pics_en_commun = sorted([float(pic) for pic in pics_en_commun])    

print("Les spectres de chaque note présentent", len(pics_en_commun), "fréquences communes :", pics_en_commun, "c'est à dire les notes :", [freq_to_note(pic) for pic in pics_en_commun])

# --- Affichage ---

plt.figure(figsize=(10, 4))
plt.plot(freqs, spectre_C4, color = 'pink', alpha = 1, linestyle='-', label = 'C4')
plt.plot(freqs, spectre_E4,color = 'orange', alpha = 1, linestyle='-', label = 'E4')
plt.plot(freqs, spectre_G4,color = 'blue', alpha = 1, linestyle='-', label = 'G4')
#plt.plot(freqs, (spectre_C4+spectre_E4+spectre_G4)/3,color = 'black', alpha = 1, linestyle='-', label = 'Moyenne')

plt.scatter(x=dico["Audio_YT/C4 YT.mp3"][3][0], y=2000,color = 'pink', alpha = 1,marker = '+',label = 'pics C4')
for pic in dico["Audio_YT/C4 YT.mp3"][3][1:] :
    plt.scatter(x=pic,y=2000, color = 'pink', alpha = 1, marker = '+')

plt.scatter(x=dico["Audio_YT/E4 YT.mp3"][3][0], y=2100,color = 'orange', alpha = 1,marker = '+',label = 'pics E4')
for pic in dico["Audio_YT/E4 YT.mp3"][3][1:] :
    plt.scatter(x=pic,y=2100, color = 'orange', alpha = 1, marker = '+')

plt.scatter(x=dico["Audio_YT/G4 YT.mp3"][3][0], y=2200,color = 'blue', alpha = 1,marker = '+',label = 'pics G4')
for pic in dico["Audio_YT/G4 YT.mp3"][3][1:] :
    plt.scatter(x=pic,y=2200, color = 'blue', alpha = 1, marker = '+')


plt.scatter(x=pics_en_commun[0], y=500,color = 'red', alpha = 1,marker = '+',label = 'pics com')
for pic in pics_en_commun[1:] :
    plt.scatter(x=pic,y=500, color = 'red', alpha = 1, marker = '+')



plt.xlim(0, 3000)
plt.grid(True)
plt.legend()
plt.show()
