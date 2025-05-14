import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# PROBLEME

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
def trouve_pics(audio, sr, tolerance_hz=5):
    amplitude = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    
    freq_resolution = freqs[1]-freqs[0]
    min_distance = int(tolerance_hz/freq_resolution)
    
    i_peaks, _ = find_peaks(amplitude, height=np.max(amplitude) * 0.1, distance=min_distance)
    freq_peaks = sorted(freqs[i_peaks])
    
    return freq_peaks

# --- Imports des audios ---
fichiers = ["Audio_YT/C4 YT.mp3","Audio_YT/E4 YT.mp3","Audio_YT/G4 YT.mp3"]

dico = {}
spectre_combiné = None
for f in fichiers :
    dico[f]=[]
    audio, sr = librosa.load(f, sr = 44100)
    audio = audio[2*sr:4*sr]   # trim
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    spectre = np.abs(np.fft.rfft(audio))
    spectre /= max(spectre)
    dico[f].append(audio)
    dico[f].append(freqs)
    dico[f].append(spectre)

    if spectre_combiné is None : 
        spectre_combiné = spectre
    else : 
        spectre_combiné *= spectre


freqs= dico["Audio_YT/C4 YT.mp3"][1]

i_peaks, _ = find_peaks(spectre_combiné, height=np.max(spectre_combiné) * 0.1)
freq_peaks = freqs[i_peaks]


# --- Affichage ---
plt.figure(figsize=(10, 4))

for f in freq_peaks:
    plt.axvline(x=f, color='r', linestyle='--', alpha=0.6)

plt.plot(freqs, spectre_combiné,label='res')

plt.plot(dico["Audio_YT/C4 YT.mp3"][1], dico["Audio_YT/C4 YT.mp3"][2],label='C4')
plt.plot(dico["Audio_YT/E4 YT.mp3"][1], dico["Audio_YT/E4 YT.mp3"][2],label='E4')
plt.plot(dico["Audio_YT/G4 YT.mp3"][1], dico["Audio_YT/G4 YT.mp3"][2],label='G4')

plt.xlim(0, 4000)
plt.grid(True)
plt.legend()
plt.show()