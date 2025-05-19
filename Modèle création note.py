import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Paramètres audio
sr = 44100          # fréquence d'échantillonnage
duration = 2.0      # durée en secondes
t = np.linspace(0, duration, int(sr * duration), endpoint=False)


notes_freqs = {
    'C4': 261.63,
    'E4': 329.63,
    'G4': 392.00
}


def generer_note(freq, t, num_harmonics=6, decay='1/n^2'):
    signal = np.zeros_like(t)
    for n in range(1, num_harmonics + 1):
        harmonic_freq = freq * n
        if decay == '1/n':
            amplitude = 1 / n
        elif decay == '1/n^2':
            amplitude = 1 / (n ** 2)
        else:
            amplitude = 1 / n  # valeur par défaut
        signal += amplitude * np.sin(2 * np.pi * harmonic_freq * t)
    return signal

# Générer les 3 notes avec harmoniques
note_signals = []
for name, freq in notes_freqs.items():
    note = generer_note(freq, t, num_harmonics=6, decay='1/n')
    note_signals.append(note)

# Combinaison des 3 notes pour créer l'accord
chord = sum(note_signals)

# Normalisation
chord /= np.max(np.abs(chord))

# Écoute / Sauvegarde
sf.write('accord_do_majeur_1n2.wav', chord, sr)
print("Fichier 'accord_do_majeur.wav' généré.")

# (optionnel) Affichage d'un extrait du signal
plt.figure(figsize=(10, 4))
plt.plot(t[:1000], chord[:1000])
plt.title("Accord de Do majeur (forme d'onde, début)")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()
plt.show()
