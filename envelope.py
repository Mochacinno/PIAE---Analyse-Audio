import numpy as np
import librosa
import matplotlib.pyplot as plt

def compute_envelope(y, sr, win_len_sec=0.01):
    """Computation of a signal's envelopes
    Args:
        y (np.ndarray): Signal (waveform) to be analyzed
        win_len_sec (float): Length (seconds) of the window (Default value = 0.01)
        sr (scalar): Sampling rate (Default value = 22050)

    Returns:
        env (np.ndarray): Magnitude envelope
    """
    win_len_half = round(win_len_sec * sr * 0.5)
    N = y.shape[0]
    env = np.zeros(N)
    for i in range(N):
        i_start = max(0, i - win_len_half)
        i_end = min(N, i + win_len_half)
        env[i] = np.amax(np.abs(y)[i_start:i_end])
    return env

def prepare_audio(file_name, start_time, end_time, sr):
    y, sr = librosa.load(file_name, sr=sr)
    y = y[round(sr*start_time):round(sr*end_time)]
    return y

#sr = 22050
#y_f4 = prepare_audio("Audio_HD/F4.wav", 1, 7) #F4 2.21e4, 0 ; 2.32 0.036 ; 3.72 0.0068 ; 11.6 0.0021 ; 11.8 0
# y_f4_2 = prepare_audio("Audio_HD/F4.wav", 8, 16) #Another F4 2.14e5 0 ; 2.15 0.036 ; 2.3 0.0064 ; 3.31  0.0019 ; 3.37 0
# y_c2 = prepare_audio("Audio_HD/C2.wav", 0, 10) #C2 3.89e4 0 ; 4.06 0.0624 ; 5.13 0.0292 ; 21.5 0.00276
y_c6 = prepare_audio("Audio_HD/C6.wav", 1, 7) #C6 4.14e4 0; 4.45e4 0.019 ; 5.24e4 0.0016 ; 15.2 0
# y_g2 = prepare_audio("Audio/Guitar.wav", 30, 37) #Guitar 2.516e5 0; 2.52 0.682; 2.68 0.18; 3.45 0.09 3.49 0
# # guitar 2 (higher in pitch) 3.87e5 0; 3.88 0.79; 3.99 0.114; 5.03 0.026; 5.05 0

# # pour la guitare
# sr = 44100
# y_guitar = prepare_audio("Audio/Guitar.wav", 17, 23, sr) 

# x_guitar = np.arange(17*sr,23*sr)
# env_guitar = compute_envelope(y_guitar, sr)
# # figure 1 - enregistrement audio avec son envelope
# # Create figure
# fig, ax = plt.subplots(1,1, figsize=(5,4))
# ax.plot(x_guitar/sr, y_guitar, alpha=0.5, label="enregistrement audio")
# ax.plot(x_guitar/sr, env_guitar, 'orange', label="enveloppe supérieur")
# ax.set_title("Enregistrement Audio en fonction du temps avec enveloppe marqué",  fontsize=11)
# ax.set_xlabel(r"Temps $s$", fontsize=10)
# ax.set_ylabel("Amplitude", fontsize=10)
# ax.legend(fontsize=10)
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # pour la guitare 2
# sr = 44100
# y_guitar = prepare_audio("Audio/Guitar.wav", 3, 10.7, sr) 
# librosa.display.waveshow(y_guitar)
# plt.show()
# x_guitar = np.arange(3*sr,10.7*sr)
# env_guitar = compute_envelope(y_guitar, sr)
# # figure 1 - enregistrement audio avec son envelope
# # Create figure
# fig, ax = plt.subplots(1,1, figsize=(5,4))
# ax.plot(x_guitar/sr, y_guitar, alpha=0.5, label="enregistrement audio")
# ax.plot(x_guitar/sr, env_guitar, 'orange', label="enveloppe supérieur")
# ax.set_title("Enregistrement Audio en fonction du temps avec enveloppe marqué",  fontsize=11)
# ax.set_xlabel(r"Temps $s$", fontsize=10)
# ax.set_ylabel("Amplitude", fontsize=10)
# ax.legend(fontsize=10)
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# pour la guitare E2 1ere fois
sr = 44100
y_guitar = prepare_audio("Audio/GuitarAplucklength.wav", 11.4, 14.2, sr) 
# librosa.display.waveshow(y_guitar, sr=sr)
# plt.show()

x_guitar = np.arange(11.4*sr,14.2*sr)
env_guitar = compute_envelope(y_guitar, sr)
# figure 1 - enregistrement audio avec son envelope
# Create figure
fig, ax = plt.subplots(1,1, figsize=(5,4))
ax.plot(x_guitar/sr, y_guitar, alpha=0.5, label="enregistrement audio")
ax.plot(x_guitar/sr, env_guitar, 'orange', label="enveloppe supérieur")
ax.set_title("Enregistrement Audio en fonction du temps avec enveloppe marqué",  fontsize=11)
ax.set_xlabel(r"Temps $s$", fontsize=10)
ax.set_ylabel("Amplitude", fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# pour la guitare E2 2eme fois
sr = 44100
y_guitar2 = prepare_audio("Audio/GuitarAplucklength.wav", 15, 19.5, sr) 
# librosa.display.waveshow(y_guitar, sr=sr)
# plt.show()

x_guitar = np.arange(15*sr,19.5*sr)
env_guitar = compute_envelope(y_guitar2, sr)
# figure 1 - enregistrement audio avec son envelope
# Create figure
fig, ax = plt.subplots(1,1, figsize=(5,4))
ax.plot(x_guitar/sr, y_guitar2, alpha=0.5, label="enregistrement audio")
ax.plot(x_guitar/sr, env_guitar, 'orange', label="enveloppe supérieur")
ax.set_title("Enregistrement Audio en fonction du temps avec enveloppe marqué",  fontsize=11)
ax.set_xlabel(r"Temps $s$", fontsize=10)
ax.set_ylabel("Amplitude", fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

adsr_f4 = np.array([[2.21, 0], [2.32, 0.036], [3.72, 0.0068], [11.6, 0.0021], [11.8, 0]])
adsr_f4_2 = np.array([[21.4, 0], [21.5, 0.036], [23, 0.0064], [33.1, 0.0019], [33.7, 0]])
adsr_c2 = np.array([[3.89, 0], [4.06, 0.0624], [5.13, 0.0292], [21.5, 0.00276], [21.5, 0.00276]])
adsr_c6 = np.array([[4.14, 0], [4.45, 0.019], [5.24, 0.0016], [15.2, 0], [15.2, 0]])

def process_adsr_norm(arr):
    arr[:,0] = arr[:,0]-arr[0,0] # relative to 0
    arr[:,1] = arr[:,1]/arr[1,1] # normal the max value to 1

process_adsr_norm(adsr_c2)
process_adsr_norm(adsr_c6)
process_adsr_norm(adsr_f4)
process_adsr_norm(adsr_f4_2)


env_f4 = compute_envelope(y_f4)
#env_f4_2 = compute_envelope(y_f4_2)
# env_c2 = compute_envelope(y_c2)
# env_g = compute_envelope(y_g2)

# adsr
#env_f4_0_index = 0
#env_f4_a_index = np.argmax(env_f4)
#env_f4_d_arr = env_f4[env_f4_a_index:]
##env_f4_d_index = np.where(env_f4_d_arr[env_f4_d_arr > 0.01][0])

x1 = np.arange(1*sr,7*sr)
# x2 = np.arange(8*sr,16*sr)
# x3 = np.arange(1*sr,7*sr)
# xc2 = np.arange(0*sr,10*sr)
# xg = np.arange(30*sr,37*sr)

# figure 1 - enregistrement audio avec son envelope
# Create figure
fig, ax = plt.subplots(1,1, figsize=(10,6))
ax.plot(x1/sr, y_f4, alpha=0.5, label="enregistrement audio")
ax.plot(x1/sr, env_f4, 'orange', label="enveloppe supérieur")
ax.set_title("Enregistrement Audio en fonction du temps avec enveloppe supérieur marqué",  fontsize=14)
ax.set_xlabel(r"Temps $s$", fontsize=12)
ax.set_ylabel("Amplitude", fontsize=12)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figure 1")
plt.show()

# figure 2 - definition des points
adsr_points_f4 = np.array([[2.21e4, 0], [2.32e4, 0.036], [3.72e4, 0.0068], [11.6e4, 0.0021], [11.8e4, 0]])

# Convert to seconds
adsr_points_f4_sec = adsr_points_f4.copy()
adsr_points_f4_sec[:,0] = adsr_points_f4[:,0]/sr

# Create figure
fig, ax = plt.subplots(1,1, figsize=(10,6))

# Plot the envelope
ax.plot(x1/sr, env_f4, 'orange', alpha=0.5, label='Enveloppe')

# Plot ADSR stages with different colors
# Attack (from start to peak)
ax.plot(adsr_points_f4_sec[0:2,0], adsr_points_f4_sec[0:2,1], 'r-', linewidth=2, label='Attack')
# Decay (from peak to sustain level)
ax.plot(adsr_points_f4_sec[1:3,0], adsr_points_f4_sec[1:3,1], 'g-', linewidth=2, label='Decay')
# Sustain (constant level)
ax.plot(adsr_points_f4_sec[2:4,0], adsr_points_f4_sec[2:4,1], 'b-', linewidth=2, label='Sustain')
# Release (from sustain to zero)
ax.plot(adsr_points_f4_sec[3:5,0], adsr_points_f4_sec[3:5,1], 'm-', linewidth=2, label='Release')

# Mark the key points
ax.scatter(adsr_points_f4_sec[:,0], adsr_points_f4_sec[:,1], c=['black', 'red', 'green', 'blue', 'purple'], s=100, zorder=5)

# Add labels and title
ax.set_title("l'Enveloppe Supérieur avec les 4 parties de l'enveloppe identifiées", fontsize=14)
ax.set_xlabel(r"Temps $s$", fontsize=12)
ax.set_ylabel("Amplitude", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figure 2")
plt.show()

# figure 3 - avec courbe simplifie de l'enveloppe
fig, ax = plt.subplots(1,1, figsize=(10,6))
ax.plot(adsr_f4[:,0]/sr/10e-5, adsr_f4[:,1])
ax.set_title("Enveloppe normalisé des differents enregistrement audio", fontsize=14)
ax.set_xlabel(r"Temps $s$", fontsize=12)
ax.set_ylabel("Amplitude", fontsize=12)
plt.plot(adsr_c2[:,0]/sr/10e-5, adsr_c2[:,1], 'r-', label="note C2")
#ax.scatter(adsr_c2[:,0]/sr/10e-5, adsr_c2[:,1], c=['black', 'red', 'green', 'blue', 'purple'], s=100, zorder=5)
plt.plot(adsr_c6[:,0]/sr/10e-5, adsr_c6[:,1], 'g-', label="note C6")
plt.plot(adsr_f4[:,0]/sr/10e-5, adsr_f4[:,1], 'b-', label="note F4")
plt.plot(adsr_f4_2[:,0]/sr/10e-5, adsr_f4_2[:,1], 'm-', label="note F4 (2ème enregistrement)")

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure 3")
plt.show()



# spectre des freq
