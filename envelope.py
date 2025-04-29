import numpy as np
import librosa
import matplotlib.pyplot as plt

def compute_envelope(y, win_len_sec=0.01, sr=22050):
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

def prepare_audio(file_name, start_time, end_time, sr=22050):
    y, sr = librosa.load(file_name, sr=sr)
    y = y[round(sr*start_time):round(sr*end_time)]
    return y

sr = 22050
y_f4 = prepare_audio("Audio_HD/F4.wav", 1, 7) #F4 2.21e4, 0 ; 2.32 0.036 ; 3.72 0.0068 ; 11.6 0.0021 ; 11.8 0
# y_f4_2 = prepare_audio("Audio_HD/F4.wav", 8, 16) #Another F4 2.14e5 0 ; 2.15 0.036 ; 2.3 0.0064 ; 3.31  0.0019 ; 3.37 0
# y_c2 = prepare_audio("Audio_HD/C2.wav", 0, 10) #C2 3.89e4 0 ; 4.06 0.0624 ; 5.13 0.0292 ; 21.5 0.00276
# y_c6 = prepare_audio("Audio_HD/C6.wav", 1, 7) #C6 4.14e4 0; 4.45e4 0.019 ; 5.24e4 0.0016 ; 15.2 0
# y_g2 = prepare_audio("Audio/Guitar.wav", 30, 37) #Guitar 2.516e5 0; 2.52 0.682; 2.68 0.18; 3.45 0.09 3.49 0
# # guitar 2 (higher in pitch) 3.87e5 0; 3.88 0.79; 3.99 0.114; 5.03 0.026; 5.05 0

adsr_f4 = np.array([[2.21, 0], [2.32, 0.036], [3.72, 0.0068], [11.6, 0.0021], [11.8, 0]])
adsr_f4_2 = np.array([[21.4, 0], [21.5, 0.036], [23, 0.0064], [33.1, 0.0019], [33.7, 0]])
adsr_c2 = np.array([[3.89, 0], [4.06, 0.0624], [5.13, 0.0292], [21.5, 0.00276]])
adsr_c6 = np.array([[4.14, 0], [4.45, 0.019], [5.24, 0.0016], [15.2, 0]])

def process_adsr_norm(arr):
    arr[:,0] = arr[:,0]-arr[0,0] # relative to 0
    arr[:,1] = arr[:,1]/arr[1,1] # normal the max value to 1

# process_adsr(adsr_c2)
# process_adsr(adsr_c6)
# process_adsr_norm(adsr_f4)
# process_adsr(adsr_f4_2)


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
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(x1/sr, y_f4, alpha=0.5)
ax.plot(x1/sr, env_f4)
ax.set_title("Enregistrement Audio en fonction du temps")
ax.set_xlabel(r"Temps $s$")
ax.set_ylabel("Amplitude")
#plt.show()
plt.savefig("figure 1")

# figure 2 - avec courbe simplifie de l'enveloppe
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(x1/sr, env_f4, alpha=0.5)
ax.plot(adsr_f4[:,0]/sr/10e-5, adsr_f4[:,1])
ax.set_title("Enregistrement Audio en fonction du temps - envelopp adsr")
ax.set_xlabel(r"Temps $s$")
ax.set_ylabel("Amplitude")
#plt.show()
plt.savefig("figure 2")

# fig, axs = plt.subplots(2, 2, figsize=(8,6))
#axs[0][0].plot(x2, y_f4_2)
# axs[0][0].plot(x2, env_f4_2, color='r')
#axs[0][0].plot(x2, y_f4_2)
# axs[0][0].plot(x1, env_f4, color='b')
# #axs[0][1].plot(x2, y_f4_2)
# axs[0][0].plot(xg, env_g, color='r')
#axs[0][0].plot(x1[envs1_attack_index], envs1[1][envs1_attack_index], color='b', marker='o')
#axs[0][0].plot(np.arange(x1[envs1_attack_index], x1[envs1_attack_index] + len(envs1_decay_arr)), envs1_decay_arr, color='g')
# plt.show()
#axs[0][1].plot(x2, y2)
#axs[0][1].plot(x2, envs2[1], color='r')
#axs[1][0].plot(x3, y3)
#axs[1][0].plot(x3, envs3[1], color='r')
# plt.plot(adsr_c2[:,0]/sr/10e-5, adsr_c2[:,1])
# plt.plot(adsr_c6[:,0]/sr/10e-5, adsr_c6[:,1])
# plt.plot(adsr_f4[:,0]/sr/10e-5, adsr_f4[:,1])
# plt.plot(adsr_f4_2[:,0]/sr/10e-5, adsr_f4_2[:,1])
# plt.show()