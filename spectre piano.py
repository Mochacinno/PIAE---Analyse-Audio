import numpy as np
import matplotlib.pyplot as plt
import librosa

def plot_fourier(y, sr):
    N = len(y) # N is the signal length (number of samples)
    dt = 1/sr
    df= 1/((N-1)*dt) 
    frequencies = np.arange(0, sr/2, df)

    fft_result = np.fft.fft(y)

    # Take only positive frequencies
    half_idx = len(frequencies) // 2
    frequencies = frequencies[:half_idx]
    magnitude = np.abs(fft_result[:half_idx])

    return frequencies, magnitude

def prepare_audio(file_name, start_time, end_time, sr=22050):
    y, sr = librosa.load(file_name, sr=sr)
    y = y[round(sr*start_time):round(sr*end_time)]
    return y

y_f4 = prepare_audio("Audio_HD/F4.wav", 1, 7, sr=44100) #F4 2.21e4, 0 ; 2.32 0.036 ; 3.72 0.0068 ; 11.6 0.0021 ; 11.8 0
y_f4_2 = prepare_audio("Audio_HD/F4.wav", 8, 16, sr=44100) #Another F4 2.14e5 0 ; 2.15 0.036 ; 2.3 0.0064 ; 3.31  0.0019 ; 3.37 0
y_c2 = prepare_audio("Audio_HD/C2.wav", 0, 10, sr=44100) #C2 3.89e4 0 ; 4.06 0.0624 ; 5.13 0.0292 ; 21.5 0.00276
y_c6 = prepare_audio("Audio_HD/C6.wav", 1, 7, sr=44100) #C6 4.14e4 0; 4.45e4 0.019 ; 5.24e4 0.0016 ; 15.2 0

y_f4_freq, y_f4_mag = plot_fourier(y_f4, 44100)
y_f4_2_freq, y_f4_2_mag = plot_fourier(y_f4_2, 44100)
y_f4_freq, y_f4_mag = plot_fourier(y_f4, 44100)
y_f4_2_freq, y_f4_2_mag = plot_fourier(y_f4_2, 44100)
y_c2_freq, y_c2_mag = plot_fourier(y_c2, 44100)
y_c6_freq, y_c6_mag = plot_fourier(y_c6, 44100)


# Plot FFT with peaks marked
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(y_f4_freq, y_f4_mag, label="FFT f4")
ax.plot(y_f4_2_freq, y_f4_2_mag, label="FFT f4_2")
ax.axvline(350, color="black", linestyle="dashed", alpha=0.5, label="Fréquence du note joué")
ax.set_title("FFT f4")
ax.set_xlabel(r"Fréquence $s^{-1}$")
ax.set_ylabel("Magnitude")
ax.set_xlim(0, 2000)  # Limit frequency range for better visualization
ax.legend()
plt.show()

fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(y_c2_freq, y_c2_mag, label="FFT c2")
#ax.axvline(350, color="black", linestyle="dashed", alpha=0.5, label="Fréquence du note joué")
ax.set_title("FFT c2")
ax.set_xlabel(r"Fréquence $s^{-1}$")
ax.set_ylabel("Magnitude")
ax.set_xlim(0, 500)  # Limit frequency range for better visualization
ax.legend()
plt.show()

fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(y_c6_freq, y_c6_mag, label="FFT c2")
#ax.axvline(350, color="black", linestyle="dashed", alpha=0.5, label="Fréquence du note joué")
ax.set_title("FFT c6")
ax.set_xlabel(r"Fréquence $s^{-1}$")
ax.set_ylabel("Magnitude")
ax.set_xlim(0, 10000)  # Limit frequency range for better visualization
ax.legend()
plt.show()