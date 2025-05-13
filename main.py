import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.signal
import soundfile as sf
import scipy

rec_sr = 44100 
# 44100 pour HD ou 22050
duration = 2.0
freq = 440.0

def create_sine_wave_wav(sr, duration, freq):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * freq * t)
    #sine_wave2 = 0.5 * np.sin(2 * np.pi * freq * 5/4 * t)
    #sine_wave3 = 0.5 * np.sin(2 * np.pi * freq * 3/2 * t)
    #sine_wave4 = 0.5 * np.sin(2 * np.pi * freq * 4 * t)
    #final_wave = sine_wave + sine_wave2 + sine_wave3
    final_wave /= np.max(np.abs(final_wave))  # Normalize to range [-1, 1]

    sf.write("sine_wave_maj2.wav", final_wave, sr)

def create_sine_wave_et_wav(sr, duration, freq): # equal temperatement (on a piano)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * freq * t)
    sine_wave2 = 0.5 * np.sin(2 * np.pi * freq * 2**(4/12) * t)
    sine_wave3 = 0.5 * np.sin(2 * np.pi * freq * 2**(7/12) * t)
    #sine_wave4 = 0.5 * np.sin(2 * np.pi * freq * 4 * t)
    final_wave = sine_wave + sine_wave2 + sine_wave3
    final_wave /= np.max(np.abs(final_wave))  # Normalize to range [-1, 1]

    sf.write("sine_wave_maj_et.wav", final_wave, sr)

#create_sine_wave_et_wav(sr, duration, freq)

def analyse_audio(y, sr):
    N = len(y) # N is the signal length (number of samples)
    dt = 1/sr
    df= 1/((N-1)*dt) 
    frequencies = np.arange(0, sr/2, df)

    fft_result = np.fft.fft(y)

    # Take only positive frequencies
    half_idx = len(frequencies) // 2
    frequencies = frequencies[:half_idx]
    magnitude = np.abs(fft_result[:half_idx])

    # Find peaks with minimum distance between them (e.g., 50 Hz apart)
    # min_peak_distance = 50  # minimum distance in Hz between peaks
    # min_peak_idx_distance = int(min_peak_distance / df)  # convert to index distance
    
    # # Find peaks with height threshold (optional) and minimum distance
    # mag_ipeaks, _ = scipy.signal.find_peaks(
    #     magnitude, 
    #     height=np.max(magnitude)*0.01,  # 10% of max magnitude as threshold (adjust as needed) # 1% for guitar
    #     distance=min_peak_idx_distance
    # )
    
    # # Get the frequencies and magnitudes of the peaks
    # peak_frequencies = frequencies[mag_ipeaks]
    # peak_amplitudes = magnitude[mag_ipeaks]
    
    # # Sort peaks by amplitude (descending order)
    # sorted_indices = np.argsort(peak_amplitudes)[::-1]
    # peak_frequencies = peak_frequencies[sorted_indices]
    # peak_amplitudes = peak_amplitudes[sorted_indices]
    
    # Print the peaks
    # print("Detected peaks (frequency, amplitude):")
    # for freq, amp in zip(peak_frequencies, peak_amplitudes):
    #     print(f"{freq:.2f} Hz: {amp:.2f}")


    # Plot FFT with peaks marked
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(frequencies, magnitude)
    # ax[0].scatter(peak_frequencies, peak_amplitudes, color='red', marker='x', label='Peaks')
    ax[0].set_title("FFT of Sine Wave with Peaks")
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("Magnitude")
    ax[0].set_xlim(0, 2000)  # Limit frequency range for better visualization
    ax[0].legend()
    
    # ax[1].scatter(peak_frequencies, peak_amplitudes, color='red', marker='o', label='Peaks')
    # ax[1].set_xscale('log')
    # ax[1].set_yscale('log')

    # --- Fit a line to the log-transformed data ---
    # Avoid log(0) by filtering out zeros (if any)
    # valid_mask = (peak_frequencies > 0) & (peak_amplitudes > 0)
    # log_freq = np.log10(peak_frequencies[valid_mask])
    # log_amp = np.log10(peak_amplitudes[valid_mask])

    # # Perform linear regression (degree=1 for straight line)
    # slope, intercept = np.polyfit(log_freq, log_amp, 1)

    # # Generate points for the best-fit line
    # x_fit = np.linspace(np.min(log_freq), np.max(log_freq), 100)
    # y_fit = slope * x_fit + intercept

    # # Plot the best-fit line (convert back from log space)
    # ax[1].plot(10**x_fit, 10**y_fit, 'b--', 
    #          label=f'Best-fit line (slope={slope:.2f})')
    # print(slope)
    plt.show()

def analyse_audio_guitar():
    y, sr = librosa.load("Guitar.wav", sr=rec_sr)
    y = y[round(22050*6):round(22050*6.5)] # 6s to 7s played a C3
    # Waveform (Amplitude over Time)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform of Sine Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    # FFT
    N = len(y)             # N is the signal length (number of samples)
    dt = 1/sr
    df= 1/((N-1)*dt) 
    frequencies = np.arange(0, sr/2, df)

    fft_result = np.fft.fft(y)

    # Take only positive frequencies
    half_idx = len(frequencies) // 2
    frequencies = frequencies[:half_idx]
    magnitude = np.abs(fft_result[:half_idx])

    #smooth_mag = scipy.ndimage.gaussian_filter1d(magnitude, 30)
    # mag_ipeaks, _ = scipy.signal.find_peaks(magnitude)
    # amplitudes = magnitude[mag_ipeaks]
    # amplitudes = amplitudes[amplitudes > 24]
    # print(amplitudes)
    # n = np.array([1, 2, 3])
    # m, b = np.polyfit(n, amplitudes, 1)
    #poly = np.poly1d(m, b)
    # plt.loglog(n, amplitudes, marker='o')
    # plt.title("FFT of Sine Wave")
    # plt.plot(n, m*n+b)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude")
    # plt.show()

    # Plot FFT (Frequency Domain)
    plt.figure(figsize=(12, 4))
    plt.plot(frequencies, magnitude)
    plt.title("FFT of Sine Wave")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    # plt.xlim(0, 2000)  # Limit frequency range for better visualization
    plt.show()


# à faire
# for timbre
# spectrogram
# envelope
# transient
#https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S3_Timbre.html
# https://ar5iv.labs.arxiv.org/html/1306.2859
#https://www.dafx.de/paper-archive/2000/pdf/Caroline_Traube.pdf
# https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S3_Timbre.html
# guitar plucking position matters because naer the bridge the higher harmonics are louder and towards the center, the fundamental is higher
# donc CL plutot le clavecin avec position ou on frappe 

#analyse_audio("Audio_HD/F4.wav", 44100, 1.5, 2.5)
#analyse_audio("Guitar.wav", 22050, 31, 32)
#analyse_audio("Guitar.wav", 22050, 18, 19)
#analyse_audio_guitar()

def gen_note_piano(freq):
    sr = 22050
    duration = 5
    b = 2.2*10**-2
    c = 2*10**-2
    u0 = 5*10**-2
    a = 1*1e-2
    L = 62.5*1e-2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = 0
    for n in range(1, 5):
        cn = 2*u0*b/n/np.pi/c*np.sin(n*np.pi*a/L)
        sine_wave += cn * np.sin(2 * np.pi * freq*n * t)
    #final_wave /= np.max(np.abs(final_wave))  # Normalize to range [-1, 1]
    sf.write("piano_note2.wav", sine_wave, sr)

def gen_note_guitar(freq):
    pass
#gen_note_piano(261.63)

# Waveform (Amplitude over Time)
# plt.figure(figsize=(12, 4))
# librosa.display.waveshow(y, sr=sr)
# plt.title("Waveform of Sine Wave")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")

def plot_fourier(y, sr, target_freq):
    N = len(y) # N is the signal length (number of samples)
    dt = 1/sr
    df= 1/((N-1)*dt) 
    frequencies = np.arange(0, sr/2, df)

    fft_result = np.fft.fft(y)

    # Take only positive frequencies
    half_idx = len(frequencies) // 2
    frequencies = frequencies[:half_idx]
    magnitude = np.abs(fft_result[:half_idx])

    # Find peaks with minimum distance between them (e.g., 50 Hz apart)
    # min_peak_distance = 50  # minimum distance in Hz between peaks
    # min_peak_idx_distance = int(min_peak_distance / df)  # convert to index distance
    
    # # Find peaks with height threshold (optional) and minimum distance
    # mag_ipeaks, _ = scipy.signal.find_peaks(
    #     magnitude, 
    #     height=np.max(magnitude)*0.01,  # 10% of max magnitude as threshold (adjust as needed) # 1% for guitar
    #     distance=min_peak_idx_distance
    # )
    
    # # Get the frequencies and magnitudes of the peaks
    # peak_frequencies = frequencies[mag_ipeaks]
    # peak_amplitudes = magnitude[mag_ipeaks]
    
    # # Sort peaks by amplitude (descending order)
    # sorted_indices = np.argsort(peak_amplitudes)[::-1]
    # peak_frequencies = peak_frequencies[sorted_indices]
    # peak_amplitudes = peak_amplitudes[sorted_indices]
    
    # Print the peaks
    # print("Detected peaks (frequency, amplitude):")
    # for freq, amp in zip(peak_frequencies, peak_amplitudes):
    #     print(f"{freq:.2f} Hz: {amp:.2f}")


    # Plot FFT with peaks marked
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.plot(frequencies, magnitude, label="FFT")
    ax.axvline(target_freq, color="black", linestyle="dashed", alpha=0.5, label="Fréquence du note joué")
    ax.set_title("FFT")
    ax.set_xlabel(r"Fréquence $s^{-1}$")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, 2000)  # Limit frequency range for better visualization
    ax.legend()
    plt.show()

sr = 44100
# y_f4, sr = librosa.load("Audio_HD/F4.wav", sr=sr)
# y_f4 = y_f4[round(sr*1):round(sr*7)]
# plot_fourier(y_f4, sr, 349.22)

# y_c2, sr = librosa.load("Audio_HD/C2.wav", sr=sr)
# y_c2 = y_c2[round(sr*0):round(sr*10)]
# plot_fourier(y_c2, sr, 65.41)

# y_c6, sr = librosa.load("Audio_HD/C6.wav", sr=sr)
# y_c6 = y_c6[round(sr*1):round(sr*7)]
# plot_fourier(y_c6, sr, 1046.502)
y_guitar, sr = librosa.load("Audio/GuitarAplucklength.wav", sr=sr)
y_guitar_low = y_guitar[round(sr*15):round(sr*19)]
y_guitar_low2 = y_guitar[round(sr*12):round(sr*14)]
y_guitar_mid = y_guitar[round(sr*37):round(sr*41)]
y_guitar_mid2 = y_guitar[round(sr*42):round(sr*44)]
y_guitar_supr = y_guitar[round(sr*50):round(sr*52)]
plot_fourier(y_guitar_low, sr, 110)
plot_fourier(y_guitar_low2, sr, 110)
plot_fourier(y_guitar_mid, sr, 110)
plot_fourier(y_guitar_mid, sr, 110)
plot_fourier(y_guitar_mid2, sr, 110)
plot_fourier(y_guitar_supr, sr, 110)