import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

sr = 22050 
duration = 2.0
freq = 440.0

def create_sine_wave_wav(sr, duration, freq):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * freq * t)
    sine_wave2 = 0.5 * np.sin(2 * np.pi * freq * 5/4 * t)
    sine_wave3 = 0.5 * np.sin(2 * np.pi * freq * 3/2 * t)
    #sine_wave4 = 0.5 * np.sin(2 * np.pi * freq * 4 * t)
    final_wave = sine_wave + sine_wave2 + sine_wave3
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

create_sine_wave_et_wav(sr, duration, freq)

def analyse_audio():
    y, sr = librosa.load("sine_wave4.wav", sr=sr)

    # Waveform (Amplitude over Time)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform of Sine Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    # FFT
    fft_result = np.fft.fft(y)
    frequencies = np.fft.fftfreq(len(fft_result), d=1/sr)

    # Take only positive frequencies
    half_idx = len(frequencies) // 2
    frequencies = frequencies[:half_idx]
    magnitude = np.abs(fft_result[:half_idx])

    # Plot FFT (Frequency Domain)
    plt.figure(figsize=(12, 4))
    plt.plot(frequencies, magnitude)
    plt.title("FFT of Sine Wave")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 2000)  # Limit frequency range for better visualization
    plt.show()

# à faire
# for timbre
# spectrogram
# envelope
# transient

