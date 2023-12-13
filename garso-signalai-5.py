from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
import wave


def readSoundFile(filename):
    with wave.open(filename, 'rb') as soundfile:
        num_channels = soundfile.getnchannels()
        frame_rate = soundfile.getframerate()
        total_frames = soundfile.getnframes()

        start_frame = max(0, soundfile.getnframes() // 2 - int(total_frames * 0.1))  
        soundfile.setpos(start_frame)

        audio_data = np.frombuffer(soundfile.readframes(-1), dtype=np.int16)


        time_frame = int(frame_rate * 0.2)  
        audio_data = audio_data[:time_frame]

        if num_channels == 2:
            if np.array_equal(audio_data[::2], audio_data[1::2]):
                audio_data = audio_data[::2].reshape(-1, 1)
                num_channels = 1
            else:
                left_channel = audio_data[::2]
                right_channel = audio_data[1::2]
                audio_data = np.column_stack((left_channel, right_channel))
        elif num_channels == 1:
            audio_data = audio_data.reshape(-1, 1)

    return audio_data, num_channels, frame_rate, time_frame

def plotSignal(audio_data, num_channels, frame_rate, filename):
    time = np.arange(len(audio_data)) / frame_rate 

    plt.figure(figsize=(10, 6))

    if num_channels == 1:
        plt.plot(time, audio_data[:, 0])
        plt.grid(True)
        plt.title(f"Original audio (Mono) of {filename.split('/')[-1]}")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
    elif num_channels == 2:
        time_stereo = np.arange(audio_data.shape[0]) / frame_rate
        plt.plot(time_stereo, audio_data[:, 0], label="Left channel")
        plt.plot(time_stereo, audio_data[:, 1], label="Right channel")
        plt.title(f"Stereo audio of {filename.split('/')[-1]}")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()

    plt.show()

def plotSpectrum(audio_data, frame_rate, filename):
    # Perform FFT on the audio data
    fft_result = np.fft.fft(audio_data[:, 0])
    fft_freq = np.fft.fftfreq(len(fft_result), d=1/frame_rate)

    # Keep only positive frequencies
    positive_freq_mask = fft_freq >= 0
    fft_freq = fft_freq[positive_freq_mask]
    fft_result = fft_result[positive_freq_mask]

    # Plot the spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(fft_freq, 20 * np.log10(np.abs(fft_result)))
    plt.title(f"Spectrum of {filename.split('/')[-1]}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)
    plt.show()

def processFrequencyDomain(audio_data, frame_rate, filename, target_frequency):
    # Perform FFT on the audio data
    fft_result = np.fft.fft(audio_data[:, 0])
    fft_freq = np.fft.fftfreq(len(fft_result), d=1/frame_rate)

    # Identify the index of the target frequency in the positive frequencies
    target_index = np.argmin(np.abs(fft_freq - target_frequency))

    # Define a small neighborhood around the target frequency to zero out
    neighborhood_size = 10  # Adjust as needed
    fft_result[target_index - neighborhood_size: target_index + neighborhood_size + 1] = 0

    # Reconstruct the time-domain signal using the modified FFT result
    processed_signal = np.fft.ifft(fft_result).real

    # Plot the original and processed spectra with only positive frequencies
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(fft_freq[fft_freq >= 0], 20 * np.log10(np.abs(fft_result[fft_freq >= 0])))
    plt.title(f"Original Spectrum of {filename.split('/')[-1]}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    processed_spectrum = np.fft.fft(processed_signal)
    plt.plot(fft_freq[fft_freq >= 0], 20 * np.log10(np.abs(processed_spectrum[fft_freq >= 0])))
    plt.title(f"Processed Spectrum - Target Frequency Removed")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def fullyReconstructSignal(original_audio_data, processed_audio_data, frame_rate, target_frequency):
    # Perform FFT on the original and processed audio data
    fft_original = np.fft.fft(original_audio_data[:, 0])
    fft_processed = np.fft.fft(processed_audio_data[:, 0])

    # Obtain the phase spectrum of the original signal
    phase_original = np.angle(fft_original)

    # Create the modified complex spectrum using the processed amplitude spectrum and original phase spectrum
    modified_spectrum = np.abs(fft_processed) * np.exp(1j * phase_original)

    # Perform the inverse Fourier transform to obtain the reconstructed signal
    reconstructed_signal = np.fft.ifft(modified_spectrum).real

    # Plot the original and reconstructed signals
    plt.figure(figsize=(12, 6))

    time_original = np.arange(len(original_audio_data)) / frame_rate
    time_reconstructed = np.arange(len(reconstructed_signal)) / frame_rate

    plt.subplot(2, 1, 1)
    plt.plot(time_original, original_audio_data[:, 0], label="Original Signal")
    plt.title("Original Signal")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_reconstructed, reconstructed_signal, label="Reconstructed Signal")
    plt.title("Reconstructed Signal")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()    

# Specify the target frequency to remove (adjust as needed)
target_frequency_to_remove = 500

filename = askopenfilename()
original_audio_data, num_channels, frame_rate, _ = readSoundFile(filename)

# Perform the processing on the original audio data
processed_audio_data, _, frame_rate, _ = readSoundFile(filename)

plotSignal(original_audio_data, num_channels, frame_rate, filename)
plotSpectrum(original_audio_data, frame_rate, filename)
processFrequencyDomain(processed_audio_data, frame_rate, filename, target_frequency_to_remove)
fullyReconstructSignal(original_audio_data, processed_audio_data, frame_rate, target_frequency_to_remove)

