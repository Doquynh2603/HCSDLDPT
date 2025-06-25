
import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack
from scipy.signal import get_window
from scipy.fft import fft

def mfcc_function(file_path, n_mfcc=13, sr_target=44100, frame_length=0.025, hop_length=0.010):
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def mel_filterbank(num_filters, fft_size, sr):
        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(sr / 2)
        mel_points = np.linspace(mel_min, mel_max, num_filters + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((fft_size + 1) * hz_points / sr).astype(int)

        filters = np.zeros((num_filters, fft_size // 2 + 1))
        for i in range(1, num_filters + 1):
            start, center, end = bin_points[i - 1], bin_points[i], bin_points[i + 1]
            if center == start: center += 1
            if end == center: end += 1
            filters[i - 1, start:center] = (np.arange(start, center) - start) / (center - start)
            filters[i - 1, center:end] = (end - np.arange(center, end)) / (end - center)
        return filters

    # 1. Đọc file WAV
    sr, signal = wav.read(file_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)  # stereo → mono

    # 2. Lọc tiền nhấn (pre_emphasis)
    pre_emphasis = 0.97
    signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    # 3. Cắt khung và tạo cửa sổ
    frame_size = int(frame_length * sr)
    hop_size = int(hop_length * sr)
    window = get_window("hamming", frame_size, fftbins=True)
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    frames = np.stack([
        signal[i * hop_size:i * hop_size + frame_size] * window
        for i in range(num_frames)
    ])

    # 4. FFT và phổ công suất
    NFFT = 512
    magnitude_spectrum = np.abs(fft(frames, n=NFFT))[:, :NFFT // 2 + 1]
    power_spectrum = (1.0 / NFFT) * (magnitude_spectrum ** 2)

    # 5. Áp dụng bộ lọc Mel
    num_mel_filters = 26
    filter_bank = mel_filterbank(num_mel_filters, NFFT, sr)
    mel_energies = np.dot(power_spectrum, filter_bank.T)
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)

    # 6. Log năng lượng Mel
    log_mel = np.log(mel_energies)

    # 7. DCT để lấy MFCC
    mfcc = scipy.fftpack.dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    # 8. Trung bình qua các khung
    mfcc_mean = np.mean(mfcc, axis=0)

    return mfcc_mean