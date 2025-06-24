import librosa
# import numpy as np

# def mfcc_function(file_path):
#     n_mfcc=13
#     sr=44100
#     frame_length=0.025
#     hop_length=0.010
#     # Bước 1: Đọc tín hiệu âm thanh
#     y, sr = librosa.load(file_path, sr=sr)
    
#     # Bước 2-7: Tính MFCC (librosa tự động thực hiện các bước này)
#     # - Chia khung (frame_length=25ms, hop_length=10ms)
#     # - Áp dụng cửa sổ Hamming
#     # - Tính FFT
#     # - Áp dụng bộ lọc Mel
#     # - Tính log năng lượng Mel
#     # - Áp dụng DCT để thu MFCC
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, 
#                                 n_fft=int(frame_length * sr), 
#                                 hop_length=int(hop_length * sr))
    
#     # Bước 8: Tính trung bình MFCC trên tất cả các khung
#     mfcc_mean = np.mean(mfcc, axis=1)
    
#     return mfcc_mean

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
    # if sr != sr_target:
    #     raise ValueError(f"Expected sampling rate {sr_target}, but got {sr}")
    # signal, sr = librosa.load(file_path, sr=sr_target)  # tự động chuyển về mono và 44100Hz

    # 2. Cắt khung và tạo cửa sổ
    frame_size = int(frame_length * sr)
    hop_size = int(hop_length * sr)
    window = get_window("hamming", frame_size, fftbins=True)
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    frames = np.stack([
        signal[i * hop_size:i * hop_size + frame_size] * window
        for i in range(num_frames)
    ])

    # 3. FFT và phổ công suất
    NFFT = 512
    magnitude_spectrum = np.abs(fft(frames, n=NFFT))[:, :NFFT // 2 + 1]
    power_spectrum = (1.0 / NFFT) * (magnitude_spectrum ** 2)

    # 4. Áp dụng bộ lọc Mel
    num_mel_filters = 26
    filter_bank = mel_filterbank(num_mel_filters, NFFT, sr)
    mel_energies = np.dot(power_spectrum, filter_bank.T)
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)

    # 5. Log năng lượng Mel
    log_mel = np.log(mel_energies)

    # 6. DCT để lấy MFCC
    mfcc = scipy.fftpack.dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    # 7. Trung bình qua các khung
    mfcc_mean = np.mean(mfcc, axis=0)

    return mfcc_mean