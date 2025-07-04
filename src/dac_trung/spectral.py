import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import get_window
from scipy.fft import fft

def spectral_centroid_function(file_path):
    # Đọc file WAV, chuyển stereo sang mono nếu cần
    sr, signal = wav.read(file_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Định nghĩa kích thước khung và bước nhảy (theo giây)
    frame_length = int(0.025 * sr)  # 25 ms
    hop_length = int(0.010 * sr)    # 10 ms

    # Tạo cửa sổ Hamming
    window = get_window("hamming", frame_length, fftbins=False)

    num_frames = 1 + (len(signal) - frame_length) // hop_length
    spectral_centroids = []

    for i in range(num_frames):
        start = i * hop_length
        frame = signal[start:start + frame_length] * window

        # FFT
        NFFT = 512
        magnitude_spectrum = np.abs(fft(frame, n=NFFT))[:NFFT // 2 + 1]

        # Tần số tương ứng mỗi bin
        freqs = np.linspace(0, sr / 2, len(magnitude_spectrum))

        # Tính spectral centroid: sum(frequency * magnitude) / sum(magnitude)
        if np.sum(magnitude_spectrum) > 0:
            centroid = np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum)
        else:
            centroid = 0
        spectral_centroids.append(centroid)

    # Trung bình spectral centroid trên tất cả các khung
    spectral_centroid_mean = np.mean(spectral_centroids)

    return spectral_centroid_mean