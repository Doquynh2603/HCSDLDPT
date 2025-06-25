import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import get_window

def rmse_function(file_path):
    # Đọc file WAV (đảm bảo mono)
    sr, signal = wav.read(file_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)  # Chuyển stereo thành mono

    # Định nghĩa kích thước khung và bước nhảy (theo giây)
    frame_length = int(0.025 * sr)  # 25 ms
    hop_length = int(0.010 * sr)    # 10 ms

    # Tạo cửa sổ Hamming
    window = get_window("hamming", frame_length, fftbins=False)

    # Tách tín hiệu thành các khung và tính RMSE cho từng khung
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    rmse_values = []
    for i in range(num_frames):
        start = i * hop_length
        frame = signal[start:start + frame_length] * window
        rmse = np.sqrt(np.mean(frame ** 2))
        rmse_values.append(rmse)

    # Tính giá trị RMSE trung bình trên tất cả các khung
    rmse_mean = np.mean(rmse_values)

    return rmse_mean