import numpy as np
import scipy.io.wavfile as wav

def zcr_function(file_path):
    # Bước 1: Đọc file WAV, chuyển stereo sang mono nếu cần
    sr, signal = wav.read(file_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Bước 2: Cấu hình khung và bước nhảy
    frame_length = int(0.025 * sr)  # 25ms
    hop_length = int(0.010 * sr)    # 10ms

    num_frames = 1 + (len(signal) - frame_length) // hop_length

    zcr_values = []
    for i in range(num_frames):
        start = i * hop_length
        frame = signal[start:start + frame_length]
        # Bước 3: Tính số lần đổi dấu trong khung (Zero Crossing Rate)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_length)
        zcr_values.append(zero_crossings)

    zcr_values = np.array(zcr_values)

    # Bước 4: Tính trung bình ZCR trên tất cả các khung
    zcr_mean = np.mean(zcr_values)

    return zcr_mean