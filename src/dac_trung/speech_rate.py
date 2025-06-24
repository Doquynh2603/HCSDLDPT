# import librosa
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# def speach_rate_function(file_path): 
#     # 1. Load file âm thanh
#     y, sr = librosa.load(file_path, sr=44100)

#     # 2. Cấu hình frame
#     frame_length = int(0.025 * sr)  # 25ms
#     hop_length = int(0.010 * sr)    # 10ms (overlap)
#     duration_sec = len(y) / sr

#     # 3. Tính năng lượng RMS cho mỗi frame
#     rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

#     # 4. Z-score normalization
#     z_scores = (rmse - np.mean(rmse)) / np.std(rmse)

#     # 5. Phân cụm KMeans để tự động chọn ngưỡng (threshold)
#     z_scores_reshape = z_scores.reshape(-1, 1)
#     kmeans = KMeans(n_clusters=2, random_state=0,n_init=10).fit(z_scores_reshape)
#     centers = kmeans.cluster_centers_.flatten()
#     threshold = np.mean(centers)

#     # 6. Đếm số đoạn voiced (nơi vượt qua ngưỡng)
#     voiced = z_scores > threshold
#     changes = np.diff(voiced.astype(int))
#     onsets = np.where(changes == 1)[0]

#     # 7. Tính nhịp độ
#     syllable_estimate = len(onsets)
#     speech_rate = syllable_estimate / duration_sec
#     return speech_rate

import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import get_window

def speech_rate_function(file_path):
    # Đọc file WAV, chuyển stereo sang mono nếu cần
    sr, signal = wav.read(file_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    # if sr != 44100:
    #     raise ValueError(f"Expected sampling rate 44100 Hz, but got {sr}")

    # Cấu hình khung và bước nhảy
    frame_length = int(0.025 * sr)  # 25ms
    hop_length = int(0.010 * sr)    # 10ms

    window = get_window("hamming", frame_length, fftbins=False)

    num_frames = 1 + (len(signal) - frame_length) // hop_length

    # Tính năng lượng RMS cho mỗi khung
    rms_values = []
    for i in range(num_frames):
        start = i * hop_length
        frame = signal[start:start + frame_length] * window
        rms = np.sqrt(np.mean(frame ** 2))
        rms_values.append(rms)
    rms_values = np.array(rms_values)

    # Z-score normalization
    rms_mean = np.mean(rms_values)
    rms_std = np.std(rms_values)
    z_scores = (rms_values - rms_mean) / rms_std if rms_std > 0 else rms_values - rms_mean

    # Chọn threshold bằng trung bình hai cụm (giả sử phân bố gần bimodal)
    threshold = np.mean([np.min(z_scores), np.max(z_scores)])

    # Xác định các khung voiced dựa trên threshold
    voiced = z_scores > threshold
    changes = np.diff(voiced.astype(int))
    onsets = np.where(changes == 1)[0]

    # Tính tốc độ nói (số âm tiết ước tính trên giây)
    duration_sec = len(signal) / sr
    syllable_estimate = len(onsets)
    speech_rate = syllable_estimate / duration_sec if duration_sec > 0 else 0

    return speech_rate