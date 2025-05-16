import librosa
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def speach_rate_function(file_path): 
    # 1. Load file âm thanh
    y, sr = librosa.load(file_path, sr=44100)

    # 2. Cấu hình frame
    frame_length = int(0.025 * sr)  # 25ms
    hop_length = int(0.010 * sr)    # 10ms (overlap)
    duration_sec = len(y) / sr

    # 3. Tính năng lượng RMS cho mỗi frame
    rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # 4. Z-score normalization
    z_scores = (rmse - np.mean(rmse)) / np.std(rmse)

    # 5. Phân cụm KMeans để tự động chọn ngưỡng (threshold)
    z_scores_reshape = z_scores.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0,n_init=10).fit(z_scores_reshape)
    centers = kmeans.cluster_centers_.flatten()
    threshold = np.mean(centers)

    # 6. Đếm số đoạn voiced (nơi vượt qua ngưỡng)
    voiced = z_scores > threshold
    changes = np.diff(voiced.astype(int))
    onsets = np.where(changes == 1)[0]

    # 7. Tính nhịp độ
    syllable_estimate = len(onsets)
    speech_rate = syllable_estimate / duration_sec
    return speech_rate

