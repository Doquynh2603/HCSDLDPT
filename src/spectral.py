import librosa
import numpy as np
import os
def spectral_function(file_path):
    y, sr = librosa.load(file_path, sr=None)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    # Lấy giá trị trung bình hoặc bạn có thể lấy theo từng frame
    mean_centroid = np.mean(spectral_centroids)
    return mean_centroid
