import librosa
import numpy as np
import os
def zcr_function(file_path):
    y, sr = librosa.load(file_path, sr=None)
    # Tính Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    # Tính trung bình ZCR toàn bộ file
    zcr_mean = np.mean(zcr)
    return zcr_mean