import librosa
import numpy as np
import os
def mel_function(file_path):
    y, sr = librosa.load(file_path, sr=None)  # giữ nguyên sample rate gốc
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    # Chuyển sang dB để dễ dùng hơn
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # Có thể tính trung bình theo trục thời gian để lấy vector đặc trưng
    mean_mel = np.mean(mel_spectrogram_db, axis=1)  # vector có n_mels phần tử
    return mean_mel  # trả về 1 vector (128 chiều nếu n_mels=128)
