import os
import librosa
import numpy as np

# Hàm trích rút pitch cho một file âm thanh
def pitch_function(file_path):
    # Tải file âm thanh
    y, sr = librosa.load(file_path, sr=None)  # sr=None giữ nguyên sample rate ban đầu

    # Trích xuất pitch (F0) sử dụng pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))

    # Chọn tần số cơ bản (F0) của các khung (frame) có tần số
    pitch = f0[voiced_flag]  # Chỉ lấy những giá trị có tần số (không phải NaN)

    return pitch