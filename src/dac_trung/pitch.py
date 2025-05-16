import librosa
import numpy as np

def pitch_function(path):
    # Load file âm thanh và chuẩn hóa sample rate về 44100 Hz
    y, sr = librosa.load(path, sr=44100)
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    # Trích xuất cao độ bằng phương pháp pyin với frame_length = 400 (25ms) và hop_length = 160 (10ms)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('E3'),  # Tần số thấp nhất (khoảng 165 Hz)
        fmax=librosa.note_to_hz('G5'),  # Tần số cao nhất (khoảng 784 Hz)
        frame_length = frame_length,  
        hop_length = hop_length  
    )

    # Lọc ra các giá trị f0 hợp lệ (voiced)
    f0 = f0[~np.isnan(f0)]

    if len(f0) > 0:
        return float(np.mean(f0))
    else:
        return 0.0  # Không tìm thấy pitch nào hợp lệ
