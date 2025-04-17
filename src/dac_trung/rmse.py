import librosa
import numpy as np

def rmse_function(file_path):
    sr=16000
    frame_length=0.025
    hop_length=0.010
    # Bước 1: Đọc tín hiệu âm thanh với tần số lấy mẫu 44.1 kHz
    y, sr = librosa.load(file_path, sr=sr)
    
    # Bước 2-4: Tính RMSE cho mỗi khung
    # - Độ dài khung: 25ms × 44100 Hz = 1102.5 mẫu
    # - Độ dài bước: 10ms × 44100 Hz = 441 mẫu
    rmse = librosa.feature.rms(y=y, frame_length=int(frame_length * sr), 
                               hop_length=int(hop_length * sr))
    
    # Bước 5: Tính trung bình RMSE trên tất cả các khung
    rmse_mean = np.mean(rmse)
    
    return rmse_mean

