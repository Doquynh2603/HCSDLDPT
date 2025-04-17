import librosa
import numpy as np

def spectral_centroid_function(file_path):
    sr=16000
    frame_length=0.025
    hop_length=0.010
    #Đọc tín hiệu âm thanh với tần số lấy mẫu 44.1 kHz
    y, sr = librosa.load(file_path, sr=sr)
    
    #Tính Spectral Centroid cho mỗi khung
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                          n_fft=int(frame_length * sr), 
                                                          hop_length=int(hop_length * sr))
    
    #Tính trung bình Spectral Centroid trên tất cả các khung
    spectral_centroid_mean = np.mean(spectral_centroid)
    
    return spectral_centroid_mean