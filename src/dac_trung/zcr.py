import librosa
import numpy as np

def zcr_function(file_path):
    sr=44100
    frame_length=0.025
    hop_length=0.010
    #Đọc tín hiệu âm thanh với tần số lấy mẫu 44.1 kHz
    y, sr = librosa.load(file_path, sr=sr)
    
    #Tính ZCR cho mỗi khung
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=int(frame_length * sr), 
                                             hop_length=int(hop_length * sr))
    
    #Tính trung bình ZCR trên tất cả các khung
    zcr_mean = np.mean(zcr)
    
    return zcr_mean