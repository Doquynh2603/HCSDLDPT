import librosa
import numpy as np

def mfcc_function(file_path):
    n_mfcc=13
    sr=16000
    frame_length=0.025
    hop_length=0.010
    # Bước 1: Đọc tín hiệu âm thanh
    y, sr = librosa.load(file_path, sr=sr)
    
    # Bước 2-7: Tính MFCC (librosa tự động thực hiện các bước này)
    # - Chia khung (frame_length=25ms, hop_length=10ms)
    # - Áp dụng cửa sổ Hamming
    # - Tính FFT
    # - Áp dụng bộ lọc Mel
    # - Tính log năng lượng Mel
    # - Áp dụng DCT để thu MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, 
                                n_fft=int(frame_length * sr), 
                                hop_length=int(hop_length * sr))
    
    # Bước 8: Tính trung bình MFCC trên tất cả các khung
    mfcc_mean = np.mean(mfcc, axis=1)
    
    return mfcc_mean

# file_path = "D:\\NAM4\\KY2\\HCSDLDPT\\HCSDLDPT\\src\\Dataset\\F01\\F01-S001.wav"
# mfcc_vector = compute_mfcc(file_path)
# print("Vector MFCC (13 chiều):", mfcc_vector)