import librosa
import numpy as np
import os
def mffc_function(file_path):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13).T, axis=0)
    return mfcc
