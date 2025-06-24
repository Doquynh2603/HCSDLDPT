from waveform_display import show_multiple_waveforms
from mfcc_display import show_multiple_mfcc
from spectrogram import show_spectrogram
from attribute import extract_features
import os
input_audio_path = 'D:\\NAM4\\KY2\\HCSDLDPT\\HCSDLDPT\\src\\TestData\\VIVOSSPK01_R021.wav'
nearest = extract_features(input_audio_path)
print("\n🔍 3 file gần nhất:")
for i in nearest:
    print(i)

show_multiple_waveforms(input_audio_path,nearest)
show_multiple_mfcc(input_audio_path,nearest)
show_spectrogram(input_audio_path,nearest)