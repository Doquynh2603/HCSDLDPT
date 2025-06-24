import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import numpy as np
def show_spectrogram(input_audio_path,nearest_files): 
    y_input, sr_input = librosa.load(input_audio_path, sr=44100)

    D_input = librosa.amplitude_to_db(librosa.stft(y_input), ref=np.max)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    librosa.display.specshow(D_input, x_axis='time', y_axis='log', sr=sr_input, ax=axes[0])
    axes[0].set_title("Spectrogram của Input Audio")
    axes[0].set_xlabel("Thời gian (s)")
    axes[0].set_ylabel("Tần số (Hz)")


    for i, item in enumerate(nearest_files):
        audio_path = os.path.join('Dataset', item['file_name'])
        y_output, sr_output = librosa.load(audio_path, sr=44100)
        D_output = librosa.amplitude_to_db(librosa.stft(y_output), ref=np.max)
        
        librosa.display.specshow(D_output, x_axis='time', y_axis='log', sr=sr_output, ax=axes[i+1])
        axes[i + 1].set_title(f"Spectrogram của {item['file_name']}")
        axes[i+1].set_xlabel("Thời gian (s)")
        axes[i+1].set_ylabel("Tần số (Hz)")

    plt.tight_layout()
    plt.show()