import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

def show_multiple_waveforms(input_audio_path, nearest_files):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    y, sr = librosa.load(input_audio_path, sr=44100)
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title("Waveform của Input Audio")
    axes[0].set_xlabel("Thời gian (s)")
    axes[0].set_ylabel("Biên độ")

    for i, item in enumerate(nearest_files):
        audio_path = os.path.join('Dataset', item['file_name'])
        y, sr = librosa.load(audio_path, sr=44100)
        librosa.display.waveshow(y, sr=sr, ax=axes[i + 1])
        axes[i + 1].set_title(f"Waveform của {item['file_name']}")
        axes[i + 1].set_xlabel("Thời gian (s)")
        axes[i + 1].set_ylabel("Biên độ")

    plt.tight_layout()
    plt.show()