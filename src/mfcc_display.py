import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
def show_multiple_mfcc(input_audio_path, nearest_files): 
    y_input, sr_input = librosa.load(input_audio_path, sr=44100)
    mfcc_input = librosa.feature.mfcc(y=y_input, sr=sr_input, n_mfcc=13)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    # Vẽ MFCC input
    librosa.display.specshow(mfcc_input, x_axis='time', sr=sr_input, ax=axes[0])
    axes[0].set_title("MFCC của Input Audio")
    axes[0].set_xlabel("Thời gian (s)")
    axes[0].set_ylabel("MFCC Coefficients")

    # Vẽ MFCC cho các output files
    for i, item in enumerate(nearest_files):
        audio_path = os.path.join('Dataset', item['file_name'])
        y_output, sr_output = librosa.load(audio_path, sr=44100)
        mfcc_output = librosa.feature.mfcc(y=y_output, sr=sr_output, n_mfcc=13)

        librosa.display.specshow(mfcc_output, x_axis='time', sr=sr_output, ax=axes[i+1])
        axes[i + 1].set_title(f"MFCC của {item['file_name']}")
        axes[i+1].set_xlabel("Thời gian (s)")
        axes[i+1].set_ylabel("MFCC Coefficients")

    plt.tight_layout()
    plt.show()