import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_audio_spectrograms(file_path, output_base_path, cqt_dir):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Create CQT spectrogram
    size = (20, 6)
    plt.figure(figsize=size)
    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    C = librosa.cqt(y_harmonic, sr=sr)
    C_magnitude = np.abs(C)

    # Apply per-frequency median normalization
    median_magnitudes = np.median(C_magnitude, axis=1, keepdims=True)
    C_normalized = C_magnitude / median_magnitudes
    C_db = librosa.amplitude_to_db(C_normalized, ref=np.max)

    librosa.display.specshow(C_db, sr=sr, hop_length=64, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.f dB')
    plt.title('Harmonic Component - Normalized Constant-Q Transform Spectrogram')
    plt.savefig(os.path.join(cqt_dir, f"{os.path.basename(output_base_path)}_cqt.jpg"))
    plt.close()


# Process all MP3 files in playlist directory
playlist_dir = 'playlist'
output_dir = 'output'
cqt_dir = os.path.join(output_dir, 'cqt-hop64-hpss')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cqt_dir, exist_ok=True)

for filename in os.listdir(playlist_dir):
    if filename.endswith('.mp3'):
        file_path = os.path.join(playlist_dir, filename)
        output_base_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}")
        print(f"Processing {filename}...")
        plot_audio_spectrograms(file_path, output_base_path, cqt_dir)
