import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from gpu_audio import compute_cqt_db  # GPU-aware CQT (torch+torchaudio if available)


def plot_audio_spectrograms(file_path, output_base_path, cqt_dir):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Create CQT spectrogram
    size = (20, 6)
    plt.figure(figsize=size)
    # Separate harmonic and percussive components (CPU)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # Heavy CQT computation: GPU if available, otherwise CPU
    hop_length = 64
    C_db = compute_cqt_db(y_harmonic, sr=sr, bins_per_octave=126, hop_length=hop_length, n_bins=84)
    librosa.display.specshow(C_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.f dB')
    plt.title('Harmonic Component - Constant-Q Transform Spectrogram')
    plt.savefig(os.path.join(cqt_dir, f"{os.path.basename(output_base_path)}.jpg"))
    plt.close()


# Process all MP3 files in playlist directory
playlist_dir = 'playlist'
output_dir = 'output'
cqt_dir = os.path.join(output_dir, 'cqt-hop64-hpss-bpo126')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cqt_dir, exist_ok=True)

for filename in os.listdir(playlist_dir):
    if filename.endswith('.mp3'):
        file_path = os.path.join(playlist_dir, filename)
        output_base_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}")
        print(f"Processing {filename}...")
        plot_audio_spectrograms(file_path, output_base_path, cqt_dir)

# After batch processing, open interactive window for last track and keep it open
try:
    import interactive_cqt_slider
    interactive_cqt_slider.interactive_cqt_for_last_track('playlist')
except Exception as e:
    print(f"Failed to open interactive window: {e}")
