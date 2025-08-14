# This module now delegates GPU/CPU CQT computation to the unified implementation in audio_cqt.
# Kept for backward compatibility of imports.
import numpy as np
from audio_cqt import CQTComputer, default_cqt_computer

# Create a single shared computer instance to reuse any internal caching/logging semantics
_COMPUTER: CQTComputer = default_cqt_computer()


def compute_cqt_db(y_harmonic: np.ndarray, sr: int, bins_per_octave: int, hop_length: int = 64, n_bins: int = 84) -> np.ndarray:
    """
    Backward-compatible wrapper. Computes CQT (in dB) using the unified GPU-aware implementation.
    """
    return _COMPUTER.compute_cqt_db(y_harmonic=y_harmonic, sr=sr, bins_per_octave=bins_per_octave, hop_length=hop_length, n_bins=n_bins)
