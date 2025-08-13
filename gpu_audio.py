import os
import numpy as np


# We only import librosa at module level (CPU). Torch + torchaudio are optional and imported lazily.
import librosa

# Environment-controlled logging to help diagnose acceleration usage
# Set DJAI_ACCEL_LOG=1 to enable detailed logs.
_ACCEL_LOG = str(os.getenv("DJAI_ACCEL_LOG", "0")).lower() not in ("0", "false", "no", "")
# Strict mode: require GPU; if unavailable, raise an error instead of falling back.
_REQUIRE_GPU = str(os.getenv("DJAI_REQUIRE_GPU", "0")).lower() not in ("0", "false", "no", "")
# Opt-in flag: try importing torch/torchaudio for GPU acceleration only if explicitly enabled.
# This avoids crashes in environments with broken torch installations. Set DJAI_TRY_TORCH=1 to enable.
_TRY_TORCH = str(os.getenv("DJAI_TRY_TORCH", "0")).lower() not in ("0", "false", "no", "")
_logged_env_once = False
_warned_cpu_once = False


def _log_once_env(torch, torchaudio):
    global _logged_env_once
    if _logged_env_once or not _ACCEL_LOG:
        return
    try:
        print("[ACCEL] torch version:", getattr(torch, "__version__", "unknown"))
        print("[ACCEL] torchaudio version:", getattr(torchaudio, "__version__", "unknown"))
        print("[ACCEL] CUDA available:", torch.cuda.is_available())
        print("[ACCEL] torch.version.cuda:", getattr(torch.version, "cuda", None))
        if torch.cuda.is_available():
            try:
                n = torch.cuda.device_count()
                print(f"[ACCEL] CUDA device count: {n}")
                for i in range(n):
                    print(f"[ACCEL] Device {i}: {torch.cuda.get_device_name(i)}")
            except Exception as e:
                print(f"[ACCEL] Failed to enumerate CUDA devices: {e}")
    finally:
        _logged_env_once = True



def _gpu_cqt_db(y_harmonic: np.ndarray, sr: int, bins_per_octave: int, hop_length: int, n_bins: int = 84):
    """
    Attempt to compute CQT on GPU using torch + torchaudio if available.
    Returns (C_db_numpy, used_gpu: bool). Falls back to CPU if anything fails.
    """
    try:
        import torch  # type: ignore
        import torchaudio  # type: ignore
        _log_once_env(torch, torchaudio)

        if not torch.cuda.is_available():
            if _ACCEL_LOG:
                print("[ACCEL] GPU path skipped: torch.cuda.is_available() is False.")
            return None, False

        device = torch.device("cuda:0")

        # Prepare input tensor on GPU
        y_t = torch.as_tensor(y_harmonic, dtype=torch.float32, device=device)

        # torchaudio CQT transform
        # Match librosa default fmin ~ C1 ~ 32.703 Hz
        fmin = 32.703195662574764
        # Some torchaudio versions do not provide transforms.CQT. Guard before constructing.
        if not hasattr(getattr(torchaudio, "transforms", object()), "CQT"):
            if _ACCEL_LOG:
                print("[ACCEL] torchaudio.transforms.CQT not available in this torchaudio version; skipping GPU CQT.")
            return None, False
        try:
            cqt = torchaudio.transforms.CQT(
                sample_rate=sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                pad_mode="reflect",
            )
        except Exception as e:
            if _ACCEL_LOG:
                print(f"[ACCEL] Failed to construct torchaudio CQT: {e}")
            return None, False

        # Ensure the module runs on GPU
        cqt = cqt.to(device)

        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        with torch.no_grad():
            # Input expected shape (..., time); 1D is acceptable
            C = cqt(y_t)  # shape: (freq, time)
            mag = torch.abs(C)
            # Compute dB scaling similar to librosa.amplitude_to_db with ref=np.max
            eps = torch.finfo(mag.dtype).tiny
            ref = torch.max(mag).clamp_min(eps)
            C_db = 20.0 * torch.log10((mag / ref).clamp_min(eps))
            C_db_np = C_db.detach().cpu().numpy()
            if _ACCEL_LOG:
                print(f"[ACCEL] CQT computed on GPU ({device}). Output shape: {C_db_np.shape}")
            return C_db_np, True
    except Exception as e:
        if _ACCEL_LOG:
            print(f"[ACCEL] GPU CQT failed, will fall back to CPU. Reason: {e}")
        return None, False



def compute_cqt_db(y_harmonic: np.ndarray, sr: int, bins_per_octave: int, hop_length: int = 64, n_bins: int = 84) -> np.ndarray:
    """
    Compute CQT and return decibel-scaled magnitude.
    - If torch+torchaudio with CUDA are available, perform CQT and dB scaling on the GPU.
    - Otherwise, fall back to librosa on CPU.

    General/other tasks (HPSS, I/O, plotting) remain on CPU.
    """
    # Try GPU path first, but only if explicitly requested or strictly required
    if _REQUIRE_GPU or _TRY_TORCH:
        C_db_gpu, used_gpu = _gpu_cqt_db(y_harmonic, sr, bins_per_octave, hop_length, n_bins=n_bins)
        if used_gpu and C_db_gpu is not None:
            return C_db_gpu
        if _REQUIRE_GPU:
            # If GPU was required but not used successfully, raise
            raise RuntimeError("DJAI_REQUIRE_GPU=1 is set but GPU acceleration is unavailable. Install CUDA-enabled torch+torchaudio and ensure CUDA drivers are present.")
    else:
        if _ACCEL_LOG:
            print("[ACCEL] Skipping GPU path (DJAI_TRY_TORCH not set). Using CPU.")

    # CPU fallback path

    global _warned_cpu_once
    if not _warned_cpu_once:
        print("[ACCEL] Warning: Using CPU fallback (librosa.cqt). Set DJAI_ACCEL_LOG=1 for details, or DJAI_REQUIRE_GPU=1 to enforce GPU usage.")
        _warned_cpu_once = True
    if _ACCEL_LOG:
        print("[ACCEL] Using CPU fallback (librosa.cqt).")
    # CPU fallback using librosa (original behavior)
    C = librosa.cqt(y_harmonic, sr=sr, bins_per_octave=bins_per_octave, hop_length=hop_length, n_bins=n_bins)
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    return C_db
