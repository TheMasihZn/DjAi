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
        # Build a CQT-like transform compatible with multiple torchaudio versions (including 2.5.1)
        ta_trans = getattr(torchaudio, "transforms", None)
        if ta_trans is None:
            if _ACCEL_LOG:
                print("[ACCEL] torchaudio.transforms not present; skipping GPU CQT.")
            return None, False

        cqt = None
        base_kwargs = dict(
            sample_rate=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
        kwarg_variants = [
            {**base_kwargs, "pad_mode": "reflect"},
            {**base_kwargs},
            {**base_kwargs, "center": True},
            {**base_kwargs, "center": True, "trainable": False},
        ]
        chosen_name = None
        for name in ("CQT", "CQT1992v2", "CQT1992", "VQT"):
            if hasattr(ta_trans, name):
                cls = getattr(ta_trans, name)
                for kw in kwarg_variants:
                    try:
                        cqt_candidate = cls(**kw)
                        cqt = cqt_candidate
                        chosen_name = name
                        if _ACCEL_LOG:
                            print(f"[ACCEL] Using torchaudio.transforms.{name} with kwargs {list(kw.keys())}.")
                        break
                    except Exception as e:
                        if _ACCEL_LOG:
                            print(f"[ACCEL] Failed to construct torchaudio.transforms.{name} with {list(kw.keys())}: {e}")
                        cqt = None
                        continue
            if cqt is not None:
                break
        if cqt is None:
            if _ACCEL_LOG:
                print("[ACCEL] No compatible torchaudio CQT/VQT transform available; skipping GPU CQT.")
            return None, False

        # Ensure the module runs on GPU if possible; otherwise determine its device
        module_device = torch.device("cpu")
        try:
            cqt = cqt.to(device)
            module_device = device
        except Exception:
            try:
                first_param = next(cqt.parameters(), None)
                if first_param is not None:
                    module_device = first_param.device
            except Exception:
                module_device = torch.device("cpu")

        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        with torch.no_grad():
            # Align input to module device
            x = y_t
            if str(module_device) != str(x.device):
                try:
                    x = y_t.to(module_device)
                except Exception:
                    x = y_t
            # Ensure shape (..., time)
            squeeze_back = False
            if x.dim() == 1:
                x = x.unsqueeze(0)
                squeeze_back = True
            C = cqt(x)
            if C.dim() == 3 and squeeze_back:
                C = C.squeeze(0)
            # Move to GPU for dB math if possible
            try:
                C = C.to(device)
            except Exception:
                pass
            mag = torch.abs(C)
            eps = torch.finfo(mag.dtype).tiny
            ref = torch.max(mag).clamp_min(eps)
            C_db = 20.0 * torch.log10((mag / ref).clamp_min(eps))
            C_db_np = C_db.detach().cpu().numpy()
            if _ACCEL_LOG:
                print(f"[ACCEL] CQT computed via torchaudio ({chosen_name}). Output shape: {C_db_np.shape}")
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
