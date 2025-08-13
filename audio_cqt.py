"""
audio_cqt.py
Encapsulated, GPU-aware CQT computation.

Public API:
    CQTComputer(...)  -- create object that knows how to compute CQT (GPU if available)
    .compute_cqt_db(y_harmonic, sr, bins_per_octave, hop_length=64, n_bins=84) -> np.ndarray
"""

from typing import Optional, Tuple
import os
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Environment-controlled flags (same semantics as original)
_ACCEL_LOG = str(os.getenv("DJAI_ACCEL_LOG", "0")).lower() not in ("0", "false", "no", "")
_REQUIRE_GPU = str(os.getenv("DJAI_REQUIRE_GPU", "0")).lower() not in ("0", "false", "no", "")
_TRY_TORCH = str(os.getenv("DJAI_TRY_TORCH", "0")).lower() not in ("0", "false", "no", "")


class CQTComputer:
    """
    Compute constant-Q transform in dB, using GPU via torch+torchaudio when configured and available.
    Falls back to librosa.cqt on CPU otherwise.

    Parameters
    ----------
    try_torch : bool
        If True, attempt to use torch+torchaudio (requires DJAI_TRY_TORCH or explicit True here).
    require_gpu : bool
        If True and GPU path cannot be used, raise an error.
    accel_log : bool
        If True, log acceleration environment details.
    """

    def __init__(self, try_torch: bool = _TRY_TORCH, require_gpu: bool = _REQUIRE_GPU, accel_log: bool = _ACCEL_LOG):
        self.try_torch = try_torch
        self.require_gpu = require_gpu
        self.accel_log = accel_log
        self._logged_env_once = False
        self._warned_cpu_once = False

    def _log_env_once(self, torch_mod, torchaudio_mod):
        if not self.accel_log or self._logged_env_once:
            return
        try:
            logger.info("torch version: %s", getattr(torch_mod, "__version__", "unknown"))
            logger.info("torchaudio version: %s", getattr(torchaudio_mod, "__version__", "unknown"))
            logger.info("CUDA available: %s", getattr(torch_mod.cuda, "is_available", lambda: False)())
            logger.info("torch.version.cuda: %s", getattr(torch_mod.version, "cuda", None))
            if torch_mod.cuda.is_available():
                try:
                    n = torch_mod.cuda.device_count()
                    logger.info("CUDA device count: %d", n)
                    for i in range(n):
                        try:
                            logger.info("Device %d: %s", i, torch_mod.cuda.get_device_name(i))
                        except Exception:
                            logger.info("Device %d: (name unknown)", i)
                except Exception as e:
                    logger.warning("Failed to enumerate CUDA devices: %s", e)
        finally:
            self._logged_env_once = True

    def _gpu_cqt_db(
            self, y_harmonic: np.ndarray, sr: int, bins_per_octave: int, hop_length: int, n_bins: int
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Try to compute CQT on GPU. Returns (C_db_numpy, True) on success,
        (None, False) on failure.
        """
        try:
            import torch  # type: ignore
            import torchaudio  # type: ignore
        except Exception as e:
            if self.accel_log:
                logger.info("Torch/torchaudio import failed: %s", e)
            return None, False

        self._log_env_once(torch, torchaudio)

        try:
            if not torch.cuda.is_available():
                if self.accel_log:
                    logger.info("torch.cuda.is_available() is False, skipping GPU path.")
                return None, False
        except Exception as e:
            if self.accel_log:
                logger.info("Error checking CUDA availability: %s", e)
            return None, False

        device = torch.device("cuda:0")
        try:
            # prepare tensor on GPU
            y_t = torch.as_tensor(y_harmonic, dtype=torch.float32, device=device)
            fmin = 32.703195662574764  # consistent with librosa behavior in original code
            # Build a CQT-like transform compatible with multiple torchaudio versions (2.5.1+ included)
            ta_trans = getattr(torchaudio, "transforms", None)
            if ta_trans is None:
                if self.accel_log:
                    logger.info("torchaudio.transforms not present; skipping GPU CQT.")
                return None, False

            cqt = None
            base_kwargs = dict(
                sample_rate=sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
            )
            # Try multiple kwarg variants to satisfy differing signatures across versions
            kwarg_variants = [
                {**base_kwargs, "pad_mode": "reflect"},
                {**base_kwargs},
                {**base_kwargs, "center": True},
                {**base_kwargs, "center": True, "trainable": False},
            ]
            chosen_name = None
            # Try known class names across torchaudio versions
            for name in ("CQT", "CQT1992v2", "CQT1992", "VQT"):
                if not hasattr(ta_trans, name):
                    continue
                cls = getattr(ta_trans, name)
                for kw in kwarg_variants:
                    try:
                        cqt_candidate = cls(**kw)
                        cqt = cqt_candidate
                        chosen_name = name
                        if self.accel_log:
                            logger.info("Using torchaudio.transforms.%s with kwargs=%s for GPU CQT.", name, list(kw.keys()))
                        break
                    except Exception as e:
                        if self.accel_log:
                            logger.info("Failed to construct torchaudio.transforms.%s with kwargs %s: %s", name, list(kw.keys()), e)
                        cqt = None
                        continue
                if cqt is not None:
                    break

            if cqt is None:
                if self.accel_log:
                    logger.info("No compatible torchaudio CQT/VQT transform available; skipping GPU CQT.")
                return None, False

            # Ensure the module runs on GPU if possible; otherwise note its device
            module_device = torch.device("cpu")
            try:
                cqt = cqt.to(device)
                module_device = device
            except Exception:
                # Some versions might not support .to on the transform; infer device from parameters if any
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
                # Match input device to module
                x = y_t
                if str(module_device) != str(x.device):
                    try:
                        x = y_t.to(module_device)
                    except Exception:
                        x = y_t
                # Ensure expected shape (..., time). Some versions prefer (batch, time)
                squeeze_back = False
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                    squeeze_back = True
                C = cqt(x)
                # Handle possible (batch, freq, time) output
                if C.dim() == 3 and squeeze_back:
                    C = C.squeeze(0)
                # Move to GPU (if available) for dB scaling to keep math accelerated
                try:
                    C = C.to(device)
                except Exception:
                    pass
                mag = torch.abs(C)
                eps = torch.finfo(mag.dtype).tiny
                ref = torch.max(mag).clamp_min(eps)
                C_db = 20.0 * torch.log10((mag / ref).clamp_min(eps))
                C_db_np = C_db.detach().cpu().numpy()
                if self.accel_log:
                    logger.info("CQT computed via torchaudio (%s), output shape: %s", str(chosen_name), str(C_db_np.shape))
                return C_db_np, True
        except Exception as e:
            if self.accel_log:
                logger.info("GPU CQT failed: %s", e)
            return None, False

    def compute_cqt_db(self, y_harmonic: np.ndarray, sr: int, bins_per_octave: int, hop_length: int = 64, n_bins: int = 84) -> np.ndarray:
        """
        Compute and return CQT in decibel scale (numpy array).
        Attempts GPU path first if configured; falls back to librosa.cqt otherwise.
        """
        if self.require_gpu or self.try_torch:
            cqt_gpu, used_gpu = self._gpu_cqt_db(y_harmonic, sr, bins_per_octave, hop_length, n_bins)
            if used_gpu and cqt_gpu is not None:
                return cqt_gpu
            if self.require_gpu:
                raise RuntimeError("GPU required but unavailable or failed. Set DJAI_REQUIRE_GPU=0 to allow CPU fallback.")

        if self.accel_log:
            logger.info("Using CPU fallback (librosa.cqt).")

        if not self._warned_cpu_once:
            logger.warning("CPU fallback: using librosa.cqt. To enable GPU try setting DJAI_TRY_TORCH=1.")
            self._warned_cpu_once = True

        C = librosa.cqt(y_harmonic, sr=sr, bins_per_octave=bins_per_octave, hop_length=hop_length, n_bins=n_bins)
        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        return C_db


# convenience factory
def default_cqt_computer() -> CQTComputer:
    return CQTComputer()
