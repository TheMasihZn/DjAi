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

    def compute_cqt_db(self, y_harmonic: np.ndarray, sr: int, bins_per_octave: int, hop_length: int = 64, n_bins: int = 84) -> np.ndarray:
        """
        Public API: compute CQT and return decibel-scaled magnitude array (freq x time).
        Tries GPU path first if enabled, falls back to librosa CPU implementation.
        """
        # Try GPU if requested or required
        if self.require_gpu or self.try_torch:
            C_db_gpu, used_gpu = self._gpu_cqt_db(y_harmonic, sr, bins_per_octave, hop_length, n_bins)
            if used_gpu and C_db_gpu is not None:
                return C_db_gpu
            if self.require_gpu:
                raise RuntimeError(
                    "DJAI_REQUIRE_GPU=1 is set but GPU acceleration is unavailable. Install CUDA-enabled torch+torchaudio and ensure CUDA drivers are present."
                )
        else:
            if self.accel_log:
                logger.info("Skipping GPU path (DJAI_TRY_TORCH not set). Using CPU.")

        # CPU fallback using librosa
        if not self._warned_cpu_once:
            logger.info("Using CPU fallback (librosa.cqt). Set DJAI_ACCEL_LOG=1 for debug or DJAI_REQUIRE_GPU=1 to enforce GPU usage.")
            self._warned_cpu_once = True
        C = librosa.cqt(y_harmonic, sr=sr, bins_per_octave=bins_per_octave, hop_length=hop_length, n_bins=n_bins)
        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        return C_db

    def _gpu_cqt_db(
            self, y_harmonic: np.ndarray, sr: int, bins_per_octave: int, hop_length: int, n_bins: int
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Delegate to the unified GPU CQT implementation that tries torchaudio (stable and prototype)
        and then nnAudio as a fallback.
        """
        return _gpu_cqt_db(self, y_harmonic, sr, bins_per_octave, hop_length, n_bins)

def _gpu_cqt_db(
        self, y_harmonic: np.ndarray, sr: int, bins_per_octave: int, hop_length: int, n_bins: int
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Try to compute CQT on GPU. Returns (C_db_numpy, True) on success,
    (None, False) on failure.
    Order: torchaudio.transforms -> torchaudio.prototype.transforms -> nnAudio.
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
        fmin = 32.703195662574764  # consistent with librosa behavior

        cqt = None
        chosen_name = None

        # ---------- Try torchaudio stable transforms ----------
        ta_trans = getattr(torchaudio, "transforms", None)
        if ta_trans is not None:
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
            for name in ("CQT", "CQT1992v2", "CQT1992", "VQT"):
                if not hasattr(ta_trans, name):
                    continue
                cls = getattr(ta_trans, name)
                for kw in kwarg_variants:
                    try:
                        cqt_candidate = cls(**kw)
                        cqt = cqt_candidate
                        chosen_name = f"torchaudio.transforms.{name}"
                        if self.accel_log:
                            logger.info("Using %s with kwargs=%s for GPU CQT.", chosen_name, list(kw.keys()))
                        break
                    except Exception as e:
                        if self.accel_log:
                            logger.info("Failed to construct torchaudio.transforms.%s with kwargs %s: %s", name, list(kw.keys()), e)
                        cqt = None
                        continue
                if cqt is not None:
                    break

        # ---------- Try torchaudio.prototype.transforms ----------
        if cqt is None:
            ta_proto_t = getattr(getattr(torchaudio, "prototype", None), "transforms", None)
            if ta_proto_t is not None:
                for name in ("CQT", "CQT1992v2", "CQT1992", "VQT"):
                    if not hasattr(ta_proto_t, name):
                        continue
                    cls = getattr(ta_proto_t, name)
                    try:
                        cqt_candidate = cls(
                            sample_rate=sr,
                            hop_length=hop_length,
                            fmin=fmin,
                            n_bins=n_bins,
                            bins_per_octave=bins_per_octave,
                        )
                        cqt = cqt_candidate
                        chosen_name = f"torchaudio.prototype.transforms.{name}"
                        if self.accel_log:
                            logger.info("Using %s for GPU CQT.", chosen_name)
                        break
                    except Exception as e:
                        if self.accel_log:
                            logger.info("Failed to construct torchaudio.prototype.transforms.%s: %s", name, e)
                        cqt = None

        # ---------- Try nnAudio GPU backend ----------
        if cqt is None:
            try:
                from nnAudio.features import CQT1992v2 as NN_CQT  # type: ignore
                cqt = NN_CQT(
                    sr=sr,               # nnAudio uses sr
                    hop_length=hop_length,
                    fmin=fmin,
                    n_bins=n_bins,
                    bins_per_octave=bins_per_octave,
                    norm=None,           # closest to librosa amplitude behavior
                    trainable=False,     # fixed kernels
                ).to(device)
                chosen_name = "nnAudio.features.CQT1992v2"
                if self.accel_log:
                    logger.info("Using %s for GPU CQT.", chosen_name)
            except Exception as e:
                if self.accel_log:
                    logger.info("nnAudio CQT unavailable or failed to construct: %s", e)
                cqt = None

        if cqt is None:
            if self.accel_log:
                logger.info("No compatible GPU CQT/VQT transform available; skipping GPU CQT.")
            return None, False

        # Ensure the module runs on GPU if possible; otherwise note its device
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
            # Match input device to module
            x = y_t
            if str(module_device) != str(x.device):
                try:
                    x = y_t.to(module_device)
                except Exception:
                    x = y_t

            squeeze_back = False
            if x.dim() == 1:
                x = x.unsqueeze(0)  # (batch, time)
                squeeze_back = True

            C = cqt(x)

            if hasattr(C, "dim") and C.dim() == 3 and squeeze_back:
                C = C.squeeze(0)

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
                logger.info("CQT computed via %s, output shape: %s", str(chosen_name), str(C_db_np.shape))
            return C_db_np, True
    except Exception as e:
        if self.accel_log:
            logger.info("GPU CQT failed: %s", e)
        return None, False

# convenience factory
def default_cqt_computer() -> CQTComputer:
    return CQTComputer()
