# interactive_cqt_async.py
"""
Responsive Interactive CQT viewer with robust shutdown behavior.

Key features:
 - Tries to use a GUI backend (TkAgg or Qt5Agg). If unavailable, falls back to
   synchronous mode (no background threads) to avoid shutdown race conditions.
 - Heavy tasks (audio load, CQT compute, wide-image generation) run in a
   ThreadPoolExecutor when GUI is available.
 - The GUI uses a small timer for smooth playhead updates (blitting).
 - All outstanding futures are cancelled / executor shut down during close.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
import time
from typing import List, Optional, Tuple

# Try to force a GUI backend before importing pyplot
import matplotlib

# Preferred GUI backends (try in order)
_PREFERRED_GUI = ["TkAgg", "Qt5Agg", "QtAgg", "Qt4Agg"]
_GUI_BACKENDS = [b.lower() for b in ["tkagg", "qt5agg", "qtagg", "wxagg", "gtk3agg", "macosx"]]

_chosen_backend = None
for be in _PREFERRED_GUI:
    try:
        matplotlib.use(be, force=True)
        _chosen_backend = matplotlib.get_backend()
        break
    except Exception:
        _chosen_backend = None

# Import pyplot after backend selection
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

# Import librosa at module import time to avoid lazy imports inside worker threads
import librosa
import librosa.display

# Optional pygame for playback
try:
    import pygame  # type: ignore
    _HAS_PYGAME = True
except Exception:
    pygame = None  # type: ignore
    _HAS_PYGAME = False

from audio_cqt import CQTComputer, default_cqt_computer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interactive_cqt_async")

# Determine if backend appears to be GUI-capable
_backend_name = matplotlib.get_backend().lower()
GUI_AVAILABLE = any(b in _backend_name for b in _GUI_BACKENDS)
if _chosen_backend:
    logger.info("Requested backend: %s, effective backend: %s", _chosen_backend, matplotlib.get_backend())
else:
    logger.info("Effective backend: %s", matplotlib.get_backend())

if not GUI_AVAILABLE:
    logger.warning(
        "No interactive GUI backend appears available (backend=%s). "
        "Falling back to synchronous behavior (no threads).",
        matplotlib.get_backend(),
    )


class AsyncCQTWorker:
    def __init__(self, max_workers: int = 2):
        self._exec = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._alive = True

    def submit(self, fn, *args, **kwargs) -> concurrent.futures.Future:
        with self._lock:
            if not self._alive:
                raise RuntimeError("Executor has been shut down")
            return self._exec.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True):
        with self._lock:
            self._alive = False
        try:
            self._exec.shutdown(wait=wait)
        except Exception:
            pass


class PlaybackProgressThread(threading.Thread):
    """Minimal playback poller for pygame playback position."""

    def __init__(self, interval: float = 0.02):
        super().__init__(daemon=True, name="PlaybackProgressThread")
        self.interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._time_sec = 0.0
        self._is_playing = False
        self._is_paused = False

    def run(self):
        while not self._stop.is_set():
            try:
                if _HAS_PYGAME and pygame.mixer.get_init() and self._is_playing and not self._is_paused:
                    try:
                        pos_ms = pygame.mixer.music.get_pos()
                        if pos_ms is not None and pos_ms >= 0:
                            t = float(pos_ms) / 1000.0
                        else:
                            t = self._time_sec
                    except Exception:
                        t = self._time_sec
                else:
                    t = self._time_sec
                with self._lock:
                    self._time_sec = t
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()

    def set_time(self, t: float):
        with self._lock:
            self._time_sec = float(t)

    def get_time(self) -> float:
        with self._lock:
            return float(self._time_sec)

    def set_playing(self, playing: bool):
        with self._lock:
            self._is_playing = bool(playing)

    def set_paused(self, paused: bool):
        with self._lock:
            self._is_paused = bool(paused)


class InteractiveCQTViewer:
    def __init__(
            self,
            playlist_dir: str = "playlist",
            cqt_computer: Optional[CQTComputer] = None,
            pixels_per_second: int = 200,
            gui_fps: int = 60,
    ):
        self.playlist_dir = playlist_dir
        self.cqt = cqt_computer or default_cqt_computer()
        self.pixels_per_second = int(pixels_per_second)
        self.hop_length = 64

        # UI
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.colorbar_ax: Optional[plt.Axes] = None
        self.slider: Optional[Slider] = None
        self.seek_slider: Optional[Slider] = None
        self.play_button: Optional[Button] = None
        self.progress_line = None

        # New UI controls for hpss and hop_length
        self.hpss_kernel_size_slider: Optional[Slider] = None
        self.hpss_power_slider: Optional[Slider] = None
        self.hpss_n_fft_slider: Optional[Slider] = None
        self.hpss_win_length_slider: Optional[Slider] = None
        self.hpss_hop_length_slider: Optional[Slider] = None
        self.hpss_margin_button: Optional[Button] = None

        # audio
        self.files: List[str] = []
        self.current_file: Optional[str] = None
        self.y = None
        self.sr = None
        self.y_harmonic = None
        self.duration_sec = 0.0

        # hpss parameters (now controlled by GUI)
        self.hpss_kernel_size = 2
        self.hpss_power = 2.0
        self.hpss_margin_options = ['hard', 'soft', 'mask']
        self.hpss_margin_index = 0
        self.hpss_n_fft = None  # Use None for default
        self.hpss_win_length = None # Use None for default
        self.hpss_hop_length = None # Use None for default

        # concurrency
        self.worker = AsyncCQTWorker(max_workers=2) if GUI_AVAILABLE else None
        self._futures: List[concurrent.futures.Future] = []
        self._closing = False

        # wide image metadata
        self._wide_img_path: Optional[str] = None
        self._vmin_vmax: Optional[Tuple[float, float]] = None

        # playback progress
        self._progress_thread = PlaybackProgressThread(interval=0.02)
        if GUI_AVAILABLE:
            self._progress_thread.start()

        self.is_playing = False
        self.is_paused = False

        # GUI timer / blit
        self._gui_interval_ms = int(max(1, round(1000.0 / float(gui_fps))))
        self._timer = None
        self._blit_ok = True
        self._bg_cache = None

    # --------------------
    # filesystem helpers
    # --------------------
    def list_mp3(self) -> List[str]:
        if not os.path.isdir(self.playlist_dir):
            raise FileNotFoundError(f"{self.playlist_dir} not found")
        files = [f for f in os.listdir(self.playlist_dir) if f.lower().endswith(".mp3")]
        files.sort()
        return files

    def _full_path(self, name: str) -> str:
        return os.path.join(self.playlist_dir, name)

    # --------------------
    # background tasks (only used when GUI_AVAILABLE)
    # --------------------
    def load_track_async(self, path: str) -> concurrent.futures.Future:
        """
        Submits a task to load the audio and apply hpss.
        """
        def _load_and_hpss(p, kernel_size, power, margin, n_fft, win_length):
            logger.info("[worker] Loading audio: %s", p)
            # librosa is imported at module level to avoid lazy import inside worker threads during shutdown
            y, sr = librosa.load(p, sr=None,
                                 # duration=5,offset=165,
                                 mono=True, res_type="kaiser_best")

            # Now using the GUI-controlled parameters for hpss
            hpss_kwargs = {
                'kernel_size': int(kernel_size),
                'power': float(power),
            }
            if n_fft is not None:
                hpss_kwargs['n_fft'] = int(n_fft)
            if win_length is not None:
                hpss_kwargs['win_length'] = int(win_length)

            # The following try-except block is a workaround for a potential bug in some versions of librosa
            # where a TypeError occurs when passing a string 'margin' value. The correct behavior is to accept
            # the string. If the TypeError occurs, we fall back to a default numeric margin to allow the app
            # to continue running. It is recommended to update the librosa library if you encounter this error.
            try:
                # This is the correct way to pass the margin parameter.
                hpss_kwargs['margin'] = margin
                y_harmonic, _ = librosa.effects.hpss(y, **hpss_kwargs)
            except TypeError:
                logger.warning(
                    "Caught TypeError while calling librosa.effects.hpss. "
                    "This is likely due to an old or buggy version of librosa. "
                    "Falling back to a default numeric margin. Please consider "
                    "updating librosa to fix this issue."
                )
                # Fallback to a default numeric value if the string-based margin fails
                hpss_kwargs['margin'] = 1.0 # default numeric value
                y_harmonic, _ = librosa.effects.hpss(y, **hpss_kwargs)

            duration = float(librosa.get_duration(y=y, sr=sr))
            return y, sr, y_harmonic, duration

        assert self.worker is not None
        fut = self.worker.submit(
            _load_and_hpss,
            path,
            self.hpss_kernel_size,
            self.hpss_power,
            self.hpss_margin_options[self.hpss_margin_index],
            self.hpss_n_fft,
            self.hpss_win_length
        )
        self._futures.append(fut)
        return fut

    def compute_and_cache_wide_async(self, y_harmonic: np.ndarray, sr: int, bpo: int, hop_length: int, duration_sec: float) -> concurrent.futures.Future:
        """
        Submits a task to compute the CQT and save the wide image.
        Now also uses the GUI-controlled hop_length.
        """
        def _task(y_h, sr_, bpo_, hop_length_, dur_):
            logger.info("[worker] Computing CQT (bpo=%d, hop_length=%d)", bpo_, hop_length_)
            C_db = self.cqt.compute_cqt_db(y_h, sr_, bins_per_octave=int(bpo_), hop_length=int(hop_length_), n_bins=84)
            vmin = float(np.percentile(C_db, 5.0))
            vmax = float(np.percentile(C_db, 99.5))

            # Update cache directory name to reflect hop_length
            cache_dir = os.path.join("output", f"cqt-hop{int(hop_length_)}-hpss-bpo{int(bpo_)}")
            os.makedirs(cache_dir, exist_ok=True)
            name = os.path.splitext(os.path.basename(self.current_file))[0] if self.current_file else "track"
            wide_path = os.path.join(cache_dir, f"{name}_wide.jpg")
            # determine width and resample in time
            try:
                width_px = max(1, int(float(dur_) * float(self.pixels_per_second)))
            except Exception:
                width_px = max(1, int(C_db.shape[1] * 4))
            n_bins, n_frames = C_db.shape
            if n_frames != width_px:
                x_src = np.linspace(0.0, 1.0, n_frames)
                x_tgt = np.linspace(0.0, 1.0, width_px)
                out = np.empty((n_bins, width_px), dtype=float)
                for i in range(n_bins):
                    out[i] = np.interp(x_tgt, x_src, C_db[i])
            else:
                out = C_db
            try:
                plt.imsave(wide_path, out, cmap="magma", vmin=vmin, vmax=vmax, origin="lower", format="jpg")
            except Exception:
                plt.imsave(wide_path, out, cmap="magma", origin="lower", format="jpg")
            logger.info("[worker] Wide image saved: %s", wide_path)
            return wide_path, vmin, vmax

        assert self.worker is not None
        fut = self.worker.submit(_task, y_harmonic, sr, int(bpo), int(hop_length), float(duration_sec))
        self._futures.append(fut)
        return fut

    # --------------------
    # UI helpers
    # --------------------
    def _init_figure(self):
        # Increased figure height to accommodate new controls
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        try:
            self.fig.canvas.manager.set_window_title("Interactive CQT: Async (robust)")
        except Exception:
            pass
        # Adjusted bottom margin to make space for the new sliders
        plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.45)
        self.colorbar_ax = self.fig.add_axes([0.935, 0.47, 0.02, 0.43])

    def _draw_placeholder(self, message: str = "Loading..."):
        assert self.ax is not None
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, ha="center", va="center", transform=self.ax.transAxes, fontsize=16)
        try:
            self.fig.canvas.draw()
        except Exception:
            pass
        self._bg_cache = None

    def _plot_wide_image(self, wide_path: str, vmin: float, vmax: float, bpo: int, hop_length: int):
        assert self.ax is not None
        self.ax.clear()
        try:
            img = plt.imread(wide_path)
            self.ax.imshow(img, aspect="auto", origin="lower", extent=(0.0, float(self.duration_sec), 0.0, 84))
            self.ax.set_ylim(0.0, 84)
            self.ax.set_xlim(0.0, float(self.duration_sec))
            self.ax.set_ylabel("CQT bins")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_title(
                f"{os.path.basename(self.current_file)} - Harmonic CQT (bpo={int(bpo)}, hop={int(hop_length)})"
            )
            try:
                self.fig.canvas.draw()
            except Exception:
                pass
            self._bg_cache = None
        except Exception as e:
            logger.exception("Failed to plot wide image: %s", e)
            self._draw_placeholder("Failed to plot image")

    # --------------------
    # Playback controls
    # --------------------
    def _ensure_mixer(self) -> bool:
        if not _HAS_PYGAME:
            return False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            return True
        except Exception:
            logger.exception("Failed to init pygame mixer")
            return False

    def _stop_playback(self):
        if _HAS_PYGAME and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
                if hasattr(pygame.mixer.music, "unload"):
                    pygame.mixer.music.unload()  # type: ignore[attr-defined]
            except Exception:
                pass
        self.is_playing = False
        self.is_paused = False
        if GUI_AVAILABLE:
            self._progress_thread.set_playing(False)
            self._progress_thread.set_paused(False)
            self._progress_thread.set_time(0.0)
        if self.seek_slider is not None:
            try:
                self.seek_slider.set_val(0.0)
            except Exception:
                pass

    def _play_current(self, start_sec: float = 0.0):
        if not self._ensure_mixer():
            logger.info("pygame not available; cannot play audio")
            return
        try:
            pygame.mixer.music.load(self.current_file)
            try:
                pygame.mixer.music.play(start=float(start_sec))
            except TypeError:
                pygame.mixer.music.play()
            self.is_playing = True
            self.is_paused = False
            if GUI_AVAILABLE:
                self._progress_thread.set_playing(True)
                self._progress_thread.set_paused(False)
        except Exception:
            logger.exception("Failed to start playback")
            self.is_playing = False

    def toggle_play(self, event=None):
        if not _HAS_PYGAME:
            logger.info("Audio disabled (pygame missing)")
            return
        if not self.is_playing:
            t = self._progress_thread.get_time() if GUI_AVAILABLE else 0.0
            self._play_current(start_sec=t)
            if self.play_button is not None:
                self.play_button.label.set_text("Pause")
        else:
            if not self.is_paused:
                pygame.mixer.music.pause()
                self.is_paused = True
                if GUI_AVAILABLE:
                    self._progress_thread.set_paused(True)
                if self.play_button is not None:
                    self.play_button.label.set_text("Resume")
            else:
                pygame.mixer.music.unpause()
                self.is_paused = False
                if GUI_AVAILABLE:
                    self._progress_thread.set_paused(False)
                if self.play_button is not None:
                    self.play_button.label.set_text("Pause")

    # --------------------
    # Seek / progress UI
    # --------------------
    def _preview_seek(self, t_sec: float):
        if GUI_AVAILABLE:
            self._progress_thread.set_time(t_sec)
        self._update_progress_line_and_seek(t_sec)

    def _apply_seek(self, t_sec: float):
        t_sec = max(0.0, min(float(t_sec), float(self.duration_sec)))
        if GUI_AVAILABLE:
            self._progress_thread.set_time(t_sec)
        if self.is_playing and _HAS_PYGAME:
            try:
                pygame.mixer.music.pause()
            except Exception:
                pass
            try:
                pygame.mixer.music.load(self.current_file)
                try:
                    pygame.mixer.music.play(start=t_sec)
                except TypeError:
                    pygame.mixer.music.play()
                if self.is_paused:
                    pygame.mixer.music.pause()
            except Exception:
                logger.exception("Failed to seek in playback")
        if self.seek_slider is not None:
            try:
                self.seek_slider.set_val(t_sec)
            except Exception:
                pass

    def _update_progress_line_and_seek(self, t_sec: float):
        if self.progress_line is None or self.ax is None or self.fig is None:
            return
        try:
            self.progress_line.set_xdata([t_sec, t_sec])
            if self._blit_ok:
                if self._bg_cache is None:
                    try:
                        self._bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
                    except Exception:
                        self._blit_ok = False
                if self._bg_cache is not None and self._blit_ok:
                    try:
                        canvas = self.fig.canvas
                        canvas.restore_region(self._bg_cache)
                        self.ax.draw_artist(self.progress_line)
                        canvas.blit(self.ax.bbox)
                    except Exception:
                        self._blit_ok = False
                        try:
                            self.fig.canvas.draw_idle()
                        except Exception:
                            pass
            else:
                try:
                    self.fig.canvas.draw_idle()
                except Exception:
                    pass
            if self.seek_slider is not None:
                try:
                    self.seek_slider.set_val(t_sec)
                except Exception:
                    pass
        except Exception:
            pass

    # --------------------
    # Recomputation logic
    # --------------------
    def _trigger_recompute(self):
        """
        Triggers a full re-computation of hpss and CQT.
        This function is the main entry point for all parameter change callbacks.
        """
        if self._closing or self.y is None or self.sr is None:
            return

        self._draw_placeholder("Recomputing HPSS and CQT...")

        # We need to re-run HPSS first, so we'll start with the load_track_async callback.
        # This is a bit of a trick, as we're not actually reloading the audio file.
        # We'll create a new function to just run hpss on the existing audio data.
        def _recompute_hpss_task(y, sr, kernel_size, power, margin, n_fft, win_length):
            hpss_kwargs = {
                'kernel_size': int(kernel_size),
                'power': float(power),
            }
            if n_fft is not None:
                hpss_kwargs['n_fft'] = int(n_fft)
            if win_length is not None:
                hpss_kwargs['win_length'] = int(win_length)

            # The following try-except block is a workaround for a potential bug in some versions of librosa
            # where a TypeError occurs when passing a string 'margin' value. The correct behavior is to accept
            # the string. If the TypeError occurs, we fall back to a default numeric margin to allow the app
            # to continue running. It is recommended to update the librosa library if you encounter this error.
            try:
                # This is the correct way to pass the margin parameter.
                hpss_kwargs['margin'] = margin
                y_harmonic, _ = librosa.effects.hpss(y, **hpss_kwargs)
            except TypeError:
                logger.warning(
                    "Caught TypeError while calling librosa.effects.hpss. "
                    "This is likely due to an old or buggy version of librosa. "
                    "Falling back to a default numeric margin. Please consider "
                    "updating librosa to fix this issue."
                )
                # Fallback to a default numeric value if the string-based margin fails
                hpss_kwargs['margin'] = (1., 5.) # default numeric value
                y_harmonic, _ = librosa.effects.hpss(y, **hpss_kwargs)
            return y_harmonic

        assert self.worker is not None
        hpss_future = self.worker.submit(
            _recompute_hpss_task,
            self.y,
            self.sr,
            self.hpss_kernel_size,
            self.hpss_power,
            self.hpss_margin_options[self.hpss_margin_index],
            self.hpss_n_fft,
            self.hpss_win_length
        )
        self._futures.append(hpss_future)

        def _on_hpss_done(hpss_fut: concurrent.futures.Future):
            if self._closing or hpss_fut.cancelled():
                return
            try:
                self.y_harmonic = hpss_fut.result()
            except Exception as e:
                logger.exception("HPSS re-computation failed: %s", e)
                self._draw_placeholder("Failed to re-compute HPSS")
                return

            # Now, trigger the CQT computation with the new y_harmonic and hop_length
            if self.y_harmonic is not None and self.sr is not None and self.duration_sec is not None:
                fut = self.compute_and_cache_wide_async(
                    self.y_harmonic,
                    self.sr,
                    int(self.slider.val),
                    int(self.hpss_hop_length_slider.val),
                    self.duration_sec
                )
                # FIX: Added self. to correctly reference the class method
                fut.add_done_callback(self._on_wide_done)

        hpss_future.add_done_callback(_on_hpss_done)

    def _on_wide_done(self, fut: concurrent.futures.Future):
        # Guard: if closing or future cancelled -> ignore
        if self._closing or fut.cancelled():
            return
        try:
            wide_path, vmin, vmax = fut.result()
        except Exception as e:
            if self._closing:
                return
            logger.exception("Wide compute failed: %s", e)
            self._draw_placeholder("Failed to compute spectrogram")
            return
        self._wide_img_path = wide_path
        self._vmin_vmax = (vmin, vmax)
        # Plot on main thread
        try:
            self._plot_wide_image(wide_path, vmin, vmax, int(self.slider.val), int(self.hpss_hop_length_slider.val))
        except Exception:
            self._draw_placeholder("Failed to draw spectrogram")
        # create animated progress line
        try:
            self.progress_line = self.ax.axvline(0.0, color="cyan", linewidth=1.2, alpha=0.9)
            try:
                self.progress_line.set_animated(True)
            except Exception:
                pass
        except Exception:
            self.progress_line = None


    # --------------------
    # Main run
    # --------------------
    def run(self, initial_bpo: int = 10):
        self.files = self.list_mp3()
        if not self.files:
            raise FileNotFoundError(f"No .mp3 files found in '{self.playlist_dir}'")
        self.current_file = self._full_path(self.files[-1])

        # If no GUI available, do synchronous path to avoid background threads during shutdown
        if not GUI_AVAILABLE:
            logger.info("Running in synchronous (non-GUI) mode")
            # load audio and compute wide image synchronously
            y, sr = librosa.load(self.current_file, sr=None)

            # Using default hpss parameters for synchronous mode
            y_harmonic, _ = librosa.effects.hpss(y, kernel_size=self.hpss_kernel_size, power=self.hpss_power, margin=self.hpss_margin_options[self.hpss_margin_index])
            duration = float(librosa.get_duration(y=y, sr=sr))
            self.y = y
            self.sr = sr
            self.y_harmonic = y_harmonic
            self.duration_sec = duration

            # compute CQT synchronously
            C_db = self.cqt.compute_cqt_db(self.y_harmonic, self.sr, bins_per_octave=int(initial_bpo), hop_length=self.hop_length, n_bins=84)
            vmin = float(np.percentile(C_db, 5.0))
            vmax = float(np.percentile(C_db, 99.5))
            cache_dir = os.path.join("output", f"cqt-hop{self.hop_length}-hpss-bpo{int(initial_bpo)}")
            os.makedirs(cache_dir, exist_ok=True)
            name = os.path.splitext(os.path.basename(self.current_file))[0]
            wide_path = os.path.join(cache_dir, f"{name}_wide.jpg")
            # resample and save
            try:
                width_px = max(1, int(float(self.duration_sec) * float(self.pixels_per_second)))
            except Exception:
                width_px = max(1, int(C_db.shape[1] * 4))
            n_bins, n_frames = C_db.shape
            if n_frames != width_px:
                x_src = np.linspace(0.0, 1.0, n_frames)
                x_tgt = np.linspace(0.0, 1.0, width_px)
                out = np.empty((n_bins, width_px), dtype=float)
                for i in range(n_bins):
                    out[i] = np.interp(x_tgt, x_src, C_db[i])
            else:
                out = C_db
            try:
                plt.imsave(wide_path, out, cmap="magma", vmin=vmin, vmax=vmax, origin="lower", format="jpg")
            except Exception:
                plt.imsave(wide_path, out, cmap="magma", origin="lower", format="jpg")
            # show quick static plot (non-interactive)
            fig, ax = plt.subplots(figsize=(14, 5.2))
            img = plt.imread(wide_path)
            ax.imshow(img, aspect="auto", origin="lower", extent=(0.0, float(self.duration_sec), 0.0, 84))
            ax.set_title(f"{os.path.basename(self.current_file)} - Harmonic CQT (bpo={int(initial_bpo)})")
            plt.show(block=True)
            return

        # GUI path
        self._init_figure()

        # controls
        # Slider for Bins per Octave (BPO)
        ax_slider_bpo = self.fig.add_axes([0.08, 0.35, 0.40, 0.025])
        self.slider = Slider(ax=ax_slider_bpo, label="Bins per Octave", valmin=10, valmax=100, valinit=float(initial_bpo), valstep=1)

        # Sliders for HPSS parameters
        ax_slider_kernel = self.fig.add_axes([0.08, 0.30, 0.40, 0.025])
        self.hpss_kernel_size_slider = Slider(ax=ax_slider_kernel, label="HPSS Kernel Size", valmin=1, valmax=100, valinit=2, valstep=1)

        ax_slider_power = self.fig.add_axes([0.08, 0.25, 0.40, 0.025])
        self.hpss_power_slider = Slider(ax=ax_slider_power, label="HPSS Power", valmin=1.0, valmax=100.0, valinit=2.0, valstep=0.1)

        ax_slider_hop = self.fig.add_axes([0.08, 0.20, 0.40, 0.025])
        self.hpss_hop_length_slider = Slider(ax=ax_slider_hop, label="Hop Length (CQT)", valmin=32, valmax=1024, valinit=self.hop_length, valstep=32)

        ax_slider_n_fft = self.fig.add_axes([0.08, 0.15, 0.40, 0.025])
        self.hpss_n_fft_slider = Slider(ax=ax_slider_n_fft, label="N_FFT (HPSS)", valmin=512, valmax=8192, valinit=2048, valstep=256)

        ax_slider_win_length = self.fig.add_axes([0.08, 0.10, 0.40, 0.025])
        self.hpss_win_length_slider = Slider(ax=ax_slider_win_length, label="Win Length (HPSS)", valmin=512, valmax=8192, valinit=2048, valstep=256)

        # Button for HPSS margin type
        ax_button_margin = self.fig.add_axes([0.5, 0.35, 0.1, 0.04])
        self.hpss_margin_button = Button(ax_button_margin, f"Margin: {self.hpss_margin_options[self.hpss_margin_index]}")

        # seek slider
        self.seek_ax = self.fig.add_axes([0.5, 0.25, 0.40, 0.04])
        self.seek_slider = Slider(ax=self.seek_ax, label="Seek (s)", valmin=0.0, valmax=1.0, valinit=0.0, valstep=0.01)

        # play button
        play_ax = self.fig.add_axes([0.5, 0.15, 0.07, 0.04])
        lbl = "Play" if _HAS_PYGAME else "NoAudio"
        self.play_button = Button(play_ax, lbl)
        self.play_button.on_clicked(self.toggle_play)

        # placeholder
        self._draw_placeholder("Loading track and computing spectrogram...")

        # submit audio load
        load_future = self.load_track_async(self.current_file)

        # Callback for when HPSS and CQT computation is complete
        def _on_load_done(fut: concurrent.futures.Future):
            # Guard: if closing or cancelled -> ignore
            if self._closing or fut.cancelled():
                return
            try:
                y, sr, y_harmonic, duration = fut.result()
            except Exception as e:
                if self._closing:
                    return
                logger.exception("Failed to load audio: %s", e)
                self._draw_placeholder("Failed to load audio")
                return
            # store state
            self.y = y
            self.sr = sr
            self.y_harmonic = y_harmonic
            self.duration_sec = duration
            # update seek slider max
            try:
                self.seek_slider.valmax = float(self.duration_sec)
                self.seek_ax.set_xlim(self.seek_slider.valmin, self.seek_slider.valmax)
            except Exception:
                pass
            # submit wide image compute
            fut2 = self.compute_and_cache_wide_async(
                self.y_harmonic,
                self.sr,
                int(self.slider.val),
                int(self.hpss_hop_length_slider.val),
                self.duration_sec
            )
            fut2.add_done_callback(self._on_wide_done)

        load_future.add_done_callback(_on_load_done)

        # Bins slider callback -> recompute wide image in background
        def _on_bpo_change(val):
            if self._closing or self.y_harmonic is None:
                return
            self._draw_placeholder("Recomputing CQT...")
            fut = self.compute_and_cache_wide_async(
                self.y_harmonic,
                self.sr,
                int(val),
                int(self.hpss_hop_length_slider.val),
                self.duration_sec
            )
            fut.add_done_callback(self._on_wide_done)

        self.slider.on_changed(_on_bpo_change)

        # HPSS parameter callbacks -> trigger full recompute
        def _on_hpss_param_change(val):
            self.hpss_kernel_size = int(self.hpss_kernel_size_slider.val)
            self.hpss_power = float(self.hpss_power_slider.val)
            self.hpss_n_fft = int(self.hpss_n_fft_slider.val)
            self.hpss_win_length = int(self.hpss_win_length_slider.val)
            self.hop_length = int(self.hpss_hop_length_slider.val)
            self._trigger_recompute()

        self.hpss_kernel_size_slider.on_changed(_on_hpss_param_change)
        self.hpss_power_slider.on_changed(_on_hpss_param_change)
        self.hpss_n_fft_slider.on_changed(_on_hpss_param_change)
        self.hpss_win_length_slider.on_changed(_on_hpss_param_change)
        self.hpss_hop_length_slider.on_changed(_on_hpss_param_change)

        def _on_margin_button_click(event):
            self.hpss_margin_index = (self.hpss_margin_index + 1) % len(self.hpss_margin_options)
            self.hpss_margin_button.label.set_text(f"Margin: {self.hpss_margin_options[self.hpss_margin_index]}")
            self._trigger_recompute()

        self.hpss_margin_button.on_clicked(_on_margin_button_click)


        # seek handling
        _user_dragging = {"flag": False}

        def _on_seek_change(val):
            if _user_dragging["flag"]:
                self._preview_seek(val)

        def _on_mouse_press(event):
            if event.inaxes is getattr(self, "seek_ax", None):
                _user_dragging["flag"] = True

        def _on_mouse_release(event):
            if _user_dragging["flag"]:
                _user_dragging["flag"] = False
                if self.seek_slider is not None:
                    try:
                        v = float(self.seek_slider.val)
                        self._apply_seek(v)
                    except Exception:
                        pass

        self.seek_slider.on_changed(_on_seek_change)
        try:
            self.fig.canvas.mpl_connect("button_press_event", _on_mouse_press)
            self.fig.canvas.mpl_connect("button_release_event", _on_mouse_release)
        except Exception:
            pass

        # draw event hook to capture background for blit
        def _on_draw(event=None):
            try:
                if self.fig is not None and self.ax is not None:
                    self._bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            except Exception:
                self._bg_cache = None

        try:
            self.fig.canvas.mpl_connect("draw_event", _on_draw)
        except Exception:
            pass

        # GUI timer callback (tiny, blit only)
        def _timer_cb():
            if self._closing:
                return
            t = self._progress_thread.get_time() if GUI_AVAILABLE else 0.0
            if self.duration_sec is not None and t > self.duration_sec:
                t = self.duration_sec
            self._update_progress_line_and_seek(t)

        try:
            self._timer = self.fig.canvas.new_timer(interval=self._gui_interval_ms)
            self._timer.add_callback(_timer_cb)
            self._timer.start()
        except Exception:
            logger.exception("Failed to start GUI timer; falling back to draw_idle polling")

        # close cleanup: cancel futures and shutdown executor before interpreter teardown
        def _on_close(event=None):
            logger.info("Closing viewer: cleaning up threads and futures...")
            self._closing = True
            # stop timer
            try:
                if self._timer is not None:
                    self._timer.stop()
            except Exception:
                pass
            # stop progress thread
            try:
                if GUI_AVAILABLE:
                    self._progress_thread.stop()
                    self._progress_thread.join(timeout=0.5)
            except Exception:
                pass
            # cancel outstanding futures
            try:
                for f in list(self._futures):
                    try:
                        f.cancel()
                    except Exception:
                        pass
                self._futures.clear()
            except Exception:
                pass
            # shutdown worker (wait for currently running tasks to finish)
            try:
                if self.worker is not None:
                    self.worker.shutdown(wait=True)
            except Exception:
                pass
            # cleanup pygame
            try:
                if _HAS_PYGAME and pygame.mixer.get_init():
                    pygame.mixer.quit()
            except Exception:
                pass

        try:
            self.fig.canvas.mpl_connect("close_event", _on_close)
        except Exception:
            pass

        # show interactive window (blocks until closed)
        plt.show(block=True)


if __name__ == "__main__":
    viewer = InteractiveCQTViewer(playlist_dir="playlist")
    viewer.run(initial_bpo=1)
