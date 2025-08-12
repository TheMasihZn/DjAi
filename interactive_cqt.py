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

        # audio
        self.files: List[str] = []
        self.current_file: Optional[str] = None
        self.y = None
        self.sr = None
        self.y_harmonic = None
        self.duration_sec = 0.0

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
        def _load(p):
            logger.info("[worker] Loading audio: %s", p)
            # librosa is imported at module level to avoid lazy import inside worker threads during shutdown
            y, sr = librosa.load(p, sr=None, duration=10, mono=True, res_type="kaiser_best")
            y_harmonic, _ = librosa.effects.hpss(y, kernel_size=2)
            duration = float(librosa.get_duration(y=y, sr=sr))
            return y, sr, y_harmonic, duration

        assert self.worker is not None
        fut = self.worker.submit(_load, path)
        self._futures.append(fut)
        return fut

    def compute_and_cache_wide_async(self, y_harmonic: np.ndarray, sr: int, bpo: int, duration_sec: float) -> concurrent.futures.Future:
        def _task(y_h, sr_, bpo_, dur_):
            logger.info("[worker] Computing CQT (bpo=%d)", bpo_)
            C_db = self.cqt.compute_cqt_db(y_h, sr_, bins_per_octave=int(bpo_), hop_length=self.hop_length, n_bins=84)
            vmin = float(np.percentile(C_db, 5.0))
            vmax = float(np.percentile(C_db, 99.5))
            cache_dir = os.path.join("output", f"cqt-hop{self.hop_length}-hpss-bpo{int(bpo_)}")
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
        fut = self.worker.submit(_task, y_harmonic, sr, int(bpo), float(duration_sec))
        self._futures.append(fut)
        return fut

    # --------------------
    # UI helpers
    # --------------------
    def _init_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 5.2))
        try:
            self.fig.canvas.manager.set_window_title("Interactive CQT: Async (robust)")
        except Exception:
            pass
        plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.30)
        self.colorbar_ax = self.fig.add_axes([0.935, 0.32, 0.02, 0.56])

    def _draw_placeholder(self, message: str = "Loading..."):
        assert self.ax is not None
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, ha="center", va="center", transform=self.ax.transAxes, fontsize=16)
        try:
            self.fig.canvas.draw()
        except Exception:
            pass
        self._bg_cache = None

    def _plot_wide_image(self, wide_path: str, vmin: float, vmax: float, bpo: int):
        assert self.ax is not None
        self.ax.clear()
        try:
            img = plt.imread(wide_path)
            self.ax.imshow(img, aspect="auto", origin="lower", extent=(0.0, float(self.duration_sec), 0.0, 84))
            self.ax.set_ylim(0.0, 84)
            self.ax.set_xlim(0.0, float(self.duration_sec))
            self.ax.set_ylabel("CQT bins")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_title(f"{os.path.basename(self.current_file)} - Harmonic CQT (bpo={int(bpo)})")
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
            y_harmonic, _ = librosa.effects.hpss(y)
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
        ax_slider = self.fig.add_axes([0.12, 0.08, 0.78, 0.06])
        self.slider = Slider(ax=ax_slider, label="Bins per Octave", valmin=10, valmax=100, valinit=float(initial_bpo), valstep=1)

        # seek slider
        self.seek_ax = self.fig.add_axes([0.12, 0.18, 0.78, 0.06])
        self.seek_slider = Slider(ax=self.seek_ax, label="Seek (s)", valmin=0.0, valmax=1.0, valinit=0.0, valstep=0.01)

        # play button
        play_ax = self.fig.add_axes([0.03, 0.08, 0.07, 0.06])
        lbl = "Play" if _HAS_PYGAME else "NoAudio"
        self.play_button = Button(play_ax, lbl)
        self.play_button.on_clicked(self.toggle_play)

        # placeholder
        self._draw_placeholder("Loading track and computing spectrogram...")

        # submit audio load
        load_future = self.load_track_async(self.current_file)

        def _on_wide_done(fut: concurrent.futures.Future):
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
                self._plot_wide_image(wide_path, vmin, vmax, int(self.slider.val))
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
            fut2 = self.compute_and_cache_wide_async(self.y_harmonic, self.sr, int(self.slider.val), self.duration_sec)
            fut2.add_done_callback(_on_wide_done)

        load_future.add_done_callback(_on_load_done)

        # bins slider callback -> recompute wide image in background
        def _on_bins_change(val):
            if self._closing:
                return
            self._draw_placeholder("Recomputing CQT...")
            if self.y_harmonic is None:
                return
            fut = self.compute_and_cache_wide_async(self.y_harmonic, self.sr, int(val), self.duration_sec)
            fut.add_done_callback(_on_wide_done)

        self.slider.on_changed(_on_bins_change)

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
