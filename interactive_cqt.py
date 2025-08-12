"""
interactive_cqt.py
Interactive CQT viewer wrapped in a class. Uses CQTComputer for CQT computation.
Features:
  - file selector (tkinter OptionMenu if available)
  - bins-per-octave slider
  - seek slider and optional pygame playback
  - wide-image caching (resampled) to speed redraws
"""

from typing import Optional, List, Tuple
import os
import time
import threading
import numpy as np
import matplotlib
# ensure GUI backend if available
try:
    _backend = matplotlib.get_backend().lower()
except Exception:
    _backend = ""
_gui_backends = ["tkagg", "qt5agg", "qtagg", "wxagg", "gtk3agg", "macosx"]
if not any(b in _backend for b in _gui_backends):
    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import librosa
import librosa.display
import logging

# optional pygame for playback
try:
    import pygame  # type: ignore
    _HAS_PYGAME = True
except Exception:
    pygame = None
    _HAS_PYGAME = False

from audio_cqt import CQTComputer, default_cqt_computer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InteractiveCQTViewer:
    """
    Interactive viewer for harmonic CQT of mp3 tracks in a directory.

    Parameters
    ----------
    playlist_dir : str
        Directory containing .mp3 files.
    cqt_computer : Optional[CQTComputer]
        Inject a CQTComputer instance; useful for testing or controlling GPU usage.
    pixels_per_second : int
        Width scaling for the "wide" cached image.
    """

    def __init__(self, playlist_dir: str = "playlist", cqt_computer: Optional[CQTComputer] = None, pixels_per_second: int = 200):
        self.playlist_dir = playlist_dir
        self.cqt = cqt_computer or default_cqt_computer()
        self.pixels_per_second = int(pixels_per_second)
        self.hop_length = 64

        # GUI state
        self.fig = None
        self.ax = None
        self.colorbar_ax = None
        self.slider = None
        self.seek_slider = None
        self.play_button = None

        # audio / track state
        self.files: List[str] = []
        self.current_file: Optional[str] = None
        self.y = None
        self.sr = None
        self.y_harmonic = None
        self.duration_sec = 0.0

        # caching / drawing state
        self._wide_img = None
        self._wide_img_path = None
        self._vmin_vmax: Optional[Tuple[float, float]] = None
        self._bg_cache = None
        self._blit_enabled = True

        # playback/progress state
        self.last_time_sec = 0.0
        self._thread_time_sec = 0.0
        self._thread_track_ended = False
        self._thread_stop_evt = threading.Event()
        self._thread_lock = threading.Lock()
        self._progress_thread = None
        self.is_playing = False
        self.is_paused = False

        # seek UI flags
        self._seek_user_dragging = False
        self._seek_internal_update = False
        self._seek_pending_time = None

    # --------------------------
    # File / audio utilities
    # --------------------------
    def list_mp3(self) -> List[str]:
        if not os.path.isdir(self.playlist_dir):
            raise FileNotFoundError(f"{self.playlist_dir} not found")
        files = [f for f in os.listdir(self.playlist_dir) if f.lower().endswith(".mp3")]
        files.sort()
        return files

    def _full_path(self, name: str) -> str:
        return os.path.join(self.playlist_dir, name)

    def _load_track(self, path: str):
        logger.info("Loading %s", path)
        y, sr = librosa.load(path)
        self.y = y
        self.sr = sr
        self.y_harmonic, _ = librosa.effects.hpss(y)
        self.duration_sec = float(librosa.get_duration(y=y, sr=sr))
        self.last_time_sec = 0.0
        with self._thread_lock:
            self._thread_time_sec = 0.0
        self._wide_img = None
        self._wide_img_path = None
        self._vmin_vmax = None

    # --------------------------
    # Wide cache helpers
    # --------------------------
    def _cache_dir(self, bpo: int) -> str:
        base = os.path.join("output", f"cqt-hop{self.hop_length}-hpss-bpo{int(bpo)}")
        os.makedirs(base, exist_ok=True)
        return base

    def _wide_cache_path(self, bpo: int) -> str:
        if not self.current_file:
            raise RuntimeError("current_file not set")
        name = os.path.splitext(os.path.basename(self.current_file))[0]
        return os.path.join(self._cache_dir(bpo), f"{name}_wide.jpg")

    def _resample_time(self, arr: np.ndarray, target_frames: int) -> np.ndarray:
        n_bins, n_frames = arr.shape
        if target_frames <= 0 or n_frames == target_frames:
            return arr
        x_src = np.linspace(0.0, 1.0, n_frames)
        x_tgt = np.linspace(0.0, 1.0, target_frames)
        out = np.empty((n_bins, target_frames), dtype=float)
        for i in range(n_bins):
            out[i] = np.interp(x_tgt, x_src, arr[i])
        return out

    def _ensure_wide_cache(self, bpo: int):
        self._wide_img_path = self._wide_cache_path(bpo)
        if os.path.exists(self._wide_img_path):
            return
        # compute CQT and save wide image
        C_db = self.cqt.compute_cqt_db(self.y_harmonic, self.sr, bins_per_octave=int(bpo), hop_length=self.hop_length, n_bins=84)
        vmin = float(np.percentile(C_db, 5.0))
        vmax = float(np.percentile(C_db, 99.5))
        self._vmin_vmax = (vmin, vmax)
        try:
            width_px = max(1, int(float(self.duration_sec) * float(self.pixels_per_second)))
        except Exception:
            width_px = max(1, int(C_db.shape[1] * 4))
        C_rs = self._resample_time(C_db, width_px)
        try:
            plt.imsave(self._wide_img_path, C_rs, cmap="magma", vmin=vmin, vmax=vmax, origin="lower", format="jpg")
        except Exception:
            plt.imsave(self._wide_img_path, C_rs, cmap="magma", origin="lower", format="jpg")

    def _load_wide_image(self, bpo: int):
        self._ensure_wide_cache(bpo)
        try:
            self._wide_img = plt.imread(self._wide_img_path)
        except Exception as e:
            logger.warning("Failed to load wide image: %s. Regenerating.", e)
            try:
                if os.path.exists(self._wide_img_path):
                    os.remove(self._wide_img_path)
            except Exception:
                pass
            self._ensure_wide_cache(bpo)
            self._wide_img = plt.imread(self._wide_img_path)

    # --------------------------
    # Plotting / UI
    # --------------------------
    def _init_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 5.2))
        try:
            self.fig.canvas.manager.set_window_title("Interactive CQT: Select track and adjust BPO")
        except Exception:
            pass
        plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.30)
        self.colorbar_ax = self.fig.add_axes([0.935, 0.32, 0.02, 0.56])

    def _plot_cqt(self, bpo: int):
        self.ax.clear()
        self.ax.set_position([0.08, 0.32, 0.82, 0.56])
        try:
            self.colorbar_ax.set_position([0.935, 0.32, 0.02, 0.56])
        except Exception:
            pass

        C_db = self.cqt.compute_cqt_db(self.y_harmonic, self.sr, bins_per_octave=int(bpo), hop_length=self.hop_length, n_bins=84)
        img = librosa.display.specshow(C_db, sr=self.sr, hop_length=self.hop_length, x_axis="time", y_axis="cqt_note", ax=self.ax)
        self.ax.set_title(f"{os.path.basename(self.current_file)} - Harmonic CQT (bins_per_octave={int(bpo)})")
        # manage persistent colorbar
        if not hasattr(self, "_colorbar_ref") or getattr(self, "_colorbar_ref") is None:
            self._colorbar_ref = self.fig.colorbar(img, cax=self.colorbar_ax, format="%+2.0f dB")
        else:
            self._colorbar_ref.update_normal(img)
        # draw and reset blit cache
        try:
            self.fig.canvas.draw()
        except Exception:
            pass
        self._bg_cache = None

    def _on_bins_change(self, val):
        self._plot_cqt(int(val))

    # --------------------------
    # Playback helpers (pygame)
    # --------------------------
    def _ensure_mixer(self) -> bool:
        if not _HAS_PYGAME:
            return False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
                try:
                    pygame.mixer.set_num_channels(1)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Failed to initialize pygame mixer: %s", e)
            return False
        return True

    def _stop_playback(self):
        if _HAS_PYGAME and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            try:
                pygame.mixer.stop()
            except Exception:
                pass
            try:
                if hasattr(pygame.mixer.music, "unload"):
                    pygame.mixer.music.unload()  # type: ignore[attr-defined]
            except Exception:
                pass
        self.is_playing = False
        self.is_paused = False
        self.last_time_sec = 0.0
        self._bg_cache = None
        with self._thread_lock:
            self._thread_time_sec = 0.0
        try:
            self._seek_internal_update = True
            if self._seek_slider_exists():
                self.seek_slider.set_val(0.0)
        finally:
            self._seek_internal_update = False
        if _HAS_PYGAME and getattr(self, "play_button", None) is not None:
            try:
                self.play_button.label.set_text("Play")
                self.fig.canvas.draw_idle()
            except Exception:
                pass

    def _play_current(self):
        if not self._ensure_mixer():
            logger.info("pygame not available; cannot play audio.")
            return
        self._stop_playback()
        try:
            start_t = max(0.0, float(self.last_time_sec))
            pygame.mixer.music.load(self.current_file)
            try:
                pygame.mixer.music.play(start=start_t)
            except TypeError:
                if start_t > 0.05:
                    logger.info("Precise seeking unsupported by pygame/codec; starting from beginning.")
                pygame.mixer.music.play()
            self.is_playing = True
            self.is_paused = False
            if getattr(self, "play_button", None):
                try:
                    self.play_button.label.set_text("Pause")
                    self.fig.canvas.draw_idle()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Failed to play audio: %s", e)
            self.is_playing = False
            self.is_paused = False

    def _toggle_play(self, event=None):
        if not _HAS_PYGAME:
            logger.info("pygame not installed. Playback disabled.")
            return
        if getattr(self._toggle_play, "_busy", False):
            return
        setattr(self._toggle_play, "_busy", True)
        try:
            if not self.is_playing:
                self._play_current()
            else:
                if not self.is_paused:
                    pygame.mixer.music.pause()
                    self.is_paused = True
                    self.play_button.label.set_text("Resume")
                    self.fig.canvas.draw_idle()
                else:
                    pygame.mixer.music.unpause()
                    self.is_paused = False
                    self._t0_perf = time.perf_counter()
                    self.play_button.label.set_text("Pause")
                    self.fig.canvas.draw_idle()
        finally:
            setattr(self._toggle_play, "_busy", False)

    # --------------------------
    # Progress thread
    # --------------------------
    def _progress_worker(self):
        last_ref_sync = 0.0
        while not self._thread_stop_evt.is_set():
            try:
                if self._seek_user_dragging:
                    t = float(self.last_time_sec)
                else:
                    t = float(self.last_time_sec)
                    if self.is_playing and not self.is_paused:
                        t_smooth = float(self._play_start_pos_sec) if hasattr(self, "_play_start_pos_sec") else 0.0
                        if hasattr(self, "_t0_perf") and self._t0_perf is not None:
                            t_smooth += max(0.0, time.perf_counter() - self._t0_perf)
                        t_ref = None
                        now = time.perf_counter()
                        if now - last_ref_sync >= 0.05:
                            last_ref_sync = now
                            try:
                                if _HAS_PYGAME and pygame.mixer.get_init():
                                    pos_ms = pygame.mixer.music.get_pos()
                                    if pos_ms is not None and pos_ms >= 0:
                                        t_ref = max(0.0, float(pos_ms) / 1000.0)
                            except Exception:
                                t_ref = None
                        if t_ref is None:
                            t = t_smooth
                        else:
                            alpha = 0.85
                            t = alpha * t_smooth + (1.0 - alpha) * t_ref
                if self.duration_sec is not None and t > self.duration_sec:
                    t = self.duration_sec
                with self._thread_lock:
                    self._thread_time_sec = t
                if _HAS_PYGAME and pygame.mixer.get_init():
                    if self.is_playing and not self.is_paused and not pygame.mixer.music.get_busy():
                        self._thread_track_ended = True
            except Exception:
                pass
            time.sleep(0.005)

    def _start_progress_thread(self):
        if self._progress_thread is None or not self._progress_thread.is_alive():
            self._thread_stop_evt.clear()
            self._progress_thread = threading.Thread(target=self._progress_worker, name="ProgressUpdater", daemon=True)
            self._progress_thread.start()

    def _stop_progress_thread(self):
        if self._progress_thread is not None and self._progress_thread.is_alive():
            self._thread_stop_evt.set()
            try:
                self._progress_thread.join(timeout=0.5)
            except Exception:
                pass

    # --------------------------
    # Seek handling / UI helpers
    # --------------------------
    def _clamp_time(self, t_sec: float) -> float:
        try:
            if self.duration_sec is not None:
                return min(max(0.0, float(t_sec)), float(self.duration_sec))
        except Exception:
            pass
        return max(0.0, float(t_sec))

    def _preview_seek(self, t_sec: float):
        t_sec = self._clamp_time(t_sec)
        self._seek_pending_time = t_sec
        self.last_time_sec = float(t_sec)
        # update vertical line via blit if available
        try:
            if hasattr(self, "progress_line") and self.progress_line is not None and self.progress_line.axes is not None:
                self.progress_line.set_xdata([t_sec, t_sec])
                if self._blit_enabled and self._bg_cache is not None:
                    try:
                        canvas = self.fig.canvas
                        canvas.restore_region(self._bg_cache)
                        self.ax.draw_artist(self.progress_line)
                        canvas.blit(self.ax.bbox)
                    except Exception:
                        pass
        except Exception:
            pass

    def _apply_seek_to_audio(self, t_sec: float):
        t_sec = self._clamp_time(t_sec)
        self.last_time_sec = float(t_sec)
        self._play_start_pos_sec = float(t_sec)
        self._t0_perf = time.perf_counter()
        try:
            if hasattr(self, "progress_line") and self.progress_line is not None and self.progress_line.axes is not None:
                self.progress_line.set_xdata([t_sec, t_sec])
        except Exception:
            pass
        self._thread_track_ended = False
        with self._thread_lock:
            self._thread_time_sec = t_sec
        if self.is_playing and _HAS_PYGAME and pygame is not None:
            try:
                if pygame.mixer.get_init():
                    try:
                        pygame.mixer.music.pause()
                    except Exception:
                        pass
                    try:
                        pygame.mixer.music.load(self.current_file)
                        try:
                            pygame.mixer.music.play(start=t_sec)
                            if self.is_paused:
                                pygame.mixer.music.pause()
                        except TypeError:
                            if t_sec > 0.05:
                                logger.info("Precise seek unsupported; restarting from beginning.")
                            pygame.mixer.music.play()
                            if self.is_paused:
                                pygame.mixer.music.pause()
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            self._seek_internal_update = True
            if self._seek_slider_exists():
                self.seek_slider.set_val(t_sec)
        finally:
            self._seek_internal_update = False

    def _on_seek_changed(self, val):
        if self._seek_internal_update:
            return
        if self._seek_user_dragging:
            self._preview_seek(val)

    def _on_mouse_press(self, event):
        if event.inaxes is getattr(self, "seek_ax", None):
            self._seek_user_dragging = True

    def _on_mouse_release(self, event):
        if self._seek_user_dragging:
            self._seek_user_dragging = False
            if self._seek_pending_time is not None:
                self._apply_seek_to_audio(self._seek_pending_time)
            self._seek_pending_time = None

    def _seek_slider_exists(self) -> bool:
        return getattr(self, "seek_slider", None) is not None

    # --------------------------
    # Public run method
    # --------------------------
    def run(self, initial_bpo: int = 48):
        self.files = self.list_mp3()
        if not self.files:
            raise FileNotFoundError(f"No .mp3 files found in '{self.playlist_dir}'")
        self.current_file = self._full_path(self.files[-1])

        # load initial track
        self._load_track(self.current_file)

        # build GUI
        self._init_figure()

        # sliders and buttons
        ax_slider = self.fig.add_axes([0.12, 0.08, 0.78, 0.06])
        self.slider = Slider(ax=ax_slider, label="Bins per Octave", valmin=10, valmax=100, valinit=float(initial_bpo), valstep=1)
        self.slider.on_changed(self._on_bins_change)

        # seek slider
        self.seek_ax = self.fig.add_axes([0.12, 0.18, 0.78, 0.06])
        self.seek_slider = Slider(ax=self.seek_ax, label="Seek (s)", valmin=0.0, valmax=self.duration_sec, valinit=0.0, valstep=0.01)
        self.seek_slider.on_changed(self._on_seek_changed)

        # play/pause
        play_ax = self.fig.add_axes([0.03, 0.08, 0.07, 0.06])
        label = "Play" if _HAS_PYGAME else "NoAudio"
        self.play_button = Button(play_ax, label)
        self.play_button.on_clicked(self._toggle_play)

        # plot initial
        self._plot_cqt(int(self.slider.val))

        # progress vertical line
        try:
            self.progress_line = self.ax.axvline(0.0, color="cyan", linewidth=1.5, alpha=0.9)
            try:
                self.progress_line.set_animated(True)
            except Exception:
                pass
        except Exception:
            self.progress_line = None

        # drawing cache hook
        def on_draw(event=None):
            try:
                self._bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            except Exception:
                self._bg_cache = None
        try:
            self.fig.canvas.mpl_connect("draw_event", on_draw)
        except Exception:
            pass

        # mouse hook for seek dragging
        try:
            self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
            self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        except Exception:
            pass

        # dropdown selector (tkinter OptionMenu) if available
        labels = [os.path.splitext(f)[0] for f in self.files]

        def on_select(label: str):
            if label not in labels:
                return
            idx = labels.index(label)
            # stop playback & load track
            self._stop_playback()
            self.current_file = self._full_path(self.files[idx])
            self._load_track(self.current_file)
            # update seek slider limits and redraw spectrogram
            try:
                self.seek_slider.valmax = float(self.duration_sec)
                self.seek_ax.set_xlim(self.seek_slider.valmin, self.seek_slider.valmax)
                self._seek_internal_update = True
                self.seek_slider.set_val(0.0)
            finally:
                self._seek_internal_update = False
            try:
                self.fig.canvas.manager.set_window_title(f"Interactive CQT: {os.path.basename(self.current_file)}")
            except Exception:
                pass
            self._plot_cqt(int(self.slider.val))
            self._bg_cache = None

        try:
            import tkinter as tk
            from tkinter import ttk
            root = self.fig.canvas.manager.window  # type: ignore
            if root is not None:
                sel_var = tk.StringVar(master=root, value=labels[-1])
                option = ttk.OptionMenu(root, sel_var, labels[-1], *labels, command=lambda val: on_select(val))
                option.place(x=10, y=10)
        except Exception:
            logger.info("tkinter dropdown unavailable; skip selector overlay.")

        # progress thread + timer
        self._start_progress_thread()
        try:
            progress_timer = self.fig.canvas.new_timer(interval=16)
            def _update_progress():
                # read thread value
                try:
                    with self._thread_lock:
                        t = float(self._thread_time_sec)
                except Exception:
                    t = float(self.last_time_sec)
                if self.duration_sec is not None and t > self.duration_sec:
                    t = self.duration_sec
                self.last_time_sec = t
                # update seek slider
                try:
                    if not self._seek_user_dragging:
                        self._seek_internal_update = True
                        self.seek_slider.set_val(t)
                finally:
                    self._seek_internal_update = False
                # update the progress line with blit if possible
                try:
                    if self.progress_line is not None and self.progress_line.axes is not None:
                        self.progress_line.set_xdata([t, t])
                        if self._blit_enabled:
                            if self._bg_cache is None:
                                try:
                                    self._bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
                                except Exception:
                                    self._blit_enabled = False
                            if self._bg_cache is not None and self._blit_enabled:
                                try:
                                    canvas = self.fig.canvas
                                    canvas.restore_region(self._bg_cache)
                                    self.ax.draw_artist(self.progress_line)
                                    canvas.blit(self.ax.bbox)
                                    return
                                except Exception:
                                    self._blit_enabled = False
                except Exception:
                    pass
        except Exception:
            pass
