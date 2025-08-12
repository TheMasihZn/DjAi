# interactive_cqt_offline.py
"""
Interactive CQT viewer with an offline (pre-computed) scrolling display.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Optional
import collections

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

# Import librosa at module import time
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
logger = logging.getLogger("interactive_cqt_offline")

_backend_name = matplotlib.get_backend().lower()
GUI_AVAILABLE = any(b in _backend_name for b in _GUI_BACKENDS)
if _chosen_backend:
    logger.info("Requested backend: %s, effective backend: %s", _chosen_backend, matplotlib.get_backend())
else:
    logger.info("Effective backend: %s", matplotlib.get_backend())

if not GUI_AVAILABLE:
    logger.warning(
        "No interactive GUI backend appears available (backend=%s). "
        "This script requires a GUI and will not run in synchronous mode.",
        matplotlib.get_backend(),
    )


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
                            if self._is_playing:
                                logger.info("Playback seems to have stopped, resetting position.")
                                self._is_playing = False
                                self._time_sec = 0.0
                                continue
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
            gui_fps: int = 60,
            view_duration_sec: int = 10,
    ):
        self.playlist_dir = playlist_dir
        self.cqt = cqt_computer or default_cqt_computer()
        self.hop_length = 64

        self.view_duration_sec = float(view_duration_sec)

        self.cqt_data_full = None  # This will hold the full pre-computed CQT

        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.im: Optional[plt.imshow] = None
        self.playhead_line = None
        self.colorbar_ax: Optional[plt.Axes] = None
        self.slider: Optional[Slider] = None
        self.seek_slider: Optional[Slider] = None
        self.play_button: Optional[Button] = None

        self.hpss_kernel_size_slider: Optional[Slider] = None
        self.hpss_power_slider: Optional[Slider] = None
        self.hpss_hop_length_slider: Optional[Slider] = None
        self.hpss_margin_button: Optional[Button] = None

        self.files: List[str] = []
        self.current_file: Optional[str] = None
        self.sr: Optional[int] = None
        self.duration_sec: float = 0.0

        self.hpss_kernel_size = 2
        self.hpss_power = 2.0
        self.hpss_margin_options = ['hard', 'soft', 'mask']
        self.hpss_margin_index = 0

        self._closing = False
        self._progress_thread = PlaybackProgressThread(interval=0.01)
        if GUI_AVAILABLE:
            self._progress_thread.start()

        self.is_playing = False
        self.is_paused = False

        self._is_seeking = False

        self._gui_interval_ms = int(max(1, round(1000.0 / float(gui_fps))))
        self._timer: Optional[plt.Timer] = None

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

    # --------------------------------
    # --- Offline Processing Method ---
    # --------------------------------

    def _pre_process_audio(self):
        """Pre-processes the entire audio file to generate the CQT spectrogram."""
        if not self.current_file:
            return

        self.ax.set_title("Pre-processing audio... This may take a moment.")
        self.fig.canvas.draw_idle()

        y_full, self.sr = librosa.load(self.current_file, sr=None, mono=True, res_type="kaiser_best")

        # HPSS is performed on the entire audio file
        hpss_kwargs = {
            'kernel_size': int(self.hpss_kernel_size_slider.val),
            'power': float(self.hpss_power_slider.val),
        }
        try:
            hpss_kwargs['margin'] = self.hpss_margin_options[self.hpss_margin_index]
            y_harmonic, _ = librosa.effects.hpss(y_full, **hpss_kwargs)
        except TypeError:
            logger.warning("Caught TypeError with string-based HPSS margin... Falling back to numeric.")
            hpss_kwargs['margin'] = 1.0
            y_harmonic, _ = librosa.effects.hpss(y_full, **hpss_kwargs)

        # Compute CQT for the entire harmonic signal
        cqt_data = self.cqt.compute_cqt_db(
            y_harmonic, self.sr,
            bins_per_octave=int(self.slider.val),
            hop_length=int(self.hpss_hop_length_slider.val),
            n_bins=84
        )
        self.cqt_data_full = cqt_data.astype(np.float32)

        self.duration_sec = librosa.get_duration(y=y_full, sr=self.sr)

        # Correctly set the slider's maximum value and reset its current position
        if self.seek_slider is not None:
            self.seek_slider.valmax = self.duration_sec
            self.seek_slider.ax.set_xlim(self.seek_slider.valmin, self.seek_slider.valmax)
            self.seek_slider.set_val(0.0) # Reset to the beginning

        logger.info("Pre-processing complete. Spectrogram data is ready.")
        self.ax.set_title(
            f"{os.path.basename(self.current_file)} - BPO={int(self.slider.val)}, Hop={int(self.hpss_hop_length_slider.val)}"
        )
        self.fig.canvas.draw_idle()

    def _update_display(self):
        """Updates the display based on the playback time and pre-computed data."""
        if self.im is None or self.fig is None or self.ax is None or self._closing or self.cqt_data_full is None:
            return

        current_time = self._progress_thread.get_time()
        cqt_hop_length = int(self.hpss_hop_length_slider.val)
        cqt_fps = self.sr / cqt_hop_length if self.sr else 44100 / cqt_hop_length

        # Calculate the start and end frames for the current view
        view_start_time = max(0, current_time - self.view_duration_sec / 2)
        view_end_time = view_start_time + self.view_duration_sec

        view_start_frame = int(view_start_time * cqt_fps)
        view_end_frame = int(view_end_time * cqt_fps)

        # We'll use a placeholder until data is ready
        view_width_frames = int(self.view_duration_sec * cqt_fps)
        rolling_buffer = np.full((84, view_width_frames), -100.0)

        # Slice the pre-computed data to get the current view
        visible_data = self.cqt_data_full[:, view_start_frame:view_end_frame]

        # Pad with empty data if we are at the end of the file
        if visible_data.shape[1] < rolling_buffer.shape[1]:
            rolling_buffer[:, :visible_data.shape[1]] = visible_data
        else:
            rolling_buffer = visible_data

        try:
            self.im.set_data(rolling_buffer)
            self.im.set_clim(-80, 20)
            self.fig.canvas.draw_idle()
        except Exception as e:
            logger.warning("Failed to update display: %s", e, exc_info=False)


    # --------------------
    # UI and Playback
    # --------------------

    def _init_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("Interactive CQT: Offline")
        plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.45)
        self.colorbar_ax = self.fig.add_axes([0.935, 0.47, 0.02, 0.43])

        # We'll use a placeholder until data is ready
        view_width_frames = int(self.view_duration_sec * (44100 / self.hop_length))
        initial_data = np.full((84, view_width_frames), -100.0)

        self.im = self.ax.imshow(initial_data, aspect='auto', origin='lower', cmap='magma', animated=True, vmin=-80, vmax=20)
        plt.colorbar(self.im, cax=self.colorbar_ax)

        self.ax.set_ylabel("CQT bins")
        self.ax.set_xlabel(f"Time Window ({self.view_duration_sec}s)")

        self.ax.set_xticks([0, view_width_frames / 2, view_width_frames])
        self.ax.set_xticklabels([f"-{self.view_duration_sec/2:.1f}s", "Playhead", f"+{self.view_duration_sec/2:.1f}s"])

        self.playhead_line = self.ax.axvline(view_width_frames / 2, color="cyan", linewidth=1.5)

    def _ensure_mixer(self) -> bool:
        if not _HAS_PYGAME: return False
        try:
            if not pygame.mixer.get_init(): pygame.mixer.init(frequency=self.sr or 44100)
            return True
        except Exception:
            logger.exception("Failed to init pygame mixer")
            return False

    def _stop_playback(self):
        if _HAS_PYGAME and pygame.mixer.get_init():
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                if hasattr(pygame.mixer.music, "unload"):
                    pygame.mixer.music.unload()
            except Exception: pass
        self.is_playing = False
        self.is_paused = False
        if GUI_AVAILABLE:
            self._progress_thread.set_playing(False)
            self._progress_thread.set_paused(False)
            self._progress_thread.set_time(0.0)

        if self.seek_slider is not None:
            self._is_seeking = True
            self.seek_slider.set_val(0.0)
            self._is_seeking = False

    def _play_current(self, start_sec: float = 0.0):
        if self.cqt_data_full is None:
            logger.warning("Spectrogram data is not ready yet. Cannot play.")
            return

        if not self._ensure_mixer():
            logger.info("pygame not available; cannot play audio")
            return

        self._stop_playback()

        try:
            pygame.mixer.music.load(self.current_file)
            pygame.mixer.music.play(start=float(start_sec))
            self.is_playing = True
            self.is_paused = False
            if GUI_AVAILABLE:
                self._progress_thread.set_playing(True)
                self._progress_thread.set_paused(False)
        except Exception:
            logger.exception("Failed to start playback")
            self.is_playing = False

    def toggle_play(self, event=None):
        if not _HAS_PYGAME or self.cqt_data_full is None: return
        if not self.is_playing:
            t = self._progress_thread.get_time() if GUI_AVAILABLE else 0.0
            self._play_current(start_sec=t)
            if self.play_button: self.play_button.label.set_text("Pause")
        else:
            if not self.is_paused:
                pygame.mixer.music.pause()
                self.is_paused = True
                if GUI_AVAILABLE: self._progress_thread.set_paused(True)
                if self.play_button: self.play_button.label.set_text("Resume")
            else:
                pygame.mixer.music.unpause()
                self.is_paused = False
                if GUI_AVAILABLE: self._progress_thread.set_paused(False)
                if self.play_button: self.play_button.label.set_text("Pause")

    def _apply_seek(self, t_sec: float):
        if self._is_seeking or self.cqt_data_full is None:
            return

        t_sec = max(0.0, min(float(t_sec), float(self.duration_sec)))
        if GUI_AVAILABLE:
            self._progress_thread.set_time(t_sec)

        logger.info("Seeking to %.2f s. No recompute needed.", t_sec)

        if self.is_playing and _HAS_PYGAME:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(self.current_file)
            pygame.mixer.music.play(start=t_sec)
            if self.is_paused:
                pygame.mixer.music.pause()

    def _trigger_recompute(self):
        """Called when parameters are changed to trigger a full re-processing."""
        if self._closing:
            return
        logger.info("Parameters changed. Triggering a full re-compute.")
        self._stop_playback()
        self._pre_process_audio()


    # --------------------
    # Main run
    # --------------------
    def run(self, initial_bpo: int = 10):
        self.files = self.list_mp3()
        if not self.files:
            raise FileNotFoundError(f"No .mp3 files found in '{self.playlist_dir}'")
        self.current_file = self._full_path(self.files[-1])

        if not GUI_AVAILABLE:
            print("This script now requires an interactive GUI backend to run.")
            return

        self._ensure_mixer()
        self._init_figure()

        ax_slider_bpo = self.fig.add_axes([0.08, 0.35, 0.40, 0.025])
        self.slider = Slider(ax=ax_slider_bpo, label="Bins/Octave", valmin=10, valmax=100, valinit=float(initial_bpo), valstep=1)

        ax_slider_kernel = self.fig.add_axes([0.08, 0.30, 0.40, 0.025])
        self.hpss_kernel_size_slider = Slider(ax=ax_slider_kernel, label="HPSS Kernel", valmin=1, valmax=100, valinit=self.hpss_kernel_size, valstep=2)

        ax_slider_power = self.fig.add_axes([0.08, 0.25, 0.40, 0.025])
        self.hpss_power_slider = Slider(ax=ax_slider_power, label="HPSS Power", valmin=1.0, valmax=100.0, valinit=self.hpss_power, valstep=0.1)

        ax_slider_hop = self.fig.add_axes([0.08, 0.20, 0.40, 0.025])
        self.hpss_hop_length_slider = Slider(ax=ax_slider_hop, label="Hop Length", valmin=64, valmax=1024, valinit=self.hop_length, valstep=64)

        ax_button_margin = self.fig.add_axes([0.5, 0.35, 0.1, 0.04])
        self.hpss_margin_button = Button(ax_button_margin, f"Margin: {self.hpss_margin_options[self.hpss_margin_index]}")

        self.seek_ax = self.fig.add_axes([0.5, 0.25, 0.40, 0.04])
        self.seek_slider = Slider(ax=self.seek_ax, label="Seek", valmin=0.0, valmax=1.0, valinit=0.0) # Placeholder max value

        play_ax = self.fig.add_axes([0.5, 0.15, 0.07, 0.04])
        self.play_button = Button(play_ax, "Play" if _HAS_PYGAME else "NoAudio")

        self.play_button.on_clicked(self.toggle_play)
        self.slider.on_changed(lambda val: self._trigger_recompute())
        self.hpss_kernel_size_slider.on_changed(lambda val: self._trigger_recompute())
        self.hpss_power_slider.on_changed(lambda val: self._trigger_recompute())
        self.hpss_hop_length_slider.on_changed(lambda val: self._trigger_recompute())

        def _on_margin_button_click(event):
            self.hpss_margin_index = (self.hpss_margin_index + 1) % len(self.hpss_margin_options)
            self.hpss_margin_button.label.set_text(f"Margin: {self.hpss_margin_options[self.hpss_margin_index]}")
            self._trigger_recompute()
        self.hpss_margin_button.on_clicked(_on_margin_button_click)
        self.seek_slider.on_changed(lambda val: self._apply_seek(val))

        # Start the pre-processing
        self._pre_process_audio()

        def _timer_cb():
            if self._closing: return

            self._is_seeking = True
            t = self._progress_thread.get_time()
            if self.seek_slider and abs(self.seek_slider.val - t) > 0.1:
                self.seek_slider.set_val(t)
            self._is_seeking = False

            self._update_display()

        self._timer = self.fig.canvas.new_timer(interval=self._gui_interval_ms)
        self._timer.add_callback(_timer_cb)
        self._timer.start()

        def _on_close(event=None):
            logger.info("Closing viewer: cleaning up threads...")
            self._closing = True
            if self._timer: self._timer.stop()
            if GUI_AVAILABLE:
                self._progress_thread.stop()
                self._progress_thread.join(timeout=0.5)
            if _HAS_PYGAME and pygame.mixer.get_init():
                pygame.mixer.quit()

        self.fig.canvas.mpl_connect("close_event", _on_close)
        plt.show(block=True)


if __name__ == "__main__":
    viewer = InteractiveCQTViewer(playlist_dir="playlist")
    viewer.run(initial_bpo=24)
