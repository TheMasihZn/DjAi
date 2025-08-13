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
    """Playback clock thread that tracks time independently of the audio backend."""
    def __init__(self, interval: float = 0.01): # 10 ms
        super().__init__(daemon=True, name="PlaybackProgressThread")
        self.interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._time_sec = 0.0
        self._is_playing = False
        self._is_paused = False
        self._last_tick = None  # type: Optional[float]

    def run(self):
        while not self._stop.is_set():
            try:
                with self._lock:
                    if self._is_playing and not self._is_paused:
                        now = time.perf_counter()
                        if self._last_tick is None:
                            self._last_tick = now
                        dt = now - self._last_tick
                        self._last_tick = now
                        self._time_sec += float(dt)
                    else:
                        # Not playing or paused: reset last tick so we don't accumulate a big dt later
                        self._last_tick = None
                
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()

    def set_time(self, t: float):
        with self._lock:
            self._time_sec = float(t)
            self._last_tick = time.perf_counter() if self._is_playing and not self._is_paused else None

    def get_time(self) -> float:
        with self._lock:
            return float(self._time_sec)

    def set_playing(self, playing: bool):
        with self._lock:
            self._is_playing = bool(playing)
            self._last_tick = time.perf_counter() if self._is_playing and not self._is_paused else None

    def set_paused(self, paused: bool):
        with self._lock:
            self._is_paused = bool(paused)
            self._last_tick = time.perf_counter() if self._is_playing and not self._is_paused else None


class InteractiveCQTViewer:
    def __init__(
            self,
            playlist_dir: str = "playlist",
            cqt_computer: Optional[CQTComputer] = None,
            gui_fps: int = 100, # Changed GUI frames per second to 100
            view_duration_sec: int = 10,
    ):
        self.playlist_dir = playlist_dir
        self.cqt = cqt_computer or default_cqt_computer()
        self.hop_length = 64

        self.view_duration_sec = float(view_duration_sec)

        # Two-track data containers
        self.tracks = [
            {
                'file': None,          # type: Optional[str]
                'sr': None,            # type: Optional[int]
                'y_full': None,        # type: Optional[np.ndarray]
                'duration': 0.0,       # type: float
                'cqt': None            # type: Optional[np.ndarray]
            },
            {
                'file': None,
                'sr': None,
                'y_full': None,
                'duration': 0.0,
                'cqt': None
            }
        ]

        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.im1: Optional[plt.imshow] = None
        self.im2: Optional[plt.imshow] = None
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
        self.current_file: Optional[str] = None  # kept for compatibility, points to last file
        self.sr: Optional[int] = None
        self.duration_sec: float = 0.0  # max duration across tracks

        self.hpss_kernel_size = 2
        self.hpss_power = 2.0
        # Use numeric margins by default for librosa HPSS to avoid TypeError with string-based options
        self.hpss_margin_options = [1.0, 1.5, 2.0]
        self.hpss_margin_index = 0

        self._closing = False
        self._progress_thread = PlaybackProgressThread(interval=0.01)
        if GUI_AVAILABLE:
            self._progress_thread.start()

        self.is_playing = False
        self.is_paused = False
        self._is_processing = False  # Flag to prevent race conditions

        self._is_seeking = False

        self._gui_interval_ms = int(max(1, round(1000.0 / float(gui_fps))))
        self._timer: Optional[plt.Timer] = None

        # Audio playback (mixed) state
        self._channel = None  # type: Optional[object]
        self._current_sound = None  # type: Optional[object]
        self._mix_sr = None  # type: Optional[int]

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
        """Pre-process the last two audio files to generate CQT spectrograms for both."""
        if not self.tracks[0]['file'] or not self.tracks[1]['file']:
            return

        self._is_processing = True
        self.ax.set_title("Pre-processing 2 audio files... This may take a moment.")
        self.fig.canvas.draw_idle()

        # Process both tracks sequentially (simpler and safe for memory)
        for i in range(2):
            path = self.tracks[i]['file']
            y_full, sr = librosa.load(path, sr=None, mono=True, res_type="kaiser_best")
            self.tracks[i]['y_full'] = y_full
            self.tracks[i]['sr'] = sr
            self.tracks[i]['duration'] = librosa.get_duration(y=y_full, sr=sr)

            # HPSS on entire audio for CQT only
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

            cqt_data = self.cqt.compute_cqt_db(
                y_harmonic, sr,
                bins_per_octave=int(self.slider.val),
                hop_length=int(self.hpss_hop_length_slider.val),
                n_bins=84
            )
            self.tracks[i]['cqt'] = cqt_data.astype(np.float32)

        # set global sr to the first track's sr (used only for default fps calc)
        self.sr = self.tracks[0]['sr']
        # Seek slider max is the max duration among both tracks
        self.duration_sec = max(self.tracks[0]['duration'], self.tracks[1]['duration'])

        if self.seek_slider is not None:
            self._is_seeking = True
            try:
                self.seek_slider.valmax = self.duration_sec
                self.seek_slider.ax.set_xlim(self.seek_slider.valmin, self.seek_slider.valmax)
                # Preserve current time if available
                t_restore = 0.0
                try:
                    t_restore = float(self._progress_thread.get_time()) if GUI_AVAILABLE else 0.0
                except Exception:
                    t_restore = 0.0
                t_restore = max(0.0, min(float(t_restore), float(self.duration_sec)))
                self.seek_slider.set_val(t_restore)
            finally:
                self._is_seeking = False

        logger.info("Pre-processing complete for two files. Spectrogram data ready.")
        base_a = os.path.basename(self.tracks[0]['file'])
        base_b = os.path.basename(self.tracks[1]['file'])
        self.ax.set_title(
            f"A: {base_a}  |  B: {base_b}  —  BPO={int(self.slider.val)}, Hop={int(self.hpss_hop_length_slider.val)}"
        )
        self._is_processing = False
        self.fig.canvas.draw_idle()

    def _update_display(self):
        """
        Updates the display based on the playback time and pre-computed data.
        The playhead is now fixed to the right of the viewing window.
        """
        if self.im1 is None or self.im2 is None or self.fig is None or self.ax is None or self._closing or self._is_processing:
            return
        if self.tracks[0]['cqt'] is None or self.tracks[1]['cqt'] is None:
            return

        current_time = self._progress_thread.get_time()
        current_time = min(current_time, self.duration_sec)

        cqt_hop_length = int(self.hpss_hop_length_slider.val)

        # Helper to compute rolling buffer for a track
        def make_buffer(track_idx: int):
            cqt_full = self.tracks[track_idx]['cqt']
            sr_i = self.tracks[track_idx]['sr'] or 44100
            cqt_fps_i = sr_i / cqt_hop_length
            total_frames_i = cqt_full.shape[1]

            view_width_frames_i = int(self.view_duration_sec * cqt_fps_i)
            view_half_frames_i = int(view_width_frames_i / 2)
            current_frame_i = int(current_time * cqt_fps_i)

            view_start_i = max(0, current_frame_i - view_half_frames_i)
            view_end_i = view_start_i + view_width_frames_i
            if view_end_i > total_frames_i:
                view_end_i = total_frames_i
                view_start_i = max(0, view_end_i - view_width_frames_i)

            visible_i = cqt_full[:, view_start_i:view_end_i]
            buffer_i = np.full((cqt_full.shape[0], view_width_frames_i), -100.0)
            start_col_i = view_width_frames_i - visible_i.shape[1]
            buffer_i[:, start_col_i:] = visible_i
            return buffer_i, view_width_frames_i, view_start_i, view_end_i, cqt_fps_i, current_frame_i

        buf1, vw1, vs1, ve1, fps1, cf1 = make_buffer(0)
        buf2, vw2, vs2, ve2, fps2, cf2 = make_buffer(1)

        # For a single x-axis, use track 0's windowing
        try:
            self.im1.set_data(buf1)
            self.im1.set_clim(-80, 20)
            self.im2.set_data(buf2)
            self.im2.set_clim(-80, 20)

            # Compute per-pixel alpha masks for sharp mix: show only the stronger spectrogram per pixel
            if buf1.shape == buf2.shape:
                # Strength based on dB magnitude (higher is stronger)
                mask1 = (buf1 >= buf2)
                mask2 = ~mask1
                # Use 1.0/0.0 alpha for a crisp, non-averaging composite
                self.im1.set_alpha(mask1.astype(float))
                self.im2.set_alpha(mask2.astype(float))
            else:
                # If shapes mismatch (different SR/fps), fall back to equal alpha for safety
                # Users typically have the same SR; this avoids errors if not
                self.im1.set_alpha(0.5)
                self.im2.set_alpha(0.5)

            playhead_pos_in_window = cf1 - vs1
            self.playhead_line.set_xdata([playhead_pos_in_window, playhead_pos_in_window])

            start_time = vs1 / fps1
            end_time = ve1 / fps1
            self.ax.set_xticks(np.linspace(0, vw1, 5))
            self.ax.set_xticklabels([f"{t:.1f}s" for t in np.linspace(start_time, end_time, 5)])

            self.fig.canvas.draw_idle()
        except Exception as e:
            logger.warning("Failed to update display: %s", e, exc_info=False)


    # --------------------
    # UI and Playback
    # --------------------

    def _setup_display(self, initial_hop_length):
        self.fig.canvas.manager.set_window_title("Interactive CQT: Offline")
        self.colorbar_ax = self.fig.add_axes([0.935, 0.47, 0.02, 0.43])

        view_width_frames = int(self.view_duration_sec * (44100 / initial_hop_length))
        initial_data = np.full((84, view_width_frames), -100.0)

        # Two overlaid images with different color schemes
        # We'll use per-pixel alpha masks later to create a sharp mix (no averaging of colors)
        self.im1 = self.ax.imshow(initial_data, aspect='auto', origin='lower', cmap='magma', alpha=1.0, animated=True, vmin=-80, vmax=20)
        self.im2 = self.ax.imshow(initial_data, aspect='auto', origin='lower', cmap='twilight', alpha=0.0, animated=True, vmin=-80, vmax=20)
        plt.colorbar(self.im1, cax=self.colorbar_ax)

        self.ax.set_ylabel("CQT bins")
        self.ax.set_xlabel("Time")

        # Initial playhead is at the center of the window
        playhead_pos = view_width_frames / 2
        self.playhead_line = self.ax.axvline(playhead_pos, color="cyan", linewidth=1.5, zorder=10)


    def _ensure_mixer(self) -> bool:
        if not _HAS_PYGAME:
            return False
        try:
            desired_freq = int(self.sr or 44100)
            desired_size = -16  # 16-bit signed
            desired_channels = 2
            cur = pygame.mixer.get_init()
            if cur is None:
                pygame.mixer.init(frequency=desired_freq, size=desired_size, channels=desired_channels)
                return True
            # cur is a tuple (frequency, format, channels)
            cur_freq = int(cur[0])
            cur_channels = int(cur[2])
            if cur_freq != desired_freq or cur_channels != desired_channels:
                # Stop any playing sound and reinitialize mixer to match desired settings
                try:
                    if self._channel is not None:
                        try:
                            self._channel.stop()
                        except Exception:
                            pass
                    self._current_sound = None
                except Exception:
                    pass
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass
                pygame.mixer.init(frequency=desired_freq, size=desired_size, channels=desired_channels)
            return True
        except Exception:
            logger.exception("Failed to init/reinit pygame mixer")
            return False

    def _stop_playback(self):
        """
        Stops the playback but does not reset the progress time.
        This allows for seamless resuming after a seek.
        """
        if _HAS_PYGAME and pygame.mixer.get_init():
            try:
                if self._channel is not None:
                    try:
                        self._channel.stop()
                    except Exception:
                        pass
                self._current_sound = None
            except Exception:
                pass
        self.is_playing = False
        self.is_paused = False
        if GUI_AVAILABLE:
            self._progress_thread.set_playing(False)
            self._progress_thread.set_paused(False)

    def _play_current(self, start_sec: float = 0.0):
        # Ensure spectrograms and audio buffers are ready
        if self.tracks[0]['y_full'] is None or self.tracks[1]['y_full'] is None:
            logger.warning("Audio not ready yet. Cannot play.")
            return

        # Determine common sample rate for mixing (use first track's sr)
        sr0 = int(self.tracks[0]['sr'] or 44100)
        # Ensure mixer is initialized with the intended playback sample rate
        self.sr = sr0
        if not self._ensure_mixer():
            logger.info("pygame not available; cannot play audio")
            return
        sr1 = int(self.tracks[1]['sr'] or sr0)
        # If sr1 != sr0, resample track 1 to sr0 for playback mixing
        def resample_if_needed(y, srin, srout):
            if srin == srout:
                return y
            # Simple high-quality resample via librosa
            return librosa.resample(y, orig_sr=srin, target_sr=srout, res_type="kaiser_best")

        y0 = self.tracks[0]['y_full']
        y1 = resample_if_needed(self.tracks[1]['y_full'], sr1, sr0)
        len0 = y0.shape[0]
        len1 = y1.shape[0]
        max_len = max(len0, len1)

        # Compute start sample index
        start_sample = max(0, int(round(float(start_sec) * sr0)))
        if start_sample >= max_len:
            logger.info("Start position beyond audio length; nothing to play.")
            return

        # Prepare tails from start position
        tail0 = y0[start_sample:] if start_sample < len0 else np.zeros(0, dtype=y0.dtype)
        tail1 = y1[start_sample:] if start_sample < len1 else np.zeros(0, dtype=y1.dtype)

        # Pad shorter tail
        L = max(tail0.shape[0], tail1.shape[0])
        if tail0.shape[0] < L:
            tail0 = np.pad(tail0, (0, L - tail0.shape[0]))
        if tail1.shape[0] < L:
            tail1 = np.pad(tail1, (0, L - tail1.shape[0]))

        # Mix and normalize to int16 for pygame
        mix = tail0.astype(np.float32) + tail1.astype(np.float32)
        max_abs = np.max(np.abs(mix)) if mix.size else 1.0
        if max_abs < 1e-6:
            scale = 1.0
        else:
            scale = 0.8 * (32767.0 / max_abs)
        mix_i16 = np.clip(mix * scale, -32768, 32767).astype(np.int16)

        # Create Sound and play on a channel
        try:
            # Adapt the array shape to the mixer channel configuration
            mixer_init = pygame.mixer.get_init()
            mixer_channels = 2
            if mixer_init is not None:
                try:
                    mixer_channels = int(mixer_init[2])
                except Exception:
                    mixer_channels = 2
            if mix_i16.ndim == 1 and mixer_channels == 2:
                # Duplicate mono to stereo for a stereo mixer
                mix_i16 = np.ascontiguousarray(np.column_stack((mix_i16, mix_i16)))
            elif mix_i16.ndim == 2 and mixer_channels == 1:
                # Downmix to mono if mixer is mono (future-proofing)
                mix_i16 = np.ascontiguousarray(mix_i16.mean(axis=1).astype(np.int16))
            else:
                mix_i16 = np.ascontiguousarray(mix_i16)

            snd = pygame.sndarray.make_sound(mix_i16)
            self._stop_playback()  # stop previous
            self._current_sound = snd
            self._channel = snd.play()
            self.is_playing = True
            self.is_paused = False
            if GUI_AVAILABLE:
                self._progress_thread.set_playing(True)
                self._progress_thread.set_paused(False)
        except Exception:
            logger.exception("Failed to start playback of mixed audio")
            self.is_playing = False

    def toggle_play(self, event=None):
        if not _HAS_PYGAME or self.tracks[0]['cqt'] is None or self.tracks[1]['cqt'] is None:
            return

        # Current desired time is the seek slider value
        t = self.seek_slider.val if self.seek_slider is not None else 0.0

        if not self.is_playing:
            self._play_current(start_sec=t)
            if self.play_button:
                self.play_button.label.set_text("Pause")
        else:
            if not self.is_paused:
                if self._channel is not None:
                    try:
                        self._channel.pause()
                    except Exception:
                        pass
                self.is_paused = True
                if GUI_AVAILABLE:
                    self._progress_thread.set_paused(True)
                if self.play_button:
                    self.play_button.label.set_text("Resume")
            else:
                # Resume from current slider position (in case the user sought while paused)
                self._play_current(start_sec=t)
                if self.play_button:
                    self.play_button.label.set_text("Pause")


    def _apply_seek(self, t_sec: float):
        if self._is_seeking:
            return

        t_sec = max(0.0, min(float(t_sec), float(self.duration_sec)))
        if GUI_AVAILABLE:
            self._progress_thread.set_time(t_sec)

        # If we are currently playing, we must restart mixed playback from the new position
        if self.is_playing and not self.is_paused:
            self._play_current(start_sec=t_sec)

        # Explicitly update the display here to prevent lag
        self._update_display()

        logger.info("Seeking to %.2f s.", t_sec)

    def _trigger_recompute(self):
        """Called when parameters are changed to trigger a full re-processing."""
        if self._closing:
            return
        logger.info("Parameters changed. Triggering a full re-compute.")

        # Remember current playback time and state
        try:
            cur_time = self.seek_slider.val if self.seek_slider is not None else 0.0
        except Exception:
            cur_time = 0.0
        if GUI_AVAILABLE and cur_time == 0.0:
            try:
                cur_time = float(self._progress_thread.get_time())
            except Exception:
                pass
        was_playing = bool(self.is_playing and not self.is_paused)
        was_paused = bool(self.is_paused)

        # Stop playback without resetting time
        self._stop_playback()

        # Recompute
        self._pre_process_audio()

        # Clamp time to new duration and restore
        cur_time = max(0.0, min(float(cur_time), float(self.duration_sec)))
        if GUI_AVAILABLE:
            try:
                self._progress_thread.set_time(cur_time)
            except Exception:
                pass
        if self.seek_slider is not None:
            self._is_seeking = True
            try:
                # Ensure slider max reflects new duration before setting value
                self.seek_slider.valmax = self.duration_sec
                self.seek_slider.ax.set_xlim(self.seek_slider.valmin, self.seek_slider.valmax)
                self.seek_slider.set_val(cur_time)
            finally:
                self._is_seeking = False

        # Resume playback if it was playing before
        if was_playing:
            self._play_current(start_sec=cur_time)
            if self.play_button:
                self.play_button.label.set_text("Pause")
        else:
            # Maintain button label consistent with paused/stopped state
            if self.play_button:
                if was_paused:
                    self.play_button.label.set_text("Resume")
                else:
                    self.play_button.label.set_text("Play")

        # Update the display
        self._update_display()


    # --------------------
    # Main run
    # --------------------
    def run(self, initial_bpo: int = 10):
        self.files = self.list_mp3()
        if not self.files:
            raise FileNotFoundError(f"No .mp3 files found in '{self.playlist_dir}'")
        if len(self.files) == 1:
            # Duplicate the only file if just one exists so UI still works
            last = self._full_path(self.files[-1])
            self.tracks[0]['file'] = last
            self.tracks[1]['file'] = last
            self.current_file = last
        else:
            self.tracks[0]['file'] = self._full_path(self.files[-2])
            self.tracks[1]['file'] = self._full_path(self.files[-1])
            self.current_file = self.tracks[1]['file']

        if not GUI_AVAILABLE:
            print("This script now requires an interactive GUI backend to run.")
            return

        self._ensure_mixer()

        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        plt.subplots_adjust(left=0.08, right=0.88, top=0.90, bottom=0.45)

        # Now create sliders and buttons, so they have an axes to attach to
        ax_slider_bpo = self.fig.add_axes([0.08, 0.35, 0.40, 0.025])
        self.slider = Slider(ax=ax_slider_bpo, label="Bins/Octave", valmin=10, valmax=48, valinit=float(initial_bpo), valstep=1)

        ax_slider_kernel = self.fig.add_axes([0.08, 0.30, 0.40, 0.025])
        self.hpss_kernel_size_slider = Slider(ax=ax_slider_kernel, label="HPSS Kernel", valmin=1, valmax=84, valinit=self.hpss_kernel_size, valstep=2)

        ax_slider_power = self.fig.add_axes([0.08, 0.25, 0.40, 0.025])
        self.hpss_power_slider = Slider(ax=ax_slider_power, label="HPSS Power", valmin=1.0, valmax=3.0, valinit=self.hpss_power, valstep=0.1)

        ax_slider_hop = self.fig.add_axes([0.08, 0.20, 0.40, 0.025])
        self.hpss_hop_length_slider = Slider(ax=ax_slider_hop, label="Hop Length", valmin=16, valmax=512, valinit=self.hop_length, valstep=8)

        ax_button_margin = self.fig.add_axes([0.5, 0.35, 0.1, 0.04])
        self.hpss_margin_button = Button(ax_button_margin, f"Margin: {self.hpss_margin_options[self.hpss_margin_index]}")

        self.seek_ax = self.fig.add_axes([0.5, 0.25, 0.40, 0.04])
        self.seek_slider = Slider(ax=self.seek_ax, label="Seek", valmin=0.0, valmax=1.0, valinit=0.0) # Placeholder max value

        play_ax = self.fig.add_axes([0.5, 0.15, 0.07, 0.04])
        self.play_button = Button(play_ax, "Play" if _HAS_PYGAME else "NoAudio")

        # Now that the sliders and axes exist, set up the initial display
        self._setup_display(self.hpss_hop_length_slider.val)

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
            # Stop when reaching the end of the longer track
            if t >= self.duration_sec and self.is_playing:
                self._stop_playback()
                if self.play_button:
                    self.play_button.label.set_text("Play")
                t = self.duration_sec
                self._progress_thread.set_time(t)

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
    viewer.run(initial_bpo=12)
