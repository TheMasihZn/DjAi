import os
import numpy as np
import librosa
import librosa.display
import matplotlib
# Try to ensure a GUI backend is selected before importing pyplot
try:
    _backend = matplotlib.get_backend().lower()
except Exception:
    _backend = ''
_gui_backends = ['tkagg', 'qt5agg', 'qtagg', 'wxagg', 'gtk3agg', 'macosx']
if not any(b in _backend for b in _gui_backends):
    try:
        matplotlib.use('TkAgg', force=True)
    except Exception:
        # If TkAgg is not available, keep the existing backend
        pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import time
import threading

# Optional audio playback via pygame (MP3 support)
try:
    import pygame  # type: ignore
    _HAS_PYGAME = True
except Exception:
    pygame = None
    _HAS_PYGAME = False


def find_last_mp3(playlist_dir: str) -> str:
    """
    Return the last .mp3 filename in the directory using the alphanumeric sort.
    If no mp3 is found, it raises FileNotFoundError.
    """
    files = [f for f in os.listdir(playlist_dir) if f.lower().endswith('.mp3')]
    if not files:
        raise FileNotFoundError(f"No .mp3 files found in '{playlist_dir}'")
    files.sort()  # alphanumeric sort; 'last' is the final element
    return os.path.join(playlist_dir, files[-1])


from gpu_audio import compute_cqt_db as _compute_cqt_db_gpu_aware

def compute_cqt_db(y_harmonic: np.ndarray, sr: int, bins_per_octave: int, hop_length: int = 64) -> np.ndarray:
    """
    Compute CQT (expects a harmonic signal) and return decibel-scaled magnitude.
    Uses GPU acceleration via torch+torchaudio if available; otherwise falls back to CPU (librosa).
    """
    # Keep n_bins at 84 to match librosa defaults, ensuring consistent display across backends
    return _compute_cqt_db_gpu_aware(y_harmonic, sr, bins_per_octave=bins_per_octave, hop_length=hop_length, n_bins=84)


def interactive_cqt_for_last_track(playlist_dir: str = 'playlist') -> None:
    """
    Open an interactive window with a selector for .mp3 files and a slider to adjust bins_per_octave.
    Defaults to the last .mp3 initially.
    Adds a Play/Pause control to play the selected audio.
    """
    # 1) Collect mp3 list and pick initial selection (last)
    files = [f for f in os.listdir(playlist_dir) if f.lower().endswith('.mp3')]
    if not files:
        raise FileNotFoundError(f"No .mp3 files found in '{playlist_dir}'")
    files.sort()
    def _full(p):
        return os.path.join(playlist_dir, p)
    current_file = _full(files[-1])

    print(f"Loading track: {os.path.basename(current_file)}")

    # 2) Prepare figure and axes with fixed layout areas
    # Make the window shorter to reduce vertical size while keeping readability
    fig, ax = plt.subplots(figsize=(14, 5.2))
    try:
        # Set a descriptive window title if backend supports it
        fig.canvas.manager.set_window_title('Interactive CQT: Select track and adjust BPO')
    except Exception:
        pass

    # Main plot and controls layout
    # Compact layout: main area for spectrogram, right for colorbar, bottom for slider; a small dropdown will overlay the window
    plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.30)
    # Fixed colorbar axes (to avoid layout changes)
    colorbar_ax = fig.add_axes([0.935, 0.32, 0.02, 0.56])

    # Give immediate visual feedback while loading audio (non-blocking show)
    try:
        ax.text(0.5, 0.5, 'Loading audio…', ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw_idle()
        plt.show(block=False)
    except Exception:
        pass

    # Now load the entire track to display full duration (can be slow)
    y, sr = librosa.load(current_file)
    # Precompute harmonic component once (avoid repeating HPSS on slider moves)
    y_harmonic, _ = librosa.effects.hpss(y)

    # 3) Slider axis for bins_per_octave
    slider_ax = fig.add_axes([0.12, 0.08, 0.78, 0.06])
    # Reasonable range: from 12 (semitone resolution) to 192 (fine resolution)
    slider = Slider(ax=slider_ax, label='Bins per Octave', valmin=10, valmax=100, valinit=48, valstep=1)

    # Track duration and progress state
    duration_sec = float(librosa.get_duration(y=y, sr=sr))
    last_time_sec = 0.0
    progress_line = None
    # Blitting/state for high-FPS indicator updates
    _blit_enabled = True
    _bg_cache = None  # cached background for blitting
    _last_drawn_t = -1.0  # last x position rendered
    _t0_perf = None  # perf counter timestamp when (re)started
    _play_start_pos_sec = 0.  # accumulated playhead at last (re)start
    # Threaded progress computation state
    _thread_time_sec = 0.0
    _thread_track_ended = False
    _thread_stop_evt = threading.Event()
    _thread_lock = threading.Lock()
    _progress_thread = None

    # 3a) Seek slider (time)
    seek_ax = fig.add_axes([0.12, 0.18, 0.78, 0.06])
    seek_slider = Slider(ax=seek_ax, label='Seek (s)', valmin=0.0, valmax=duration_sec, valinit=0.0, valstep=0.01)
    _seek_user_dragging = False
    _seek_internal_update = False

    # 3b) Play/Pause Button
    play_ax = fig.add_axes([0.03, 0.08, 0.07, 0.06])
    play_label = 'Play' if _HAS_PYGAME else 'NoAudio'
    play_button = Button(play_ax, play_label)

    # 4) Initial CQT plot
    hop_length = 64
    colorbar_ref = None

    # Dynamic layout positions; will be adjusted after selector is decided
    main_ax_pos = [0.08, 0.32, 0.80, 0.56]
    colorbar_pos = [0.935, 0.32, 0.02, 0.56]

    # Wide image cache settings/state
    _pixels_per_second = 200  # very wide image: increase to make aspect ~1:200
    _wide_img = None
    _wide_img_path = None
    _vmin_vmax = None  # for consistent color scaling
    _last_xlim = None

    def _cache_dir(bpo: int) -> str:
        base = os.path.join('output', f'cqt-hop{hop_length}-hpss-bpo{int(bpo)}')
        try:
            os.makedirs(base, exist_ok=True)
        except Exception:
            pass
        return base

    def _wide_cache_path(bpo: int) -> str:
        name = os.path.splitext(os.path.basename(current_file))[0]
        return os.path.join(_cache_dir(bpo), f"{name}_wide.jpg")

    def _resample_time(arr: np.ndarray, target_frames: int) -> np.ndarray:
        # arr shape: (n_bins, n_frames)
        n_bins, n_frames = arr.shape
        if target_frames <= 0 or n_frames == target_frames:
            return arr
        x_src = np.linspace(0.0, 1.0, n_frames)
        x_tgt = np.linspace(0.0, 1.0, target_frames)
        # interpolate along time axis for each bin
        out = np.empty((n_bins, target_frames), dtype=float)
        for i in range(n_bins):
            out[i] = np.interp(x_tgt, x_src, arr[i])
        return out

    def _ensure_wide_cache(bpo: int):
        nonlocal _wide_img_path, _vmin_vmax
        _wide_img_path = _wide_cache_path(bpo)
        if not os.path.exists(_wide_img_path):
            # Compute CQT in dB and save as a very wide image
            C_db = compute_cqt_db(y_harmonic, sr, bins_per_octave=int(bpo), hop_length=hop_length)
            # Determine color scaling using percentiles for robustness
            vmin = float(np.percentile(C_db, 5.0))
            vmax = float(np.percentile(C_db, 99.5))
            _vmin_vmax = (vmin, vmax)
            # Compute target width
            try:
                width_px = max(1, int(float(duration_sec) * float(_pixels_per_second)))
            except Exception:
                width_px = max(1, int(C_db.shape[1] * 4))
            C_rs = _resample_time(C_db, width_px)
            # Save as jpg using consistent colormap
            try:
                plt.imsave(_wide_img_path, C_rs, cmap='magma', vmin=vmin, vmax=vmax, origin='lower', format='jpg')
            except Exception:
                # Fallback: save without vmin/vmax
                plt.imsave(_wide_img_path, C_rs, cmap='magma', origin='lower', format='jpg')
        else:
            if _vmin_vmax is None:
                _vmin_vmax = None

    def _load_wide_image(bpo: int):
        nonlocal _wide_img
        _ensure_wide_cache(bpo)
        try:
            _wide_img = plt.imread(_wide_img_path)
        except Exception as e:
            print(f"[CACHE] Failed to load wide image, regenerating: {e}")
            try:
                if os.path.exists(_wide_img_path):
                    os.remove(_wide_img_path)
            except Exception:
                pass
            _ensure_wide_cache(bpo)
            try:
                _wide_img = plt.imread(_wide_img_path)
            except Exception:
                _wide_img = None

    def plot_cqt(bpo: int):
        nonlocal colorbar_ref, progress_line, _bg_cache, _last_drawn_t
        ax.clear()
        # Ensure the main spectrogram axes fill the plotting area (leave room for colorbar)
        ax.set_position(main_ax_pos)  # [left, bottom, width, height]
        # Ensure colorbar axes follow the chosen layout
        try:
            colorbar_ax.set_position(colorbar_pos)
        except Exception:
            pass
        C_db = compute_cqt_db(y_harmonic, sr, bins_per_octave=int(bpo), hop_length=hop_length)
        img = librosa.display.specshow(C_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='cqt_note', ax=ax)
        ax.set_title(f"{os.path.basename(current_file)} - Harmonic CQT (bins_per_octave={int(bpo)})")
        # Keep a persistent colorbar in a fixed axe to avoid layout changes
        if colorbar_ref is None:
            colorbar_ref = fig.colorbar(img, cax=colorbar_ax, format='%+2.0f dB')
        else:
            colorbar_ref.update_normal(img)
        # Recreate or update the progress line at the last known time
        try:
            t = float(last_time_sec)
        except Exception:
            t = 0.0
        try:
            progress_line = ax.axvline(t, color='cyan', linewidth=1.5, alpha=0.9)
            try:
                progress_line.set_animated(True)
            except Exception:
                pass
        except Exception:
            progress_line = None
        # Force a full draw to (re)create the background cache
        try:
            fig.canvas.draw()
        except Exception:
            pass
        _bg_cache = None
        _last_drawn_t = -1.0

    # Initial draw moved to after selector setup to avoid overlap

    # 5) Update function on slider change
    def on_change(val):
        # Replot and reset blit cache
        plot_cqt(int(val))

    slider.on_changed(on_change)

    # Playback state and helpers
    is_playing = False
    is_paused = False

    def _ensure_mixer():
        if not _HAS_PYGAME:
            return False
        try:
            desired_freq = 44100
            desired_size = -16
            desired_channels = 2
            cur = pygame.mixer.get_init()
            if cur is None:
                pygame.mixer.init(frequency=desired_freq, size=desired_size, channels=desired_channels)
            else:
                cur_freq, _, cur_channels = int(cur[0]), cur[1], int(cur[2])
                if cur_freq != desired_freq or cur_channels != desired_channels:
                    try:
                        pygame.mixer.quit()
                    except Exception:
                        pass
                    pygame.mixer.init(frequency=desired_freq, size=desired_size, channels=desired_channels)
            # Limit to a single channel for music to avoid unexpected overlaps
            try:
                pygame.mixer.set_num_channels(1)
            except Exception:
                pass
        except Exception as e:
            print(f"[AUDIO] Failed to initialize audio mixer: {e}")
            return False
        return True

    def _stop_playback():
        nonlocal is_playing, is_paused, last_time_sec, progress_line, _bg_cache, _t0_perf, _play_start_pos_sec, _thread_time_sec, _thread_track_ended
        if _HAS_PYGAME and pygame.mixer.get_init():
            try:
                # Stop music channel explicitly
                pygame.mixer.music.stop()
            except Exception:
                pass
            try:
                # Also stop any other channels to avoid overlap from stray sounds
                pygame.mixer.stop()
            except Exception:
                pass
            try:
                # Try to unload current music to release decoder/stream
                if hasattr(pygame.mixer.music, 'unload'):
                    pygame.mixer.music.unload()  # type: ignore[attr-defined]
            except Exception:
                pass
        is_playing = False
        is_paused = False
        # Reset progress
        last_time_sec = 0.0
        _t0_perf = None
        _play_start_pos_sec = 0.0
        _thread_track_ended = False
        try:
            with _thread_lock:
                _thread_time_sec = 0.0
        except Exception:
            pass
        try:
            if progress_line is not None and progress_line.axes is not None:
                progress_line.set_xdata([0.0, 0.0])
            _bg_cache = None
        except Exception:
            pass
        # Reset seek slider programmatically
        try:
            nonlocal _seek_internal_update
            _seek_internal_update = True
            seek_slider.set_val(0.0)
        except Exception:
            pass
        finally:
            _seek_internal_update = False
        if _HAS_PYGAME:
            try:
                play_button.label.set_text('Play')
                fig.canvas.draw_idle()
            except Exception:
                pass

    def _play_current():
        nonlocal is_playing, is_paused, last_time_sec, progress_line, _t0_perf, _play_start_pos_sec, _thread_time_sec, _thread_track_ended
        if not _ensure_mixer():
            print("[AUDIO] pygame is not available. Install pygame to enable playback.")
            return
        # Ensure no previous playback is lingering
        _stop_playback()
        # Start (or restart) from the current last_time_sec position
        start_t = max(0.0, float(last_time_sec))
        _play_start_pos_sec = float(start_t)
        _t0_perf = time.perf_counter()
        _thread_track_ended = False
        try:
            with _thread_lock:
                _thread_time_sec = start_t
        except Exception:
            pass
        try:
            if progress_line is not None and progress_line.axes is not None:
                progress_line.set_xdata([start_t, start_t])
            # Invalidate blit cache; will be recreated on next draw
            _bg_cache = None
        except Exception:
            pass
        try:
            pygame.mixer.music.load(current_file)
            try:
                pygame.mixer.music.play(start=start_t)
            except TypeError:
                # For older pygame versions without 'start' for mp3, approximate by restarting and fast-forwarding is unsupported; fall back to from-beginning if start_t==0
                if start_t > 0.05:
                    print("[AUDIO] Precise seek not supported by this pygame/codec. Starting from beginning.")
                pygame.mixer.music.play()
            is_playing = True
            is_paused = False
            play_button.label.set_text('Pause')
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"[AUDIO] Failed to play '{current_file}': {e}")
            is_playing = False
            is_paused = False

    def _toggle_play(event):
        nonlocal is_playing, is_paused
        if not _HAS_PYGAME:
            print("[AUDIO] pygame not installed. Can't play audio.")
            return
        # Simple reentrancy/debounce guard to avoid double-trigger on click/release
        if getattr(_toggle_play, '_busy', False):
            return
        setattr(_toggle_play, '_busy', True)
        try:
            if not is_playing:
                _play_current()
            else:
                if not is_paused:
                    pygame.mixer.music.pause()
                    is_paused = True
                    play_button.label.set_text('Resume')
                    fig.canvas.draw_idle()
                else:
                    pygame.mixer.music.unpause()
                    is_paused = False
                    # On resume, set perf anchor so we smoothly continue from last_time_sec
                    _play_start_pos_sec = float(last_time_sec)
                    _t0_perf = time.perf_counter()
                    play_button.label.set_text('Pause')
                    fig.canvas.draw_idle()
        except Exception as e:
            print(f"[AUDIO] Playback error: {e}")
        finally:
            setattr(_toggle_play, '_busy', False)

    play_button.on_clicked(_toggle_play)

    # Threaded progress updater
    def _progress_worker():
        nonlocal _thread_time_sec, _thread_track_ended, _t0_perf, _play_start_pos_sec
        last_ref_sync = 0.0
        while not _thread_stop_evt.is_set():
            try:
                if _seek_user_dragging:
                    # While dragging, hold at preview time
                    t = float(last_time_sec)
                else:
                    t = float(last_time_sec)
                    if is_playing and not is_paused:
                        # Smooth based on perf counter
                        t_smooth = float(_play_start_pos_sec)
                        if _t0_perf is not None:
                            t_smooth += max(0.0, time.perf_counter() - _t0_perf)
                        # Periodically sync with mixer position to reduce drift
                        t_ref = None
                        now = time.perf_counter()
                        if now - last_ref_sync >= 0.05:  # every 50ms
                            last_ref_sync = now
                            try:
                                if _HAS_PYGAME and pygame is not None and pygame.mixer.get_init():
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
                    # else: keep last_time_sec
                # Clamp
                try:
                    if duration_sec is not None and t > duration_sec:
                        t = duration_sec
                except Exception:
                    pass
                with _thread_lock:
                    _thread_time_sec = t
                # Track-end detection
                try:
                    if _HAS_PYGAME and pygame is not None and pygame.mixer.get_init():
                        if is_playing and not is_paused and not pygame.mixer.music.get_busy():
                            _thread_track_ended = True
                except Exception:
                    pass
            except Exception:
                # Avoid crashing the thread
                pass
            # High-frequency update without busy wait
            time.sleep(0.005)  # 200 Hz loop

    def _start_progress_thread():
        nonlocal _progress_thread
        if _progress_thread is None or not _progress_thread.is_alive():
            _thread_stop_evt.clear()
            _progress_thread = threading.Thread(target=_progress_worker, name='ProgressUpdater', daemon=True)
            _progress_thread.start()

    def _stop_progress_thread():
        if _progress_thread is not None and _progress_thread.is_alive():
            _thread_stop_evt.set()
            try:
                _progress_thread.join(timeout=0.5)
            except Exception:
                pass

    # Seek handling
    _seek_pending_time = None  # store latest target while dragging

    def _clamp_time(t_sec: float) -> float:
        try:
            if duration_sec is not None:
                return min(max(0.0, float(t_sec)), float(duration_sec))
        except Exception:
            pass
        return max(0.0, float(t_sec))

    def _preview_seek(t_sec: float):
        nonlocal last_time_sec, _seek_pending_time
        t_sec = _clamp_time(t_sec)
        _seek_pending_time = t_sec
        last_time_sec = float(t_sec)
        # Update indicator immediately without touching audio
        try:
            if progress_line is not None and progress_line.axes is not None:
                progress_line.set_xdata([t_sec, t_sec])
                # Try to use blitting for responsiveness
                if _blit_enabled and _bg_cache is not None:
                    try:
                        canvas = fig.canvas
                        canvas.restore_region(_bg_cache)
                        ax.draw_artist(progress_line)
                        canvas.blit(ax.bbox)
                    except Exception:
                        pass
        except Exception:
            pass

    def _apply_seek_to_audio(t_sec: float):
        nonlocal last_time_sec, _t0_perf, _play_start_pos_sec, _thread_time_sec, _thread_track_ended
        t_sec = _clamp_time(t_sec)
        last_time_sec = float(t_sec)
        # Adjust smoothing anchors
        _play_start_pos_sec = float(t_sec)
        _t0_perf = time.perf_counter()
        # Update indicator
        try:
            if progress_line is not None and progress_line.axes is not None:
                progress_line.set_xdata([t_sec, t_sec])
        except Exception:
            pass
        _thread_track_ended = False
        try:
            with _thread_lock:
                _thread_time_sec = t_sec
        except Exception:
            pass
        # Perform a single audio seek only once (on mouse release), but only if currently playing/paused
        if is_playing and _HAS_PYGAME and pygame is not None:
            try:
                if pygame.mixer.get_init():
                    try:
                        # Pause if needed before reloading
                        pygame.mixer.music.pause()
                    except Exception:
                        pass
                    try:
                        pygame.mixer.music.load(current_file)
                        try:
                            pygame.mixer.music.play(start=t_sec)
                            # If we were paused before seeking, remain paused
                            if is_paused:
                                pygame.mixer.music.pause()
                        except TypeError:
                            # Fallback: restart from beginning if precise seek unsupported
                            if t_sec > 0.05:
                                print("[AUDIO] Precise seek not supported by this pygame/codec. Restarting from beginning.")
                            pygame.mixer.music.play()
                            if is_paused:
                                pygame.mixer.music.pause()
                    except Exception:
                        pass
            except Exception:
                pass
        # Reflect on seek slider programmatically
        try:
            nonlocal _seek_internal_update
            _seek_internal_update = True
            seek_slider.set_val(t_sec)
        finally:
            _seek_internal_update = False

    def _on_seek_changed(val):
        # Ignore programmatic updates
        try:
            if _seek_internal_update:
                return
        except Exception:
            pass
        # While dragging: only preview; do not touch audio
        try:
            if _seek_user_dragging:
                _preview_seek(val)
        except Exception:
            pass

    def _on_mouse_press(event):
        nonlocal _seek_user_dragging
        if event.inaxes is seek_ax:
            _seek_user_dragging = True

    def _on_mouse_release(event):
        nonlocal _seek_user_dragging, _seek_pending_time
        if _seek_user_dragging:
            _seek_user_dragging = False
            try:
                if _seek_pending_time is not None:
                    _apply_seek_to_audio(_seek_pending_time)
            finally:
                _seek_pending_time = None

    seek_slider.on_changed(_on_seek_changed)
    try:
        fig.canvas.mpl_connect('button_press_event', _on_mouse_press)
        fig.canvas.mpl_connect('button_release_event', _on_mouse_release)
    except Exception:
        pass

    # Timer to update the playback progress indicator
    try:
        update_interval_ms = 16  # ~60 FPS for ultra-smooth indicator updates
        progress_timer = fig.canvas.new_timer(interval=update_interval_ms)
    except Exception:
        progress_timer = None

    def _update_progress():
        nonlocal last_time_sec, progress_line, is_playing, is_paused, _bg_cache, _last_drawn_t, _blit_enabled, _seek_internal_update, _seek_user_dragging, seek_slider, _thread_time_sec, _thread_track_ended
        # Read the latest time computed by the background thread
        try:
            with _thread_lock:
                t = float(_thread_time_sec)
        except Exception:
            t = float(last_time_sec)
        # Clamp to duration
        try:
            if duration_sec is not None and t > duration_sec:
                t = duration_sec
        except Exception:
            pass
        last_time_sec = t
        # Reflect on seek slider unless user is dragging it
        try:
            if not _seek_user_dragging:
                _seek_internal_update = True
                seek_slider.set_val(t)
        finally:
            _seek_internal_update = False
        # Move the vertical line with blitting
        try:
            if progress_line is not None and progress_line.axes is not None:
                progress_line.set_xdata([t, t])
                if _blit_enabled:
                    canvas = fig.canvas
                    if _bg_cache is None:
                        try:
                            _bg_cache = canvas.copy_from_bbox(ax.bbox)
                        except Exception:
                            _blit_enabled = False
                    if _bg_cache is not None and _blit_enabled:
                        try:
                            canvas.restore_region(_bg_cache)
                            ax.draw_artist(progress_line)
                            canvas.blit(ax.bbox)
                            _last_drawn_t = t
                            return
                        except Exception:
                            _blit_enabled = False
        except Exception:
            pass
        # Fallback redraw
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
        # Handle end-of-track signal from worker
        try:
            if _thread_track_ended:
                _thread_track_ended = False
                is_playing = False
                try:
                    play_button.label.set_text('Play')
                except Exception:
                    pass
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass
        except Exception:
            pass

    try:
        # Start background progress computation thread
        _start_progress_thread()
    except Exception:
        pass
    try:
        if progress_timer is not None:
            progress_timer.add_callback(_update_progress)
            progress_timer.start()
    except Exception:
        pass

    # 6) File selector: use a small Tk OptionMenu dropdown (no radio buttons)
    # Show base names (without extension) for compact labels
    labels = [os.path.splitext(f)[0] for f in files]

    def on_select(label: str):
        nonlocal y, sr, y_harmonic, current_file, duration_sec, last_time_sec, _bg_cache, _last_drawn_t, _thread_time_sec, _thread_track_ended
        # Map label back to file name (by index)
        try:
            idx = labels.index(label)
        except ValueError:
            return
        # Stop current playback when switching files
        _stop_playback()
        current_file = _full(files[idx])
        print(f"Loading track: {os.path.basename(current_file)}")
        # Reload preview and recompute harmonic component
        # Load the entire track when switching to show full duration
        y, sr = librosa.load(current_file)
        y_harmonic, _ = librosa.effects.hpss(y)
        duration_sec = float(librosa.get_duration(y=y, sr=sr))
        last_time_sec = 0.0
        _thread_track_ended = False
        try:
            with _thread_lock:
                _thread_time_sec = 0.0
        except Exception:
            pass
        # Update seek slider range and reset to 0
        try:
            seek_slider.valmax = float(duration_sec)
            seek_ax.set_xlim(seek_slider.valmin, seek_slider.valmax)
            nonlocal _seek_internal_update
            _seek_internal_update = True
            seek_slider.set_val(0.0)
        finally:
            _seek_internal_update = False
        # Update window title if supported
        try:
            fig.canvas.manager.set_window_title(f"Interactive CQT: {os.path.basename(current_file)}")
        except Exception:
            pass
        # Redraw spectrogram with the current slider value
        plot_cqt(int(slider.val))
        _bg_cache = None
        _last_drawn_t = -1.0

    used_dropdown = False
    # Always prefer a dropdown (Tk OptionMenu); no radio buttons
    try:
        import tkinter as tk
        from tkinter import ttk
        # Access the Tk root window from the Matplotlib figure manager
        root = fig.canvas.manager.window  # type: ignore[attr-defined]
        if root is not None:
            sel_var = tk.StringVar(master=root, value=labels[-1])
            # Create a small button-like dropdown (OptionMenu) that expands on click
            option = ttk.OptionMenu(root, sel_var, labels[-1], *labels, command=lambda val: on_select(val))
            try:
                option.configure(width=18)
            except Exception:
                pass
            # Place the small button at the top-left corner of the window
            option.place(x=10, y=10)
            used_dropdown = True
    except Exception:
        used_dropdown = False
        print("[UI] Dropdown selector unavailable (tkinter not accessible). Track selection via UI is disabled.")

    # Keep a compact layout when using a small dropdown (or when dropdown unavailable)
    main_ax_pos = [0.08, 0.32, 0.80, 0.56]
    colorbar_pos = [0.935, 0.32, 0.02, 0.56]

    # Draw initial plot after layout is finalized
    plot_cqt(int(slider.val))

    # Recreate background cache when figure is redrawn/resized (ensures blit stays valid)
    def _on_draw(event=None):
        nonlocal _bg_cache
        try:
            _bg_cache = fig.canvas.copy_from_bbox(ax.bbox)
        except Exception:
            _bg_cache = None
    try:
        fig.canvas.mpl_connect('draw_event', _on_draw)
    except Exception:
        pass

    # Ensure mixer is quit when window closes
    def _on_close(event=None):
        try:
            _stop_playback()
        finally:
            try:
                if 'progress_timer' in locals() and progress_timer is not None:
                    progress_timer.stop()
            except Exception:
                pass
            try:
                _stop_progress_thread()
            except Exception:
                pass
            if _HAS_PYGAME and pygame.mixer.get_init():
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass

    try:
        fig.canvas.mpl_connect('close_event', _on_close)
    except Exception:
        pass

    # 7) Show interactive window (blocking to keep it open for retweaks)
    plt.show(block=True)


if __name__ == '__main__':
    # Run for default 'playlist' directory relative to this script
    interactive_cqt_for_last_track('playlist')
