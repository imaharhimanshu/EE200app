import librosa
import librosa.display
import numpy as np
import scipy.ndimage as ndimage
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Locked parameters — used for BOTH building the DB and querying ──
N_FFT          = 2048
HOP_LENGTH     = 512    # 512
NEIGHBORHOOD   = (20, 20)
PERCENTILE     = 75
FAN            = 5
MAX_DT         = 2.0
SAMPLE_RATE    = 22050          # fixed for everything — indexing AND querying

_db_cache = None

def load_db(db_path="database/song_db.pkl"):
    global _db_cache
    if _db_cache is None:
        with open(db_path, 'rb') as f:
            try:
                _db_cache = pickle.load(f)
            except Exception:
                f.seek(0)
                _db_cache = pickle.load(f, encoding='latin1')
    return _db_cache

def load_audio(path, duration=None):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=duration)
    return y, sr

def compute_spectrogram(y, sr):
    D    = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db

def get_peaks(S_db, sr):
    local_max   = ndimage.maximum_filter(S_db, size=NEIGHBORHOOD) == S_db
    loud_enough = S_db > np.percentile(S_db, PERCENTILE)
    peak_fi, peak_ti = np.where(local_max & loud_enough)
    times = librosa.frames_to_time(peak_ti, sr=sr, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)[peak_fi]
    return times, freqs

def build_hashes(peak_times, peak_freqs):
    peaks_sorted = sorted(zip(peak_times, peak_freqs), key=lambda x: x[0])
    hashes = []
    for i, (t1, f1) in enumerate(peaks_sorted):
        for j in range(1, FAN + 1):
            if i + j < len(peaks_sorted):
                t2, f2 = peaks_sorted[i + j]
                dt = round(t2 - t1, 3)
                if 0 < dt <= MAX_DT:
                    hashes.append(((round(f1, 1), round(f2, 1), dt), t1))
    return hashes

def fingerprint_file(path, duration=None):
    y, sr = load_audio(path, duration=duration)
    S_db = compute_spectrogram(y, sr)
    p_times, p_freqs = get_peaks(S_db, sr)
    hashes = build_hashes(p_times, p_freqs)
    return hashes, S_db, p_times, p_freqs

def build_database(songs_folder, db_path="database/song_db.pkl"):
    db = {}
    songs = [f for f in os.listdir(songs_folder)
             if f.lower().endswith(('.mp3', '.wav', '.flac'))]
    print(f"Found {len(songs)} songs")
    for fname in songs:
        fpath = os.path.join(songs_folder, fname)
        song_name = os.path.splitext(fname)[0]
        hashes, _, _, _ = fingerprint_file(fpath, duration=None)
        for hk, t in hashes:
            db.setdefault(hk, []).append((song_name, t))
        print(f"  {song_name}: {len(hashes)} hashes")
    with open(db_path, 'wb') as f:
        pickle.dump(db, f, protocol=2)
    print(f"Saved {len(db)} unique hash keys to {db_path}")
    return db

def plot_spectrogram(S_db, sr, ylim=3500, xlim=None):
    """
    Spectrogram plot in the same visual style as the reference snippet:
    magma colormap, dB colorbar, frequency-limited y-axis.
    Returns a matplotlib Figure (caller is responsible for st.pyplot / plt.close).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis='time', y_axis='hz', cmap='magma', ax=ax
    )
    ax.set_ylim(0, ylim)
    if xlim is not None:
        ax.set_xlim(0, xlim)
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.tight_layout()
    return fig

def plot_constellation(S_db, p_times, p_freqs, sr, ylim=3500, xlim=None):
    """
    Spectrogram + constellation peaks overlaid, in the same visual style as
    the reference snippet: magma background, cyan unfilled circle markers.
    Returns a matplotlib Figure (caller is responsible for st.pyplot / plt.close).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis='time', y_axis='hz', cmap='magma', ax=ax
    )
    ax.scatter(
        p_times, p_freqs,
        facecolors='none', edgecolors='cyan', s=30,
        label='Constellation Peaks'
    )
    ax.set_ylim(0, ylim)
    if xlim is not None:
        ax.set_xlim(0, xlim)
    ax.set_title('Spectrogram with Constellation Peaks')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig

def identify(query_path, db_path="database/song_db.pkl"):
    db = load_db(db_path)
    hashes, S_db, p_times, p_freqs = fingerprint_file(query_path, duration=None)

    offset_tallies = {}
    for hk, q_time in hashes:
        if hk in db:
            for song_name, db_time in db[hk]:
                offset = round(db_time - q_time, 2)
                offset_tallies.setdefault(song_name, {})
                offset_tallies[song_name][offset] = offset_tallies[song_name].get(offset, 0) + 1

    best_song, best_count = "Unknown", 0
    all_scores = {}
    for song, offsets in offset_tallies.items():
        top = max(offsets.values())
        all_scores[song] = top
        if top > best_count:
            best_count, best_song = top, song

    return best_song, S_db, p_times, p_freqs, all_scores, offset_tallies.get(best_song, {})
